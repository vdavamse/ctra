"""Tests for ModelStore versioning, artifact persistence, and promotion.

Uses ``tmp_path`` for all file I/O and mock objects for model artifacts
to avoid heavy ML dependencies.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ctra.mlops.model_store import ModelStore
from ctra.search.objectives import ObjectiveResult

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_registry(model_name: str = "xgboost") -> MagicMock:
    """Create a mock ModelRegistry that returns a picklable wrapper."""
    registry = MagicMock()
    # The wrapper must be picklable for joblib -- a simple dict works.
    wrapper = {"model_type": model_name, "params": {"n_estimators": 100}}

    def _get_wrapper(name: str) -> dict:
        if name == model_name:
            return wrapper
        raise KeyError(name)

    registry.get_wrapper.side_effect = _get_wrapper
    return registry


def _make_objective_result(
    accuracy: float = 0.82,
    parsimony: float = 0.75,
) -> ObjectiveResult:
    """Create a minimal ObjectiveResult."""
    return ObjectiveResult(
        values=np.array([accuracy, parsimony]),
        names=["accuracy", "parsimony"],
        details={"accuracy": accuracy, "parsimony": parsimony},
    )


def _make_feature_plans() -> list[dict[str, Any]]:
    return [
        {
            "name": "drug_target_count",
            "dtype": "int",
            "description": "Number of known drug targets",
            "extraction_prompt": "Count drug targets from ChEMBL.",
            "sources": ["chembl"],
        },
        {
            "name": "prior_phase_success_rate",
            "dtype": "float",
            "description": "Historical success rate for this phase",
            "extraction_prompt": "Look up success rates from CTG.",
            "sources": ["ctg"],
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    """save_model -> load_model produces an equivalent artifact."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        store = ModelStore(store_dir=str(tmp_path))
        registry = _make_mock_registry("xgboost")
        plans = _make_feature_plans()
        obj_result = _make_objective_result()

        version = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=plans,
            objective_result=obj_result,
        )

        wrapper, loaded_plans = store.load_model(version)
        assert wrapper["model_type"] == "xgboost"
        assert wrapper["params"]["n_estimators"] == 100
        assert len(loaded_plans) == 2
        assert loaded_plans[0]["name"] == "drug_target_count"
        assert loaded_plans[1]["name"] == "prior_phase_success_rate"

    def test_load_latest_without_explicit_version(self, tmp_path: Path) -> None:
        """load_model(version=None) loads the latest version."""
        store = ModelStore(store_dir=str(tmp_path))
        registry = _make_mock_registry("xgboost")
        plans = _make_feature_plans()

        v1 = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=plans,
            objective_result=_make_objective_result(accuracy=0.70),
        )

        # Small delay to ensure different timestamp
        time.sleep(1.1)

        v2 = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=plans,
            objective_result=_make_objective_result(accuracy=0.85),
        )

        assert v1 != v2
        wrapper, _ = store.load_model()  # should load v2
        assert wrapper["model_type"] == "xgboost"


class TestVersionStringFormat:
    """Verify the version string matches the expected pattern."""

    def test_version_format(self, tmp_path: Path) -> None:
        store = ModelStore(store_dir=str(tmp_path))
        registry = _make_mock_registry()

        version = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=_make_feature_plans(),
            objective_result=_make_objective_result(),
        )

        # Format: v_YYYYMMDDTHHMMSS
        pattern = r"^v_\d{8}T\d{6}$"
        assert re.match(pattern, version), (
            f"Version {version!r} does not match expected pattern {pattern}"
        )


class TestListVersions:
    """list_versions returns metadata dicts sorted newest first."""

    def test_sorted_newest_first(self, tmp_path: Path) -> None:
        store = ModelStore(store_dir=str(tmp_path))
        registry = _make_mock_registry()
        plans = _make_feature_plans()

        v1 = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=plans,
            objective_result=_make_objective_result(accuracy=0.70),
        )
        time.sleep(1.1)
        v2 = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=plans,
            objective_result=_make_objective_result(accuracy=0.85),
        )

        versions = store.list_versions()
        assert len(versions) == 2
        assert versions[0]["version"] == v2, "Newest version should be first"
        assert versions[1]["version"] == v1


class TestGetLatestVersion:
    """get_latest_version returns the newest version string."""

    def test_returns_newest(self, tmp_path: Path) -> None:
        store = ModelStore(store_dir=str(tmp_path))
        registry = _make_mock_registry()
        plans = _make_feature_plans()

        store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=plans,
            objective_result=_make_objective_result(),
        )
        time.sleep(1.1)
        v2 = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=plans,
            objective_result=_make_objective_result(),
        )

        assert store.get_latest_version() == v2

    def test_empty_store_raises(self, tmp_path: Path) -> None:
        store = ModelStore(store_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No model versions found"):
            store.get_latest_version()


class TestPromote:
    """Promotion sets stage in metadata.json and demotes previous."""

    def test_promote_sets_stage(self, tmp_path: Path) -> None:
        store = ModelStore(store_dir=str(tmp_path))
        registry = _make_mock_registry()

        version = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=_make_feature_plans(),
            objective_result=_make_objective_result(),
        )

        store.promote(version, stage="production")

        meta_path = tmp_path / version / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["stage"] == "production"

    def test_promote_demotes_previous_production(self, tmp_path: Path) -> None:
        store = ModelStore(store_dir=str(tmp_path))
        registry = _make_mock_registry()
        plans = _make_feature_plans()

        v1 = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=plans,
            objective_result=_make_objective_result(accuracy=0.70),
        )
        time.sleep(1.1)
        v2 = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=plans,
            objective_result=_make_objective_result(accuracy=0.85),
        )

        store.promote(v1, stage="production")
        store.promote(v2, stage="production")

        # v1 should now be demoted to "none"
        meta_v1_path = tmp_path / v1 / "metadata.json"
        with open(meta_v1_path) as f:
            meta_v1 = json.load(f)
        assert meta_v1["stage"] == "none", "Previous production version should be demoted to 'none'"

        # v2 should be "production"
        meta_v2_path = tmp_path / v2 / "metadata.json"
        with open(meta_v2_path) as f:
            meta_v2 = json.load(f)
        assert meta_v2["stage"] == "production"

    def test_promote_invalid_stage_raises(self, tmp_path: Path) -> None:
        store = ModelStore(store_dir=str(tmp_path))
        registry = _make_mock_registry()

        version = store.save_model(
            model_name="xgboost",
            registry=registry,
            feature_plans=_make_feature_plans(),
            objective_result=_make_objective_result(),
        )

        with pytest.raises(ValueError, match="Invalid stage"):
            store.promote(version, stage="invalid_stage")
