"""Tests for WrappedFeatureBuilder cache logic.

Covers:
- feature store integration: same plans -> same hash, different plans -> different hash
- __call__ store hit: write a JSON store file, verify it loads correctly
- __call__ store miss + exception: verify returns all-None dict
- upfront batch query short-circuit for compute_features
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

try:
    from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
    from ctra.agents.feature_builder import WrappedFeatureBuilder
    from ctra.agents.feature_store import _plan_content_hash, put_cached_feature

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan(name: str, idea: str = "test idea") -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=idea,
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[{"value": "1.0"}],
        possible_values={},
        feature_instructions="Extract value.",
    )


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / "feature_store"


@pytest.fixture()
def wrapper(store_dir: Path) -> WrappedFeatureBuilder:
    return WrappedFeatureBuilder(
        task_description="Predict trial success",
        feature_store_dir=store_dir,
        task_namespace="test",
        feature_store_enabled=True,
    )


# ======================================================================
# _plan_content_hash (per-feature)
# ======================================================================


class TestPlanContentHash:
    def test_same_plan_same_hash(self) -> None:
        plan_a = _make_plan("feat_a")
        hash1 = _plan_content_hash(plan_a)
        hash2 = _plan_content_hash(plan_a)
        assert hash1 == hash2

    def test_different_plans_different_hash(self) -> None:
        plan1 = _make_plan("feat_a")
        plan2 = _make_plan("feat_b")
        hash1 = _plan_content_hash(plan1)
        hash2 = _plan_content_hash(plan2)
        assert hash1 != hash2

    def test_different_idea_different_hash(self) -> None:
        plan1 = _make_plan("feat_a", idea="idea one")
        plan2 = _make_plan("feat_a", idea="idea two")
        hash1 = _plan_content_hash(plan1)
        hash2 = _plan_content_hash(plan2)
        assert hash1 != hash2

    def test_hash_is_16_char_hex(self) -> None:
        plan = _make_plan("feat_a")
        h = _plan_content_hash(plan)
        assert isinstance(h, str)
        # 16-char hex digest (64-bit key space)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# ======================================================================
# __call__ — store hit
# ======================================================================


class TestStoreHit:
    def test_loads_from_existing_store(
        self,
        wrapper: WrappedFeatureBuilder,
        store_dir: Path,
    ) -> None:
        """Pre-populate the store, then verify __call__ reads it."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctid = "NCT001"

        # Pre-populate store via put_cached_feature
        feature_values = {"feat_a": {"value": 1.5}}
        put_cached_feature(store_dir, "test", nctid, "feat_a", plan, feature_values)

        # Call the wrapper — should hit store
        result_nctid, result_values, result_meta = wrapper((nctid, plans))

        assert result_nctid == nctid
        # Note: result_values may be wrapped as {"feat_a": {...}} from the store hit
        assert result_values["feat_a"] == feature_values["feat_a"]
        assert result_meta == {}  # No meta on store hit

    def test_store_hit_does_not_call_builder(
        self,
        wrapper: WrappedFeatureBuilder,
        store_dir: Path,
    ) -> None:
        """On all-feature store hit, FeatureBuilder should not be instantiated."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctid = "NCT002"

        # Pre-populate store
        feature_values = {"feat_a": {"value": 2.0}}
        put_cached_feature(store_dir, "test", nctid, "feat_a", plan, feature_values)

        with patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls:
            wrapper((nctid, plans))
            mock_builder_cls.assert_not_called()


# ======================================================================
# __call__ — cache miss + exception
# ======================================================================


class TestCacheMissException:
    def test_exception_returns_all_none(
        self,
        wrapper: WrappedFeatureBuilder,
    ) -> None:
        """When FeatureBuilder raises, return all-None values."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctid = "NCT_FAIL"

        with (
            patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls,
            patch(
                "ctra.agents.feature_builder.ResettingRefine",
                side_effect=lambda module, **kw: module,
            ),
        ):
            mock_instance = MagicMock()
            mock_instance.side_effect = RuntimeError("LLM error")
            mock_builder_cls.return_value = mock_instance

            result_nctid, result_values, result_meta = wrapper((nctid, plans))

        assert result_nctid == nctid
        # All sub-feature keys should be None
        assert result_values == {"feat_a": {"value": None}}
        assert result_meta == {}

    def test_multi_plan_exception_returns_all_none(
        self,
        wrapper: WrappedFeatureBuilder,
    ) -> None:
        """Multiple plans with multi-valued types — all should be None on error."""
        plan_a = FeaturePlan(
            feature_name="drug_profile",
            feature_idea="Drug profile",
            feature_type={
                "mechanism": FeatureType.CATEGORICAL,
                "target_count": FeatureType.INTEGER,
            },
            data_sources=[FeatureSource.CHEMBL],
            example_values=[],
            possible_values={"mechanism": ["kinase", "antibody"]},
            feature_instructions="Look up drug profile.",
        )
        plan_b = _make_plan("safety_score")
        plans = {"drug_profile": plan_a, "safety_score": plan_b}
        nctid = "NCT_MULTI_FAIL"

        with (
            patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls,
            patch(
                "ctra.agents.feature_builder.ResettingRefine",
                side_effect=lambda module, **kw: module,
            ),
        ):
            mock_instance = MagicMock()
            mock_instance.side_effect = RuntimeError("LLM error")
            mock_builder_cls.return_value = mock_instance

            result_nctid, result_values, _result_meta = wrapper((nctid, plans))

        assert result_nctid == nctid
        assert result_values["drug_profile"] == {
            "mechanism": None,
            "target_count": None,
        }
        assert result_values["safety_score"] == {"value": None}
