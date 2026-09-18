"""Tests for the global feature value store (feature_store.py).

Covers:
- _plan_content_hash: stability, field-order invariance, REFINE divergence
- get_cached_feature / put_cached_feature: roundtrip, schema version
- get_cached_features_batch: mixed hits/misses
- get_store_stats: counting features and trials
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
from ctra.agents.feature_store import (
    _plan_content_hash,
    _sort_nested,
    get_cached_feature,
    get_cached_features_batch,
    get_store_stats,
    put_cached_feature,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan(
    name: str = "feat_a",
    idea: str = "test idea",
    instructions: str = "Extract value.",
) -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=idea,
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[{"value": "1.0"}],
        possible_values={},
        feature_instructions=instructions,
    )


# ======================================================================
# _plan_content_hash
# ======================================================================


class TestPlanContentHashStability:
    """Hash must be stable and deterministic."""

    def test_same_plan_same_hash(self) -> None:
        plan = _make_plan()
        h1 = _plan_content_hash(plan)
        h2 = _plan_content_hash(plan)
        assert h1 == h2

    def test_hash_is_16_char_hex(self) -> None:
        plan = _make_plan()
        h = _plan_content_hash(plan)
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_feature_name_affects_hash(self) -> None:
        plan1 = _make_plan(name="feat_a")
        plan2 = _make_plan(name="feat_b")
        assert _plan_content_hash(plan1) != _plan_content_hash(plan2)

    def test_feature_idea_affects_hash(self) -> None:
        plan1 = _make_plan(idea="idea one")
        plan2 = _make_plan(idea="idea two")
        assert _plan_content_hash(plan1) != _plan_content_hash(plan2)

    def test_feature_instructions_affects_hash(self) -> None:
        plan1 = _make_plan(instructions="instruction one")
        plan2 = _make_plan(instructions="instruction two")
        assert _plan_content_hash(plan1) != _plan_content_hash(plan2)

    def test_whitespace_normalized_in_idea(self) -> None:
        """Leading/trailing spaces should be stripped."""
        plan1 = _make_plan(idea="  idea  ")
        plan2 = _make_plan(idea="idea")
        assert _plan_content_hash(plan1) == _plan_content_hash(plan2)

    def test_whitespace_normalized_in_instructions(self) -> None:
        plan1 = _make_plan(instructions="  instruction  ")
        plan2 = _make_plan(instructions="instruction")
        assert _plan_content_hash(plan1) == _plan_content_hash(plan2)

    def test_example_values_do_not_affect_hash(self) -> None:
        """example_values are illustrative, not definitional."""
        plan1 = FeaturePlan(
            feature_name="feat_a",
            feature_idea="idea",
            feature_type={"value": FeatureType.FLOAT},
            data_sources=[FeatureSource.PUBMED],
            example_values=[{"value": "1.0"}],
            possible_values={},
            feature_instructions="Extract.",
        )
        plan2 = FeaturePlan(
            feature_name="feat_a",
            feature_idea="idea",
            feature_type={"value": FeatureType.FLOAT},
            data_sources=[FeatureSource.PUBMED],
            example_values=[{"value": "2.0"}],  # Different
            possible_values={},
            feature_instructions="Extract.",
        )
        assert _plan_content_hash(plan1) == _plan_content_hash(plan2)

    def test_refine_chain_diverges(self) -> None:
        """REFINE appends "\\n---\\nrefinement"; hash must differ."""
        original_idea = "original feature idea"
        refined_idea = f"{original_idea}\n---\nrefined variant"

        plan1 = _make_plan(idea=original_idea)
        plan2 = _make_plan(idea=refined_idea)

        hash1 = _plan_content_hash(plan1)
        hash2 = _plan_content_hash(plan2)
        assert hash1 != hash2


class TestSortNested:
    """Helper for sorting nested dicts."""

    def test_empty_none(self) -> None:
        assert _sort_nested(None) == {}

    def test_empty_dict(self) -> None:
        assert _sort_nested({}) == {}

    def test_single_list(self) -> None:
        result = _sort_nested({"a": ["z", "a"]})
        assert result == {"a": ["a", "z"]}

    def test_multiple_lists_sorted(self) -> None:
        result = _sort_nested({"z": ["2", "1"], "a": ["y", "x"]})
        assert result == {"a": ["x", "y"], "z": ["1", "2"]}


# ======================================================================
# get_cached_feature / put_cached_feature
# ======================================================================


class TestGetPutRoundtrip:
    """Basic store get/put roundtrip."""

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctid = "NCT001"

        result = get_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan)
        assert result is None

    def test_put_then_get_roundtrip(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctid = "NCT001"
        feature_values = {"feat_a": {"value": 3.14}}

        put_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan, feature_values)
        result = get_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan)

        assert result is not None
        assert result == feature_values

    def test_complex_values_survive_dill(self, tmp_path: Path) -> None:
        """numpy scalars, lists of dicts should survive dill roundtrip."""
        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctid = "NCT001"
        feature_values = {
            "feat_a": {
                "scalar": 2.5,
                "list_of_dicts": [{"key": "val1"}, {"key": "val2"}],
                "nested": {"a": [1, 2, 3]},
            }
        }

        put_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan, feature_values)
        result = get_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan)

        assert result is not None
        assert result == feature_values

    def test_non_json_values_survive_roundtrip(self, tmp_path: Path) -> None:
        """Non-JSON-native values (np.nan, np.int64, datetime) must not crash put.

        Regression test: an earlier version of the payload duplicated
        feature_values as a raw JSON field, which crashed json.dump on
        these types and bubbled up as a silent LLM-failure in the wrapper.
        """
        import math
        from datetime import datetime

        import numpy as np

        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctid = "NCT001"
        dt = datetime(2025, 1, 15, 12, 30)
        feature_values = {
            "feat_a": {
                "nan_float": float("nan"),
                "np_float": np.float64(3.14),
                "np_int": np.int64(5),
                "np_array": np.array([1.0, 2.0, 3.0]),
                "timestamp": dt,
            }
        }

        # Must not raise
        put_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan, feature_values)
        result = get_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan)

        assert result is not None
        inner = result["feat_a"]
        assert math.isnan(inner["nan_float"])
        assert inner["np_float"] == np.float64(3.14)
        assert inner["np_int"] == np.int64(5)
        assert list(inner["np_array"]) == [1.0, 2.0, 3.0]
        assert inner["timestamp"] == dt

    def test_schema_version_mismatch_returns_none(self, tmp_path: Path) -> None:
        """Manually write JSON with wrong schema_version."""
        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctid = "NCT001"

        # Create the directory and manually write a JSON with wrong version
        plan_hash = _plan_content_hash(plan)
        store_path = store_dir / "test_ns" / f"feat_a--{plan_hash}" / f"{nctid}.json"
        store_path.parent.mkdir(parents=True, exist_ok=True)

        # Write with schema_version = 99 (wrong)
        bad_data = {
            "schema_version": 99,
            "feature_name": "feat_a",
            "nctid": nctid,
            "plan_hash": plan_hash,
            "feature_values": {"feat_a": {"value": 1.0}},
            "feature_values_b": "deadbeef",
            "metadata": {},
        }
        store_path.write_text(json.dumps(bad_data, indent=2))

        result = get_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan)
        assert result is None

    def test_corrupt_json_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Manually write corrupt JSON."""
        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctid = "NCT001"

        plan_hash = _plan_content_hash(plan)
        store_path = store_dir / "test_ns" / f"feat_a--{plan_hash}" / f"{nctid}.json"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text("{ invalid json }")

        result = get_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan)
        assert result is None
        assert "Corrupt feature store entry" in caplog.text

    def test_corrupt_dill_payload_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Valid JSON + correct schema_version but garbage feature_values_b."""
        import logging

        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctid = "NCT001"

        plan_hash = _plan_content_hash(plan)
        store_path = store_dir / "test_ns" / f"feat_a--{plan_hash}" / f"{nctid}.json"
        store_path.parent.mkdir(parents=True, exist_ok=True)

        # Valid JSON, correct schema, but feature_values_b is hex-garbage
        # that decodes to bytes which are not a valid dill pickle.
        bad_data = {
            "schema_version": 1,
            "feature_name": "feat_a",
            "nctid": nctid,
            "plan_hash": plan_hash,
            "feature_values": {"feat_a": {"value": 1.0}},
            "feature_values_b": "deadbeef",  # valid hex, invalid dill
            "metadata": {},
        }
        store_path.write_text(json.dumps(bad_data, indent=2))

        with caplog.at_level(logging.WARNING):
            result = get_cached_feature(store_dir, "test_ns", nctid, "feat_a", plan)
        assert result is None
        assert "Failed to decode feature values" in caplog.text


# ======================================================================
# get_cached_features_batch
# ======================================================================


class TestBatchQuery:
    """Bulk probe across multiple nctids."""

    def test_empty_store_returns_empty_dict(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctids = ["NCT001", "NCT002"]

        result = get_cached_features_batch(store_dir, "test_ns", nctids, "feat_a", plan)
        assert result == {}

    def test_all_hit_returns_all_nctids(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctids = ["NCT001", "NCT002", "NCT003"]

        # Pre-populate all
        for nctid in nctids:
            put_cached_feature(
                store_dir,
                "test_ns",
                nctid,
                "feat_a",
                plan,
                {"feat_a": {"value": float(nctid.replace("NCT", ""))}},
            )

        result = get_cached_features_batch(store_dir, "test_ns", nctids, "feat_a", plan)
        assert set(result.keys()) == set(nctids)
        assert len(result) == 3

    def test_partial_hit_returns_only_hits(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        plan = _make_plan()
        nctids = ["NCT001", "NCT002", "NCT003"]

        # Pre-populate only first two
        for nctid in nctids[:2]:
            put_cached_feature(
                store_dir,
                "test_ns",
                nctid,
                "feat_a",
                plan,
                {"feat_a": {"value": 1.0}},
            )

        result = get_cached_features_batch(store_dir, "test_ns", nctids, "feat_a", plan)
        assert set(result.keys()) == {"NCT001", "NCT002"}
        assert len(result) == 2


# ======================================================================
# get_store_stats
# ======================================================================


class TestStoreStats:
    """Statistics and monitoring."""

    def test_empty_store_returns_zeros(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        stats = get_store_stats(store_dir, task_namespace="test_ns")
        assert stats == {"features": 0, "trials": 0, "total_entries": 0}

    def test_populated_store_counts_correctly(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"

        # Add 2 features x 3 trials = 6 entries
        plans = [_make_plan(name=f"feat_{i}") for i in range(2)]
        nctids = [f"NCT{i:03d}" for i in range(3)]

        for plan in plans:
            for nctid in nctids:
                put_cached_feature(
                    store_dir,
                    "test_ns",
                    nctid,
                    plan.feature_name,
                    plan,
                    {plan.feature_name: {"value": 1.0}},
                )

        stats = get_store_stats(store_dir, task_namespace="test_ns")
        assert stats["features"] == 2
        assert stats["trials"] == 3
        assert stats["total_entries"] == 6

    def test_stats_aggregates_across_namespaces(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        plan = _make_plan()

        # Add to namespace 1
        put_cached_feature(store_dir, "ns1", "NCT001", "feat_a", plan, {"feat_a": {"value": 1.0}})
        put_cached_feature(store_dir, "ns1", "NCT002", "feat_a", plan, {"feat_a": {"value": 2.0}})

        # Add to namespace 2
        put_cached_feature(store_dir, "ns2", "NCT003", "feat_a", plan, {"feat_a": {"value": 3.0}})

        # Query without namespace filter: should aggregate
        stats = get_store_stats(store_dir)
        assert stats["features"] == 1
        assert stats["trials"] == 3
        assert stats["total_entries"] == 3

        # Query with namespace filter: should be subset
        stats_ns1 = get_store_stats(store_dir, task_namespace="ns1")
        assert stats_ns1["trials"] == 2
        assert stats_ns1["total_entries"] == 2
