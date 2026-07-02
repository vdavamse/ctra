"""Tests for compute_features aggregation logic.

Covers:
- Single plan skips grouper
- Multiple plans calls grouper
- Result aggregation (nctid merging via |=)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

try:
    from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
    from ctra.agents.feature_builder import compute_features

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan(name: str) -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[{"value": "1.0"}],
        possible_values={},
        feature_instructions=f"Extract {name}.",
    )


@pytest.fixture()
def mock_wrapper():
    """Patch WrappedFeatureBuilder to return deterministic results."""

    def _factory(feature_values_fn):
        """Create a mock wrapper whose __call__ uses feature_values_fn."""
        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            instance = MagicMock()
            instance.side_effect = feature_values_fn
            mock_cls.return_value = instance
            yield mock_cls, instance

    return _factory


# ======================================================================
# Single plan — skips grouper
# ======================================================================


class TestSinglePlanSkipsGrouper:
    def test_single_plan_does_not_call_grouper(self, tmp_path: Path) -> None:
        """When there's only one plan, grouper should NOT be called."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctids = ["NCT001", "NCT002"]

        grouper = MagicMock()

        # Mock the wrapper to return feature values
        def mock_call(arg):
            nctid, _plan_group = arg
            return (
                nctid,
                {"feat_a": {"value": 1.0}},
                {"none_feature_explanations": {}},
            )

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            raw_features, _none_explanations, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        # Grouper should NOT have been called
        grouper.assert_not_called()

        # Results should still be returned for each nctid
        assert "NCT001" in raw_features
        assert "NCT002" in raw_features


# ======================================================================
# Multiple plans — calls grouper
# ======================================================================


class TestMultiplePlansCallGrouper:
    def test_multiple_plans_calls_grouper(self, tmp_path: Path) -> None:
        """When there are multiple plans, grouper should be called."""
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        plans = {"feat_a": plan_a, "feat_b": plan_b}
        nctids = ["NCT001"]

        # Grouper returns a single group containing both features
        grouper = MagicMock(return_value=[plans])

        def mock_call(arg):
            nctid, plan_group = arg
            values = {name: {"value": 1.0} for name in plan_group}
            return (nctid, values, {"none_feature_explanations": {}})

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            _raw_features, _, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        # Grouper should have been called
        grouper.assert_called_once_with(task="test task", feature_plans=plans)

    def test_grouper_splits_into_two_groups(self, tmp_path: Path) -> None:
        """When grouper splits plans into two groups, both groups are processed."""
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        plans = {"feat_a": plan_a, "feat_b": plan_b}
        nctids = ["NCT001"]

        # Grouper splits into two groups
        group1 = {"feat_a": plan_a}
        group2 = {"feat_b": plan_b}
        grouper = MagicMock(return_value=[group1, group2])

        call_count = {"n": 0}

        def mock_call(arg):
            nctid, plan_group = arg
            call_count["n"] += 1
            values = {name: {"value": float(call_count["n"])} for name in plan_group}
            return (nctid, values, {})

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            _raw_features, _, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        # Should have been called twice (1 nctid x 2 groups)
        assert call_count["n"] == 2


# ======================================================================
# Result aggregation (nctid merging via |=)
# ======================================================================


class TestResultAggregation:
    def test_features_merged_across_groups(self, tmp_path: Path) -> None:
        """Features from separate groups should be merged into one dict per nctid."""
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        plans = {"feat_a": plan_a, "feat_b": plan_b}
        nctids = ["NCT001"]

        group1 = {"feat_a": plan_a}
        group2 = {"feat_b": plan_b}
        grouper = MagicMock(return_value=[group1, group2])

        def mock_call(arg):
            nctid, plan_group = arg
            if "feat_a" in plan_group:
                return (nctid, {"feat_a": {"value": 1.0}}, {})
            else:
                return (nctid, {"feat_b": {"value": 2.0}}, {})

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            raw_features, _, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        assert "feat_a" in raw_features["NCT001"]
        assert "feat_b" in raw_features["NCT001"]
        assert raw_features["NCT001"]["feat_a"]["value"] == 1.0
        assert raw_features["NCT001"]["feat_b"]["value"] == 2.0

    def test_none_explanations_merged(self, tmp_path: Path) -> None:
        """None explanations from separate groups should be merged."""
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        plans = {"feat_a": plan_a, "feat_b": plan_b}
        nctids = ["NCT001"]

        group1 = {"feat_a": plan_a}
        group2 = {"feat_b": plan_b}
        grouper = MagicMock(return_value=[group1, group2])

        def mock_call(arg):
            nctid, plan_group = arg
            if "feat_a" in plan_group:
                return (
                    nctid,
                    {"feat_a": {"value": None}},
                    {"none_feature_explanations": {"feat_a": "No data found"}},
                )
            else:
                return (
                    nctid,
                    {"feat_b": {"value": None}},
                    {"none_feature_explanations": {"feat_b": "Ambiguous data"}},
                )

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            _, none_explanations, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        assert none_explanations["NCT001"]["feat_a"] == "No data found"
        assert none_explanations["NCT001"]["feat_b"] == "Ambiguous data"

    def test_multiple_nctids(self, tmp_path: Path) -> None:
        """Each nctid should get its own entry in the output."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctids = ["NCT001", "NCT002", "NCT003"]

        grouper = MagicMock()  # not called for single plan

        counter = {"n": 0}

        def mock_call(arg):
            nctid, _ = arg
            counter["n"] += 1
            return (nctid, {"feat_a": {"value": float(counter["n"])}}, {})

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            raw_features, _, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        assert len(raw_features) == 3
        assert "NCT001" in raw_features
        assert "NCT002" in raw_features
        assert "NCT003" in raw_features
