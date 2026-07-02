"""Unit tests for dspy.Refine reward functions used in the agent pipeline.

Each reward function validates LLM output quality and returns 1.0 (valid)
or 0.0 (invalid), driving the Refine retry loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    from ctra.agents.data_models import (
        AgentOutput,
        FeatureOp,
        FeaturePlan,
        FeatureSource,
        FeatureType,
        ProposerOutput,
    )
    from ctra.agents.feature_builder import _builder_reward
    from ctra.agents.orchestrator import (
        _grouper_reward,
        _planner_reward,
        _proposer_reward,
    )

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


def _make_plan(name: str, ftype: FeatureType = FeatureType.FLOAT) -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type={"value": ftype},
        data_sources=[FeatureSource.PUBMED],
        example_values=[{"value": "1.0"}],
        possible_values={},
        feature_instructions=f"Extract {name}.",
    )


# ======================================================================
# _proposer_reward
# ======================================================================


class TestProposerReward:
    def _make_previous_output(self, existing_names: list[str]) -> MagicMock:
        prev = MagicMock(spec=AgentOutput)
        prev.feature_plans = {n: _make_plan(n) for n in existing_names}
        return prev

    def test_add_new_name_valid(self) -> None:
        kwargs = {"previous_output": self._make_previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.ADD,
            feature_name="feat_b",
            feature_explanation="new feature",
        )
        assert _proposer_reward(kwargs, result) == 1.0

    def test_add_duplicate_name_invalid(self) -> None:
        kwargs = {"previous_output": self._make_previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.ADD,
            feature_name="feat_a",
            feature_explanation="duplicate",
        )
        assert _proposer_reward(kwargs, result) == 0.0

    def test_remove_existing_name_valid(self) -> None:
        kwargs = {"previous_output": self._make_previous_output(["feat_a", "feat_b"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.REMOVE,
            feature_name="feat_a",
            feature_explanation="remove",
        )
        assert _proposer_reward(kwargs, result) == 1.0

    def test_remove_nonexistent_name_invalid(self) -> None:
        kwargs = {"previous_output": self._make_previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.REMOVE,
            feature_name="feat_z",
            feature_explanation="remove nonexistent",
        )
        assert _proposer_reward(kwargs, result) == 0.0

    def test_refine_existing_name_valid(self) -> None:
        kwargs = {"previous_output": self._make_previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.REFINE,
            feature_name="feat_a",
            feature_explanation="improve",
        )
        assert _proposer_reward(kwargs, result) == 1.0

    def test_exception_returns_zero(self) -> None:
        assert _proposer_reward({}, None) == 0.0


# ======================================================================
# _planner_reward
# ======================================================================


class TestPlannerReward:
    def test_valid_plan(self) -> None:
        plan = FeaturePlan(
            feature_name="test",
            feature_idea="idea",
            feature_type={"cat_field": FeatureType.CATEGORICAL, "num_field": FeatureType.FLOAT},
            data_sources=[FeatureSource.PUBMED],
            example_values=[],
            possible_values={"cat_field": ["a", "b"]},
            feature_instructions="test",
        )
        result = (plan, MagicMock())
        assert _planner_reward({}, result) == 1.0

    def test_possible_values_key_not_in_feature_type(self) -> None:
        plan = FeaturePlan(
            feature_name="test",
            feature_idea="idea",
            feature_type={"num_field": FeatureType.FLOAT},
            data_sources=[FeatureSource.PUBMED],
            example_values=[],
            possible_values={"nonexistent_key": ["a", "b"]},
            feature_instructions="test",
        )
        result = (plan, MagicMock())
        assert _planner_reward({}, result) == 0.0

    def test_categorical_without_possible_values(self) -> None:
        plan = FeaturePlan(
            feature_name="test",
            feature_idea="idea",
            feature_type={"cat_field": FeatureType.CATEGORICAL},
            data_sources=[FeatureSource.PUBMED],
            example_values=[],
            possible_values={},
            feature_instructions="test",
        )
        result = (plan, MagicMock())
        assert _planner_reward({}, result) == 0.0

    def test_multicategorical_without_possible_values(self) -> None:
        plan = FeaturePlan(
            feature_name="test",
            feature_idea="idea",
            feature_type={"mc_field": FeatureType.MULTICATEGORICAL},
            data_sources=[FeatureSource.PUBMED],
            example_values=[],
            possible_values={},
            feature_instructions="test",
        )
        result = (plan, MagicMock())
        assert _planner_reward({}, result) == 0.0

    def test_exception_returns_zero(self) -> None:
        assert _planner_reward({}, "not a tuple") == 0.0


# ======================================================================
# _grouper_reward
# ======================================================================


class TestGrouperReward:
    def test_valid_grouping(self) -> None:
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        result = [{"feat_a": plans["feat_a"]}, {"feat_b": plans["feat_b"]}]
        assert _grouper_reward({"feature_plans": plans}, result) == 1.0

    def test_single_group_valid(self) -> None:
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        result = [{"feat_a": plans["feat_a"], "feat_b": plans["feat_b"]}]
        assert _grouper_reward({"feature_plans": plans}, result) == 1.0

    def test_empty_grouping_invalid(self) -> None:
        plans = {"feat_a": _make_plan("feat_a")}
        assert _grouper_reward({"feature_plans": plans}, []) == 0.0

    def test_missing_feature_invalid(self) -> None:
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        result = [{"feat_a": plans["feat_a"]}]  # missing feat_b
        assert _grouper_reward({"feature_plans": plans}, result) == 0.0

    def test_exception_returns_zero(self) -> None:
        assert _grouper_reward({}, None) == 0.0


# ======================================================================
# _builder_reward
# ======================================================================


class TestBuilderReward:
    def test_all_features_present(self) -> None:
        kwargs = {
            "feature_plan_group": {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        }
        result = ({"feat_a": {"value": 1.0}, "feat_b": {"value": 2.0}}, {})
        assert _builder_reward(kwargs, result) == 1.0

    def test_missing_feature_invalid(self) -> None:
        kwargs = {
            "feature_plan_group": {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        }
        result = ({"feat_a": {"value": 1.0}}, {})  # missing feat_b
        assert _builder_reward(kwargs, result) == 0.0

    def test_extra_features_still_valid(self) -> None:
        kwargs = {"feature_plan_group": {"feat_a": _make_plan("feat_a")}}
        result = ({"feat_a": {"value": 1.0}, "feat_extra": {"value": 3.0}}, {})
        assert _builder_reward(kwargs, result) == 1.0

    def test_exception_returns_zero(self) -> None:
        assert _builder_reward({}, "bad") == 0.0
