"""Unit tests for the boolean validation predicates in ``ctra.agents.reward_fns``.

These predicates are the canonical source of truth for "what valid LLM output
means"; the ``dspy.Refine`` float rewards are thin adapters over them. The
adapters themselves are covered by ``test_reward_functions.py`` (via the
orchestrator re-exports), so this module tests the predicates directly.
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
    from ctra.agents.reward_fns import (
        is_valid_grouper,
        is_valid_planner,
        is_valid_proposer,
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


def _previous_output(existing_names: list[str]) -> MagicMock:
    prev = MagicMock(spec=AgentOutput)
    prev.feature_plans = {n: _make_plan(n) for n in existing_names}
    return prev


# ======================================================================
# is_valid_proposer
# ======================================================================


class TestIsValidProposer:
    def test_add_new_name_is_valid(self) -> None:
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.ADD,
            feature_name="feat_b",
            feature_explanation="new feature",
        )
        assert is_valid_proposer(kwargs, result) is True

    def test_add_duplicate_name_is_invalid(self) -> None:
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.ADD,
            feature_name="feat_a",
            feature_explanation="duplicate",
        )
        assert is_valid_proposer(kwargs, result) is False

    def test_remove_existing_name_is_valid(self) -> None:
        kwargs = {"previous_output": _previous_output(["feat_a", "feat_b"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.REMOVE,
            feature_name="feat_a",
            feature_explanation="remove",
        )
        assert is_valid_proposer(kwargs, result) is True

    def test_remove_missing_name_is_invalid(self) -> None:
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.REMOVE,
            feature_name="feat_z",
            feature_explanation="remove missing",
        )
        assert is_valid_proposer(kwargs, result) is False

    def test_refine_existing_name_is_valid(self) -> None:
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=FeatureOp.REFINE,
            feature_name="feat_a",
            feature_explanation="improve",
        )
        assert is_valid_proposer(kwargs, result) is True

    def test_exception_returns_false(self) -> None:
        assert is_valid_proposer({}, None) is False

    # -- The StrEnum trap -------------------------------------------------
    #
    # ``ProposerOutput`` is a NamedTuple, so nothing coerces ``feature_operation``:
    # whatever the DSPy adapter produced is stored verbatim, which may be a plain
    # ``str``. The predicate must therefore not reach for ``.value`` unguarded,
    # and must not use ``str(op)`` -- on Python 3.10 ``data_models`` uses a
    # ``class StrEnum(str, Enum)`` backport where ``str(FeatureOp.ADD)`` is
    # ``"FeatureOp.ADD"``, while 3.11+ stdlib StrEnum yields ``"add"``.

    def test_add_with_raw_string_operation_is_valid(self) -> None:
        """A raw ``"add"`` string must behave exactly like ``FeatureOp.ADD``."""
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation="add",  # type: ignore[arg-type]
            feature_name="feat_b",
            feature_explanation="new feature",
        )
        assert is_valid_proposer(kwargs, result) is True

    def test_add_duplicate_with_raw_string_operation_is_invalid(self) -> None:
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation="add",  # type: ignore[arg-type]
            feature_name="feat_a",
            feature_explanation="duplicate",
        )
        assert is_valid_proposer(kwargs, result) is False

    def test_remove_with_raw_string_operation(self) -> None:
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation="remove",  # type: ignore[arg-type]
            feature_name="feat_a",
            feature_explanation="remove",
        )
        assert is_valid_proposer(kwargs, result) is True

    @pytest.mark.parametrize("bogus_op", ["Add", "ADD", "delete", "", None, 3])
    def test_unrecognized_operation_is_invalid(self, bogus_op) -> None:
        """An op the enum doesn't know must be rejected, never treated as REMOVE.

        The REMOVE/REFINE arm is a catch-all `return`, so without an explicit
        membership check an op like "Add" on an *existing* feature would be
        judged valid, skip the caller's guard, and reach the orchestrator's
        REMOVE branch -- which asserts the op really is REMOVE, or under `-O`
        silently deletes the feature the LLM asked to add.
        """
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=bogus_op,  # type: ignore[arg-type]
            feature_name="feat_a",  # exists -- the dangerous case
            feature_explanation="malformed op",
        )
        assert is_valid_proposer(kwargs, result) is False


# ======================================================================
# is_valid_planner
# ======================================================================


class TestIsValidPlanner:
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
        assert is_valid_planner({}, (plan, MagicMock())) is True

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
        assert is_valid_planner({}, (plan, MagicMock())) is False

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
        assert is_valid_planner({}, (plan, MagicMock())) is False

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
        assert is_valid_planner({}, (plan, MagicMock())) is False

    def test_non_tuple_result_returns_false(self) -> None:
        assert is_valid_planner({}, "not a tuple") is False

    def test_wrong_length_tuple_returns_false(self) -> None:
        assert is_valid_planner({}, (1, 2, 3)) is False


# ======================================================================
# is_valid_grouper
# ======================================================================


class TestIsValidGrouper:
    def test_valid_partition(self) -> None:
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        result = [{"feat_a": plans["feat_a"]}, {"feat_b": plans["feat_b"]}]
        assert is_valid_grouper({"feature_plans": plans}, result) is True

    def test_single_group_covering_everything(self) -> None:
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        result = [{"feat_a": plans["feat_a"], "feat_b": plans["feat_b"]}]
        assert is_valid_grouper({"feature_plans": plans}, result) is True

    def test_missing_feature_is_invalid(self) -> None:
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        result = [{"feat_a": plans["feat_a"]}]
        assert is_valid_grouper({"feature_plans": plans}, result) is False

    def test_empty_grouping_is_invalid(self) -> None:
        plans = {"feat_a": _make_plan("feat_a")}
        assert is_valid_grouper({"feature_plans": plans}, []) is False

    def test_exception_returns_false(self) -> None:
        assert is_valid_grouper({}, None) is False

    def test_duplicate_feature_is_invalid(self) -> None:
        """The count-aware check: set semantics alone would score this valid.

        ``feat_a`` appears in two groups, so coverage as a *set* equals the
        expected set -- but the total assigned count (3) exceeds the number of
        plans (2), which means a feature would be built twice.
        """
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        result = [
            {"feat_a": plans["feat_a"], "feat_b": plans["feat_b"]},
            {"feat_a": plans["feat_a"]},
        ]
        assert is_valid_grouper({"feature_plans": plans}, result) is False
