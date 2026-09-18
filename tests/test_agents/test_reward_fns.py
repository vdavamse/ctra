"""Unit tests for the boolean validation predicates in ``ctra.agents.reward_fns``.

These predicates are the canonical source of truth for "what valid LLM output
means"; the ``dspy.Refine`` float rewards are thin adapters over them. The
adapters themselves are covered by ``test_reward_functions.py`` (imported
directly from ``ctra.agents.reward_fns``), so this module tests the predicates
directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    import dspy

    from ctra.agents.data_models import (
        AgentOutput,
        FeatureOp,
        FeaturePlan,
        FeatureSource,
        FeatureType,
        ProposerOutput,
    )
    from ctra.agents.reward_fns import (
        is_valid_builder,
        is_valid_grouper,
        is_valid_planner,
        is_valid_proposer,
        unwrap_builder_result,
        unwrap_groups,
        unwrap_planner_result,
        unwrap_proposal,
    )
    from tests.test_agents.conftest import (
        builder_prediction,
        grouper_prediction,
        planner_prediction,
        proposer_prediction,
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

    @pytest.mark.parametrize("bogus_op", ["Add", "ADD", "delete", "obliterate", "", None, 3])
    def test_unrecognized_operation_is_invalid(self, bogus_op) -> None:
        """An op the enum doesn't know must be rejected, never treated as REMOVE.

        The REMOVE/REFINE arm is a catch-all `return`, so without an explicit
        membership check an unrecognised op on an *existing* feature would be
        judged valid, skip the caller's guard, and reach the orchestrator's
        REMOVE branch -- deleting a feature nobody asked to remove.

        Note the cases are genuinely unresolvable ops, not merely miscased ones:
        `FeatureProposer.forward` strips and lowercases before coercing, so
        "Add"/"ADD" become `FeatureOp.ADD` and never reach this predicate. They
        can still arrive from a caller that does not normalise, which is why the
        membership check lives here rather than relying on the proposer.
        """
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = ProposerOutput(
            feature_operation=bogus_op,  # type: ignore[arg-type]
            feature_name="feat_a",  # exists -- the dangerous case
            feature_explanation="malformed op",
        )
        assert is_valid_proposer(kwargs, result) is False

    def test_prediction_wrapped_proposal_is_valid(self) -> None:
        """Proposer can return dspy.Prediction(proposal=...) and pass validation."""
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = proposer_prediction(
            feature_name="feat_b",
            feature_explanation="new feature",
            feature_operation=FeatureOp.ADD,
        )
        assert is_valid_proposer(kwargs, result) is True

    def test_prediction_missing_proposal_field_is_false(self) -> None:
        """Prediction without 'proposal' field falls through to exception handler."""
        kwargs = {"previous_output": _previous_output(["feat_a"])}
        result = dspy.Prediction(some_other_field="value")
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

    def test_prediction_wrapped_plan_and_raw_is_valid(self) -> None:
        """Planner can return dspy.Prediction(plan=..., raw=...) and pass validation."""
        plan = FeaturePlan(
            feature_name="test",
            feature_idea="idea",
            feature_type={"cat_field": FeatureType.CATEGORICAL},
            data_sources=[FeatureSource.PUBMED],
            example_values=[],
            possible_values={"cat_field": ["a", "b"]},
            feature_instructions="test",
        )
        result = planner_prediction(plan, raw=MagicMock())
        assert is_valid_planner({}, result) is True

    def test_prediction_missing_plan_field_is_false(self) -> None:
        """Prediction without 'plan' field falls through to exception handler."""
        result = dspy.Prediction(raw="something")
        assert is_valid_planner({}, result) is False


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

    def test_prediction_wrapped_groups_is_valid(self) -> None:
        """Grouper can return dspy.Prediction(groups=...) and pass validation."""
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        result = grouper_prediction([{"feat_a": plans["feat_a"]}, {"feat_b": plans["feat_b"]}])
        assert is_valid_grouper({"feature_plans": plans}, result) is True

    def test_prediction_empty_groups_is_invalid(self) -> None:
        """Prediction(groups=[]) is invalid (empty partition)."""
        plans = {"feat_a": _make_plan("feat_a")}
        result = grouper_prediction([])
        assert is_valid_grouper({"feature_plans": plans}, result) is False

    def test_prediction_missing_groups_field_is_false(self) -> None:
        """Prediction without 'groups' field falls through to exception handler."""
        plans = {"feat_a": _make_plan("feat_a")}
        result = dspy.Prediction(some_other_field="value")
        assert is_valid_grouper({"feature_plans": plans}, result) is False


# ======================================================================
# is_valid_builder
# ======================================================================


class TestIsValidBuilder:
    """Completeness predicate for the builder (issue #6).

    ``FeatureBuilder.forward`` no longer raises on an incomplete group; this
    predicate is what turns a partial build into a Refine retry.
    """

    def _two_plans(self) -> dict[str, dict[str, FeaturePlan]]:
        return {
            "feature_plan_group": {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        }

    def test_complete_legacy_tuple_is_valid(self) -> None:
        result = ({"feat_a": {"value": 1.0}, "feat_b": {"value": 2.0}}, {})
        assert is_valid_builder(self._two_plans(), result) is True

    def test_missing_feature_in_legacy_tuple_is_invalid(self) -> None:
        result = ({"feat_a": {"value": 1.0}}, {})
        assert is_valid_builder(self._two_plans(), result) is False

    def test_complete_prediction_is_valid(self) -> None:
        """The discriminating case: a tuple-unpack of the Prediction binds the
        strings 'feature_values'/'metadata' and would return False here."""
        result = builder_prediction({"feat_a": {"value": 1.0}, "feat_b": {"value": 2.0}})
        assert is_valid_builder(self._two_plans(), result) is True

    def test_missing_feature_in_prediction_is_invalid(self) -> None:
        result = builder_prediction({"feat_a": {"value": 1.0}})
        assert is_valid_builder(self._two_plans(), result) is False

    def test_prediction_missing_feature_values_field_is_false(self) -> None:
        """A malformed Prediction is reported invalid, not raised."""
        result = dspy.Prediction(metadata={})
        assert is_valid_builder(self._two_plans(), result) is False

    def test_extra_features_still_valid(self) -> None:
        """issubset semantics: extra keys do not invalidate the build."""
        kwargs = {"feature_plan_group": {"feat_a": _make_plan("feat_a")}}
        result = builder_prediction({"feat_a": {"value": 1.0}, "feat_extra": {"value": 3.0}})
        assert is_valid_builder(kwargs, result) is True

    def test_exception_returns_false(self) -> None:
        assert is_valid_builder({}, "bad") is False


# ======================================================================
# unwrap_* helpers: a Prediction lacking its field must not fall through
# ======================================================================


class TestUnwrapHelpersRejectMalformedPredictions:
    """A ``dspy.Prediction`` missing the contract field raises rather than
    falling through, so it can never reach a ``plan, raw = ...`` unpack
    downstream (``Prediction`` iterates over its keys). Legacy shapes still
    pass through; the predicates above turn the TypeError into ``False``."""

    def test_unwrap_proposal_raises_on_missing_field(self) -> None:
        with pytest.raises(TypeError, match=r"FeatureProposer Prediction lacks 'proposal'"):
            unwrap_proposal(dspy.Prediction(some_other_field="value"))

    def test_unwrap_planner_result_raises_on_missing_field(self) -> None:
        with pytest.raises(TypeError, match=r"FeaturePlanner Prediction lacks 'plan'"):
            unwrap_planner_result(dspy.Prediction(raw="something"))
        with pytest.raises(TypeError, match=r"FeaturePlanner Prediction lacks 'raw'"):
            unwrap_planner_result(dspy.Prediction(plan=_make_plan("feat_a")))

    def test_unwrap_groups_raises_on_missing_field(self) -> None:
        with pytest.raises(TypeError, match=r"FeatureGrouper Prediction lacks 'groups'"):
            unwrap_groups(dspy.Prediction(some_other_field="value"))

    @pytest.mark.parametrize(
        ("prediction", "missing"),
        [
            (dspy.Prediction(feature_values={}), "metadata"),
            (dspy.Prediction(metadata={}), "feature_values"),
        ],
        ids=["only_feature_values", "only_metadata"],
    )
    def test_unwrap_builder_result_raises_on_missing_field(
        self, prediction: dspy.Prediction, missing: str
    ) -> None:
        with pytest.raises(TypeError, match=rf"FeatureBuilder Prediction lacks '{missing}'"):
            unwrap_builder_result(prediction)

    def test_unwrap_builder_result_returns_the_two_fields(self) -> None:
        values = {"feat_a": {"value": 1.0}}
        meta = {"research_results": "r"}
        assert unwrap_builder_result(builder_prediction(values, meta)) == (values, meta)

    def test_legacy_shapes_still_pass_through(self) -> None:
        legacy_prop = ProposerOutput(
            feature_operation=FeatureOp.ADD, feature_name="f", feature_explanation="e"
        )
        legacy_tuple = (_make_plan("feat_a"), None)
        legacy_groups = [{"feat_a": _make_plan("feat_a")}]
        legacy_build = ({"feat_a": {"value": 1.0}}, {})
        assert unwrap_proposal(legacy_prop) is legacy_prop
        assert unwrap_planner_result(legacy_tuple) is legacy_tuple
        assert unwrap_groups(legacy_groups) is legacy_groups
        assert unwrap_builder_result(legacy_build) is legacy_build
