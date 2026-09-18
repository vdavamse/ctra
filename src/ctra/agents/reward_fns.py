"""Reward functions and validation predicates for DSPy Refine wrappers.

This module is the single owner of validation semantics for proposer, planner, and
grouper agents. Each predicate is defined once as `is_valid_*`, and the DSPy Refine
float reward is a thin adapter over it to ensure the two never drift.

All functions wrap `try/except Exception -> 0.0/False` with debug logging to make
the next silent failure traceable.

Also provides :class:`ResettingRefine`, which every agent module should use in place
of :class:`dspy.Refine` -- see its docstring for why.

All four Refine-wrapped modules -- FeatureProposer, FeaturePlanner, FeatureGrouper
and FeatureBuilder -- return ``dspy.Prediction`` so that Refine's feedback step
(``refine.py:153`` ``dict(outputs)``) succeeds and ``OfferFeedback`` actually runs.
The four ``unwrap_*`` helpers below normalise them back to the legacy shapes at the
boundary.

``FeatureBuilder`` additionally returns **partial** coverage by design (issue #6):
``FeatureBuilder.forward`` does not raise on an incomplete group. ``is_valid_builder``
is the sole judge of completeness, and ``WrappedFeatureBuilder`` -- not the module,
not the reward -- fills the gaps *after* Refine returns, so a partial result is
still scored 0.0 and retried.

NEVER tuple-unpack a Prediction: it inherits ``Example.__iter__``, which yields
KEYS, so ``values, meta = pred`` silently binds the strings 'feature_values' and
'metadata' with no error. Always go through the matching ``unwrap_*`` helper.
"""

from __future__ import annotations

import logging
from typing import Any

import dspy

from ctra.agents.data_models import FeatureOp, FeatureType

logger = logging.getLogger(__name__)

_FEATURE_OP_VALUES = frozenset(op.value for op in FeatureOp)


class ResettingRefine(dspy.Refine):  # type: ignore[misc]
    """``dspy.Refine`` with a per-call failure budget instead of a per-lifetime one.

    ``dspy.Refine.forward`` decrements ``self.fail_count`` for every attempt it
    swallows and never restores it (``refine.py:170-174``). The counter therefore
    erodes across calls on a long-lived instance -- and ``Agent.proposer`` /
    ``planner`` / ``grouper`` are built once in ``Agent.__init__`` and reused for an
    entire MCTS run.

    Resetting per call restores graceful degradation on every call. Budget erosion
    from transient LM errors (timeout, rate limit) and exceptions in the feedback
    path persists: ``refine.py:170-174`` decrements for **any** swallowed exception
    and never restores. The instances are long-lived (shared planner in
    ``Initializer``), so multiple callers draw on one budget. Additionally, the
    now-live ``OfferFeedback`` call at ``refine.py:167`` is itself a new exception
    source inside the same ``try`` block. Without the reset, a second exception
    re-raises past the caller's ``is_valid_*`` guard instead of returning its best
    attempt -- the dead-skip-branch failure this package was fixed to eliminate.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # dspy.Refine is untyped, so annotate explicitly rather than inferring.
        self._initial_fail_count: int = self.fail_count

    def forward(self, **kwargs: Any) -> Any:
        self.fail_count = self._initial_fail_count
        return super().forward(**kwargs)


# ---------------------------------------------------------------------------
# Return-shape normalisation
# ---------------------------------------------------------------------------
# The Refine-wrapped modules return dspy.Prediction so that Refine's feedback
# step (refine.py:153 ``dict(outputs)``) succeeds and OfferFeedback actually
# runs.  Every consumer still works against the legacy raw shape; these four
# helpers are the only place that knows both.
#
# Tolerant by design: a non-Prediction passes through untouched, which keeps the
# hand-built ``is_valid_planner({}, (plan, raw))`` calls in orchestrator.py and
# initializer.py -- and the MagicMock doubles in the test suite -- valid. A
# Prediction that lacks the expected field raises TypeError instead: letting it
# fall through would hand a Prediction to the ``plan, raw = ...`` unpack below.
#
# NEVER tuple-unpack a Prediction: it inherits Example.__iter__, which yields
# KEYS, so ``plan, raw = pred`` silently binds the strings 'plan' and 'raw'.


def unwrap_proposal(result: Any) -> Any:
    """Extract proposal from dspy.Prediction or pass through legacy shape.

    If result is a dspy.Prediction, returns its 'proposal' field. A Prediction
    without that field raises TypeError (the ``is_valid_*`` predicates catch it
    and report the output as invalid). A non-Prediction is returned unchanged.

    Args:
        result: Either a dspy.Prediction(proposal=...) or a raw ProposerOutput.

    Returns:
        The ProposerOutput (or original result if not a Prediction).

    Raises:
        TypeError: If ``result`` is a Prediction lacking 'proposal'.
    """
    if isinstance(result, dspy.Prediction):
        if "proposal" not in result:
            raise TypeError(
                f"FeatureProposer Prediction lacks 'proposal': keys={list(result.keys())}"
            )
        return result["proposal"]
    return result


def unwrap_planner_result(result: Any) -> Any:
    """Extract (plan, raw) from dspy.Prediction or pass through legacy shape.

    If result is a dspy.Prediction, returns the tuple ``(plan, raw)`` built from
    its 'plan' and 'raw' fields. A Prediction missing either field raises
    TypeError (the ``is_valid_*`` predicates catch it and report the output as
    invalid). A non-Prediction is returned unchanged.

    Args:
        result: Either a dspy.Prediction(plan=..., raw=...) or a (plan, raw) tuple.

    Returns:
        The (FeaturePlan, raw) tuple (or original result if not a Prediction).

    Raises:
        TypeError: If ``result`` is a Prediction lacking 'plan' or 'raw'.
    """
    if isinstance(result, dspy.Prediction):
        for field in ("plan", "raw"):
            if field not in result:
                raise TypeError(
                    f"FeaturePlanner Prediction lacks '{field}': keys={list(result.keys())}"
                )
        return (result["plan"], result["raw"])
    return result


def unwrap_groups(result: Any) -> Any:
    """Extract groups from dspy.Prediction or pass through legacy shape.

    If result is a dspy.Prediction, returns its 'groups' field. A Prediction
    without that field raises TypeError (the ``is_valid_*`` predicates catch it
    and report the output as invalid). A non-Prediction is returned unchanged.

    Args:
        result: Either a dspy.Prediction(groups=...) or a raw list[dict].

    Returns:
        The list of groups (or original result if not a Prediction).

    Raises:
        TypeError: If ``result`` is a Prediction lacking 'groups'.
    """
    if isinstance(result, dspy.Prediction):
        if "groups" not in result:
            raise TypeError(f"FeatureGrouper Prediction lacks 'groups': keys={list(result.keys())}")
        return result["groups"]
    return result


def unwrap_builder_result(result: Any) -> Any:
    """Extract (feature_values, metadata) from dspy.Prediction or pass through.

    If result is a dspy.Prediction, returns the tuple ``(feature_values, metadata)``
    built from its fields. A Prediction missing either field raises TypeError
    (``is_valid_builder`` catches it and reports the output as invalid;
    ``WrappedFeatureBuilder.__call__`` lets it fall into the builder_exception
    path). A non-Prediction -- the legacy ``(values, meta)`` tuple that ~8 test
    doubles still return -- is returned unchanged.

    Note the field is ``feature_values``, never ``values``: ``Example.values``
    is a reserved method and a Prediction built with it is silently broken.

    Args:
        result: Either a dspy.Prediction(feature_values=..., metadata=...) or a
            ``(values, meta)`` tuple.

    Returns:
        The ``(feature_values, metadata)`` tuple (or original result if not a
        Prediction).

    Raises:
        TypeError: If ``result`` is a Prediction lacking 'feature_values' or
            'metadata'.
    """
    if isinstance(result, dspy.Prediction):
        for field in ("feature_values", "metadata"):
            if field not in result:
                raise TypeError(
                    f"FeatureBuilder Prediction lacks '{field}': keys={list(result.keys())}"
                )
        return (result["feature_values"], result["metadata"])
    return result


# ---------------------------------------------------------------------------
# Boolean validation predicates (canonical source of truth)
# ---------------------------------------------------------------------------


def is_valid_proposer(kwargs: Any, result: Any) -> bool:
    """Validate a proposer output against the current feature set.

    Returns True if the proposed operation is valid:
    - For ADD: proposed feature_name is not already in the feature set
    - For REMOVE/REFINE: proposed feature_name exists in the feature set

    Invariant callers may rely on: a ``True`` result implies
    ``result.feature_operation`` is a **recognised** ``FeatureOp`` value (enum
    member or its raw ``str``). ``ProposerOutput`` is a plain ``NamedTuple`` with
    no coercion, and ``FeatureProposer.forward`` keeps the raw value when
    ``FeatureOp()`` cannot resolve it, so this predicate is the only thing
    standing between an unrecognised operation and:

    - ``Agent.forward``'s operation dispatch, whose REMOVE arm would otherwise
      run for an op that is not REMOVE; and
    - ``AgentOutput.operation``, which reaches persisted state (MLflow logging,
      MCTS node serialization) and should never carry an unknown verb.

    Relaxing this predicate re-arms both. Keep the ``_FEATURE_OP_VALUES``
    membership check below.

    Args:
        kwargs: Dict with "previous_output" containing current feature_plans.
        result: The ProposerOutput (NamedTuple) or the ``dspy.Prediction``
            wrapping it.

    Returns:
        True if valid operation, False otherwise.
    """
    try:
        result = unwrap_proposal(result)
        previous_output = kwargs["previous_output"]
        existing = set(previous_output.feature_plans.keys())

        # Robustly extract operation value: handle both FeatureOp enum and raw strings.
        # On Python 3.10, FeatureOp is a StrEnum backport where str() returns "FeatureOp.ADD".
        # On Python 3.11+, stdlib StrEnum would return just "add". Use .value on the enum,
        # but defend against a raw string being passed.
        op = result.feature_operation
        op_value = op.value if isinstance(op, FeatureOp) else op

        # An operation the enum does not recognise is never valid. Without this
        # guard the REMOVE/REFINE branch below acts as a catch-all: an op the
        # proposer's coercion could not resolve (an unknown verb, a non-string,
        # or any raw value from a caller that does not normalise) would be
        # judged valid whenever the feature already exists, skipping the
        # caller's guard and reaching the REMOVE branch in the orchestrator
        # -- which would then delete the very feature the LLM asked to add.
        if op_value not in _FEATURE_OP_VALUES:
            return False

        if op_value == FeatureOp.ADD.value:
            return result.feature_name not in existing
        return result.feature_name in existing
    except Exception:
        logger.debug("is_valid_proposer failed", exc_info=True)
        return False


def is_valid_planner(kwargs: Any, result: Any) -> bool:
    """Validate a planner output for schema consistency.

    Returns True if the feature plan is internally consistent:
    - All keys in possible_values exist in feature_type
    - All categorical/multicategorical features have possible_values defined

    Args:
        kwargs: Dict (not used, plan comes from result).
        result: Tuple of (FeaturePlan, raw LLM output) or the ``dspy.Prediction``
            wrapping it.

    Returns:
        True if plan schema is valid, False otherwise.
    """
    try:
        result = unwrap_planner_result(result)
        if not isinstance(result, tuple) or len(result) != 2:
            return False

        plan, _raw = result
        pv_keys = set(plan.possible_values.keys())
        ft_keys = set(plan.feature_type.keys())

        # Require: possible_values keys ⊆ feature_type keys
        if not pv_keys.issubset(ft_keys):
            return False

        # Require: all categorical/multicategorical keys have possible_values defined
        for key, keytype in plan.feature_type.items():
            if (
                keytype.value
                in (
                    FeatureType.MULTICATEGORICAL.value,
                    FeatureType.CATEGORICAL.value,
                )
                and key not in plan.possible_values
            ):
                return False

        return True
    except Exception:
        logger.debug("is_valid_planner failed", exc_info=True)
        return False


def is_valid_grouper(kwargs: Any, result: Any) -> bool:
    """Validate a grouper output for complete feature coverage (count-aware).

    Returns True if the grouping partitions all features exactly once:
    - Non-empty result
    - All features in feature_plans are assigned to a group
    - No feature is missing or duplicated across groups
    - Exact count: sum(len(group) for group in result) == len(feature_plans)

    Args:
        kwargs: Dict with "feature_plans" containing all features to be grouped.
        result: List of groups, each a dict/set of feature names, or the
            ``dspy.Prediction`` wrapping it.

    Returns:
        True if grouping covers all features exactly once, False otherwise.
    """
    try:
        result = unwrap_groups(result)
        feature_plans = kwargs["feature_plans"]

        # Must be a non-empty sequence of groups
        if not result or not isinstance(result, list):
            return False

        # Groups must be name -> plan mappings. A list-shaped group (bare names,
        # no plans) is *not* acceptable: the only consumer, ``compute_features``,
        # iterates ``group.items()`` and would die on it. Judging such a grouping
        # valid here would skip the repair that makes it safe.
        if not all(isinstance(group, dict) for group in result):
            return False

        all_features: set[str] = set()
        for group in result:
            all_features.update(group.keys())

        # Exact coverage: every feature present, and none assigned twice. The
        # count arm is what set equality alone cannot catch -- a duplicate would
        # otherwise be built twice.
        expected = set(feature_plans.keys())
        total_count = sum(len(group) for group in result)

        return all_features == expected and total_count == len(expected)
    except Exception:
        logger.debug("is_valid_grouper failed", exc_info=True)
        return False


def is_valid_builder(kwargs: Any, result: Any) -> bool:
    """Validate that the builder produced every planned feature in the group.

    ``FeatureBuilder.forward`` no longer raises on an incomplete build (issue #6):
    it returns whatever the Construct step produced, and this predicate is what
    turns an incomplete group into a Refine retry. A False here costs one rollout;
    if all N are spent, ``WrappedFeatureBuilder.__call__`` fills the stragglers with
    all-None values and a ``builder_omitted`` explanation.

    ``issubset`` (not equality) is deliberate: extra keys stay valid, matching the
    semantics of the private reward this replaced in ``feature_builder``.

    Args:
        kwargs: Dict with "feature_plan_group" -- the plans requested of the builder.
        result: ``dspy.Prediction(feature_values=..., metadata=...)`` or the legacy
            ``(values, meta)`` tuple.

    Returns:
        True if every planned feature name is present in the built values.
    """
    try:
        values, _meta = unwrap_builder_result(result)
        planned = set(kwargs["feature_plan_group"].keys())
        return planned.issubset(set(values.keys()))
    except Exception:
        logger.debug("is_valid_builder failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Float reward adapters (thin wrappers over validation predicates)
# ---------------------------------------------------------------------------


def proposer_reward(kwargs: Any, result: Any) -> float:
    """Reward function for DSPy Refine on FeatureProposer (MCTS reward signal).

    Returns 1.0 if the proposed operation is valid, 0.0 otherwise.
    Used in MCTS node expansion to guide which feature operations are worth exploring.

    Args:
        kwargs: Dict with "previous_output" containing current feature_plans.
        result: The FeatureProposerSignature output with operation and feature_name.

    Returns:
        1.0 if valid operation, 0.0 otherwise.
    """
    return 1.0 if is_valid_proposer(kwargs, result) else 0.0


def planner_reward(kwargs: Any, result: Any) -> float:
    """Reward function for DSPy Refine on FeaturePlanner (MCTS reward signal).

    Returns 1.0 if the feature plan is internally consistent, 0.0 otherwise.
    Ensures plans can be executed by FeatureBuilder without ambiguity.
    Used in MCTS to filter out invalid plan specifications.

    Args:
        kwargs: Dict (not used, plan comes from result).
        result: Tuple of (plan object, raw LLM output).

    Returns:
        1.0 if plan schema is valid, 0.0 otherwise.
    """
    return 1.0 if is_valid_planner(kwargs, result) else 0.0


def grouper_reward(kwargs: Any, result: Any) -> float:
    """Reward function for DSPy Refine on FeatureGrouper (MCTS reward signal).

    Returns 1.0 if the grouping partitions all features exactly once, 0.0 otherwise.
    Ensures FeatureBuilder will have a complete assignment and can parallelize
    feature construction efficiently. Used in MCTS to accept/reject grouping proposals.

    Args:
        kwargs: Dict with "feature_plans" containing all features to be grouped.
        result: List of groups, each a dict/set of feature names.

    Returns:
        1.0 if grouping covers all features exactly once, 0.0 otherwise.
    """
    return 1.0 if is_valid_grouper(kwargs, result) else 0.0


def builder_reward(kwargs: Any, result: Any) -> float:
    """Reward function for DSPy Refine on FeatureBuilder (MCTS reward signal).

    Returns 1.0 if every planned feature was built, 0.0 otherwise. A 0.0 here is
    a *retry request*, not a failure: Refine re-runs the builder with
    ``OfferFeedback`` guidance, and only after all N attempts does
    ``WrappedFeatureBuilder`` fill what is still missing.

    Args:
        kwargs: Dict with "feature_plan_group" -- the plans requested of the builder.
        result: ``dspy.Prediction(feature_values=..., metadata=...)`` or the legacy
            ``(values, meta)`` tuple.

    Returns:
        1.0 if every planned feature is present, 0.0 otherwise.
    """
    return 1.0 if is_valid_builder(kwargs, result) else 0.0
