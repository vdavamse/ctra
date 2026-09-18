"""Builders for the ``dspy.Prediction`` values the agent modules return.

Single source of truth for the return contract in test doubles: if
``FeatureProposer.forward`` ever renames its field, one edit here moves every
mock with it. ``test_refine_feedback_path.py`` pins these against the real
modules so the helpers cannot silently drift from production.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import dspy
    from ctra.agents.data_models import FeatureOp, FeaturePlan, ProposerOutput
    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

if TYPE_CHECKING:
    pass


def proposer_prediction(
    *, feature_name: str, feature_explanation: str, feature_operation: str | FeatureOp
) -> dspy.Prediction:
    """Build a dspy.Prediction matching FeatureProposer.forward() return.

    Constructs the ProposerOutput internally so call sites read as a drop-in
    for today's ``ProposerOutput(...)`` but gain type correctness when the
    contract changes.

    Args:
        feature_name: The proposed feature name.
        feature_explanation: The explanation for the proposal.
        feature_operation: The operation (add/remove/refine).

    Returns:
        A ``dspy.Prediction(proposal=ProposerOutput(...))`` matching the
        real module's return.
    """
    if not _HAS_DSPY:
        raise ImportError("dspy not available")
    return dspy.Prediction(
        proposal=ProposerOutput(
            feature_name=feature_name,
            feature_explanation=feature_explanation,
            feature_operation=feature_operation,
        )
    )


def planner_prediction(plan: FeaturePlan, raw: object = None) -> dspy.Prediction:
    """Build a dspy.Prediction matching FeaturePlanner.forward() return.

    Mimics the (plan, raw) idiom: ``planner_prediction(plan, MagicMock())`` or
    ``planner_prediction(plan, None)`` replaces hand-built tuples.

    Args:
        plan: The FeaturePlan object.
        raw: The unparsed planner result (optional, defaults to None).

    Returns:
        A ``dspy.Prediction(plan=..., raw=...)`` matching the real module's
        return.
    """
    if not _HAS_DSPY:
        raise ImportError("dspy not available")
    return dspy.Prediction(plan=plan, raw=raw)


def grouper_prediction(groups: list[dict[str, FeaturePlan]]) -> dspy.Prediction:
    """Build a dspy.Prediction matching FeatureGrouper.forward() return.

    Args:
        groups: A list of grouped feature plans (dicts keyed by feature name).

    Returns:
        A ``dspy.Prediction(groups=...)`` matching the real module's return.
    """
    if not _HAS_DSPY:
        raise ImportError("dspy not available")
    return dspy.Prediction(groups=groups)
