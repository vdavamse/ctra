"""DSPy ChainOfThought agent for feature schema generation.

**FeaturePlanner**: Multi-valued ``FeaturePlan`` output with sub-features,
schema validation via ``is_valid_planner`` predicate, and ``possible_values``
enforcement.

The planner uses the primary LM (Claude Opus 4.6) configured via
:func:`ctra.agents.lm_config.configure_lm`.

Returns a ``dspy.Prediction`` carrying the best-effort plan. Note: ``forward()``
does not raise on invalid LLM output. Validation is deferred to the caller (e.g.,
``dspy.Refine`` rewards and post-checks in the orchestrator).
"""

from __future__ import annotations

import logging

import dspy

from ctra.agents.data_models import FeaturePlan
from ctra.agents.signatures import FeaturePlannerSignature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Planner — multi-valued features
# ---------------------------------------------------------------------------


class FeaturePlanner(dspy.Module):  # type: ignore[misc]
    """Plan features with multi-valued schema validation.

    Produces ``FeaturePlan`` objects with ``feature_type: dict[str, FeatureType]``
    supporting sub-features.

    Schema validation via ``is_valid_planner`` (does not raise):
    - ``possible_values`` keys must be a subset of ``feature_type`` keys.
    - Categorical / multi-categorical keys must have ``possible_values`` defined.

    Parameters:
        task_description: Text description of the prediction task.
    """

    def __init__(self, task_description: str) -> None:
        super().__init__()
        self.task_description = task_description
        self.planner = dspy.ChainOfThought(FeaturePlannerSignature)

    def forward(self, feature_name: str, feature_idea: str) -> dspy.Prediction:
        """Plan a single feature.

        Parameters:
            feature_name: Snake-cased feature name.
            feature_idea: Description / idea for the feature.  For REFINE
                operations, this is the concatenation of old + new ideas.

        Returns:
            ``dspy.Prediction`` with fields ``plan`` (``FeaturePlan``) and ``raw``
            (the unparsed ``ChainOfThought`` result). **Do not tuple-unpack it**
            -- ``Prediction`` iterates its keys. Use ``reward_fns.unwrap_planner_result``.
        """
        planner_result = self.planner(
            task=self.task_description,
            feature_name=feature_name,
            feature_idea=feature_idea,
        )

        plan = FeaturePlan(
            feature_name=feature_name,
            feature_idea=feature_idea,
            feature_type=planner_result.feature_type,
            data_sources=planner_result.data_sources,
            example_values=planner_result.example_values,
            possible_values=planner_result.possible_values,
            feature_instructions=planner_result.feature_instructions,
        )

        logger.debug("Planned feature: %s (types=%s)", feature_name, plan.feature_type)
        return dspy.Prediction(plan=plan, raw=planner_result)
