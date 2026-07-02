"""DSPy ChainOfThought agent for feature schema generation.

**FeaturePlanner**: Multi-valued ``FeaturePlan`` output with sub-features,
``ValueError`` validation (retried via ``dspy.Refine``), and ``possible_values``
enforcement.

The planner uses the primary LM (Claude Opus 4.6) configured via
:func:`ctra.agents.lm_config.configure_lm`.
"""

from __future__ import annotations

import logging
from typing import Any

import dspy

from ctra.agents.data_models import FeaturePlan, FeatureType
from ctra.agents.signatures import FeaturePlannerSignature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Planner — multi-valued features
# ---------------------------------------------------------------------------


class FeaturePlanner(dspy.Module):  # type: ignore[misc]
    """Plan features with multi-valued schema and assertion validation.

    Produces ``FeaturePlan`` objects with ``feature_type: dict[str, FeatureType]``
    supporting sub-features.

    Validation (``ValueError`` on failure, retried via ``dspy.Refine``
    in the orchestrator):
    - ``possible_values`` keys must be a subset of ``feature_type`` keys.
    - Categorical / multi-categorical keys must have ``possible_values`` defined.

    Parameters:
        task_description: Text description of the prediction task.
    """

    def __init__(self, task_description: str) -> None:
        super().__init__()
        self.task_description = task_description
        self.planner = dspy.ChainOfThought(FeaturePlannerSignature)

    def forward(self, feature_name: str, feature_idea: str) -> tuple[FeaturePlan, Any]:
        """Plan a single feature.

        Parameters:
            feature_name: Snake-cased feature name.
            feature_idea: Description / idea for the feature.  For REFINE
                operations, this is the concatenation of old + new ideas.

        Returns:
            Tuple of ``(FeaturePlan, raw_planner_result)``.
        """
        planner_result = self.planner(
            task=self.task_description,
            feature_name=feature_name,
            feature_idea=feature_idea,
        )

        # --- Validation: possible_values keys ⊆ feature_type keys ---
        if not set(planner_result.possible_values.keys()).issubset(
            set(planner_result.feature_type.keys())
        ):
            raise ValueError(
                f"possible_values keys {list(planner_result.possible_values.keys())} "
                f"not subset of feature_type keys "
                f"{list(planner_result.feature_type.keys())}"
            )

        # --- Validation: categorical keys must have possible_values ---
        for key, keytype in planner_result.feature_type.items():
            if (
                keytype.value
                in (
                    FeatureType.MULTICATEGORICAL.value,
                    FeatureType.CATEGORICAL.value,
                )
                and key not in planner_result.possible_values
            ):
                raise ValueError(
                    f"Categorical key '{key}' missing from possible_values "
                    f"{list(planner_result.possible_values.keys())}"
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
        return plan, planner_result
