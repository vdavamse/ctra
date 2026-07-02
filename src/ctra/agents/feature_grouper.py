"""Feature grouping module for batch research efficiency.

Groups features by data dependency so that a single ReAct research pass
can serve multiple features, reducing LLM calls by ~5x.

Mirrors AutoCT's ``FeatureGrouper`` from ``agent.py:1089-1136``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import dspy

from ctra.agents.feature_utils import dump_as_json
from ctra.agents.signatures import FeatureGroupingSignature

if TYPE_CHECKING:
    from ctra.agents.data_models import FeaturePlan

logger = logging.getLogger(__name__)


class FeatureGrouper(dspy.Module):  # type: ignore[misc]
    """Group features by data dependency for batch research.

    Uses ``FeatureGroupingSignature`` to ask the LLM to cluster features
    that rely on similar data.  Each group has at most 5 features.

    Three validations (``ValueError`` on failure, retried via ``dspy.Refine``
    in the orchestrator) ensure the grouping is complete:
    1. At least one group is produced.
    2. Total features across groups matches input count.
    3. No features are missing from the groups.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_grouper = dspy.ChainOfThought(FeatureGroupingSignature)

    def forward(
        self,
        task: str,
        feature_plans: dict[str, FeaturePlan],
    ) -> list[dict[str, FeaturePlan]]:
        """Group feature plans for batch research.

        Parameters:
            task: Task description string.
            feature_plans: All feature plans keyed by name.

        Returns:
            List of grouped plan dicts, each with at most 5 features.
        """
        serialized_feature_plans = {
            feature_name: json.loads(dump_as_json(plan))
            for feature_name, plan in feature_plans.items()
        }

        grouped = self.feature_grouper(
            task=task,
            feature_plans=serialized_feature_plans,
        )

        if len(grouped.groups) == 0:
            raise ValueError("No groups were generated. Expected at least one group.")

        if sum(len(group) for group in grouped.groups) != len(serialized_feature_plans):
            raise ValueError(
                f"Feature count mismatch: groups have "
                f"{sum(len(group) for group in grouped.groups)}, "
                f"expected {len(serialized_feature_plans)}"
            )

        grouped_features: list[list[str]] = grouped.groups
        grouped_feature_all = {fn for fns in grouped_features for fn in fns}
        missing = set(feature_plans.keys()) - grouped_feature_all

        if missing:
            raise ValueError(f"Features missing from groups: {', '.join(missing)}")

        grouped_feature_plans = []
        for group in grouped.groups:
            grouped_feature_plan = {
                feature_name: feature_plans[feature_name] for feature_name in group
            }
            grouped_feature_plans.append(grouped_feature_plan)

        logger.info(
            "Grouped %d features into %d groups",
            len(feature_plans),
            len(grouped_feature_plans),
        )
        return grouped_feature_plans
