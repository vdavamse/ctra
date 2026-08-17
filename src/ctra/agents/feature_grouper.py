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

    Filtering (``forward()`` does not raise, validated via ``is_valid_grouper``):
    1. Drops groups with names not in feature_plans.
    2. Drops features already claimed by an earlier group (de-duplication).
    3. Drops empty groups after filtering.
    4. Returns the remaining valid groups (may be partial coverage).
    """

    def __init__(self, task_description: str | None = None) -> None:
        super().__init__()
        # Optional: ``forward()`` receives the task per call, so this is only a
        # default for callers that would rather configure it once. Keep it
        # defaulted -- ``scripts/predict.py`` constructs the grouper with no args.
        self.task_description = task_description
        self.feature_grouper = dspy.ChainOfThought(FeatureGroupingSignature)

    def forward(
        self,
        feature_plans: dict[str, FeaturePlan],
        task: str | None = None,
    ) -> list[dict[str, FeaturePlan]]:
        """Group feature plans for batch research (filtering, non-raising).

        Filters the LLM output to drop:
        - Feature names not in feature_plans (stray names).
        - Features already claimed by an earlier group (de-duplication).
        - Empty groups after filtering.

        Returns the remaining valid groups, which may have partial coverage
        if the LLM produced invalid assignments. Validation is deferred to the
        caller via ``is_valid_grouper`` (e.g., in orchestrator post-checks and
        Site 3 fallback).

        Parameters:
            feature_plans: All feature plans keyed by name.
            task: Task description string. Falls back to the ``task_description``
                given at construction.

        Returns:
            List of grouped plan dicts (may be empty or partial coverage).
        """
        task_text = task if task is not None else self.task_description
        if task_text is None:
            raise ValueError(
                "FeatureGrouper needs a task description: pass task= to forward() "
                "or task_description= to __init__."
            )

        serialized_feature_plans = {
            feature_name: json.loads(dump_as_json(plan))
            for feature_name, plan in feature_plans.items()
        }

        grouped = self.feature_grouper(
            task=task_text,
            feature_plans=serialized_feature_plans,
        )

        # Filter and deduplicate: build groups with only valid, unclaimed features
        valid_feature_names = set(feature_plans.keys())
        claimed_features: set[str] = set()
        grouped_feature_plans: list[dict[str, FeaturePlan]] = []

        for group in grouped.groups:
            filtered_group = {}
            for feature_name in group:
                # Skip: (1) not in feature_plans, (2) already claimed
                if feature_name in valid_feature_names and feature_name not in claimed_features:
                    filtered_group[feature_name] = feature_plans[feature_name]
                    claimed_features.add(feature_name)

            # Only append non-empty groups
            if filtered_group:
                grouped_feature_plans.append(filtered_group)

        logger.info(
            "Grouped %d features into %d groups (claimed %d, valid %d)",
            len(feature_plans),
            len(grouped_feature_plans),
            len(claimed_features),
            len(valid_feature_names),
        )
        return grouped_feature_plans
