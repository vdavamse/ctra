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

# Maximum features per research batch. ``FeatureGroupingSignature`` instructs the
# LLM to respect this ("There should be a maximum of 5 features in each group"),
# but nothing downstream checked it: an oversized group was researched as a single
# batch, which is what the grouping exists to avoid. ``forward()`` now enforces it.
# Keep in sync with the prompt in ``signatures.py``.
_MAX_GROUP_SIZE = 5


class FeatureGrouper(dspy.Module):  # type: ignore[misc]
    """Group features by data dependency for batch research.

    Uses ``FeatureGroupingSignature`` to ask the LLM to cluster features
    that rely on similar data.  Each group has at most ``_MAX_GROUP_SIZE``
    features — enforced here, not merely requested of the LLM.

    Filtering (``forward()`` does not raise, validated via ``is_valid_grouper``):
    1. Drops groups with names not in feature_plans.
    2. Drops features already claimed by an earlier group (de-duplication).
    3. Drops empty groups after filtering.
    4. Splits any group larger than ``_MAX_GROUP_SIZE`` into chunks of that size.
    5. Returns the remaining valid groups (may be partial coverage).
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

        Then splits any surviving group larger than ``_MAX_GROUP_SIZE``.

        Returns the remaining valid groups, which may have partial coverage
        if the LLM produced invalid assignments. Validation is deferred to the
        caller via ``is_valid_grouper`` (e.g., in orchestrator post-checks and
        Site 3 fallback).

        Never raises: a missing task description is logged as an error and
        returns ``[]``, which the caller repairs into one group per feature.

        Parameters:
            feature_plans: All feature plans keyed by name.
            task: Task description string. Falls back to the ``task_description``
                given at construction.

        Returns:
            List of grouped plan dicts (may be empty or partial coverage).
        """
        task_text = task if task is not None else self.task_description
        if task_text is None:
            # Degrade rather than raise. This is a wiring error, not bad LLM
            # output, but ``forward()`` runs inside a ``Refine`` wrapper that
            # deepcopies the module and retries three times before re-raising,
            # burying the real message under three "Attempt failed" lines.
            # Returning an empty grouping is judged invalid by
            # ``is_valid_grouper``, so ``compute_features`` repairs it into
            # one-feature-per-group: every feature still gets built.
            #
            # What this trade actually costs, measured against a DummyLM: the
            # error is logged once per Refine attempt (three times, not once),
            # and because ``forward()`` no longer raises, Refine reaches its
            # feedback block -- ``dict([])`` succeeds where a non-empty
            # ``list[dict]`` would not -- spending 2 ``OfferFeedback`` LM calls
            # the raise-path skipped. So a misconfiguration is now a recurring
            # cost (those calls, plus permanently losing ~5x batching) instead
            # of a fail-fast. Reachability is low: ``compute_features`` always
            # passes ``task=``, and it is the only caller.
            logger.error(
                "FeatureGrouper has no task description (pass task= to forward() or "
                "task_description= to __init__); returning an empty grouping, which "
                "the caller will repair into one group per feature."
            )
            return []

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

            # Enforce the size cap. The LLM is asked for it but does not always
            # comply, and neither ``is_valid_grouper`` nor the caller's repair
            # would catch a violation. Chunking preserves coverage *and* total
            # count, so it cannot turn a valid partition invalid. Empty groups
            # fall out here naturally: the range below yields nothing.
            names = list(filtered_group)
            for start in range(0, len(names), _MAX_GROUP_SIZE):
                chunk = names[start : start + _MAX_GROUP_SIZE]
                grouped_feature_plans.append({name: filtered_group[name] for name in chunk})

        logger.info(
            "Grouped %d features into %d groups (claimed %d, valid %d)",
            len(feature_plans),
            len(grouped_feature_plans),
            len(claimed_features),
            len(valid_feature_names),
        )
        return grouped_feature_plans
