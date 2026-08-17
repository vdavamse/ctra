"""Test that FeatureGrouper.forward() doesn't raise on invalid LLM output.

Verifies that validation has been moved from forward() to reward functions.
Invalid groupings (empty groups, missing features, stray names) are handled
gracefully without raising ValueError.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
    from ctra.agents.feature_grouper import FeatureGrouper

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


def _make_plan(name: str) -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[{"value": "1.0"}],
        possible_values={},
        feature_instructions=f"Extract {name}.",
    )


def test_grouper_empty_groups_no_raise():
    """FeatureGrouper.forward() returns list even for empty groups.

    Previously, forward() would raise ValueError if groups were empty.
    Now it handles empty groups gracefully, filtering them out.

    Arranges: Create grouper, mock to return empty groups list.
    Acts: Call forward().
    Asserts: Empty list is returned without raising ValueError.
    """
    grouper = FeatureGrouper(
        task_description="Predict trial outcome",
    )

    feature_plans = {
        "feat_a": _make_plan("feat_a"),
        "feat_b": _make_plan("feat_b"),
    }

    with patch.object(grouper, "feature_grouper") as mock_grouper_module:
        # Mock to return empty groups
        mock_result = MagicMock()
        mock_result.groups = []
        mock_grouper_module.return_value = mock_result

        # Call forward — should NOT raise
        result = grouper.forward(
            task="Predict trial outcome",
            feature_plans=feature_plans,
        )

        # Assert we got an empty list back
        assert isinstance(result, list)
        assert len(result) == 0


def test_grouper_missing_feature_names_no_raise():
    """FeatureGrouper.forward() returns list even when groups reference missing features.

    The defensive filter in forward() drops stray names not in feature_plans,
    allowing the orchestrator to detect the incomplete partition.

    Arranges: Create grouper, mock to return group with stray feature names.
    Acts: Call forward().
    Asserts: Group is returned with stray names dropped.
    """
    grouper = FeatureGrouper(
        task_description="Predict trial outcome",
    )

    feature_plans = {
        "feat_a": _make_plan("feat_a"),
        "feat_b": _make_plan("feat_b"),
    }

    with patch.object(grouper, "feature_grouper") as mock_grouper_module:
        # Mock to return groups with a stray feature name
        mock_result = MagicMock()
        mock_result.groups = [
            ["feat_a", "stray_feature"],  # stray_feature not in feature_plans
            ["feat_b"],
        ]
        mock_grouper_module.return_value = mock_result

        # Call forward — should NOT raise
        result = grouper.forward(
            task="Predict trial outcome",
            feature_plans=feature_plans,
        )

        # Assert stray name was dropped from the first group
        assert len(result) == 2
        assert "feat_a" in result[0]
        assert "stray_feature" not in result[0]
        assert "feat_b" in result[1]


def test_grouper_feature_count_mismatch_no_raise():
    """FeatureGrouper.forward() returns list even if total count doesn't match input.

    A missing feature results in fewer total features in groups.
    The orchestrator will catch this with is_valid_grouper.

    Arranges: Create grouper, mock to return groups with fewer features than input.
    Acts: Call forward().
    Asserts: List is returned without raising ValueError (orchestrator validation handles it).
    """
    grouper = FeatureGrouper(
        task_description="Predict trial outcome",
    )

    feature_plans = {
        "feat_a": _make_plan("feat_a"),
        "feat_b": _make_plan("feat_b"),
        "feat_c": _make_plan("feat_c"),
    }

    with patch.object(grouper, "feature_grouper") as mock_grouper_module:
        # Mock to return groups that skip feat_c
        mock_result = MagicMock()
        mock_result.groups = [
            ["feat_a", "feat_b"],
        ]
        mock_grouper_module.return_value = mock_result

        # Call forward — should NOT raise
        result = grouper.forward(
            task="Predict trial outcome",
            feature_plans=feature_plans,
        )

        # Assert we got back a list (orchestrator will detect feat_c is missing)
        assert isinstance(result, list)
        assert len(result) == 1
        assert set(result[0].keys()) == {"feat_a", "feat_b"}


def test_grouper_all_stray_group_is_dropped():
    """A group consisting entirely of unknown names is dropped, not emitted empty.

    Arranges: Mock the grouper to return one all-stray group and one real group.
    Acts: Call forward().
    Asserts: Only the real group survives -- no empty dict is emitted.
    """
    grouper = FeatureGrouper(task_description="Predict trial outcome")

    feature_plans = {"feat_a": _make_plan("feat_a")}

    with patch.object(grouper, "feature_grouper") as mock_grouper_module:
        mock_result = MagicMock()
        mock_result.groups = [
            ["ghost_one", "ghost_two"],  # nothing here exists
            ["feat_a"],
        ]
        mock_grouper_module.return_value = mock_result

        result = grouper.forward(task="Predict trial outcome", feature_plans=feature_plans)

    assert len(result) == 1
    assert set(result[0].keys()) == {"feat_a"}
    assert all(group for group in result), "no empty group should be emitted"


def test_grouper_duplicate_feature_is_claimed_once():
    """A feature listed in two groups is assigned to the first only.

    Building the same feature twice wastes LLM calls and risks a double merge,
    so ``forward()`` de-duplicates across groups.
    """
    grouper = FeatureGrouper(task_description="Predict trial outcome")

    feature_plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}

    with patch.object(grouper, "feature_grouper") as mock_grouper_module:
        mock_result = MagicMock()
        mock_result.groups = [
            ["feat_a", "feat_b"],
            ["feat_a"],  # duplicate claim
        ]
        mock_grouper_module.return_value = mock_result

        result = grouper.forward(task="Predict trial outcome", feature_plans=feature_plans)

    # Second group becomes empty after de-duplication and is dropped.
    assert len(result) == 1
    assert set(result[0].keys()) == {"feat_a", "feat_b"}
    total_assigned = sum(len(group) for group in result)
    assert total_assigned == len(feature_plans)
