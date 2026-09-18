"""Test that FeatureGrouper.forward() doesn't raise on invalid LLM output.

Verifies that validation has been moved from forward() to reward functions.
Invalid groupings (empty groups, missing features, stray names) are handled
gracefully without raising ValueError.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import dspy

from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
from ctra.agents.feature_grouper import FeatureGrouper
from ctra.agents.reward_fns import is_valid_grouper


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
    """FeatureGrouper.forward() returns dspy.Prediction(groups=[]).

    Previously, forward() would raise ValueError if groups were empty.
    Now it handles empty groups gracefully, filtering them out.

    Arranges: Create grouper, mock to return empty groups list.
    Acts: Call forward().
    Asserts: dspy.Prediction(groups=[]) is returned without raising ValueError.
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
        groups = result.groups

        # Assert we got a dspy.Prediction with empty groups
        assert isinstance(result, dspy.Prediction)
        assert isinstance(groups, list)
        assert len(groups) == 0


def test_grouper_missing_feature_names_no_raise():
    """FeatureGrouper.forward() returns dspy.Prediction with stray names dropped.

    The defensive filter in forward() drops stray names not in feature_plans,
    allowing the orchestrator to detect the incomplete partition.

    Arranges: Create grouper, mock to return group with stray feature names.
    Acts: Call forward().
    Asserts: dspy.Prediction(groups=...) is returned with stray names dropped.
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
        groups = result.groups

        # Assert dspy.Prediction is returned with stray name dropped from first group
        assert isinstance(result, dspy.Prediction)
        assert len(groups) == 2
        assert "feat_a" in groups[0]
        assert "stray_feature" not in groups[0]
        assert "feat_b" in groups[1]


def test_grouper_feature_count_mismatch_no_raise():
    """FeatureGrouper.forward() returns dspy.Prediction even if count doesn't match input.

    A missing feature results in fewer total features in groups.
    The orchestrator will catch this with is_valid_grouper.

    Arranges: Create grouper, mock to return groups with fewer features than input.
    Acts: Call forward().
    Asserts: dspy.Prediction is returned without raising ValueError (orchestrator validation handles it).
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
        groups = result.groups

        # Assert we got back a dspy.Prediction (orchestrator will detect feat_c is missing)
        assert isinstance(result, dspy.Prediction)
        assert len(groups) == 1
        assert set(groups[0].keys()) == {"feat_a", "feat_b"}


def test_grouper_all_stray_group_is_dropped():
    """A group consisting entirely of unknown names is dropped, not emitted empty.

    Arranges: Mock the grouper to return one all-stray group and one real group.
    Acts: Call forward().
    Asserts: dspy.Prediction returned with only the real group -- no empty dict is emitted.
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
        groups = result.groups

    assert isinstance(result, dspy.Prediction)
    assert len(groups) == 1
    assert set(groups[0].keys()) == {"feat_a"}
    assert all(group for group in groups), "no empty group should be emitted"


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
        groups = result.groups

    # Second group becomes empty after de-duplication and is dropped.
    assert isinstance(result, dspy.Prediction)
    assert len(groups) == 1
    assert set(groups[0].keys()) == {"feat_a", "feat_b"}
    total_assigned = sum(len(group) for group in groups)
    assert total_assigned == len(feature_plans)


def test_grouper_missing_task_description_degrades_instead_of_raising(caplog):
    """A missing task description is logged and yields dspy.Prediction(groups=[]), never a raise.

    ``forward()`` runs inside a ``Refine`` wrapper that deepcopies the module
    and retries three times before re-raising, so a raise here surfaced as three
    "Attempt failed" lines with the real cause buried. Returning ``dspy.Prediction(groups=[])``
    is judged invalid by ``is_valid_grouper``, and ``compute_features`` repairs it into
    one group per feature -- every feature still gets built.
    """
    grouper = FeatureGrouper()  # no task_description, and none passed to forward()

    feature_plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}

    with patch.object(grouper, "feature_grouper") as mock_grouper_module, caplog.at_level("ERROR"):
        result = grouper.forward(feature_plans=feature_plans)
        groups = result.groups

    assert isinstance(result, dspy.Prediction)
    assert groups == []
    assert "task description" in caplog.text
    assert not mock_grouper_module.called, "must not spend an LM call when misconfigured"


def test_grouper_splits_oversized_group():
    """A group larger than the cap is chunked, not researched as one batch.

    ``FeatureGroupingSignature`` asks the LLM for at most 5 per group, but the
    LLM does not always comply and nothing downstream checked it -- so a single
    oversized group silently undid the batching this module exists to provide.
    """
    from ctra.agents.feature_grouper import _MAX_GROUP_SIZE

    grouper = FeatureGrouper(task_description="Predict trial outcome")

    names = [f"feat_{i:02d}" for i in range(12)]
    feature_plans = {n: _make_plan(n) for n in names}

    with patch.object(grouper, "feature_grouper") as mock_grouper_module:
        mock_result = MagicMock()
        mock_result.groups = [names]  # one 12-feature group
        mock_grouper_module.return_value = mock_result

        result = grouper.forward(task="Predict trial outcome", feature_plans=feature_plans)
        groups = result.groups

    assert isinstance(result, dspy.Prediction)
    assert len(groups) == 3, "12 features at a cap of 5 must split into 5 + 5 + 2"
    assert all(len(group) <= _MAX_GROUP_SIZE for group in groups)

    # Chunking must preserve coverage *and* count, or it could turn a valid
    # partition invalid -- which would send compute_features into its repair path.
    assigned = [name for group in groups for name in group]
    assert sorted(assigned) == sorted(names)
    assert len(assigned) == len(set(assigned)) == len(feature_plans)


def test_grouper_chunked_output_stays_valid():
    """The chunked partition must still satisfy ``is_valid_grouper``."""
    grouper = FeatureGrouper(task_description="Predict trial outcome")

    names = [f"feat_{i:02d}" for i in range(7)]
    feature_plans = {n: _make_plan(n) for n in names}

    with patch.object(grouper, "feature_grouper") as mock_grouper_module:
        mock_result = MagicMock()
        mock_result.groups = [names]
        mock_grouper_module.return_value = mock_result

        result = grouper.forward(task="Predict trial outcome", feature_plans=feature_plans)

    # is_valid_grouper is tolerant of both Prediction and legacy list shape
    assert is_valid_grouper({"feature_plans": feature_plans}, result) is True


def test_grouper_group_at_cap_is_not_split():
    """Exactly ``_MAX_GROUP_SIZE`` is within the cap -- an off-by-one guard."""
    from ctra.agents.feature_grouper import _MAX_GROUP_SIZE

    grouper = FeatureGrouper(task_description="Predict trial outcome")

    names = [f"feat_{i}" for i in range(_MAX_GROUP_SIZE)]
    feature_plans = {n: _make_plan(n) for n in names}

    with patch.object(grouper, "feature_grouper") as mock_grouper_module:
        mock_result = MagicMock()
        mock_result.groups = [names]
        mock_grouper_module.return_value = mock_result

        result = grouper.forward(task="Predict trial outcome", feature_plans=feature_plans)
        groups = result.groups

    assert isinstance(result, dspy.Prediction)
    assert len(groups) == 1
    assert set(groups[0]) == set(names)
