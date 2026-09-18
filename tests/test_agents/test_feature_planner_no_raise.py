"""Test that FeaturePlanner.forward() doesn't raise on invalid LLM output.

Verifies that validation has been moved from forward() to reward functions,
so invalid plans are returned as-is (not raised as ValueError).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    import dspy

    from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
    from ctra.agents.feature_planner import FeaturePlanner
    from ctra.agents.reward_fns import unwrap_planner_result

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


def test_planner_missing_possible_values_for_categorical_no_raise():
    """FeaturePlanner.forward() returns dspy.Prediction wrapping (plan, raw).

    Previously, forward() would raise ValueError if a categorical feature lacked
    possible_values. Now it returns the FeaturePlan and lets the orchestrator's
    is_valid_planner check handle the validation.

    Arranges: Create planner, mock dspy.ChainOfThought to return plan with missing possible_values.
    Acts: Call forward().
    Asserts: dspy.Prediction(plan=..., raw=...) is returned without raising ValueError.
    """
    planner = FeaturePlanner(
        task_description="Predict trial outcome",
    )

    with patch.object(planner, "planner") as mock_planner_module:
        # Mock to return a plan with categorical but no possible_values
        mock_result = MagicMock()
        mock_result.feature_type = {"status": FeatureType.CATEGORICAL}
        mock_result.data_sources = [FeatureSource.PUBMED]
        mock_result.example_values = [{"status": "active"}]
        mock_result.possible_values = {}  # Invalid: missing keys for categorical
        mock_result.feature_instructions = "Extract status field."
        mock_planner_module.return_value = mock_result

        # Call forward — should NOT raise
        pred = planner.forward(
            feature_name="trial_status",
            feature_idea="Status of the trial",
        )
        plan, raw = pred.plan, pred.raw

        # Assert we got a dspy.Prediction with plan and raw fields
        assert isinstance(pred, dspy.Prediction)
        assert isinstance(plan, FeaturePlan)
        assert plan.feature_name == "trial_status"
        assert plan.feature_type == {"status": FeatureType.CATEGORICAL}
        assert plan.possible_values == {}  # Invalid, but returned as-is


def test_planner_possible_values_key_not_in_feature_type_no_raise():
    """FeaturePlanner.forward() returns dspy.Prediction wrapping (plan, raw).

    Arranges: Create planner, mock to return plan where possible_values has a key
              not in feature_type (schema mismatch).
    Acts: Call forward().
    Asserts: dspy.Prediction(plan=..., raw=...) is returned without raising ValueError.
    """
    planner = FeaturePlanner(
        task_description="Predict trial outcome",
    )

    with patch.object(planner, "planner") as mock_planner_module:
        # Mock to return a plan with mismatched possible_values keys
        mock_result = MagicMock()
        mock_result.feature_type = {"status": FeatureType.CATEGORICAL}
        mock_result.data_sources = [FeatureSource.PUBMED]
        mock_result.example_values = [{"status": "active"}]
        mock_result.possible_values = {
            "status": ["active", "inactive"],
            "extra_key": ["value"],  # Not in feature_type!
        }
        mock_result.feature_instructions = "Extract status."
        mock_planner_module.return_value = mock_result

        # Call forward — should NOT raise
        pred = planner.forward(
            feature_name="trial_status",
            feature_idea="Status of the trial",
        )
        plan, raw = pred.plan, pred.raw

        # Assert we got a dspy.Prediction with plan and raw fields
        assert isinstance(pred, dspy.Prediction)
        assert isinstance(plan, FeaturePlan)
        assert plan.feature_name == "trial_status"
        # Note: the mismatch is present, but returned as-is for orchestrator validation
        assert "extra_key" in plan.possible_values

        # Verify unwrap_planner_result extracts the tuple correctly
        unwrapped = unwrap_planner_result(pred)
        assert unwrapped == (pred.plan, pred.raw)
