"""Test that FeatureProposer.forward() doesn't raise on invalid LLM output.

Verifies that validation has been moved from forward() to reward functions,
so invalid outputs are returned as-is (not raised as ValueError).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    from ctra.agents.data_models import (
        AgentOutput,
        EvalOutput,
        FeatureOp,
        FeaturePlan,
        FeatureSource,
        FeatureType,
        ModelEvalResult,
        ProposerOutput,
    )
    from ctra.agents.feature_proposer import FeatureProposer

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


def _make_output() -> AgentOutput:
    er = ModelEvalResult(
        roc_auc=0.8,
        f1=0.75,
        pr_auc=0.7,
        interaction_values={},
        wrong_idxs=[0],
        wrong_preds=[1],
        wrong_df=pd.DataFrame({"id": ["NCT001"]}, index=[0]),
        pipeline=None,
    )
    eval_outputs = {
        "xgboost": EvalOutput(
            model_eval_result=er,
            suggestions=["Add a new feature"],
        ),
    }
    return AgentOutput(
        eval_outputs=eval_outputs,
        test_eval_outputs={"xgboost": er},
        operation=None,
        feature_plans={"feat_a": _make_plan("feat_a")},
        df=pd.DataFrame({"feat_a--value": [1.0]}),
        val_df=pd.DataFrame({"feat_a--value": [2.0]}),
        suggestion_index=0,
        raw_features={"NCT001": {"feat_a": {"value": 1.0}}},
        raw_val_features={"NCT002": {"feat_a": {"value": 2.0}}},
        raw_test_features={"NCT003": {"feat_a": {"value": 3.0}}},
        none_explanations={},
        builder_meta={},
    )


def test_proposer_add_with_existing_name_no_raise():
    """FeatureProposer.forward() returns ProposerOutput even for ADD with existing name.

    Previously, forward() would raise ValueError if the proposed feature name
    already existed. Now it returns the output and lets the orchestrator's
    is_valid_proposer check handle the validation.

    Arranges: Create proposer, mock dspy.Predict to return ADD with existing name.
    Acts: Call forward().
    Asserts: ProposerOutput is returned without raising ValueError.
    """
    proposer = FeatureProposer(
        task_description="Predict trial outcome",
    )

    previous = _make_output()

    with patch.object(proposer, "proposer") as mock_proposer_module:
        # Mock the inner dspy.Predict to return a result with ADD and existing name
        mock_result = MagicMock()
        mock_result.operation = FeatureOp.ADD.value
        mock_result.feature_name = "feat_a"  # Already exists in previous_output
        mock_result.operation_description = "Some explanation"
        mock_proposer_module.return_value = mock_result

        # Call forward — should NOT raise, should return ProposerOutput
        result = proposer.forward(previous_output=previous)

        # Assert we got a ProposerOutput back
        assert isinstance(result, ProposerOutput)
        assert result.feature_name == "feat_a"
        assert result.feature_operation == FeatureOp.ADD


def test_proposer_remove_nonexistent_name_no_raise():
    """FeatureProposer.forward() returns ProposerOutput even for REMOVE of nonexistent name.

    Arranges: Create proposer, mock dspy.Predict to return REMOVE with nonexistent name.
    Acts: Call forward().
    Asserts: ProposerOutput is returned without raising ValueError.
    """
    proposer = FeatureProposer(
        task_description="Predict trial outcome",
    )

    previous = _make_output()

    with patch.object(proposer, "proposer") as mock_proposer_module:
        # Mock to return REMOVE of nonexistent feature
        mock_result = MagicMock()
        mock_result.operation = FeatureOp.REMOVE.value
        mock_result.feature_name = "nonexistent_feature"
        mock_result.operation_description = "Remove explanation"
        mock_proposer_module.return_value = mock_result

        # Call forward — should NOT raise
        result = proposer.forward(previous_output=previous)

        # Assert we got a ProposerOutput back
        assert isinstance(result, ProposerOutput)
        assert result.feature_name == "nonexistent_feature"
        assert result.feature_operation == FeatureOp.REMOVE
