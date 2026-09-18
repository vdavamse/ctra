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
    """FeatureProposer.forward() returns dspy.Prediction wrapping ProposerOutput.

    Previously, forward() would raise ValueError if the proposed feature name
    already existed. Now it returns the output and lets the orchestrator's
    is_valid_proposer check handle the validation.

    Arranges: Create proposer, mock dspy.Predict to return ADD with existing name.
    Acts: Call forward().
    Asserts: dspy.Prediction(proposal=ProposerOutput) is returned without raising ValueError.
    """
    import dspy

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

        # Call forward — should NOT raise, should return dspy.Prediction
        result = proposer.forward(previous_output=previous)

        # Assert we got a dspy.Prediction with proposal field
        assert isinstance(result, dspy.Prediction)
        proposal = result.proposal
        assert isinstance(proposal, ProposerOutput)
        assert proposal.feature_name == "feat_a"
        assert proposal.feature_operation == FeatureOp.ADD


def test_proposer_remove_nonexistent_name_no_raise():
    """FeatureProposer.forward() returns dspy.Prediction wrapping ProposerOutput.

    Arranges: Create proposer, mock dspy.Predict to return REMOVE with nonexistent name.
    Acts: Call forward().
    Asserts: dspy.Prediction(proposal=ProposerOutput) is returned without raising ValueError.
    """
    import dspy

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

        # Assert we got a dspy.Prediction with proposal field
        assert isinstance(result, dspy.Prediction)
        proposal = result.proposal
        assert isinstance(proposal, ProposerOutput)
        assert proposal.feature_name == "nonexistent_feature"
        assert proposal.feature_operation == FeatureOp.REMOVE


# ======================================================================
# Operation normalisation
# ======================================================================


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("add", FeatureOp.ADD),
        ("Add", FeatureOp.ADD),
        ("ADD", FeatureOp.ADD),
        ("  add  ", FeatureOp.ADD),
        ("Remove", FeatureOp.REMOVE),
        ("REFINE", FeatureOp.REFINE),
        (FeatureOp.ADD, FeatureOp.ADD),
    ],
)
def test_proposer_normalises_operation_case_and_whitespace(raw, expected):
    """A miscased/padded op is recovered rather than left as a raw string.

    Without this, ``FeatureOp()`` coercion fails, the raw string survives into
    ``ProposerOutput``, ``is_valid_proposer`` rejects it as unrecognised, and the
    iteration is skipped -- burning three LM calls over a formatting slip.
    """
    proposer = FeatureProposer(task_description="Predict trial outcome")
    previous = _make_output()

    with patch.object(proposer, "proposer") as mock_proposer_module:
        mock_result = MagicMock()
        mock_result.operation = raw
        mock_result.feature_name = "feat_new"
        mock_result.operation_description = "explanation"
        mock_proposer_module.return_value = mock_result

        result = proposer.forward(previous_output=previous)

    assert result.proposal.feature_operation is expected


def test_proposer_keeps_unrecognisable_operation_as_raw_value():
    """A genuinely unknown op is passed through, not raised on.

    ``forward()`` must stay non-raising; ``is_valid_proposer`` is what rejects it.
    """
    proposer = FeatureProposer(task_description="Predict trial outcome")
    previous = _make_output()

    with patch.object(proposer, "proposer") as mock_proposer_module:
        mock_result = MagicMock()
        mock_result.operation = "obliterate"
        mock_result.feature_name = "feat_a"
        mock_result.operation_description = "explanation"
        mock_proposer_module.return_value = mock_result

        result = proposer.forward(previous_output=previous)

    assert result.proposal.feature_operation == "obliterate"


# ======================================================================
# Empty-suggestion degradation (must not raise inside dspy.Refine)
# ======================================================================


def test_proposer_with_no_suggestions_does_not_raise():
    """``get_next_suggestion()`` raising must not escape ``forward()``.

    ``evaluator`` returns ``suggestions=[]`` when every analysis step fails. A
    raise here propagates through ``dspy.Refine`` past the caller's
    ``is_valid_proposer`` guard -- the dead-skip-branch failure this package was
    fixed to eliminate. Degrade to an empty suggestion instead.
    """
    import dspy

    proposer = FeatureProposer(task_description="Predict trial outcome")

    base = _make_output()
    empty = base._replace(
        eval_outputs={
            name: EvalOutput(model_eval_result=ev.model_eval_result, suggestions=[])
            for name, ev in base.eval_outputs.items()
        }
    )

    with patch.object(proposer, "proposer") as mock_proposer_module:
        mock_result = MagicMock()
        mock_result.operation = FeatureOp.ADD.value
        mock_result.feature_name = "feat_new"
        mock_result.operation_description = "explanation"
        mock_proposer_module.return_value = mock_result

        result = proposer.forward(previous_output=empty)

    assert isinstance(result, dspy.Prediction)
    assert isinstance(result.proposal, ProposerOutput)
    # The inner predictor still ran, with an empty suggestion.
    assert mock_proposer_module.call_args.kwargs["suggestion"] == ""
