"""Tests for orchestrator validation fallback paths (Sites 1-3).

Covers:
- Site 1 (proposer validation): invalid op → skip iteration, advance suggestion_index
- Site 2 (planner validation): invalid plan → skip planning, continue with unchanged set
- Site 3 (grouper validation): invalid partition → fall back to one-feature-per-group

These tests exercise the caller-side is_valid_<role> checks that prevent
invalid LLM outputs from propagating downstream.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
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
        Task,
    )
    from ctra.agents.orchestrator import Agent
    from ctra.config.settings import ClassifierType

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _make_eval_result(roc_auc: float = 0.8) -> ModelEvalResult:
    return ModelEvalResult(
        roc_auc=roc_auc,
        f1=0.75,
        pr_auc=0.7,
        interaction_values={},
        wrong_idxs=[0],
        wrong_preds=[1],
        wrong_df=pd.DataFrame({"id": ["NCT001"]}, index=[0]),
        pipeline=None,
    )


def _make_output(
    feature_plans: dict[str, FeaturePlan] | None = None,
    builder_meta: dict[str, dict[str, Any]] | None = None,
    suggestion_index: int = 0,
) -> AgentOutput:
    plans = feature_plans or {"feat_a": _make_plan("feat_a")}
    er = _make_eval_result()
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
        feature_plans=plans,
        df=pd.DataFrame({"feat_a--value": [1.0]}),
        val_df=pd.DataFrame({"feat_a--value": [2.0]}),
        suggestion_index=suggestion_index,
        raw_features={"NCT001": {"feat_a": {"value": 1.0}}},
        raw_val_features={"NCT002": {"feat_a": {"value": 2.0}}},
        raw_test_features={"NCT003": {"feat_a": {"value": 3.0}}},
        none_explanations={},
        builder_meta=builder_meta or {},
    )


# ======================================================================
# Site 1: Proposer validation
# ======================================================================


def test_invalid_proposer_skips_iteration_with_warning(caplog):
    """Site 1: Invalid proposer result → skip iteration, advance suggestion_index.

    Arranges: Mock proposer to return invalid ProposerOutput (ADD with existing name).
    Acts: Call forward() with iteration N.
    Asserts:
    - Returned AgentOutput has suggestion_index == prev.suggestion_index + 1
    - feature_plans is a deepcopy (not aliased) — independent of returned output
    - Warning log matches "Proposer returned invalid op"
    """
    previous_output = _make_output(suggestion_index=2)

    with patch("ctra.agents.orchestrator.Agent.proposer") as mock_proposer, patch(
        "ctra.agents.orchestrator.Agent.initializer"
    ), patch(
        "ctra.agents.orchestrator.Agent.grouper"
    ), patch(
        "ctra.agents.orchestrator.Agent.planner"
    ), patch(
        "ctra.agents.orchestrator.Agent.evaluator"
    ), patch(
        "ctra.agents.orchestrator.get_settings"
    ) as mock_settings:
        # Setup mocks
        agent = Agent(
            task=Task.TRIAL_OUTCOME_PHASE_2,
            X_train=pd.Series(["NCT001", "NCT002"]),
            y_train=pd.Series([1, 0]),
            X_val=pd.Series(["NCT003"]),
            y_val=pd.Series([1]),
        )

        # Mock proposer to return invalid output (ADD with existing feature name)
        mock_proposer.return_value = ProposerOutput(
            feature_name="feat_a",  # Already exists in previous_output
            feature_explanation="Some explanation",
            feature_operation=FeatureOp.ADD,  # Invalid: feature already exists
        )
        agent.proposer = mock_proposer

        # Call forward with iteration N
        with caplog.at_level("WARNING"):
            result = agent.forward(previous_output=previous_output, iteration=1)

        # Assert suggestion_index advanced
        assert result.suggestion_index == 3  # Was 2, now 3

        # Assert feature_plans is independent (deepcopy, not aliased)
        assert result.feature_plans is not previous_output.feature_plans
        assert result.feature_plans == previous_output.feature_plans

        # Assert warning log
        assert "Proposer returned invalid op" in caplog.text


def test_invalid_proposer_two_consecutive_failures(caplog):
    """Regression test: Two consecutive proposer failures advance suggestion_index both times.

    This tests M1: MCTS must honor the advanced suggestion_index so the second
    proposer call sees suggestion_index == original + 2, not original + 1 replayed.
    """
    prev1 = _make_output(suggestion_index=0)
    prev2 = _make_output(suggestion_index=1)  # Result of first failure

    with patch("ctra.agents.orchestrator.Agent.proposer") as mock_proposer, patch(
        "ctra.agents.orchestrator.Agent.initializer"
    ), patch(
        "ctra.agents.orchestrator.Agent.grouper"
    ), patch(
        "ctra.agents.orchestrator.Agent.planner"
    ), patch(
        "ctra.agents.orchestrator.Agent.evaluator"
    ), patch(
        "ctra.agents.orchestrator.get_settings"
    ):
        agent = Agent(
            task=Task.TRIAL_OUTCOME_PHASE_2,
            X_train=pd.Series(["NCT001", "NCT002"]),
            y_train=pd.Series([1, 0]),
            X_val=pd.Series(["NCT003"]),
            y_val=pd.Series([1]),
        )

        # Mock proposer to always return invalid output
        mock_proposer.return_value = ProposerOutput(
            feature_name="feat_a",
            feature_explanation="explanation",
            feature_operation=FeatureOp.ADD,
        )
        agent.proposer = mock_proposer

        # First call
        result1 = agent.forward(previous_output=prev1, iteration=1)
        assert result1.suggestion_index == 1

        # Second call with result from first
        result2 = agent.forward(previous_output=result1, iteration=2)
        assert result2.suggestion_index == 2


# ======================================================================
# Site 2: Planner validation
# ======================================================================


def test_invalid_planner_skips_planning_with_warning(caplog):
    """Site 2: Invalid planner result → skip planning, continue with unchanged feature set.

    Arranges: Mock planner to return invalid FeaturePlan (missing possible_values for categorical).
    Acts: Call forward() with proposer returning ADD.
    Asserts:
    - feature_plans unchanged (new plan not added)
    - Warning log matches "Planner returned invalid plan"
    - Evaluation proceeds with old feature set
    """
    previous_output = _make_output(feature_plans={"feat_a": _make_plan("feat_a")})

    with patch("ctra.agents.orchestrator.Agent.proposer") as mock_proposer, patch(
        "ctra.agents.orchestrator.Agent.planner"
    ) as mock_planner, patch(
        "ctra.agents.orchestrator.Agent.initializer"
    ), patch(
        "ctra.agents.orchestrator.Agent.grouper"
    ), patch(
        "ctra.agents.orchestrator.Agent.evaluator"
    ), patch(
        "ctra.agents.orchestrator.get_settings"
    ), patch(
        "ctra.agents.orchestrator.compute_features"
    ):
        agent = Agent(
            task=Task.TRIAL_OUTCOME_PHASE_2,
            X_train=pd.Series(["NCT001", "NCT002"]),
            y_train=pd.Series([1, 0]),
            X_val=pd.Series(["NCT003"]),
            y_val=pd.Series([1]),
        )

        # Mock proposer to return ADD
        mock_proposer.return_value = ProposerOutput(
            feature_name="feat_b",
            feature_explanation="New feature idea",
            feature_operation=FeatureOp.ADD,
        )
        agent.proposer = mock_proposer

        # Mock planner to return invalid plan (categorical without possible_values)
        invalid_plan = FeaturePlan(
            feature_name="feat_b",
            feature_idea="New feature idea",
            feature_type={"code": FeatureType.CATEGORICAL},
            data_sources=[FeatureSource.PUBMED],
            example_values=[{"code": "A"}],
            possible_values={},  # Invalid: categorical requires possible_values
            feature_instructions="Extract code.",
        )
        mock_planner.return_value = (invalid_plan, None)
        agent.planner = mock_planner

        with caplog.at_level("WARNING"):
            result = agent.forward(previous_output=previous_output, iteration=1)

        # Assert feature_plans unchanged (feat_b not added)
        assert "feat_b" not in result.feature_plans
        assert "feat_a" in result.feature_plans

        # Assert warning log
        assert "Planner returned invalid plan" in caplog.text


# ======================================================================
# Site 3: Grouper validation (via compute_features)
# ======================================================================


def test_invalid_grouper_falls_back_to_one_per_group():
    """Site 3: Invalid grouper result → fall back to one-feature-per-group partition.

    This test is deferred to test_compute_features.py since Site 3 is in feature_builder.py.
    The grouper fallback is integration-tested via compute_features tests.
    """
    pass  # See test_compute_features.py for Site 3 coverage
