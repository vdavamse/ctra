"""Tests for Agent orchestrator — _extract_nctids and forward() dispatch.

Covers:
- _extract_nctids: Series, DataFrame with various column names
- forward() iteration 0 vs N dispatch with fully mocked agents
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

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
from tests.test_agents.conftest import planner_prediction, proposer_prediction

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
        suggestion_index=0,
        raw_features={"NCT001": {"feat_a": {"value": 1.0}}},
        raw_val_features={"NCT002": {"feat_a": {"value": 2.0}}},
        raw_test_features={"NCT003": {"feat_a": {"value": 3.0}}},
        none_explanations={},
        builder_meta=builder_meta or {},
    )


# ======================================================================
# _extract_nctids
# ======================================================================


class TestExtractNctids:
    """Tests for Agent._extract_nctids static method."""

    def test_series_input(self) -> None:
        s = pd.Series(["NCT001", "NCT002", "NCT003"])
        result = Agent._extract_nctids(s)
        assert result == ["NCT001", "NCT002", "NCT003"]

    def test_series_int_converted_to_str(self) -> None:
        s = pd.Series([1, 2, 3])
        result = Agent._extract_nctids(s)
        assert result == ["1", "2", "3"]

    def test_dataframe_nctId_column(self) -> None:
        df = pd.DataFrame({"nctId": ["NCT001", "NCT002"], "other": [1, 2]})
        result = Agent._extract_nctids(df)
        assert result == ["NCT001", "NCT002"]

    def test_dataframe_nct_id_column(self) -> None:
        df = pd.DataFrame({"nct_id": ["NCT003", "NCT004"], "other": [1, 2]})
        result = Agent._extract_nctids(df)
        assert result == ["NCT003", "NCT004"]

    def test_dataframe_nctid_lowercase_column(self) -> None:
        df = pd.DataFrame({"nctid": ["NCT005", "NCT006"]})
        result = Agent._extract_nctids(df)
        assert result == ["NCT005", "NCT006"]

    def test_dataframe_priority_nctId_first(self) -> None:
        """nctId should be checked before nct_id."""
        df = pd.DataFrame(
            {
                "nctId": ["NCT001"],
                "nct_id": ["NCT999"],
            }
        )
        result = Agent._extract_nctids(df)
        assert result == ["NCT001"]

    def test_dataframe_fallback_first_column(self) -> None:
        """If no known column, use the first column."""
        df = pd.DataFrame({"trial_identifier": ["NCT007"], "label": [1]})
        result = Agent._extract_nctids(df)
        assert result == ["NCT007"]

    def test_empty_series(self) -> None:
        s = pd.Series([], dtype=str)
        result = Agent._extract_nctids(s)
        assert result == []


# ======================================================================
# Agent.__init__ — Task enum vs string
# ======================================================================


class TestAgentTaskParam:
    """Tests for Agent accepting Task enum or plain string."""

    @patch("ctra.agents.orchestrator.ResettingRefine", side_effect=lambda module, **kw: module)
    @patch("ctra.agents.orchestrator.Initializer")
    @patch("ctra.agents.orchestrator.FeatureProposer")
    @patch("ctra.agents.orchestrator.FeaturePlanner")
    @patch("ctra.agents.orchestrator.Evaluator")
    @patch("ctra.agents.orchestrator.FeatureGrouper")
    def test_task_enum_sets_both_attrs(self, *mocks: MagicMock) -> None:
        """When a Task enum is passed, both .task and .task_description are set."""
        for m in mocks:
            m.return_value = MagicMock()

        task = Task.TRIAL_OUTCOME_PHASE_2
        agent = Agent(
            task=task,
            X_train=pd.Series(["NCT001"]),
            X_val=pd.Series(["NCT002"]),
            y_train=np.array([1]),
            y_val=np.array([0]),
            X_test=pd.Series(["NCT003"]),
            y_test=np.array([1]),
        )

        assert agent.task is Task.TRIAL_OUTCOME_PHASE_2
        assert agent.task_description == task.description
        assert "Phase 2" in agent.task_description

    @patch("ctra.agents.orchestrator.ResettingRefine", side_effect=lambda module, **kw: module)
    @patch("ctra.agents.orchestrator.Initializer")
    @patch("ctra.agents.orchestrator.FeatureProposer")
    @patch("ctra.agents.orchestrator.FeaturePlanner")
    @patch("ctra.agents.orchestrator.Evaluator")
    @patch("ctra.agents.orchestrator.FeatureGrouper")
    def test_string_sets_task_to_none(self, *mocks: MagicMock) -> None:
        """When a raw string is passed, .task is None and .task_description is the string."""
        for m in mocks:
            m.return_value = MagicMock()

        agent = Agent(
            task="Predict trial success",
            X_train=pd.Series(["NCT001"]),
            X_val=pd.Series(["NCT002"]),
            y_train=np.array([1]),
            y_val=np.array([0]),
            X_test=pd.Series(["NCT003"]),
            y_test=np.array([1]),
        )

        assert agent.task is None
        assert agent.task_description == "Predict trial success"


# ======================================================================
# forward() — iteration 0 dispatch
# ======================================================================


class TestForwardIteration0:
    """Tests for Agent.forward() when previous_output is None (iteration 0)."""

    @pytest.fixture()
    def mock_agent(self, monkeypatch: pytest.MonkeyPatch) -> Agent:
        """Create Agent with all sub-agents fully mocked."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.mcts.feature_cache_dir = "/tmp/test_cache"
        mock_settings.model.classifiers = []  # Skip model training
        monkeypatch.setattr("ctra.agents.orchestrator.get_settings", lambda: mock_settings)

        # Bypass the Refine wrapper — return the module unchanged
        monkeypatch.setattr("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)

        # Mock sub-agent constructors
        with (
            patch("ctra.agents.orchestrator.Initializer") as mock_init_cls,
            patch("ctra.agents.orchestrator.FeatureProposer") as mock_prop_cls,
            patch("ctra.agents.orchestrator.FeaturePlanner") as mock_plan_cls,
            patch("ctra.agents.orchestrator.Evaluator") as mock_eval_cls,
            patch("ctra.agents.orchestrator.FeatureGrouper") as mock_group_cls,
        ):
            for cls in [mock_prop_cls, mock_plan_cls, mock_eval_cls, mock_group_cls]:
                cls.return_value = MagicMock()

            mock_init_cls.return_value = MagicMock()

            agent = Agent(
                task="Predict trial success",
                X_train=pd.Series(["NCT001"]),
                X_val=pd.Series(["NCT002"]),
                y_train=np.array([1]),
                y_val=np.array([0]),
                X_test=pd.Series(["NCT003"]),
                y_test=np.array([1]),
            )

        return agent

    def test_iteration_0_calls_initializer(
        self,
        mock_agent: Agent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Iteration 0 should call the initializer."""
        plan = _make_plan("feat_a")
        mock_agent.initializer.return_value = {"feat_a": plan}

        with (
            patch("ctra.agents.orchestrator.compute_features") as mock_compute,
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
        ):
            mock_compute.return_value = (
                {"NCT001": {"feat_a": {"value": 1.0}}},
                {},
                {},
            )
            mock_to_df.return_value = pd.DataFrame({"id": ["NCT001"], "feat_a--value": [1.0]})

            result = mock_agent.forward(previous_output=None)

        mock_agent.initializer.assert_called_once()
        assert isinstance(result, AgentOutput)

    def test_iteration_0_computes_train_val_test(
        self,
        mock_agent: Agent,
    ) -> None:
        """Iteration 0 should call compute_features three times."""
        plan = _make_plan("feat_a")
        mock_agent.initializer.return_value = {"feat_a": plan}

        with (
            patch("ctra.agents.orchestrator.compute_features") as mock_compute,
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
        ):
            mock_compute.return_value = ({"NCT001": {}}, {}, {})
            mock_to_df.return_value = pd.DataFrame({"id": ["NCT001"]})

            mock_agent.forward(previous_output=None)

        # Should be called 3 times: train, val, test
        assert mock_compute.call_count == 3


# ======================================================================
# forward() — iteration N dispatch
# ======================================================================


class TestForwardIterationN:
    """Tests for Agent.forward() when previous_output is provided."""

    @pytest.fixture()
    def mock_agent(self, monkeypatch: pytest.MonkeyPatch) -> Agent:
        """Create Agent with all sub-agents fully mocked."""
        mock_settings = MagicMock()
        mock_settings.mcts.feature_cache_dir = "/tmp/test_cache"
        mock_settings.model.classifiers = []
        monkeypatch.setattr("ctra.agents.orchestrator.get_settings", lambda: mock_settings)

        # Bypass the Refine wrapper — return the module unchanged
        monkeypatch.setattr("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)

        with (
            patch("ctra.agents.orchestrator.Initializer") as mock_init_cls,
            patch("ctra.agents.orchestrator.FeatureProposer") as mock_prop_cls,
            patch("ctra.agents.orchestrator.FeaturePlanner") as mock_plan_cls,
            patch("ctra.agents.orchestrator.Evaluator") as mock_eval_cls,
            patch("ctra.agents.orchestrator.FeatureGrouper") as mock_group_cls,
        ):
            for cls in [mock_prop_cls, mock_plan_cls, mock_eval_cls, mock_group_cls]:
                cls.return_value = MagicMock()

            mock_init_cls.return_value = MagicMock()

            agent = Agent(
                task="Predict trial success",
                X_train=pd.Series(["NCT001"]),
                X_val=pd.Series(["NCT002"]),
                y_train=np.array([1]),
                y_val=np.array([0]),
                X_test=pd.Series(["NCT003"]),
                y_test=np.array([1]),
            )

        return agent

    def test_iteration_n_calls_proposer(
        self,
        mock_agent: Agent,
    ) -> None:
        """Iteration N should call the proposer with previous_output.

        Both doubles return the ``dspy.Prediction`` shape the real modules
        produce, so this test only passes if the orchestrator unwraps them:
        an un-unwrapped proposer Prediction fails ``is_valid_proposer`` and the
        iteration is skipped before the planner is reached.
        """
        prev_output = _make_output()

        new_plan = _make_plan("feat_b")
        mock_agent.proposer.return_value = proposer_prediction(
            feature_operation=FeatureOp.ADD,
            feature_name="feat_b",
            feature_explanation="Add a safety feature",
        )
        mock_agent.planner.return_value = planner_prediction(new_plan, MagicMock())

        with (
            patch("ctra.agents.orchestrator.compute_features") as mock_compute,
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
        ):
            mock_compute.return_value = (
                {"NCT001": {"feat_b": {"value": 5.0}}},
                {},
                {},
            )
            mock_to_df.return_value = pd.DataFrame(
                {"id": ["NCT001"], "feat_a--value": [1.0], "feat_b--value": [5.0]}
            )

            result = mock_agent.forward(previous_output=prev_output)

        mock_agent.proposer.assert_called_once()
        mock_agent.planner.assert_called_once()
        assert isinstance(result, AgentOutput)
        assert "feat_b" in result.feature_plans

    def test_remove_operation_deletes_feature(
        self,
        mock_agent: Agent,
    ) -> None:
        """REMOVE operation should remove the feature from plans and values."""
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        prev_output = _make_output(feature_plans=plans)
        # Update raw features to include both features
        prev_output = prev_output._replace(
            raw_features={
                "NCT001": {
                    "feat_a": {"value": 1.0},
                    "feat_b": {"value": 2.0},
                },
            },
            raw_val_features={
                "NCT002": {
                    "feat_a": {"value": 3.0},
                    "feat_b": {"value": 4.0},
                },
            },
            raw_test_features={
                "NCT003": {
                    "feat_a": {"value": 5.0},
                    "feat_b": {"value": 6.0},
                },
            },
        )

        # Legacy raw shape: the unwrap helpers must keep tolerating it.
        mock_agent.proposer.return_value = ProposerOutput(
            feature_operation=FeatureOp.REMOVE,
            feature_name="feat_b",
            feature_explanation="Remove unimportant feature",
        )

        with patch("ctra.agents.orchestrator.features_to_df") as mock_to_df:
            mock_to_df.return_value = pd.DataFrame({"id": ["NCT001"], "feat_a--value": [1.0]})

            result = mock_agent.forward(previous_output=prev_output)

        # feat_b should be removed from feature_plans
        assert "feat_b" not in result.feature_plans
        assert "feat_a" in result.feature_plans

    def test_add_operation_calls_planner_and_compute(
        self,
        mock_agent: Agent,
    ) -> None:
        """ADD operation should call planner and compute_features."""
        prev_output = _make_output()

        mock_agent.proposer.return_value = proposer_prediction(
            feature_operation=FeatureOp.ADD,
            feature_name="feat_new",
            feature_explanation="Add new feature",
        )
        new_plan = _make_plan("feat_new")
        # Legacy (plan, raw) tuple: the unwrap helpers must keep tolerating it.
        mock_agent.planner.return_value = (new_plan, MagicMock())

        with (
            patch("ctra.agents.orchestrator.compute_features") as mock_compute,
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
        ):
            mock_compute.return_value = (
                {"NCT001": {"feat_new": {"value": 9.0}}},
                {},
                {},
            )
            mock_to_df.return_value = pd.DataFrame(
                {"id": ["NCT001"], "feat_a--value": [1.0], "feat_new--value": [9.0]}
            )

            result = mock_agent.forward(previous_output=prev_output)

        # Planner should have been called
        mock_agent.planner.assert_called_once()
        # compute_features should be called 3 times (train, val, test) for the new feature
        assert mock_compute.call_count == 3
        # New feature should be in plans
        assert "feat_new" in result.feature_plans


# ======================================================================
# Regression: task_namespace resolution (C1 from issue #26 code review)
# ======================================================================


class TestTaskNamespaceResolution:
    """Regression tests for Agent.forward() passing `task_namespace` as a
    string (not a bound method) to compute_features.

    The original bug: `Task.output_subdir` was a plain method, not a
    @property, but orchestrator.py read it as an attribute. That silently
    sent a `<bound method>` object into `compute_features` -> `_store_path`,
    which crashes the first real MCTS run with a TypeError. It was not
    caught by existing tests because they all mock `compute_features` and
    never inspected the kwargs.

    This class asserts `task_namespace` arrives at `compute_features` as
    the correct string for each Task enum member and for the no-Task path.
    """

    def _build_agent(
        self,
        task: Task | str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> Agent:
        mock_settings = MagicMock()
        mock_settings.mcts.feature_cache_dir = "/tmp/test_cache"
        mock_settings.mcts.feature_store_dir = "/tmp/test_store"
        mock_settings.mcts.feature_store_enabled = True
        mock_settings.model.classifiers = []
        monkeypatch.setattr("ctra.agents.orchestrator.get_settings", lambda: mock_settings)
        monkeypatch.setattr("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)

        with (
            patch("ctra.agents.orchestrator.Initializer") as mock_init_cls,
            patch("ctra.agents.orchestrator.FeatureProposer") as mock_prop_cls,
            patch("ctra.agents.orchestrator.FeaturePlanner") as mock_plan_cls,
            patch("ctra.agents.orchestrator.Evaluator") as mock_eval_cls,
            patch("ctra.agents.orchestrator.FeatureGrouper") as mock_group_cls,
        ):
            for cls in [mock_prop_cls, mock_plan_cls, mock_eval_cls, mock_group_cls]:
                cls.return_value = MagicMock()
            mock_init_cls.return_value = MagicMock()

            agent = Agent(
                task=task,
                X_train=pd.Series(["NCT001"]),
                X_val=pd.Series(["NCT002"]),
                y_train=np.array([1]),
                y_val=np.array([0]),
                X_test=pd.Series(["NCT003"]),
                y_test=np.array([1]),
            )

        plan = _make_plan("feat_a")
        agent.initializer.return_value = {"feat_a": plan}
        return agent

    @pytest.mark.parametrize(
        ("task", "expected_namespace"),
        [
            (Task.TRIAL_OUTCOME_PHASE_1, "phase1"),
            (Task.TRIAL_OUTCOME_PHASE_2, "phase2"),
            (Task.TRIAL_OUTCOME_PHASE_3, "phase3"),
        ],
    )
    def test_task_enum_yields_phase_namespace(
        self,
        task: Task,
        expected_namespace: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each Task enum member must resolve to its `phaseN` subdirectory."""
        agent = self._build_agent(task, monkeypatch)

        with (
            patch("ctra.agents.orchestrator.compute_features") as mock_compute,
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
        ):
            mock_compute.return_value = ({"NCT001": {}}, {}, {})
            mock_to_df.return_value = pd.DataFrame({"id": ["NCT001"]})

            agent.forward(previous_output=None)

        # Every compute_features call must receive the phase string verbatim,
        # not a bound-method repr, not a MagicMock, not the enum name.
        assert mock_compute.call_count >= 1
        for call in mock_compute.call_args_list:
            actual = call.kwargs["task_namespace"]
            assert isinstance(actual, str), (
                f"task_namespace must be a string, got {type(actual).__name__}: {actual!r}"
            )
            assert actual == expected_namespace

    def test_string_task_falls_back_to_default_namespace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bare-string task description (no Task enum) must fall back to
        the literal string ``"default"``, not crash."""
        agent = self._build_agent("Predict trial success", monkeypatch)

        with (
            patch("ctra.agents.orchestrator.compute_features") as mock_compute,
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
        ):
            mock_compute.return_value = ({"NCT001": {}}, {}, {})
            mock_to_df.return_value = pd.DataFrame({"id": ["NCT001"]})

            agent.forward(previous_output=None)

        for call in mock_compute.call_args_list:
            assert call.kwargs["task_namespace"] == "default"


# ======================================================================
# Classifier type assertion — regression test for issue #36
# ======================================================================


class TestClassifierTypeAssertion:
    """Regression test: verify the trained classifier type matches the request.

    This would have caught the original bug where TabPFN was silently
    replaced by LogisticRegression.
    """

    @pytest.mark.parametrize(
        ("classifier_type", "expected_class_name"),
        [
            (ClassifierType.XGBOOST, "XGBClassifier"),
            # TabPFN requires CUDA -- skip in CI
            pytest.param(
                ClassifierType.TABPFN,
                "TabPFNClassifier",
                marks=pytest.mark.skip(reason="TabPFN requires CUDA"),
            ),
        ],
    )
    def test_classifier_type_matches_request(
        self,
        classifier_type: ClassifierType,
        expected_class_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ModelRegistry.create_classifier must return the correct class."""
        from ctra.config.settings import ModelConfig
        from ctra.models.model_registry import ModelRegistry

        # Use minimal config
        config = ModelConfig(
            classifiers=[classifier_type],
            shap_enabled=False,
            xgb_n_estimators=10,
            xgb_max_depth=3,
            xgb_learning_rate=0.1,
        )

        clf = ModelRegistry.create_classifier(classifier_type, config=config)
        assert type(clf).__name__ == expected_class_name
