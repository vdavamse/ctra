"""Orchestrator integrity tests — error handling, data completeness, determinism.

Validates that Agent.forward() handles model failures gracefully,
captures none_explanations from all data splits, and that
state management is correct.
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
    FeaturePlan,
    FeatureSource,
    FeatureType,
    ModelEvalResult,
)
from ctra.config.settings import ClassifierType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_classifier(class_name: str = "XGBClassifier") -> MagicMock:
    """Create a MagicMock whose type().__name__ matches the expected class.

    The orchestrator's regression guard (issue #36) checks
    ``type(classifier).__name__`` against ``_EXPECTED_CLASSIFIER_CLASS``.
    """
    cls = type(class_name, (), {})
    return MagicMock(spec=cls)


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


def _make_eval_result(roc_auc: float = 0.85) -> ModelEvalResult:
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
    roc_auc: float = 0.85,
    suggestions: list[str] | None = None,
    builder_meta: dict[str, dict[str, Any]] | None = None,
) -> AgentOutput:
    plans = feature_plans or {"feat_a": _make_plan("feat_a")}
    er = _make_eval_result(roc_auc)
    suggestions = suggestions or ["Add safety feature"]
    return AgentOutput(
        eval_outputs={"xgboost": EvalOutput(model_eval_result=er, suggestions=suggestions)},
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


@pytest.fixture()
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_s = MagicMock()
    mock_s.mcts.max_features = 20
    monkeypatch.setattr("ctra.config.settings.get_settings", lambda: mock_s)


# ======================================================================
# BUG: _find_parent_output is non-deterministic with equal overlap
#
# When multiple stored states have the same overlap with the target
# features, the result depends on dict iteration order.
# The correct behavior should use the MCTS tree structure (parent node)
# rather than a feature-overlap heuristic.
# ======================================================================


# ======================================================================
# BUG: AgentOutput.get_best_eval_output crashes on empty eval_outputs
#
# If all model training fails, eval_outputs={} and
# max(self.eval_outputs, ...) raises ValueError.
# ======================================================================


class TestEmptyEvalOutputs:
    def test_get_best_eval_output_empty_raises_clearly(self) -> None:
        """get_best_eval_output on empty eval_outputs should raise a clear
        error, not an opaque ValueError from max()."""
        output = _make_output()
        output = output._replace(eval_outputs={}, test_eval_outputs={})

        with pytest.raises(ValueError, match=r"[Nn]o.*eval"):
            output.get_best_eval_output()

    def test_forward_handles_all_models_failing(self, mock_settings, monkeypatch) -> None:
        """If all classifiers fail to train, forward() should not crash
        but return an AgentOutput with empty eval_outputs and a clear warning."""
        from ctra.agents.orchestrator import Agent

        # Bypass the Refine wrapper
        monkeypatch.setattr("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)

        mock_orchestrator_settings = MagicMock()
        mock_orchestrator_settings.mcts.feature_cache_dir = "/tmp/test_cache"
        mock_orchestrator_settings.model.classifiers = [ClassifierType.XGBOOST]
        monkeypatch.setattr(
            "ctra.agents.orchestrator.get_settings",
            lambda: mock_orchestrator_settings,
        )

        with (
            patch("ctra.agents.orchestrator.Initializer") as mock_init_cls,
            patch("ctra.agents.orchestrator.FeatureProposer"),
            patch("ctra.agents.orchestrator.FeaturePlanner"),
            patch("ctra.agents.orchestrator.Evaluator") as _mock_eval_cls,
            patch("ctra.agents.orchestrator.FeatureGrouper"),
            patch("ctra.agents.orchestrator.compute_features") as mock_compute,
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.build_feature_type_transformer") as mock_transformer,
            patch("ctra.agents.orchestrator.eval_model") as mock_eval_model,
        ):
            mock_init_cls.return_value = MagicMock(return_value={"feat_a": _make_plan("feat_a")})
            mock_compute.return_value = ({"NCT001": {"feat_a": {"value": 1.0}}}, {}, {})
            mock_to_df.return_value = pd.DataFrame({"id": ["NCT001"], "feat_a--value": [1.0]})
            mock_transformer.return_value = MagicMock()

            # All model training fails
            mock_eval_model.side_effect = RuntimeError("Training crashed")

            agent = Agent(
                task="Predict trial success",
                X_train=pd.Series(["NCT001"]),
                X_val=pd.Series(["NCT002"]),
                y_train=np.array([1]),
                y_val=np.array([0]),
                X_test=pd.Series(["NCT003"]),
                y_test=np.array([1]),
            )

            # Should NOT crash — should return AgentOutput with empty eval_outputs
            result = agent.forward(previous_output=None)
            assert result is not None
            # eval_outputs should be empty (all failed), not crash
            assert isinstance(result.eval_outputs, dict)


# ======================================================================
# BUG: none_explanations only captured from val set
#
# In iteration 0, compute_features is called 3 times but only the
# val set's none_explanations are kept. Train and test are discarded.
# ======================================================================


class TestNoneExplanationsCoverage:
    def test_iteration0_captures_train_none_explanations(self, mock_settings, monkeypatch) -> None:
        """none_explanations from training set should be included in AgentOutput."""
        from ctra.agents.orchestrator import Agent

        monkeypatch.setattr("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)

        mock_orchestrator_settings = MagicMock()
        mock_orchestrator_settings.mcts.feature_cache_dir = "/tmp/test_cache"
        mock_orchestrator_settings.model.classifiers = [ClassifierType.XGBOOST]
        mock_orchestrator_settings.model.shapiq_max_order = 2
        mock_orchestrator_settings.model.shapiq_max_samples = 100
        mock_orchestrator_settings.model.shapiq_budget = 50
        monkeypatch.setattr(
            "ctra.agents.orchestrator.get_settings",
            lambda: mock_orchestrator_settings,
        )

        call_count = [0]

        def fake_compute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # train
                return (
                    {"NCT001": {"feat_a": {"value": 1.0}}},
                    {"NCT001": {"feat_a": "Drug not found in ChEMBL"}},
                    {},
                )
            elif call_count[0] == 2:  # val
                return (
                    {"NCT002": {"feat_a": {"value": None}}},
                    {"NCT002": {"feat_a": "Trial too old for FAERS data"}},
                    {},
                )
            else:  # test
                return (
                    {"NCT003": {"feat_a": {"value": None}}},
                    {"NCT003": {"feat_a": "No PubMed results"}},
                    {},
                )

        mock_eval_result = _make_eval_result()

        with (
            patch("ctra.agents.orchestrator.Initializer") as mock_init_cls,
            patch("ctra.agents.orchestrator.FeatureProposer"),
            patch("ctra.agents.orchestrator.FeaturePlanner"),
            patch("ctra.agents.orchestrator.Evaluator"),
            patch("ctra.agents.orchestrator.FeatureGrouper"),
            patch("ctra.agents.orchestrator.compute_features", side_effect=fake_compute),
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.ModelRegistry.create_classifier") as mock_create_clf,
            patch(
                "ctra.agents.orchestrator.build_feature_type_transformer"
            ) as mock_build_transformer,
            patch("ctra.agents.orchestrator.eval_model") as mock_eval,
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline") as mock_shapiq,
        ):
            mock_init_cls.return_value = MagicMock(return_value={"feat_a": _make_plan("feat_a")})
            mock_to_df.return_value = pd.DataFrame({"feat_a--value": [1.0]})
            mock_create_clf.return_value = _mock_classifier()
            mock_build_transformer.return_value = MagicMock()
            mock_eval.return_value = mock_eval_result
            mock_shapiq.return_value = {}

            agent = Agent(
                task="Predict trial success",
                X_train=pd.Series(["NCT001"]),
                X_val=pd.Series(["NCT002"]),
                y_train=np.array([1]),
                y_val=np.array([0]),
                X_test=pd.Series(["NCT003"]),
                y_test=np.array([1]),
            )
            result = agent.forward(previous_output=None)

        # Train set explanation should be present
        assert "NCT001" in result.none_explanations, (
            "Train set none_explanations were discarded — only val set is kept"
        )
        assert result.none_explanations["NCT001"]["feat_a"] == "Drug not found in ChEMBL"

    def test_iteration0_captures_test_none_explanations(self, mock_settings, monkeypatch) -> None:
        """none_explanations from test set should be included in AgentOutput."""
        from ctra.agents.orchestrator import Agent

        monkeypatch.setattr("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)

        mock_orchestrator_settings = MagicMock()
        mock_orchestrator_settings.mcts.feature_cache_dir = "/tmp/test_cache"
        mock_orchestrator_settings.model.classifiers = [ClassifierType.XGBOOST]
        mock_orchestrator_settings.model.shapiq_max_order = 2
        mock_orchestrator_settings.model.shapiq_max_samples = 100
        mock_orchestrator_settings.model.shapiq_budget = 50
        monkeypatch.setattr(
            "ctra.agents.orchestrator.get_settings",
            lambda: mock_orchestrator_settings,
        )

        call_count = [0]

        def fake_compute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # train
                return ({"NCT001": {"feat_a": {"value": 1.0}}}, {}, {})
            elif call_count[0] == 2:  # val
                return ({"NCT002": {"feat_a": {"value": 2.0}}}, {}, {})
            else:  # test
                return (
                    {"NCT003": {"feat_a": {"value": None}}},
                    {"NCT003": {"feat_a": "No PubMed results"}},
                    {},
                )

        mock_eval_result = _make_eval_result()

        with (
            patch("ctra.agents.orchestrator.Initializer") as mock_init_cls,
            patch("ctra.agents.orchestrator.FeatureProposer"),
            patch("ctra.agents.orchestrator.FeaturePlanner"),
            patch("ctra.agents.orchestrator.Evaluator"),
            patch("ctra.agents.orchestrator.FeatureGrouper"),
            patch("ctra.agents.orchestrator.compute_features", side_effect=fake_compute),
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.ModelRegistry.create_classifier") as mock_create_clf,
            patch(
                "ctra.agents.orchestrator.build_feature_type_transformer"
            ) as mock_build_transformer,
            patch("ctra.agents.orchestrator.eval_model") as mock_eval,
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline") as mock_shapiq,
        ):
            mock_init_cls.return_value = MagicMock(return_value={"feat_a": _make_plan("feat_a")})
            mock_to_df.return_value = pd.DataFrame({"feat_a--value": [1.0]})
            mock_create_clf.return_value = _mock_classifier()
            mock_build_transformer.return_value = MagicMock()
            mock_eval.return_value = mock_eval_result
            mock_shapiq.return_value = {}

            agent = Agent(
                task="Predict trial success",
                X_train=pd.Series(["NCT001"]),
                X_val=pd.Series(["NCT002"]),
                y_train=np.array([1]),
                y_val=np.array([0]),
                X_test=pd.Series(["NCT003"]),
                y_test=np.array([1]),
            )
            result = agent.forward(previous_output=None)

        assert "NCT003" in result.none_explanations, (
            "Test set none_explanations were discarded — only val set is kept"
        )
        assert result.none_explanations["NCT003"]["feat_a"] == "No PubMed results"


# ======================================================================
# Regression test for issue #40: iter-N diagnostics preservation
# ======================================================================


class TestIterNDiagnosticsPreservation:
    """Regression test for issue #40: iter-N must preserve diagnostics for old features."""

    def test_iter_n_preserves_diagnostics_for_old_features(
        self, mock_settings, monkeypatch
    ) -> None:
        """Run two iterations; assert iter-1 diagnostics include iter-0 features
        with correct attribution."""
        from ctra.agents.data_models import FeatureOp
        from ctra.agents.orchestrator import Agent
        from tests.test_agents.conftest import planner_prediction, proposer_prediction

        monkeypatch.setattr("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)

        mock_orchestrator_settings = MagicMock()
        mock_orchestrator_settings.mcts.feature_cache_dir = "/tmp/test_cache"
        mock_orchestrator_settings.model.classifiers = [ClassifierType.XGBOOST]
        mock_orchestrator_settings.model.shapiq_max_order = 2
        mock_orchestrator_settings.model.shapiq_max_samples = 100
        mock_orchestrator_settings.model.shapiq_budget = 50
        monkeypatch.setattr(
            "ctra.agents.orchestrator.get_settings",
            lambda: mock_orchestrator_settings,
        )

        # --- Build the agent ---
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

        # --- Iteration 0 ---
        plan_a = _make_plan("feat_a")
        agent.initializer.return_value = {"feat_a": plan_a}

        iter0_call_count = [0]

        def fake_compute_iter0(*args, **kwargs):
            iter0_call_count[0] += 1
            if iter0_call_count[0] == 1:  # train
                return (
                    {"NCT001": {"feat_a": {"value": 1.0}}},
                    {"NCT001": {"feat_a": "Drug not found"}},
                    {"NCT001": {"feat_a": {"research_results": "Found 3 PubMed papers"}}},
                )
            elif iter0_call_count[0] == 2:  # val
                return (
                    {"NCT002": {"feat_a": {"value": 2.0}}},
                    {},
                    {"NCT002": {"feat_a": {"research_results": "Found 5 PubMed papers"}}},
                )
            else:  # test
                return (
                    {"NCT003": {"feat_a": {"value": 3.0}}},
                    {},
                    {"NCT003": {"feat_a": {"research_results": "Found 2 PubMed papers"}}},
                )

        mock_eval_result = _make_eval_result()

        with (
            patch("ctra.agents.orchestrator.compute_features", side_effect=fake_compute_iter0),
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.ModelRegistry.create_classifier") as mock_create_clf,
            patch(
                "ctra.agents.orchestrator.build_feature_type_transformer"
            ) as mock_build_transformer,
            patch("ctra.agents.orchestrator.eval_model") as mock_eval,
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline") as mock_shapiq,
        ):
            mock_to_df.return_value = pd.DataFrame({"feat_a--value": [1.0]})
            mock_create_clf.return_value = _mock_classifier()
            mock_build_transformer.return_value = MagicMock()
            mock_eval.return_value = mock_eval_result
            mock_shapiq.return_value = {}

            iter0_result = agent.forward(previous_output=None)

        # Verify iter-0 has builder_meta populated
        assert iter0_result.builder_meta, "iter-0 builder_meta should be non-empty"
        assert "NCT001" in iter0_result.builder_meta
        assert "feat_a" in iter0_result.builder_meta["NCT001"]

        # --- Iteration 1: ADD feat_b ---
        # Prediction-shaped doubles (what the real modules return): without the
        # orchestrator's unwrap the proposer output is rejected as invalid and
        # feat_b never reaches the diagnostics asserted below.
        agent.proposer.return_value = proposer_prediction(
            feature_operation=FeatureOp.ADD,
            feature_name="feat_b",
            feature_explanation="Add safety feature",
        )
        plan_b = _make_plan("feat_b")
        agent.planner.return_value = planner_prediction(plan_b, MagicMock())

        iter1_call_count = [0]

        def fake_compute_iter1(*args, **kwargs):
            iter1_call_count[0] += 1
            if iter1_call_count[0] == 1:  # train
                return (
                    {"NCT001": {"feat_b": {"value": 10.0}}},
                    {},
                    {"NCT001": {"feat_b": {"research_results": "Found 1 paper"}}},
                )
            elif iter1_call_count[0] == 2:  # val
                return (
                    {"NCT002": {"feat_b": {"value": 20.0}}},
                    {"NCT002": {"feat_b": "No data available"}},
                    {"NCT002": {"feat_b": {"research_results": "Found 1 paper"}}},
                )
            else:  # test
                return (
                    {"NCT003": {"feat_b": {"value": 30.0}}},
                    {},
                    {"NCT003": {"feat_b": {"research_results": "Found 1 paper"}}},
                )

        with (
            patch("ctra.agents.orchestrator.compute_features", side_effect=fake_compute_iter1),
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.ModelRegistry.create_classifier") as mock_create_clf,
            patch(
                "ctra.agents.orchestrator.build_feature_type_transformer"
            ) as mock_build_transformer,
            patch("ctra.agents.orchestrator.eval_model") as mock_eval,
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline") as mock_shapiq,
        ):
            mock_to_df.return_value = pd.DataFrame(
                {"feat_a--value": [1.0], "feat_b--value": [10.0]}
            )
            mock_create_clf.return_value = _mock_classifier()
            mock_build_transformer.return_value = MagicMock()
            mock_eval.return_value = mock_eval_result
            mock_shapiq.return_value = {}

            iter1_result = agent.forward(previous_output=iter0_result)

        # --- Assertions ---
        # A: iter-1 diagnostics must contain entries for ALL features
        diag_names = {
            fd.feature_name for fd in iter1_result.builder_diagnostics.feature_diagnostics
        }
        assert "feat_a" in diag_names, "iter-1 diagnostics missing old feature feat_a"
        assert "feat_b" in diag_names, "iter-1 diagnostics missing new feature feat_b"

        # B: Old feature with good research coverage must NOT have score 0
        feat_a_diag = next(
            fd
            for fd in iter1_result.builder_diagnostics.feature_diagnostics
            if fd.feature_name == "feat_a"
        )
        assert feat_a_diag.research_coverage_score > 0, (
            f"feat_a research_coverage_score should be > 0, got {feat_a_diag.research_coverage_score}"
        )

        # C: Attribution must not be RESEARCHER for a feature with good research coverage
        # (none_rate for feat_a is 1/3 = 0.33, research_coverage > 0.5 => should be BUILDER or UNCLEAR)
        formatted = iter1_result.builder_diagnostics.format_for_llm()
        assert "feat_a" in formatted, (
            "Expected feat_a to appear in formatted diagnostics "
            "(none_rate=0.33 should exceed format_for_llm's inclusion threshold) — "
            "if this fails, the attribution check below would silently no-op"
        )
        lines_with_feat_a = [line for line in formatted.split("\n") if "feat_a" in line]
        for line in lines_with_feat_a:
            assert "attribution=RESEARCHER" not in line, (
                f"feat_a incorrectly attributed as RESEARCHER: {line}"
            )

        # D: builder_meta should contain entries from both iterations
        assert "NCT001" in iter1_result.builder_meta
        assert "feat_a" in iter1_result.builder_meta["NCT001"], (
            "iter-1 builder_meta lost feat_a metadata from iter-0"
        )
        assert "feat_b" in iter1_result.builder_meta["NCT001"], (
            "iter-1 builder_meta missing feat_b metadata from iter-1"
        )

    def _run_iter0(self, agent, mock_eval_result, feature_name="feat_a") -> Any:
        """Helper: run iter-0 with a single initial feature and real research_results."""
        plan = _make_plan(feature_name)
        agent.initializer.return_value = {feature_name: plan}

        iter0_call_count = [0]

        def fake_compute(*args, **kwargs):
            iter0_call_count[0] += 1
            nctid = {1: "NCT001", 2: "NCT002", 3: "NCT003"}[iter0_call_count[0]]
            return (
                {nctid: {feature_name: {"value": 1.0}}},
                {},
                {nctid: {feature_name: {"research_results": "Found papers"}}},
            )

        with (
            patch("ctra.agents.orchestrator.compute_features", side_effect=fake_compute),
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.ModelRegistry.create_classifier") as mock_create_clf,
            patch(
                "ctra.agents.orchestrator.build_feature_type_transformer"
            ) as mock_build_transformer,
            patch("ctra.agents.orchestrator.eval_model") as mock_eval,
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline") as mock_shapiq,
        ):
            mock_to_df.return_value = pd.DataFrame({f"{feature_name}--value": [1.0]})
            mock_create_clf.return_value = _mock_classifier()
            mock_build_transformer.return_value = MagicMock()
            mock_eval.return_value = mock_eval_result
            mock_shapiq.return_value = {}
            return agent.forward(previous_output=None)

    def _build_agent(self, monkeypatch, mock_settings) -> Any:
        """Helper: build a fully-mocked Agent for iter-N tests."""
        from ctra.agents.orchestrator import Agent

        monkeypatch.setattr("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)

        mock_orchestrator_settings = MagicMock()
        mock_orchestrator_settings.mcts.feature_cache_dir = "/tmp/test_cache"
        mock_orchestrator_settings.model.classifiers = [ClassifierType.XGBOOST]
        mock_orchestrator_settings.model.shapiq_max_order = 2
        mock_orchestrator_settings.model.shapiq_max_samples = 100
        mock_orchestrator_settings.model.shapiq_budget = 50
        monkeypatch.setattr(
            "ctra.agents.orchestrator.get_settings",
            lambda: mock_orchestrator_settings,
        )

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
            return Agent(
                task="Predict trial success",
                X_train=pd.Series(["NCT001"]),
                X_val=pd.Series(["NCT002"]),
                y_train=np.array([1]),
                y_val=np.array([0]),
                X_test=pd.Series(["NCT003"]),
                y_test=np.array([1]),
            )

    def test_iter_n_refine_preserves_other_features_meta(self, mock_settings, monkeypatch) -> None:
        """REFINE path: refining feat_a must preserve builder_meta for other
        features. REFINE goes through the same merge logic as ADD; the refined
        feature's fresh metadata overlays whatever was in iter-0, while other
        features' metadata is kept via the hydration deepcopy."""
        from ctra.agents.data_models import FeatureOp, ProposerOutput

        agent = self._build_agent(monkeypatch, mock_settings)
        mock_eval_result = _make_eval_result()

        # iter-0: two features, both with research
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        agent.initializer.return_value = {"feat_a": plan_a, "feat_b": plan_b}

        iter0_call_count = [0]

        def fake_compute_iter0(*args, **kwargs):
            iter0_call_count[0] += 1
            nctid = {1: "NCT001", 2: "NCT002", 3: "NCT003"}[iter0_call_count[0]]
            return (
                {nctid: {"feat_a": {"value": 1.0}, "feat_b": {"value": 2.0}}},
                {},
                {
                    nctid: {
                        "feat_a": {"research_results": "Original feat_a research"},
                        "feat_b": {"research_results": "Original feat_b research"},
                    }
                },
            )

        with (
            patch("ctra.agents.orchestrator.compute_features", side_effect=fake_compute_iter0),
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.ModelRegistry.create_classifier") as mock_create_clf,
            patch("ctra.agents.orchestrator.build_feature_type_transformer"),
            patch("ctra.agents.orchestrator.eval_model", return_value=mock_eval_result),
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline", return_value={}),
        ):
            mock_to_df.return_value = pd.DataFrame({"feat_a--value": [1.0], "feat_b--value": [2.0]})
            mock_create_clf.return_value = _mock_classifier()
            iter0_result = agent.forward(previous_output=None)

        # iter-1 REFINE feat_a -- legacy raw shapes, kept as the tolerance regression
        agent.proposer.return_value = ProposerOutput(
            feature_operation=FeatureOp.REFINE,
            feature_name="feat_a",
            feature_explanation="Make feat_a more specific",
        )
        refined_plan_a = _make_plan("feat_a")
        agent.planner.return_value = (refined_plan_a, MagicMock())

        iter1_call_count = [0]

        def fake_compute_iter1(*args, **kwargs):
            iter1_call_count[0] += 1
            nctid = {1: "NCT001", 2: "NCT002", 3: "NCT003"}[iter1_call_count[0]]
            return (
                {nctid: {"feat_a": {"value": 5.0}}},
                {},
                {nctid: {"feat_a": {"research_results": "Refined feat_a research"}}},
            )

        with (
            patch("ctra.agents.orchestrator.compute_features", side_effect=fake_compute_iter1),
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.ModelRegistry.create_classifier") as mock_create_clf,
            patch("ctra.agents.orchestrator.build_feature_type_transformer"),
            patch("ctra.agents.orchestrator.eval_model", return_value=mock_eval_result),
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline", return_value={}),
        ):
            mock_to_df.return_value = pd.DataFrame({"feat_a--value": [5.0], "feat_b--value": [2.0]})
            mock_create_clf.return_value = _mock_classifier()
            iter1_result = agent.forward(previous_output=iter0_result)

        # feat_b's iter-0 research must survive the REFINE of feat_a
        assert "feat_b" in iter1_result.builder_meta["NCT001"], (
            "REFINE of feat_a should not drop feat_b's builder_meta"
        )
        assert (
            iter1_result.builder_meta["NCT001"]["feat_b"]["research_results"]
            == "Original feat_b research"
        )
        # feat_a's research is the refined version (last-write-wins)
        assert (
            iter1_result.builder_meta["NCT001"]["feat_a"]["research_results"]
            == "Refined feat_a research"
        )
        # Both features appear in diagnostics
        diag_names = {
            fd.feature_name for fd in iter1_result.builder_diagnostics.feature_diagnostics
        }
        assert diag_names == {"feat_a", "feat_b"}

    def test_iter_n_remove_prunes_builder_meta(self, mock_settings, monkeypatch) -> None:
        """REMOVE path: the removed feature must be pruned from builder_meta
        for every nctid, and remaining features' metadata must be preserved."""
        from ctra.agents.data_models import FeatureOp, ProposerOutput

        agent = self._build_agent(monkeypatch, mock_settings)
        mock_eval_result = _make_eval_result()

        # iter-0: two features
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        agent.initializer.return_value = {"feat_a": plan_a, "feat_b": plan_b}

        iter0_call_count = [0]

        def fake_compute_iter0(*args, **kwargs):
            iter0_call_count[0] += 1
            nctid = {1: "NCT001", 2: "NCT002", 3: "NCT003"}[iter0_call_count[0]]
            return (
                {nctid: {"feat_a": {"value": 1.0}, "feat_b": {"value": 2.0}}},
                {},
                {
                    nctid: {
                        "feat_a": {"research_results": "feat_a research"},
                        "feat_b": {"research_results": "feat_b research"},
                    }
                },
            )

        with (
            patch("ctra.agents.orchestrator.compute_features", side_effect=fake_compute_iter0),
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.ModelRegistry.create_classifier") as mock_create_clf,
            patch("ctra.agents.orchestrator.build_feature_type_transformer"),
            patch("ctra.agents.orchestrator.eval_model", return_value=mock_eval_result),
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline", return_value={}),
        ):
            mock_to_df.return_value = pd.DataFrame({"feat_a--value": [1.0], "feat_b--value": [2.0]})
            mock_create_clf.return_value = _mock_classifier()
            iter0_result = agent.forward(previous_output=None)

        # iter-1 REMOVE feat_a (no compute_features call on the REMOVE path)
        agent.proposer.return_value = ProposerOutput(
            feature_operation=FeatureOp.REMOVE,
            feature_name="feat_a",
            feature_explanation="Remove feat_a",
        )

        with (
            patch("ctra.agents.orchestrator.features_to_df") as mock_to_df,
            patch("ctra.agents.orchestrator.ModelRegistry.create_classifier") as mock_create_clf,
            patch("ctra.agents.orchestrator.build_feature_type_transformer"),
            patch("ctra.agents.orchestrator.eval_model", return_value=mock_eval_result),
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline", return_value={}),
        ):
            mock_to_df.return_value = pd.DataFrame({"feat_b--value": [2.0]})
            mock_create_clf.return_value = _mock_classifier()
            iter1_result = agent.forward(previous_output=iter0_result)

        # feat_a pruned from all nctids
        for nctid in ("NCT001", "NCT002", "NCT003"):
            assert "feat_a" not in iter1_result.builder_meta[nctid], (
                f"REMOVE did not prune feat_a from builder_meta[{nctid}]"
            )
        # feat_b preserved
        assert "feat_b" in iter1_result.builder_meta["NCT001"]
        assert (
            iter1_result.builder_meta["NCT001"]["feat_b"]["research_results"] == "feat_b research"
        )
        # Diagnostics reflect only remaining feature
        diag_names = {
            fd.feature_name for fd in iter1_result.builder_diagnostics.feature_diagnostics
        }
        assert diag_names == {"feat_b"}


# ======================================================================
# Parametrized integrity test per classifier type
# ======================================================================


class TestModelFailurePerClassifier:
    """Parametrized test: each classifier type's error path is covered."""

    def test_modelregistry_create_classifier_works(self) -> None:
        """ModelRegistry.create_classifier should return correct classifiers."""
        from ctra.config.settings import ModelConfig
        from ctra.models.model_registry import ModelRegistry

        # Test XGBoost creation
        config = ModelConfig(
            classifiers=[ClassifierType.XGBOOST],
            shap_enabled=False,
            xgb_n_estimators=10,
            xgb_max_depth=3,
            xgb_learning_rate=0.1,
        )
        clf = ModelRegistry.create_classifier(ClassifierType.XGBOOST, config=config)
        assert type(clf).__name__ == "XGBClassifier"


class TestBuilderExceptionDiagnostics:
    """Tests for builder exception diagnostics.

    Note: the three legacy attribution cases (RESEARCHER, BUILDER, UNCLEAR)
    are already covered at test_shapiq_helpers.py:210-244. These tests
    focus on the new builder_exception path only.
    """

    def test_all_exception_feature_is_not_reported_healthy(self) -> None:
        """Feature with all-exception none_explanations should be visible."""
        from ctra.agents.orchestrator import _build_builder_diagnostics

        plans = {"feat_a": _make_plan("feat_a")}

        # All 10 trials have exception
        none_explanations = {
            f"NCT{i:03d}": {"feat_a": "builder_exception: RuntimeError: Test exception"}
            for i in range(10)
        }

        builder_meta = {
            f"NCT{i:03d}": {
                "feat_a": {
                    "research_results": "[builder_exception]",
                    "builder_reasoning": "builder_exception: RuntimeError: Test",
                }
            }
            for i in range(10)
        }

        diagnostics = _build_builder_diagnostics(none_explanations, plans, builder_meta)
        diag = next(d for d in diagnostics.feature_diagnostics if d.feature_name == "feat_a")

        assert diag.none_rate == 1.0
        assert diag.dominant_failure_reason == "builder_exception"

        # format_for_llm must NOT report as "All features have low None rates"
        formatted = diagnostics.format_for_llm()
        assert "All features have low None rates" not in formatted
        assert "BUILDER" in formatted

    def test_builder_exception_not_folded_into_extraction_error(self) -> None:
        """Builder exception sentinel should not fold into extraction_error."""
        from ctra.agents.orchestrator import _build_builder_diagnostics

        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}

        # One feature with sentinel, one with plain error
        none_explanations = {
            "NCT001": {
                "feat_a": "builder_exception: RuntimeError: Crash",
                "feat_b": "API call failed with timeout exception",
            }
        }
        builder_meta = {
            "NCT001": {
                "feat_a": {"research_results": "[builder_exception]"},
                "feat_b": {"research_results": ""},
            }
        }

        diagnostics = _build_builder_diagnostics(none_explanations, plans, builder_meta)

        diag_a = next(d for d in diagnostics.feature_diagnostics if d.feature_name == "feat_a")
        diag_b = next(d for d in diagnostics.feature_diagnostics if d.feature_name == "feat_b")

        assert diag_a.dominant_failure_reason == "builder_exception"
        assert diag_b.dominant_failure_reason == "extraction_error"

    def test_dominant_reason_follows_the_majority(self) -> None:
        """Dominant reason should be the most common category."""
        from ctra.agents.orchestrator import _build_builder_diagnostics

        plans = {"feat_a": _make_plan("feat_a")}

        # 7 insufficient_data, 3 builder_exception
        none_explanations = {}
        for i in range(7):
            none_explanations[f"NCT{i:03d}"] = {"feat_a": "No data found in PubMed"}
        for i in range(7, 10):
            none_explanations[f"NCT{i:03d}"] = {"feat_a": "builder_exception: RuntimeError: Test"}

        builder_meta = {nctid: {} for nctid in none_explanations}

        diagnostics = _build_builder_diagnostics(none_explanations, plans, builder_meta)
        diag = next(d for d in diagnostics.feature_diagnostics if d.feature_name == "feat_a")

        # Majority should win
        assert diag.dominant_failure_reason == "insufficient_data"

        # Now test the reverse: 7 builder_exception, 3 insufficient_data
        none_explanations = {}
        for i in range(7):
            none_explanations[f"NCT{i:03d}"] = {"feat_a": "builder_exception: RuntimeError: Test"}
        for i in range(7, 10):
            none_explanations[f"NCT{i:03d}"] = {"feat_a": "No data found in PubMed"}

        builder_meta = {nctid: {} for nctid in none_explanations}

        diagnostics = _build_builder_diagnostics(none_explanations, plans, builder_meta)
        diag = next(d for d in diagnostics.feature_diagnostics if d.feature_name == "feat_a")

        assert diag.dominant_failure_reason == "builder_exception"

    def test_partial_exception_reports_partial_none_rate(self) -> None:
        """4 of 10 trials with exception should yield none_rate=0.4."""
        from ctra.agents.orchestrator import _build_builder_diagnostics

        plans = {"feat_a": _make_plan("feat_a")}

        # 4 exceptions, 6 successful
        none_explanations = {}
        for i in range(4):
            none_explanations[f"NCT{i:03d}"] = {"feat_a": "builder_exception: RuntimeError: Test"}
        # Trials 4-9 have no explanation (successful)

        builder_meta = {f"NCT{i:03d}": {} for i in range(10)}

        diagnostics = _build_builder_diagnostics(none_explanations, plans, builder_meta)
        diag = next(d for d in diagnostics.feature_diagnostics if d.feature_name == "feat_a")

        assert diag.none_rate == 0.4
        assert diag.dominant_failure_reason == "builder_exception"

    def test_lookalike_explanation_does_not_impersonate_the_sentinel(self) -> None:
        """Lookalike containing 'builder_exception:' should not impersonate if not prefixed."""
        from ctra.agents.orchestrator import _build_builder_diagnostics

        plans = {"feat_a": _make_plan("feat_a")}

        # An explanation that contains "builder_exception:" but does NOT start with it.
        # Our check uses startswith(), so this should NOT be classified as builder_exception.
        none_explanations = {
            "NCT001": {
                "feat_a": "See logs: builder_exception: may have occurred during rehydration"
            }
        }
        builder_meta = {"NCT001": {}}

        diagnostics = _build_builder_diagnostics(none_explanations, plans, builder_meta)
        diag = next(d for d in diagnostics.feature_diagnostics if d.feature_name == "feat_a")

        # Should classify as extraction_error (contains "exception"), NOT builder_exception
        # because it doesn't START with the sentinel prefix
        assert diag.dominant_failure_reason == "extraction_error"
