"""Tests for Evaluator forward flow.

Covers Evaluator forward flow with mocked dspy modules.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    from ctra.agents.data_models import (
        EvalOutput,
        FeaturePlan,
        FeatureSource,
        FeatureType,
        ModelEvalResult,
    )
    from ctra.agents.evaluator import Evaluator

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_eval_result(
    roc_auc: float = 0.8,
    wrong_idxs: list[int] | None = None,
    wrong_preds: list[int] | None = None,
    wrong_df_index: list[int] | None = None,
) -> ModelEvalResult:
    wrong_idxs = wrong_idxs or []
    wrong_preds = wrong_preds or []
    wrong_df_index = wrong_df_index if wrong_df_index is not None else wrong_idxs
    return ModelEvalResult(
        roc_auc=roc_auc,
        f1=0.75,
        pr_auc=0.7,
        interaction_values={},
        wrong_idxs=wrong_idxs,
        wrong_preds=wrong_preds,
        wrong_df=pd.DataFrame(
            {"id": [f"NCT{i:03d}" for i in wrong_idxs]},
            index=wrong_df_index,
        )
        if wrong_idxs
        else pd.DataFrame({"id": []}, index=pd.Index([], dtype=int)),
        pipeline=None,
    )


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


class TestEvaluatorForward:
    """Tests for Evaluator.forward() with fully mocked dspy modules."""

    @pytest.fixture()
    def mock_evaluator(self) -> Evaluator:
        """Create Evaluator with mocked dspy modules."""
        with patch("ctra.agents.evaluator.dspy") as mock_dspy:
            # Mock ChainOfThought to return a module-like object
            mock_cot = MagicMock()
            mock_dspy.ChainOfThought.return_value = mock_cot

            evaluator = Evaluator(task_description="Predict trial success")

            # Override the internal modules to be controllable
            evaluator.evaluator_no_example = MagicMock()
            evaluator.summarizer = MagicMock()

            return evaluator

    def test_no_wrong_predictions_returns_generic_suggestions(
        self, mock_evaluator: Evaluator
    ) -> None:
        """When there are no wrong predictions, generic suggestions go through summarizer."""
        model_result = _make_eval_result(roc_auc=0.9, wrong_idxs=[], wrong_preds=[])
        plans = {"feat_a": _make_plan("feat_a")}

        # Mock the generic evaluator
        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = ["Add safety_signal feature"]
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result

        # Configure summarizer to return summarized suggestions
        mock_summary = MagicMock()
        mock_summary.suggestions = ["Add safety_signal feature"]
        mock_evaluator.summarizer.return_value = mock_summary

        result = mock_evaluator.forward(
            feature_plans=plans,
            model_eval_result=model_result,
            none_explanations={},
        )

        assert isinstance(result, EvalOutput)
        assert result.model_eval_result.roc_auc == 0.9
        assert result.suggestions == ["Add safety_signal feature"]
        mock_evaluator.summarizer.assert_called_once()

    def test_with_wrong_predictions_runs_example_evaluators(
        self, mock_evaluator: Evaluator
    ) -> None:
        """When there are wrong predictions, should attempt per-trial analysis and summarize."""
        wrong_idxs = [0, 1]
        wrong_preds = [1, 0]
        model_result = _make_eval_result(
            roc_auc=0.7, wrong_idxs=wrong_idxs, wrong_preds=wrong_preds
        )
        plans = {"feat_a": _make_plan("feat_a")}

        # Mock the generic evaluator
        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = ["Generic suggestion"]
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result

        # Configure summarizer
        mock_summary = MagicMock()
        mock_summary.suggestions = ["Example analysis", "Generic suggestion"]
        mock_evaluator.summarizer.return_value = mock_summary

        # Mock ReAct for per-trial analysis
        mock_react_result = MagicMock()
        mock_react_result.analysis = "Example analysis"

        with (
            patch("ctra.agents.evaluator.dspy") as mock_dspy,
            patch("ctra.agents.evaluator.get_trial_info_dict") as mock_info,
            patch("ctra.agents.evaluator.make_pubmed_search") as _mock_pub,
            patch("ctra.agents.evaluator.make_nct_search") as _mock_nct,
            patch("ctra.agents.evaluator.make_chembl_search") as _mock_chembl,
            patch("ctra.agents.evaluator.make_faers_search") as _mock_faers,
            patch("ctra.agents.evaluator.make_aact_search") as _mock_aact,
            patch("ctra.agents.evaluator.make_primekg_search") as _mock_pkg,
            patch("ctra.agents.evaluator.make_drugsfda_search") as _mock_fda,
        ):
            mock_info.return_value = {"nctId": "NCT000"}
            mock_react_instance = MagicMock(return_value=mock_react_result)
            mock_dspy.ReAct.return_value = mock_react_instance

            result = mock_evaluator.forward(
                feature_plans=plans,
                model_eval_result=model_result,
                none_explanations={},
            )

        assert isinstance(result, EvalOutput)
        mock_evaluator.summarizer.assert_called_once()
        # Suggestions come from the summarizer output
        assert result.suggestions == ["Example analysis", "Generic suggestion"]

    def test_react_failure_does_not_crash(self, mock_evaluator: Evaluator) -> None:
        """If ReAct fails for a trial, it should be skipped gracefully; generic suggestions summarized."""
        wrong_idxs = [0]
        wrong_preds = [1]
        model_result = _make_eval_result(
            roc_auc=0.7, wrong_idxs=wrong_idxs, wrong_preds=wrong_preds
        )
        plans = {"feat_a": _make_plan("feat_a")}

        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = ["Generic suggestion"]
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result

        # Configure summarizer
        mock_summary = MagicMock()
        mock_summary.suggestions = ["Generic suggestion"]
        mock_evaluator.summarizer.return_value = mock_summary

        with (
            patch("ctra.agents.evaluator.dspy") as mock_dspy,
            patch("ctra.agents.evaluator.get_trial_info_dict") as mock_info,
            patch("ctra.agents.evaluator.make_pubmed_search"),
            patch("ctra.agents.evaluator.make_nct_search"),
            patch("ctra.agents.evaluator.make_chembl_search"),
            patch("ctra.agents.evaluator.make_faers_search"),
            patch("ctra.agents.evaluator.make_aact_search"),
            patch("ctra.agents.evaluator.make_primekg_search"),
            patch("ctra.agents.evaluator.make_drugsfda_search"),
        ):
            mock_info.return_value = {"nctId": "NCT000"}
            mock_dspy.ReAct.return_value = MagicMock(side_effect=RuntimeError("LLM timeout"))

            result = mock_evaluator.forward(
                feature_plans=plans,
                model_eval_result=model_result,
                none_explanations={},
            )

        # Should still return valid output with summarized generic suggestions
        assert isinstance(result, EvalOutput)
        assert result.suggestions == ["Generic suggestion"]
        mock_evaluator.summarizer.assert_called_once()

    def test_evaluator_calls_summarizer_when_suggestions_exist(
        self, mock_evaluator: Evaluator
    ) -> None:
        """Behavioral regression: summarizer must be invoked exactly once when suggestions exist."""
        model_result = _make_eval_result(roc_auc=0.85, wrong_idxs=[], wrong_preds=[])
        plans = {"feat_a": _make_plan("feat_a")}

        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = ["Suggestion A", "Suggestion B"]
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result

        mock_summary = MagicMock()
        mock_summary.suggestions = ["Merged suggestion"]
        mock_evaluator.summarizer.return_value = mock_summary

        result = mock_evaluator.forward(
            feature_plans=plans,
            model_eval_result=model_result,
            none_explanations={},
        )

        mock_evaluator.summarizer.assert_called_once_with(
            task="Predict trial success",
            analyses=["Suggestion A", "Suggestion B"],
        )
        assert result.suggestions == ["Merged suggestion"]

    def test_evaluator_skips_summarizer_when_no_suggestions(
        self, mock_evaluator: Evaluator
    ) -> None:
        """Behavioral regression: summarizer must NOT be called when all_suggestions is empty."""
        model_result = _make_eval_result(roc_auc=0.95, wrong_idxs=[], wrong_preds=[])
        plans = {"feat_a": _make_plan("feat_a")}

        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = []
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result

        result = mock_evaluator.forward(
            feature_plans=plans,
            model_eval_result=model_result,
            none_explanations={},
        )

        mock_evaluator.summarizer.assert_not_called()
        assert result.suggestions == []

    def test_suggestions_capped_at_five(self, mock_evaluator: Evaluator) -> None:
        """Contract test: summarizer output should have at most 5 suggestions."""
        model_result = _make_eval_result(roc_auc=0.6, wrong_idxs=[], wrong_preds=[])
        plans = {"feat_a": _make_plan("feat_a")}

        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = [f"Suggestion {i}" for i in range(8)]
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result

        # Summarizer returns 5 (the contract max)
        mock_summary = MagicMock()
        mock_summary.suggestions = [f"Final {i}" for i in range(5)]
        mock_evaluator.summarizer.return_value = mock_summary

        result = mock_evaluator.forward(
            feature_plans=plans,
            model_eval_result=model_result,
            none_explanations={},
        )

        assert len(result.suggestions) <= 5
        mock_evaluator.summarizer.assert_called_once()


class TestEvaluatorWrongRowSelection:
    """Tests for positional row selection in Evaluator.forward()."""

    @pytest.fixture()
    def mock_evaluator(self) -> Evaluator:
        """Create Evaluator with mocked dspy modules."""
        with patch("ctra.agents.evaluator.dspy"):
            evaluator = Evaluator(task_description="Predict trial success")
            evaluator.evaluator_no_example = MagicMock()
            return evaluator

    def test_non_range_index_does_not_raise(self, mock_evaluator: Evaluator) -> None:
        """Non-RangeIndex labels should not raise KeyError during sampling."""
        wrong_idxs = [1, 4]
        wrong_preds = [1, 0]
        wrong_df_index = [20, 50]
        model_result = _make_eval_result(
            roc_auc=0.7,
            wrong_idxs=wrong_idxs,
            wrong_preds=wrong_preds,
            wrong_df_index=wrong_df_index,
        )
        plans = {"feat_a": _make_plan("feat_a")}

        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = []
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result
        mock_evaluator.summarizer = MagicMock(return_value=MagicMock(suggestions=[]))

        # Should not raise KeyError
        with (
            patch("ctra.agents.evaluator.dspy"),
            patch("ctra.agents.evaluator.get_trial_info_dict") as mock_info,
            patch("ctra.agents.evaluator.make_pubmed_search"),
            patch("ctra.agents.evaluator.make_nct_search"),
            patch("ctra.agents.evaluator.make_chembl_search"),
            patch("ctra.agents.evaluator.make_faers_search"),
            patch("ctra.agents.evaluator.make_aact_search"),
            patch("ctra.agents.evaluator.make_primekg_search"),
            patch("ctra.agents.evaluator.make_drugsfda_search"),
        ):
            mock_info.return_value = {"nctId": "NCT000"}
            result = mock_evaluator.forward(
                feature_plans=plans,
                model_eval_result=model_result,
                none_explanations={},
            )

        assert isinstance(result, EvalOutput)

    def test_offset_wrong_idxs_select_matching_rows_and_preds(
        self, mock_evaluator: Evaluator
    ) -> None:
        """Offset wrong_idxs should pick matching rows and predictions (anti-iloc trap)."""
        wrong_idxs = [3, 4]
        wrong_preds = [1, 0]
        model_result = _make_eval_result(
            roc_auc=0.7,
            wrong_idxs=wrong_idxs,
            wrong_preds=wrong_preds,
        )
        plans = {"feat_a": _make_plan("feat_a")}

        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = []
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result

        mock_react_result = MagicMock()
        mock_react_result.analysis = "Example analysis"

        with (
            patch("ctra.agents.evaluator.dspy") as mock_dspy,
            patch("ctra.agents.evaluator.get_trial_info_dict") as mock_info,
            patch("ctra.agents.evaluator.make_pubmed_search"),
            patch("ctra.agents.evaluator.make_nct_search"),
            patch("ctra.agents.evaluator.make_chembl_search"),
            patch("ctra.agents.evaluator.make_faers_search"),
            patch("ctra.agents.evaluator.make_aact_search"),
            patch("ctra.agents.evaluator.make_primekg_search"),
            patch("ctra.agents.evaluator.make_drugsfda_search"),
        ):
            mock_info.return_value = {"nctId": "NCT000"}
            mock_react_instance = MagicMock(return_value=mock_react_result)
            mock_dspy.ReAct.return_value = mock_react_instance
            mock_evaluator.summarizer = MagicMock(
                return_value=MagicMock(suggestions=["Example analysis"])
            )

            _ = mock_evaluator.forward(
                feature_plans=plans,
                model_eval_result=model_result,
                none_explanations={},
            )

            # Verify that the mock was called with correct NCT IDs
            assert mock_react_instance.call_count == 2
            call_args_list = mock_react_instance.call_args_list
            examples = [call.kwargs.get("example", "") for call in call_args_list]

            # Both rows should be present: "## NCT003 Predicted 1, should be 0" and "## NCT004 Predicted 0, should be 1"
            example_text = "\n".join(examples)
            assert "## NCT003 Predicted 1, should be 0" in example_text
            assert "## NCT004 Predicted 0, should be 1" in example_text

    def test_duplicate_index_labels_produce_scalar_rows(self, mock_evaluator: Evaluator) -> None:
        """Duplicate index labels should be handled positionally, not with .loc."""
        wrong_idxs = [0, 1, 2]
        wrong_preds = [1, 0, 1]
        wrong_df_index = [1, 1, 2]
        model_result = _make_eval_result(
            roc_auc=0.7,
            wrong_idxs=wrong_idxs,
            wrong_preds=wrong_preds,
            wrong_df_index=wrong_df_index,
        )
        plans = {"feat_a": _make_plan("feat_a")}

        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = []
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result

        mock_react_result = MagicMock()
        mock_react_result.analysis = "Example analysis"

        with (
            patch("ctra.agents.evaluator.dspy") as mock_dspy,
            patch("ctra.agents.evaluator.get_trial_info_dict") as mock_info,
            patch("ctra.agents.evaluator.make_pubmed_search"),
            patch("ctra.agents.evaluator.make_nct_search"),
            patch("ctra.agents.evaluator.make_chembl_search"),
            patch("ctra.agents.evaluator.make_faers_search"),
            patch("ctra.agents.evaluator.make_aact_search"),
            patch("ctra.agents.evaluator.make_primekg_search"),
            patch("ctra.agents.evaluator.make_drugsfda_search"),
        ):
            mock_info.return_value = {"nctId": "NCT000"}
            mock_react_instance = MagicMock(return_value=mock_react_result)
            mock_dspy.ReAct.return_value = mock_react_instance
            mock_evaluator.summarizer = MagicMock(
                return_value=MagicMock(suggestions=["Example analysis"])
            )

            _ = mock_evaluator.forward(
                feature_plans=plans,
                model_eval_result=model_result,
                none_explanations={},
            )

            # Verify that examples are well-formed (not DataFrame repr)
            call_args_list = mock_react_instance.call_args_list
            for call in call_args_list:
                example = call.kwargs.get("example", "")
                # Should contain exactly one "Predicted" token per example
                assert example.count("Predicted") == 1
                # Should match pattern
                assert "## NCT" in example

    def test_sampling_is_seed_stable(self, mock_evaluator: Evaluator) -> None:
        """Sampling with a fixed seed (42) should produce deterministic results."""
        wrong_idxs = [3, 7, 11, 19, 23]
        wrong_preds = [1, 0, 1, 0, 1]
        model_result = _make_eval_result(
            roc_auc=0.7,
            wrong_idxs=wrong_idxs,
            wrong_preds=wrong_preds,
        )
        plans = {"feat_a": _make_plan("feat_a")}

        # Use the mock evaluator (seed is hardcoded to 42 in Evaluator.__init__)
        evaluator = mock_evaluator

        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = []
        evaluator.evaluator_no_example.return_value = mock_eval_result

        mock_react_result = MagicMock()
        mock_react_result.analysis = "Example analysis"

        with (
            patch("ctra.agents.evaluator.dspy") as mock_dspy,
            patch("ctra.agents.evaluator.get_trial_info_dict") as mock_info,
            patch("ctra.agents.evaluator.make_pubmed_search"),
            patch("ctra.agents.evaluator.make_nct_search"),
            patch("ctra.agents.evaluator.make_chembl_search"),
            patch("ctra.agents.evaluator.make_faers_search"),
            patch("ctra.agents.evaluator.make_aact_search"),
            patch("ctra.agents.evaluator.make_primekg_search"),
            patch("ctra.agents.evaluator.make_drugsfda_search"),
        ):
            mock_info.return_value = {"nctId": "NCT000"}
            mock_react_instance = MagicMock(return_value=mock_react_result)
            mock_dspy.ReAct.return_value = mock_react_instance
            evaluator.summarizer = MagicMock(
                return_value=MagicMock(suggestions=["Example analysis"])
            )

            evaluator.forward(
                feature_plans=plans,
                model_eval_result=model_result,
                none_explanations={},
            )

            # Verify seed stability: at seed 42, sampling should produce consistent results
            # rng(42).choice(5, 3) -> [4, 0, 3] -> wrong_idxs[[4, 0, 3]] -> [23, 3, 19]
            assert mock_react_instance.call_count == 3
            call_args_list = mock_react_instance.call_args_list
            captured_ids = []
            for call in call_args_list:
                example = call.kwargs.get("example", "")
                if "NCT023" in example:
                    captured_ids.append("NCT023")
                elif "NCT003" in example:
                    captured_ids.append("NCT003")
                elif "NCT019" in example:
                    captured_ids.append("NCT019")
            assert captured_ids == ["NCT023", "NCT003", "NCT019"]

    def test_misaligned_arrays_degrade_gracefully(self, mock_evaluator: Evaluator, caplog) -> None:
        """Misaligned arrays should log a warning and sample from common prefix."""
        from ctra.agents.data_models import ModelEvalResult

        wrong_idxs = [0, 1, 2]
        wrong_preds = [1, 0]  # Only 2 elements, not 3
        # Create a misaligned model result
        model_result = ModelEvalResult(
            roc_auc=0.7,
            f1=0.75,
            pr_auc=0.7,
            interaction_values={},
            wrong_idxs=wrong_idxs,
            wrong_preds=wrong_preds,
            wrong_df=pd.DataFrame({"id": ["NCT000", "NCT001"]}, index=[0, 1]),
            pipeline=None,
        )

        plans = {"feat_a": _make_plan("feat_a")}

        mock_eval_result = MagicMock()
        mock_eval_result.suggestions = []
        mock_evaluator.evaluator_no_example.return_value = mock_eval_result
        mock_evaluator.summarizer = MagicMock(return_value=MagicMock(suggestions=[]))

        # Should not raise IndexError
        with (
            patch("ctra.agents.evaluator.dspy"),
            patch("ctra.agents.evaluator.get_trial_info_dict") as mock_info,
            patch("ctra.agents.evaluator.make_pubmed_search"),
            patch("ctra.agents.evaluator.make_nct_search"),
            patch("ctra.agents.evaluator.make_chembl_search"),
            patch("ctra.agents.evaluator.make_faers_search"),
            patch("ctra.agents.evaluator.make_aact_search"),
            patch("ctra.agents.evaluator.make_primekg_search"),
            patch("ctra.agents.evaluator.make_drugsfda_search"),
        ):
            mock_info.return_value = {"nctId": "NCT000"}
            result = mock_evaluator.forward(
                feature_plans=plans,
                model_eval_result=model_result,
                none_explanations={},
            )

        assert isinstance(result, EvalOutput)
        # Should have logged a warning
        assert any("Misaligned" in record.message for record in caplog.records)
