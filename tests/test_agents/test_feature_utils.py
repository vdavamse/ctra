"""Tests for ctra.agents.feature_utils — pure utility functions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ctra.agents.feature_utils import dump_as_json, eval_model, features_to_df, soft_assert

# ---------------------------------------------------------------------------
# features_to_df
# ---------------------------------------------------------------------------


class TestFeaturesToDf:
    def test_single_valued_feature(self) -> None:
        features = {
            "NCT001": {"drug_mechanism": {"value": "kinase_inhibitor"}},
            "NCT002": {"drug_mechanism": {"value": "antibody"}},
        }
        df = features_to_df(features)
        assert "drug_mechanism--value" in df.columns
        assert "id" in df.columns
        assert len(df) == 2

    def test_multi_valued_feature(self) -> None:
        features = {
            "NCT001": {
                "drug_profile": {
                    "mechanism": "kinase_inhibitor",
                    "target_count": 3,
                },
            },
        }
        df = features_to_df(features)
        assert "drug_profile--mechanism" in df.columns
        assert "drug_profile--target_count" in df.columns
        assert df.loc[0, "drug_profile--target_count"] == 3

    def test_multiple_features(self) -> None:
        features = {
            "NCT001": {
                "feat_a": {"value": 1.0},
                "feat_b": {"sub1": "x", "sub2": "y"},
            },
        }
        df = features_to_df(features)
        assert set(df.columns) == {"id", "feat_a--value", "feat_b--sub1", "feat_b--sub2"}

    def test_empty_features(self) -> None:
        df = features_to_df({})
        assert len(df) == 0


# ---------------------------------------------------------------------------
# dump_as_json
# ---------------------------------------------------------------------------


class TestDumpAsJson:
    def test_dict(self) -> None:
        result = dump_as_json({"a": 1, "b": "c"})
        assert json.loads(result) == {"a": 1, "b": "c"}

    def test_namedtuple(self) -> None:
        from ctra.agents.data_models import FeatureOp, ProposerOutput

        obj = ProposerOutput(
            feature_operation=FeatureOp.ADD,
            feature_name="test",
            feature_explanation="explain",
        )
        result = json.loads(dump_as_json(obj))
        assert result["feature_name"] == "test"

    def test_numpy_types(self) -> None:
        result = json.loads(dump_as_json({"val": np.float64(1.5)}))
        assert result["val"] == 1.5


# ---------------------------------------------------------------------------
# soft_assert
# ---------------------------------------------------------------------------


class TestSoftAssert:
    def test_true_condition_returns_value(self) -> None:
        assert soft_assert("hello", True, "msg") == "hello"

    def test_false_condition_returns_none(self) -> None:
        assert soft_assert("hello", False, "msg") is None

    def test_preserves_type(self) -> None:
        assert soft_assert(42, True, "msg") == 42


# ---------------------------------------------------------------------------
# eval_model
# ---------------------------------------------------------------------------


class _StubPipeline:
    """Minimal pipeline stub that returns fixed predictions and probabilities."""

    def __init__(self, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
        self._y_pred = y_pred
        self._y_prob = y_prob

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._y_pred

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([1 - self._y_prob, self._y_prob])


class TestEvalModel:
    """Tests for eval_model PR-AUC computation (Issue #37)."""

    def test_pr_auc_matches_sklearn_average_precision(self) -> None:
        """Numeric oracle: eval_model pr_auc must match average_precision_score."""
        from sklearn.metrics import average_precision_score

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.7, 0.55, 0.3])
        y_pred = (y_prob >= 0.5).astype(int)

        pipeline = _StubPipeline(y_pred=y_pred, y_prob=y_prob)
        df = pd.DataFrame({"feat": np.zeros(len(y_true))})

        result = eval_model(pipeline, df, y_true)

        expected = float(average_precision_score(y_true, y_prob))
        assert abs(result.pr_auc - expected) < 1e-6, (
            f"pr_auc={result.pr_auc} != expected={expected}"
        )
        # Sanity: the correct value is ~0.903, not ~0.393 (the old buggy value)
        assert result.pr_auc > 0.85

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_pr_auc_single_class_all_zeros(self) -> None:
        """When y_true has only negative class (all 0s), pr_auc should be 0.0 (not crash)."""
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        y_pred = (y_prob >= 0.5).astype(int)

        pipeline = _StubPipeline(y_pred=y_pred, y_prob=y_prob)
        df = pd.DataFrame({"feat": np.zeros(len(y_true))})

        result = eval_model(pipeline, df, y_true)
        # average_precision_score with single class (all zeros) returns 0.0
        assert result.pr_auc == 0.0

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_pr_auc_single_class_all_ones(self) -> None:
        """When y_true has only positive class (all 1s), pr_auc should be 1.0."""
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.6, 0.7, 0.8, 0.9])
        y_pred = np.array([1, 1, 1, 1])

        pipeline = _StubPipeline(y_pred=y_pred, y_prob=y_prob)
        df = pd.DataFrame({"feat": np.zeros(len(y_true))})

        result = eval_model(pipeline, df, y_true)
        assert result.pr_auc == 1.0
        assert result.roc_auc == 0.0  # roc_auc undefined for single-class

    def test_pr_auc_perfect_predictions(self) -> None:
        """Perfect separation should yield pr_auc = 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.9, 1.0])
        y_pred = np.array([0, 0, 1, 1])

        pipeline = _StubPipeline(y_pred=y_pred, y_prob=y_prob)
        df = pd.DataFrame({"feat": np.zeros(len(y_true))})

        result = eval_model(pipeline, df, y_true)
        assert result.pr_auc == 1.0
