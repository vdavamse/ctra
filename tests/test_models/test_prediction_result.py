"""Tests for PredictionResult dataclass and compute_metrics function."""

from __future__ import annotations

import numpy as np

from ctra.models.tabpfn_classifier import PredictionResult, compute_metrics

# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.9, 1.0])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics["roc_auc"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_random_predictions_in_bounds(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        y_prob = rng.rand(100)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_single_class_no_crash(self) -> None:
        """When y_true has only one class, roc_auc should not crash."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.4, 0.7])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        # roc_auc is undefined for single class; code sets it to 0.0
        assert metrics["roc_auc"] == 0.0
        assert "accuracy" in metrics
        assert "f1" in metrics

    def test_all_zeros(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.05])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 0.0  # single class

    def test_returns_all_expected_keys(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_prob = np.array([0.3, 0.8])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert set(metrics.keys()) == {"roc_auc", "pr_auc", "f1", "accuracy"}

    def test_float_values(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.6, 0.7, 0.3, 0.4])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        for v in metrics.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------


class TestPredictionResult:
    def test_construction(self) -> None:
        pr = PredictionResult(
            probabilities=np.array([[0.3, 0.7], [0.6, 0.4]]),
            predictions=np.array([1, 0]),
            shap_values=None,
            feature_names=["feat_a", "feat_b"],
            model_name="xgboost",
        )
        assert pr.model_name == "xgboost"
        assert pr.shap_values is None
        assert len(pr.feature_names) == 2

    def test_metrics_default_empty(self) -> None:
        pr = PredictionResult(
            probabilities=np.zeros((2, 2)),
            predictions=np.zeros(2),
            shap_values=None,
            feature_names=[],
            model_name="test",
        )
        assert pr.metrics == {}

    def test_field_access(self) -> None:
        probs = np.array([[0.1, 0.9]])
        preds = np.array([1])
        shap = np.array([[0.2, -0.1]])
        pr = PredictionResult(
            probabilities=probs,
            predictions=preds,
            shap_values=shap,
            feature_names=["f1", "f2"],
            model_name="tabpfn",
            metrics={"roc_auc": 0.85},
        )
        np.testing.assert_array_equal(pr.probabilities, probs)
        np.testing.assert_array_equal(pr.predictions, preds)
        np.testing.assert_array_equal(pr.shap_values, shap)
        assert pr.metrics["roc_auc"] == 0.85
