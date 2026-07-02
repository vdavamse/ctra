"""Tests for ctra.models.model_registry — XGBoost-only (TabPFN not available in CI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctra.config.settings import ClassifierType, ModelConfig, get_settings
from ctra.models.model_registry import ModelRegistry
from ctra.models.tabpfn_classifier import PredictionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch get_settings() to return a lightweight config with XGBoost only."""
    get_settings.cache_clear()
    test_config = ModelConfig(
        classifiers=[ClassifierType.XGBOOST],
        shap_enabled=False,
        xgb_n_estimators=10,
        xgb_max_depth=3,
    )
    monkeypatch.setattr(
        "ctra.models.xgboost_classifier.get_settings",
        lambda: type("S", (), {"model": test_config})(),
    )
    monkeypatch.setattr(
        "ctra.models.model_registry.get_settings",
        lambda: type("S", (), {"model": test_config})(),
    )
    yield
    get_settings.cache_clear()


@pytest.fixture()
def dataset() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Tiny dataset for training and evaluation."""
    rng = np.random.RandomState(42)
    n = 30
    X = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
        }
    )
    y = (X["f1"] + X["f2"] > 0).astype(int).values
    X_train, X_val = X.iloc[:20], X.iloc[20:]
    y_train, y_val = y[:20], y[20:]
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_train_and_evaluate_returns_xgboost(
        self,
        dataset: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    ) -> None:
        X_train, y_train, X_val, y_val = dataset
        registry = ModelRegistry(
            classifiers=["xgboost"],
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3),
        )
        results = registry.train_and_evaluate(X_train, y_train, X_val, y_val)

        assert "xgboost" in results
        assert isinstance(results["xgboost"], PredictionResult)
        assert "roc_auc" in results["xgboost"].metrics

    def test_get_best_model_returns_string(
        self,
        dataset: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    ) -> None:
        X_train, y_train, X_val, y_val = dataset
        registry = ModelRegistry(
            classifiers=["xgboost"],
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3),
        )
        results = registry.train_and_evaluate(X_train, y_train, X_val, y_val)
        best = registry.get_best_model(results)

        assert isinstance(best, str)
        assert best == "xgboost"

    def test_get_best_model_selects_highest_metric(self) -> None:
        """When multiple results exist, the highest metric wins."""
        result_a = PredictionResult(
            probabilities=np.zeros((1, 2)),
            predictions=np.zeros(1),
            shap_values=None,
            feature_names=[],
            model_name="model_a",
            metrics={"roc_auc": 0.75, "accuracy": 0.80},
        )
        result_b = PredictionResult(
            probabilities=np.zeros((1, 2)),
            predictions=np.zeros(1),
            shap_values=None,
            feature_names=[],
            model_name="model_b",
            metrics={"roc_auc": 0.90, "accuracy": 0.70},
        )
        registry = ModelRegistry(classifiers=["xgboost"])
        best = registry.get_best_model({"a": result_a, "b": result_b}, metric="roc_auc")
        assert best == "b"

        # By accuracy, model_a wins
        best_acc = registry.get_best_model({"a": result_a, "b": result_b}, metric="accuracy")
        assert best_acc == "a"

    def test_get_best_model_raises_on_empty(self) -> None:
        registry = ModelRegistry(classifiers=["xgboost"])
        with pytest.raises(ValueError, match="empty"):
            registry.get_best_model({})

    def test_predict_with_best_produces_result(
        self,
        dataset: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    ) -> None:
        X_train, y_train, X_val, y_val = dataset
        registry = ModelRegistry(
            classifiers=["xgboost"],
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3),
        )
        results = registry.train_and_evaluate(X_train, y_train, X_val, y_val)
        prediction = registry.predict_with_best(X_val, results)

        assert isinstance(prediction, PredictionResult)
        assert prediction.probabilities.shape[0] == len(X_val)

    def test_tabpfn_gracefully_skipped(
        self,
        dataset: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    ) -> None:
        """TabPFN should be skipped when not installed, not raise."""
        X_train, y_train, X_val, y_val = dataset
        registry = ModelRegistry(
            classifiers=["xgboost", "tabpfn"],
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3),
        )
        results = registry.train_and_evaluate(X_train, y_train, X_val, y_val)

        # XGBoost should still succeed
        assert "xgboost" in results
        # TabPFN may or may not be present depending on environment,
        # but it should not crash the entire train_and_evaluate call.

    def test_get_wrapper_returns_fitted_model(
        self,
        dataset: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    ) -> None:
        X_train, y_train, X_val, y_val = dataset
        registry = ModelRegistry(
            classifiers=["xgboost"],
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3),
        )
        registry.train_and_evaluate(X_train, y_train, X_val, y_val)
        wrapper = registry.get_wrapper("xgboost")
        assert wrapper.is_fitted

    def test_get_wrapper_raises_for_missing(self) -> None:
        registry = ModelRegistry(classifiers=["xgboost"])
        with pytest.raises(KeyError, match="No fitted wrapper"):
            registry.get_wrapper("nonexistent")
