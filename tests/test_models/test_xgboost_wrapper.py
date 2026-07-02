"""Tests for ctra.models.xgboost_classifier — real XGBoost on tiny data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctra.config.settings import ModelConfig, get_settings
from ctra.models.tabpfn_classifier import PredictionResult
from ctra.models.xgboost_classifier import XGBoostWrapper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch get_settings() to return a lightweight ModelConfig."""
    get_settings.cache_clear()
    test_config = ModelConfig(
        shap_enabled=False,
        xgb_n_estimators=10,
        xgb_max_depth=3,
    )
    monkeypatch.setattr(
        "ctra.models.xgboost_classifier.get_settings",
        lambda: type("S", (), {"model": test_config})(),
    )
    yield
    get_settings.cache_clear()


@pytest.fixture()
def tiny_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """20-row dataset with 3 numeric features."""
    rng = np.random.RandomState(0)
    n = 20
    X = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "feat_c": rng.rand(n),
        }
    )
    y = (X["feat_a"] + X["feat_b"] > 0).astype(int).values
    # Split: first 14 train, last 6 test
    X_train, X_test = X.iloc[:14], X.iloc[14:]
    y_train, y_test = y[:14], y[14:]
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestXGBoostWrapper:
    def test_fit_predict_produces_prediction_result(
        self,
        tiny_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    ) -> None:
        X_train, y_train, X_test, _y_test = tiny_data
        wrapper = XGBoostWrapper(
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3)
        )
        wrapper.fit(X_train, y_train)
        result = wrapper.predict(X_test)

        assert isinstance(result, PredictionResult)
        assert result.model_name == "xgboost"

    def test_probabilities_shape_and_bounds(
        self,
        tiny_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    ) -> None:
        X_train, y_train, X_test, _y_test = tiny_data
        wrapper = XGBoostWrapper(
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3)
        )
        wrapper.fit(X_train, y_train)
        result = wrapper.predict(X_test)

        assert result.probabilities.shape == (len(X_test), 2)
        assert np.all(result.probabilities >= 0.0)
        assert np.all(result.probabilities <= 1.0)
        # Probabilities should sum to ~1 per row
        row_sums = result.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predictions_binary(
        self,
        tiny_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    ) -> None:
        X_train, y_train, X_test, _ = tiny_data
        wrapper = XGBoostWrapper(
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3)
        )
        wrapper.fit(X_train, y_train)
        result = wrapper.predict(X_test)

        unique = set(result.predictions.tolist())
        assert unique.issubset({0, 1})

    def test_metrics_computed_when_y_true_provided(
        self,
        tiny_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    ) -> None:
        X_train, y_train, X_test, y_test = tiny_data
        wrapper = XGBoostWrapper(
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3)
        )
        wrapper.fit(X_train, y_train)
        result = wrapper.predict(X_test, y_true=y_test)

        assert "roc_auc" in result.metrics
        assert "accuracy" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0

    def test_predict_before_fit_raises(self) -> None:
        wrapper = XGBoostWrapper(
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3)
        )
        X_dummy = pd.DataFrame({"a": [1.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            wrapper.predict(X_dummy)

    def test_mixed_dtypes_handled(self) -> None:
        """Categorical + numeric columns should be handled by ColumnTransformer."""
        rng = np.random.RandomState(1)
        n = 20
        X = pd.DataFrame(
            {
                "numeric_col": rng.randn(n),
                "cat_col": np.random.choice(["A", "B", "C"], size=n),
            }
        )
        y = rng.randint(0, 2, size=n)

        wrapper = XGBoostWrapper(
            config=ModelConfig(shap_enabled=False, xgb_n_estimators=10, xgb_max_depth=3)
        )
        wrapper.fit(X, y)
        result = wrapper.predict(X)

        assert isinstance(result, PredictionResult)
        assert result.probabilities.shape[0] == n
