"""Tests for ctra.mlops.objectives — objective functions and evaluator.

Pure math tests: no mocks, no external services.  Uses monkeypatch to
isolate from the global settings singleton.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig, Settings, get_settings
from ctra.mlops.objectives import (
    MultiObjectiveEvaluator,
    Parsimony,
    PredictiveAccuracy,
)

# ======================================================================
# Settings fixture — isolate from environment / .env
# ======================================================================


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace get_settings with a deterministic test configuration."""
    get_settings.cache_clear()
    settings = Settings(
        mcts=MCTSConfig(
            max_features=50,
            objectives=["accuracy", "parsimony"],
        ),
    )
    monkeypatch.setattr("ctra.mlops.objectives.get_settings", lambda: settings)
    yield
    get_settings.cache_clear()


# ======================================================================
# PredictiveAccuracy
# ======================================================================


class TestPredictiveAccuracy:
    """Tests for the accuracy (ROC-AUC) objective."""

    def test_roc_auc_passthrough(self) -> None:
        obj = PredictiveAccuracy()
        result = obj.evaluate(roc_auc=0.85)
        np.testing.assert_allclose(result, 0.85, atol=1e-6)

    def test_clamp_above_one(self) -> None:
        obj = PredictiveAccuracy()
        result = obj.evaluate(roc_auc=1.5)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_clamp_below_zero(self) -> None:
        obj = PredictiveAccuracy()
        result = obj.evaluate(roc_auc=-0.3)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_fallback_no_data(self) -> None:
        obj = PredictiveAccuracy()
        result = obj.evaluate()
        np.testing.assert_allclose(result, 0.5, atol=1e-6)


# ======================================================================
# Parsimony
# ======================================================================


class TestParsimony:
    """Tests for the parsimony (feature efficiency) objective."""

    def test_zero_features(self) -> None:
        obj = Parsimony()
        result = obj.evaluate(feature_set=[])
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_max_features(self) -> None:
        obj = Parsimony()
        features = [f"f{i}" for i in range(50)]
        result = obj.evaluate(feature_set=features)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_half_features(self) -> None:
        obj = Parsimony()
        features = [f"f{i}" for i in range(25)]
        result = obj.evaluate(feature_set=features)
        np.testing.assert_allclose(result, 0.5, atol=1e-6)

    def test_exceeds_max_clamped(self) -> None:
        obj = Parsimony()
        features = [f"f{i}" for i in range(100)]
        result = obj.evaluate(feature_set=features)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_one_feature(self) -> None:
        obj = Parsimony()
        result = obj.evaluate(feature_set=["age"])
        expected = 1.0 - 1.0 / 50.0
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_custom_max_features(self) -> None:
        obj = Parsimony(max_features=10)
        features = [f"f{i}" for i in range(5)]
        result = obj.evaluate(feature_set=features)
        np.testing.assert_allclose(result, 0.5, atol=1e-6)


# ======================================================================
# MultiObjectiveEvaluator
# ======================================================================


class TestMultiObjectiveEvaluator:
    """Tests for the combined multi-objective evaluator."""

    def test_names_match_config(self) -> None:
        ev = MultiObjectiveEvaluator()
        assert ev.names == ["accuracy", "parsimony"]

    def test_n_objectives(self) -> None:
        ev = MultiObjectiveEvaluator()
        assert ev.n_objectives == 2

    def test_evaluate_returns_two_values(self) -> None:
        ev = MultiObjectiveEvaluator()
        result = ev.evaluate(
            feature_set=["f1", "f2"],
            roc_auc=0.75,
        )
        assert len(result.values) == 2
        assert len(result.names) == 2

    def test_evaluate_accuracy_value(self) -> None:
        ev = MultiObjectiveEvaluator()
        result = ev.evaluate(
            feature_set=["f1", "f2"],
            roc_auc=0.85,
        )
        np.testing.assert_allclose(result["accuracy"], 0.85, atol=1e-6)

    def test_evaluate_parsimony_value(self) -> None:
        ev = MultiObjectiveEvaluator()
        features = [f"f{i}" for i in range(10)]
        result = ev.evaluate(feature_set=features, roc_auc=0.7)
        expected_parsimony = 1.0 - 10.0 / 50.0
        np.testing.assert_allclose(result["parsimony"], expected_parsimony, atol=1e-6)

    def test_unknown_objective_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown objective"):
            MultiObjectiveEvaluator(objectives=["accuracy", "nonexistent"])

    def test_custom_objective_subset(self) -> None:
        ev = MultiObjectiveEvaluator(objectives=["accuracy"])
        assert ev.n_objectives == 1
        assert ev.names == ["accuracy"]
