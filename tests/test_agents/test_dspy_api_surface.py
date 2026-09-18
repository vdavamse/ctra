"""Smoke tests to detect DSPy API breaking changes early.

These tests verify that the DSPy APIs used by CTRA actually exist in the
installed version, preventing silent failures masked by MagicMock in
unit tests.
"""

from __future__ import annotations

import dspy
import pytest


class TestDspyAPISurface:
    """Verify existence of all DSPy APIs used by CTRA."""

    def test_refine_exists(self) -> None:
        assert hasattr(dspy, "Refine"), "dspy.Refine missing — validation migration needed"

    def test_best_of_n_exists(self) -> None:
        assert hasattr(dspy, "BestOfN"), "dspy.BestOfN missing"

    def test_module_exists_and_callable(self) -> None:
        assert hasattr(dspy, "Module")
        assert callable(dspy.Module)

    def test_chain_of_thought_exists(self) -> None:
        assert hasattr(dspy, "ChainOfThought")

    def test_react_exists(self) -> None:
        assert hasattr(dspy, "ReAct")

    def test_predict_exists(self) -> None:
        assert hasattr(dspy, "Predict")

    def test_signature_exists(self) -> None:
        assert hasattr(dspy, "Signature")

    def test_input_output_field_exist(self) -> None:
        assert hasattr(dspy, "InputField")
        assert hasattr(dspy, "OutputField")

    def test_lm_exists(self) -> None:
        assert hasattr(dspy, "LM")

    def test_configure_exists(self) -> None:
        assert hasattr(dspy, "configure")

    def test_context_exists(self) -> None:
        assert hasattr(dspy, "context")

    def test_assert_does_not_exist(self) -> None:
        """Confirm dspy.Assert is NOT available — guards against accidental usage."""
        assert not hasattr(dspy, "Assert"), (
            "dspy.Assert exists again — remove ValueError-based validation "
            "and consider reverting to inline assertions"
        )

    def test_suggest_does_not_exist(self) -> None:
        """Confirm dspy.Suggest is NOT available."""
        assert not hasattr(dspy, "Suggest")

    def test_prediction_exists(self) -> None:
        """dspy.Prediction is load-bearing for all four Refine-wrapped modules (#5, #6)."""
        assert hasattr(dspy, "Prediction"), "dspy.Prediction missing — needed for return contracts"

    def test_global_history_importable(self) -> None:
        """GLOBAL_HISTORY is the observation channel for acceptance test of issue #5."""
        try:
            from dspy.clients.base_lm import GLOBAL_HISTORY  # noqa: F401
        except ImportError:
            pytest.fail(
                "dspy.clients.base_lm.GLOBAL_HISTORY not importable — acceptance test channel missing"
            )
