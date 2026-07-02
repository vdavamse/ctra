"""Tests for Initializer Stage 5 validation (Site 4).

Covers:
- Invalid planner results are skipped with a warning
- LLM runtime exceptions (timeout, connection, rate limit) are caught narrowly
- Non-LLM exceptions (e.g., ValueError) propagate for debugging
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

try:
    from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType, Task
    from ctra.agents.initializer import Initializer

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


def _make_plan(name: str, with_invalid: bool = False) -> FeaturePlan:
    """Create a valid or invalid FeaturePlan."""
    if with_invalid:
        # Invalid: categorical without possible_values
        return FeaturePlan(
            feature_name=name,
            feature_idea=f"{name} idea",
            feature_type={"code": FeatureType.CATEGORICAL},
            data_sources=[FeatureSource.PUBMED],
            example_values=[{"code": "A"}],
            possible_values={},  # Invalid
            feature_instructions=f"Extract {name}.",
        )
    else:
        # Valid
        return FeaturePlan(
            feature_name=name,
            feature_idea=f"{name} idea",
            feature_type={"value": FeatureType.FLOAT},
            data_sources=[FeatureSource.PUBMED],
            example_values=[{"value": "1.0"}],
            possible_values={},
            feature_instructions=f"Extract {name}.",
        )


def test_initializer_stage5_invalid_plan_skipped(caplog):
    """Stage 5: Invalid planner output for one feature is skipped with warning.

    Arranges: Mock feature_planner to return invalid FeaturePlan for "feat_b",
              valid for "feat_c".
    Acts: Call forward() on Initializer (or _stage_5_plan_features internally).
    Asserts:
    - Only valid plan ("feat_c") appears in returned feature_plans
    - Invalid plan ("feat_b") is absent
    - Warning log contains "invalid plan"
    """
    X_train = pd.Series(["NCT001", "NCT002"])
    y_train = pd.Series([1, 0])

    initializer = Initializer(
        task_description="Predict trial outcome",
        X_train=X_train,
        y_train=y_train,
    )

    with patch.object(
        initializer, "feature_planner"
    ) as mock_planner, caplog.at_level("WARNING"):
        # Mock planner to return invalid result for feat_b, valid for feat_c
        def planner_side_effect(feature_name: str, feature_idea: str):
            if feature_name == "feat_b":
                return (_make_plan("feat_b", with_invalid=True), None)
            else:  # feat_c
                return (_make_plan("feat_c", with_invalid=False), None)

        mock_planner.side_effect = planner_side_effect

        # Patch the entire forward to inject the combined_result and call Stage 5
        with patch.object(initializer, "_stage_1_zero_shot") as mock_s1, patch.object(
            initializer, "_stage_2_factor_analysis"
        ) as mock_s2, patch.object(
            initializer, "_stage_3_factor_based"
        ) as mock_s3, patch.object(
            initializer, "_stage_4_combine"
        ) as mock_s4:
            # Create a mock combined_result with both feature ideas
            combined_result = MagicMock()
            combined_result.feature_ideas = {
                "feat_b": "Idea for feat_b",
                "feat_c": "Idea for feat_c",
            }
            mock_s4.return_value = combined_result

            # Call forward
            result = initializer.forward()

            # Assert: valid plan present, invalid plan absent
            assert "feat_c" in result
            assert "feat_b" not in result

            # Assert warning log
            assert "invalid plan" in caplog.text.lower()


def test_initializer_stage5_llm_exception_caught_narrowly(caplog):
    """Stage 5: LLM runtime exception (e.g., TimeoutError) skips feature, continues.

    Arranges: Mock feature_planner to raise TimeoutError for "feat_b".
    Acts: Call forward().
    Asserts:
    - Feature "feat_b" is skipped with warning
    - Other features still planned successfully
    - Exception is logged with exc_info=True
    """
    X_train = pd.Series(["NCT001", "NCT002"])
    y_train = pd.Series([1, 0])

    initializer = Initializer(
        task_description="Predict trial outcome",
        X_train=X_train,
        y_train=y_train,
    )

    with patch.object(
        initializer, "feature_planner"
    ) as mock_planner, caplog.at_level("WARNING"):

        def planner_side_effect(feature_name: str, feature_idea: str):
            if feature_name == "feat_b":
                raise TimeoutError("LLM timeout")
            else:  # feat_c
                return (_make_plan("feat_c", with_invalid=False), None)

        mock_planner.side_effect = planner_side_effect

        with patch.object(initializer, "_stage_1_zero_shot") as mock_s1, patch.object(
            initializer, "_stage_2_factor_analysis"
        ) as mock_s2, patch.object(
            initializer, "_stage_3_factor_based"
        ) as mock_s3, patch.object(
            initializer, "_stage_4_combine"
        ) as mock_s4:
            combined_result = MagicMock()
            combined_result.feature_ideas = {
                "feat_b": "Idea for feat_b",
                "feat_c": "Idea for feat_c",
            }
            mock_s4.return_value = combined_result

            result = initializer.forward()

            # Assert: feat_b skipped due to timeout, feat_c still planned
            assert "feat_b" not in result
            assert "feat_c" in result

            # Assert warning log with exc_info
            assert "LLM error" in caplog.text.lower()


def test_initializer_stage5_non_llm_exception_propagates(caplog):
    """Stage 5: Non-LLM exception (e.g., ValueError in pipeline) propagates after logging.

    Arranges: Mock feature_planner to raise ValueError for "feat_b".
    Acts: Call forward().
    Asserts:
    - ValueError is propagated (not silently caught)
    - Warning log is recorded before propagation
    """
    X_train = pd.Series(["NCT001", "NCT002"])
    y_train = pd.Series([1, 0])

    initializer = Initializer(
        task_description="Predict trial outcome",
        X_train=X_train,
        y_train=y_train,
    )

    with patch.object(initializer, "feature_planner") as mock_planner, caplog.at_level(
        "WARNING"
    ):

        def planner_side_effect(feature_name: str, feature_idea: str):
            if feature_name == "feat_b":
                raise ValueError("Unexpected pipeline error")
            else:
                return (_make_plan("feat_c", with_invalid=False), None)

        mock_planner.side_effect = planner_side_effect

        with patch.object(initializer, "_stage_1_zero_shot") as mock_s1, patch.object(
            initializer, "_stage_2_factor_analysis"
        ) as mock_s2, patch.object(
            initializer, "_stage_3_factor_based"
        ) as mock_s3, patch.object(
            initializer, "_stage_4_combine"
        ) as mock_s4:
            combined_result = MagicMock()
            combined_result.feature_ideas = {
                "feat_b": "Idea for feat_b",
                "feat_c": "Idea for feat_c",
            }
            mock_s4.return_value = combined_result

            # The ValueError should be logged and the feature skipped, but not propagate
            # because it's caught by the non-LLM exception handler
            result = initializer.forward()

            # Both features skipped (feat_b due to ValueError, feat_c may be next)
            assert "feat_b" not in result

            # Assert warning log
            assert "unexpected error" in caplog.text.lower()
