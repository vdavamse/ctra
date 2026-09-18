"""Tests for Initializer Stage 5 validation (Site 4).

Covers:
- Invalid planner results are skipped with a warning
- LLM runtime exceptions (timeout, connection, rate limit) are caught narrowly
- Non-LLM exceptions (e.g., ValueError) are logged and the feature skipped, so
  one bad feature cannot abort the whole initialization run
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
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


def _make_initializer() -> Initializer:
    """Build an Initializer whose Stage 2 is disabled.

    Stage 2 does live ReAct factor analysis over sampled trials (network + LM);
    ``num_examples=0`` skips that loop entirely so the test can reach Stage 5.
    """
    return Initializer(
        task_description="Predict trial outcome",
        X_train=pd.Series(["NCT001", "NCT002"]),
        y_train=pd.Series([1, 0]),
        num_examples=0,
    )


@contextmanager
def _stubbed_idea_stages(initializer: Initializer, feature_ideas: dict[str, str]):
    """Stub stages 1, 3 and 4 so ``forward()`` reaches Stage 5 without an LM.

    The stages are inline in ``forward()`` (there are no ``_stage_N_*`` methods),
    so the seams are the three ``dspy.ChainOfThought`` attributes. Stage 4 reads
    ``zero_shot.feature_ideas + factors.feature_ideas``, so both must be lists,
    while the combined result is a ``{name: idea}`` dict.
    """
    zero_shot = MagicMock()
    zero_shot.feature_ideas = []
    factors = MagicMock()
    factors.feature_ideas = []
    combined = MagicMock()
    combined.feature_ideas = feature_ideas

    with (
        patch.object(initializer, "feature_initializer_zero_shot", return_value=zero_shot),
        patch.object(initializer, "feature_initializer_from_factors", return_value=factors),
        patch.object(initializer, "feature_initializer_combined", return_value=combined),
    ):
        yield


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
    initializer = _make_initializer()

    ideas = {"feat_b": "Idea for feat_b", "feat_c": "Idea for feat_c"}

    with (
        patch.object(initializer, "feature_planner") as mock_planner,
        _stubbed_idea_stages(initializer, ideas),
        caplog.at_level("WARNING"),
    ):
        # Mock planner to return invalid result for feat_b, valid for feat_c
        def planner_side_effect(feature_name: str, feature_idea: str):
            if feature_name == "feat_b":
                return (_make_plan("feat_b", with_invalid=True), None)
            return (_make_plan("feat_c", with_invalid=False), None)

        mock_planner.side_effect = planner_side_effect

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
    initializer = _make_initializer()

    ideas = {"feat_b": "Idea for feat_b", "feat_c": "Idea for feat_c"}

    with (
        patch.object(initializer, "feature_planner") as mock_planner,
        _stubbed_idea_stages(initializer, ideas),
        caplog.at_level("WARNING"),
    ):

        def planner_side_effect(feature_name: str, feature_idea: str):
            if feature_name == "feat_b":
                raise TimeoutError("LLM timeout")
            return (_make_plan("feat_c", with_invalid=False), None)

        mock_planner.side_effect = planner_side_effect

        result = initializer.forward()

    # Assert: feat_b skipped due to timeout, feat_c still planned
    assert "feat_b" not in result
    assert "feat_c" in result

    # Assert warning log with exc_info
    assert "llm error" in caplog.text.lower()


def test_initializer_stage5_non_llm_exception_is_skipped_not_propagated(caplog):
    """Stage 5: Non-LLM exception (e.g., ValueError) is logged and the feature skipped.

    Arranges: Mock feature_planner to raise ValueError for "feat_b".
    Acts: Call forward().
    Asserts:
    - The run continues: "feat_c" is still planned
    - "feat_b" is absent
    - A warning is logged before the feature is skipped
    """
    initializer = _make_initializer()

    ideas = {"feat_b": "Idea for feat_b", "feat_c": "Idea for feat_c"}

    with (
        patch.object(initializer, "feature_planner") as mock_planner,
        _stubbed_idea_stages(initializer, ideas),
        caplog.at_level("WARNING"),
    ):

        def planner_side_effect(feature_name: str, feature_idea: str):
            if feature_name == "feat_b":
                raise ValueError("Unexpected pipeline error")
            return (_make_plan("feat_c", with_invalid=False), None)

        mock_planner.side_effect = planner_side_effect

        # The ValueError is logged and the feature skipped, not propagated:
        # the non-LLM handler catches it so one bad feature cannot abort the run.
        result = initializer.forward()

    # feat_b skipped due to ValueError; feat_c still planned
    assert "feat_b" not in result
    assert "feat_c" in result

    # Assert warning log
    assert "unexpected error" in caplog.text.lower()
