"""Test that Initializer receives shared planner from Agent (identity contract).

Verifies that Agent injects its wrapped FeaturePlanner into Initializer's
constructor, so both use the same dspy.Refine instance rather than each
maintaining their own.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    from ctra.agents.data_models import Task
    from ctra.agents.orchestrator import Agent

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


def test_initializer_shares_planner_identity():
    """Assert agent.initializer.feature_planner is agent.planner (identity, not equality).

    Arranges: Construct Agent with real dspy setup.
    Acts: Access initializer.feature_planner and agent.planner.
    Asserts: They are the same object (identity via 'is', not equality via '==').
    """
    with patch("ctra.agents.orchestrator.dspy.Refine") as mock_refine, patch(
        "ctra.agents.orchestrator.dspy.ChainOfThought"
    ) as mock_cot, patch(
        "ctra.agents.orchestrator.get_settings"
    ):
        # Mock dspy.Refine to return a mock module that tracks identity
        shared_planner = MagicMock()
        mock_refine.return_value = shared_planner

        agent = Agent(
            task=Task.TRIAL_OUTCOME_PHASE_2,
            X_train=pd.Series(["NCT001", "NCT002"]),
            y_train=pd.Series([1, 0]),
            X_val=pd.Series(["NCT003"]),
            y_val=pd.Series([1]),
        )

        # Assert identity: initializer.feature_planner IS agent.planner
        assert agent.initializer.feature_planner is agent.planner
