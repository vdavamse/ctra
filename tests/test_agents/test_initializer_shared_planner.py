"""Test that Initializer receives the shared planner from Agent (identity contract).

``Agent`` injects its wrapped ``FeaturePlanner`` into ``Initializer``'s
constructor, so both use the same ``Refine`` instance rather than each
maintaining its own (and each eroding its own failure budget).
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from ctra.agents.data_models import Task
from ctra.agents.initializer import Initializer
from ctra.agents.orchestrator import Agent


def _make_agent() -> Agent:
    with patch("ctra.agents.orchestrator.get_settings"):
        return Agent(
            task=Task.TRIAL_OUTCOME_PHASE_2,
            X_train=pd.Series(["NCT001", "NCT002"]),
            y_train=pd.Series([1, 0]),
            X_val=pd.Series(["NCT003"]),
            y_val=pd.Series([1]),
            X_test=pd.Series(["NCT004"]),
            y_test=pd.Series([0]),
        )


def test_initializer_shares_planner_identity():
    """``agent.initializer.feature_planner`` IS ``agent.planner`` -- unmocked.

    Deliberately built with **no** ``Refine`` patch. An earlier version patched
    ``dspy.Refine``, which handed both construction sites the same mock and made
    the assertion pass regardless of whether the injection existed; unmocked, the
    identity was ``False``. Constructing the real object is the only way this
    assertion means anything.
    """
    agent = _make_agent()

    assert agent.initializer.feature_planner is agent.planner


def test_initializer_builds_its_own_planner_when_not_injected():
    """Standalone use (e.g. ``scripts/``) still gets a working planner."""
    init = Initializer(
        task_description="Predict trial outcome",
        X_train=pd.Series(["NCT001"]),
        y_train=pd.Series([1]),
    )

    assert init.feature_planner is not None
