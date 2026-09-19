"""``AgentOutput.cache_stats`` across ``Agent.forward`` (issue #17).

The counters ride inside the output because the feature store runs in the
``run_agent.py`` subprocess and the parent only ever sees the pickled
``AgentOutput``.  Every skip path must return a *zeroed* set: those paths
copy the parent output, and replaying the parent's counters would count the
parent's store traffic twice.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock, patch

import dill
import numpy as np
import pandas as pd
import pytest

from ctra.agents.data_models import (
    AgentOutput,
    CacheStats,
    FeatureOp,
    FeatureStoreCounters,
    ProposerOutput,
)
from ctra.agents.orchestrator import Agent
from tests.test_agents.conftest import planner_prediction, proposer_prediction
from tests.test_agents.test_orchestrator import _make_output, _make_plan


@pytest.fixture()
def agent(monkeypatch: pytest.MonkeyPatch) -> Agent:
    """An ``Agent`` with every dspy module stubbed (the test_orchestrator patch set)."""
    mock_settings = MagicMock()
    mock_settings.model.classifiers = []
    monkeypatch.setattr("ctra.agents.orchestrator.get_settings", lambda: mock_settings)
    monkeypatch.setattr("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)
    with (
        patch("ctra.agents.orchestrator.Initializer") as init_cls,
        patch("ctra.agents.orchestrator.FeatureProposer") as prop_cls,
        patch("ctra.agents.orchestrator.FeaturePlanner") as plan_cls,
        patch("ctra.agents.orchestrator.Evaluator") as eval_cls,
        patch("ctra.agents.orchestrator.FeatureGrouper") as group_cls,
    ):
        for cls in (init_cls, prop_cls, plan_cls, eval_cls, group_cls):
            cls.return_value = MagicMock()
        return Agent(
            task="Predict trial success",
            X_train=pd.Series(["NCT001"]),
            X_val=pd.Series(["NCT002"]),
            y_train=np.array([1]),
            y_val=np.array([0]),
            X_test=pd.Series(["NCT003"]),
            y_test=np.array([1]),
        )


def _counting_compute(seen_origins: list[str]) -> Any:
    """A ``compute_features`` double that records one lookup + one hit per call."""

    def fake(grouper: Any, nctids: list[str], task_description: str, plans: Any, **kw: Any) -> Any:
        counters: FeatureStoreCounters = kw["counters"]
        seen_origins.append(kw["plan_origin"])
        counters.feature_lookups += 1
        counters.record_hits(1, kw["plan_origin"])
        return ({n: {name: {"value": 1.0} for name in plans} for n in nctids}, {}, {})

    return fake


def _populated_previous() -> AgentOutput:
    prev = _make_output()
    prev.cache_stats.feature_lookups = 9
    prev.cache_stats.feature_hits = 7
    prev.cache_stats.groups_dispatched = 2
    prev.cache_stats.llm_calls_made = 40
    return prev


class TestForwardPopulatesCacheStats:
    def test_iteration_0_counts_the_three_initializer_calls(self, agent: Agent) -> None:
        agent.initializer.return_value = {"feat_a": _make_plan("feat_a")}
        origins: list[str] = []
        with patch(
            "ctra.agents.orchestrator.compute_features", side_effect=_counting_compute(origins)
        ):
            result = agent.forward(previous_output=None)
        assert origins == ["initializer"] * 3
        assert isinstance(result.cache_stats, CacheStats)
        assert result.cache_stats.feature_lookups == 3
        assert result.cache_stats.hits_from_initializer_plans == 3
        assert result.cache_stats.hits_from_planner_plans == 0
        assert result.cache_stats.llm_calls_made == 0

    def test_iteration_n_add_counts_the_three_planner_calls_afresh(self, agent: Agent) -> None:
        prev = _populated_previous()
        agent.proposer.return_value = proposer_prediction(
            feature_operation=FeatureOp.ADD,
            feature_name="feat_b",
            feature_explanation="Add a safety feature",
        )
        agent.planner.return_value = planner_prediction(_make_plan("feat_b"), MagicMock())
        origins: list[str] = []
        with patch(
            "ctra.agents.orchestrator.compute_features", side_effect=_counting_compute(origins)
        ):
            result = agent.forward(previous_output=prev)
        assert origins == ["planner"] * 3
        assert result.cache_stats is not prev.cache_stats
        assert result.cache_stats.feature_lookups == 3
        assert result.cache_stats.hits_from_planner_plans == 3
        assert result.cache_stats.hits_from_initializer_plans == 0
        # The parent's counters are untouched and not carried forward.
        assert prev.cache_stats.feature_lookups == 9
        assert result.cache_stats.llm_calls_made == 0

    def test_remove_touches_no_store_and_reports_zero(self, agent: Agent) -> None:
        prev = _populated_previous()
        agent.proposer.return_value = ProposerOutput(
            feature_operation=FeatureOp.REMOVE,
            feature_name="feat_a",
            feature_explanation="drop it",
        )
        with patch("ctra.agents.orchestrator.compute_features") as compute:
            result = agent.forward(previous_output=prev)
        compute.assert_not_called()
        assert result.cache_stats == CacheStats()


class TestSkipPathsZeroCacheStats:
    """The fence: a copied parent output must not replay the parent's counters."""

    def test_exhausted_early_skip_is_zeroed(self, agent: Agent) -> None:
        prev = _populated_previous()._replace(suggestion_index=5)
        assert prev.suggestions_exhausted
        result = agent.forward(previous_output=prev)
        agent.proposer.assert_not_called()
        assert result.cache_stats == CacheStats()
        assert result.cache_stats is not prev.cache_stats
        assert prev.cache_stats.feature_hits == 7

    def test_invalid_proposer_skip_is_zeroed(self, agent: Agent) -> None:
        prev = _populated_previous()
        # ADD of a feature that already exists is invalid for every retry.
        agent.proposer.return_value = proposer_prediction(
            feature_operation=FeatureOp.ADD,
            feature_name="feat_a",
            feature_explanation="again",
        )
        result = agent.forward(previous_output=prev)
        assert result.suggestion_index == prev.suggestion_index + 1
        assert result.cache_stats == CacheStats()
        assert prev.cache_stats.feature_hits == 7

    def test_unhandled_operation_skip_is_zeroed(self, agent: Agent) -> None:
        prev = _populated_previous()
        agent.proposer.return_value = ProposerOutput(
            feature_operation="explode",  # type: ignore[arg-type]
            feature_name="feat_a",
            feature_explanation="?",
        )
        with patch("ctra.agents.orchestrator.is_valid_proposer", return_value=True):
            result = agent.forward(previous_output=prev)
        assert result.suggestion_index == prev.suggestion_index + 1
        assert result.cache_stats == CacheStats()
        assert prev.cache_stats.feature_hits == 7


class TestOlderPicklesWithoutTheField:
    """An ``AgentOutput`` dumped before ``cache_stats`` existed must stay readable."""

    @staticmethod
    def _without_attribute() -> AgentOutput:
        out = _make_output()
        del out.__dict__["cache_stats"]
        assert not hasattr(out, "cache_stats")
        return out

    def test_getattr_default_reads_a_missing_attribute(self) -> None:
        out = self._without_attribute()
        assert getattr(out, "cache_stats", None) is None
        assert getattr(getattr(out, "cache_stats", None), "groups_dispatched", 0) == 0

    def test_dill_round_trip_backfills_a_zeroed_set(self) -> None:
        loaded = dill.loads(dill.dumps(self._without_attribute()))
        assert loaded.cache_stats == CacheStats()
        # ``_replace`` reads every field, so the backfill is what keeps the
        # resume and skip paths working on an old pickle.
        assert loaded._replace(suggestion_index=3).suggestion_index == 3

    def test_deepcopy_backfills_too(self) -> None:
        copied = deepcopy(self._without_attribute())
        assert copied.cache_stats == CacheStats()

    def test_present_values_survive_the_hook(self) -> None:
        out = _make_output()
        out.cache_stats.feature_hits = 4
        loaded = dill.loads(dill.dumps(out))
        assert loaded.cache_stats.feature_hits == 4
        assert loaded.builder_meta == {}
