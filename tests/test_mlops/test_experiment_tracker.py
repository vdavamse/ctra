"""Tests for ``ExperimentTracker.log_cache_stats`` (issue #17).

The tracker is built without touching MLflow: ``__new__`` plus a fake
``_mlflow`` module double, so the test checks what would be logged, not the
MLflow client.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ctra.agents.data_models import FeatureStoreCounters
from ctra.agents.runner import RunCacheStats
from ctra.mlops.experiment_tracker import ExperimentTracker


def _tracker() -> tuple[ExperimentTracker, MagicMock]:
    tracker = ExperimentTracker.__new__(ExperimentTracker)
    fake_mlflow = MagicMock()
    tracker._mlflow = fake_mlflow
    tracker._experiment_name = "ctra-test"
    tracker._active_run = None
    return tracker, fake_mlflow


def _stats() -> RunCacheStats:
    return RunCacheStats(
        agent_hits=1,
        agent_misses=3,
        replayed_group_builds=4,
        llm_calls_made=120,
        feature_store=FeatureStoreCounters(
            feature_lookups=40,
            feature_hits=10,
            groups_dispatched=6,
            groups_skipped=2,
            hits_from_initializer_plans=7,
            hits_from_planner_plans=3,
            store_writes=18,
        ),
    )


class TestLogCacheStats:
    def test_logs_rates_headline_counts_and_raw_counters(self) -> None:
        tracker, fake_mlflow = _tracker()
        tracker.log_cache_stats(_stats())
        fake_mlflow.log_metrics.assert_called_once()
        metrics = fake_mlflow.log_metrics.call_args.args[0]
        assert fake_mlflow.log_metrics.call_args.kwargs == {"step": None}
        assert metrics == {
            "agent_cache_hit_rate": pytest.approx(0.25),
            "feature_store_hit_rate": pytest.approx(0.25),
            "groups_skipped": 2.0,
            "llm_calls_avoided_estimate": 12.0,
            "llm_calls_made": 120.0,
            "agent_hits": 1.0,
            "agent_misses": 3.0,
            "replayed_group_builds": 4.0,
            "feature_lookups": 40.0,
            "feature_hits": 10.0,
            "groups_dispatched": 6.0,
            "hits_from_initializer_plans": 7.0,
            "hits_from_planner_plans": 3.0,
            "store_writes": 18.0,
        }
        assert all(isinstance(v, float) for v in metrics.values())

    def test_step_is_passed_through_for_per_rollout_history(self) -> None:
        tracker, fake_mlflow = _tracker()
        tracker.log_cache_stats(_stats(), step=7)
        assert fake_mlflow.log_metrics.call_args.kwargs == {"step": 7}

    def test_zero_stats_log_zero_rates(self) -> None:
        tracker, fake_mlflow = _tracker()
        tracker.log_cache_stats(RunCacheStats())
        metrics = fake_mlflow.log_metrics.call_args.args[0]
        assert metrics["agent_cache_hit_rate"] == 0.0
        assert metrics["feature_store_hit_rate"] == 0.0
        assert metrics["llm_calls_avoided_estimate"] == 0.0


class TestEndRun:
    def test_default_status_is_finished(self) -> None:
        tracker, fake_mlflow = _tracker()
        tracker._active_run = MagicMock()
        tracker.end_run()
        fake_mlflow.end_run.assert_called_once_with(status="FINISHED")
        assert tracker._active_run is None

    def test_failed_status_is_passed_through(self) -> None:
        tracker, fake_mlflow = _tracker()
        tracker._active_run = MagicMock()
        tracker.end_run(status="FAILED")
        fake_mlflow.end_run.assert_called_once_with(status="FAILED")

    def test_without_an_active_run_nothing_is_ended(self) -> None:
        tracker, fake_mlflow = _tracker()
        tracker.end_run(status="FAILED")
        fake_mlflow.end_run.assert_not_called()
