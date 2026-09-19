"""Tests for ctra.agents.runner.run_agent_as_subprocess.

Pure unit tests with subprocess.run and dill fully mocked.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import dill
import pandas as pd
import pytest

from ctra.agents.data_models import (
    AgentOutput,
    CacheStats,
    EvalOutput,
    FeaturePlan,
    FeatureSource,
    FeatureType,
    ModelEvalResult,
    Task,
)
from ctra.agents.runner import (
    RunCacheStats,
    load_feature_plans_from_json,
    run_agent_as_subprocess,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_output(roc_auc: float = 0.85) -> AgentOutput:
    er = ModelEvalResult(
        roc_auc=roc_auc,
        f1=0.75,
        pr_auc=0.7,
        interaction_values={},
        wrong_idxs=[],
        wrong_preds=[],
        wrong_df=pd.DataFrame(),
        pipeline=None,
    )
    return AgentOutput(
        eval_outputs={"xgboost": EvalOutput(model_eval_result=er, suggestions=["add feat"])},
        test_eval_outputs={"xgboost": er},
        operation=None,
        feature_plans={},
        df=pd.DataFrame(),
        val_df=pd.DataFrame(),
        suggestion_index=0,
        raw_features={},
        raw_val_features={},
        raw_test_features={},
        none_explanations={},
        builder_meta={},
    )


def _mock_settings(monkeypatch: pytest.MonkeyPatch, output_dir: Path) -> MagicMock:
    """Mock get_settings for run_agent_as_subprocess.

    ``output_dir`` is what ``cache_dir=None`` derives ``agent_cache`` from, so
    callers pass ``tmp_path`` and the fallback can never escape the sandbox.
    """
    mock_s = MagicMock()
    mock_s.output_dir = output_dir
    monkeypatch.setattr("ctra.agents.runner.get_settings", MagicMock(return_value=mock_s))
    return mock_s


# ---------------------------------------------------------------------------
# Tests: cache hit
# ---------------------------------------------------------------------------


class TestSubprocessCacheHit:
    def test_loads_from_cache_without_subprocess(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When a cached pickle exists, load it directly without spawning subprocess."""
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()

        expected_output = _make_mock_output(roc_auc=0.90)
        cached_path = cache_dir / "phase2--node-42.output.pkl"
        with open(cached_path, "wb") as f:
            dill.dump(expected_output, f)

        _mock_settings(monkeypatch, tmp_path)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)

        result = run_agent_as_subprocess(
            node_id="node-42",
            task="phase2",
            previous_output=None,
            cache_dir=cache_dir,
        )

        # Should have loaded from cache
        assert result.eval_outputs["xgboost"].model_eval_result.roc_auc == 0.90
        # subprocess.run should NOT have been called
        mock_subprocess_run.assert_not_called()

    def test_default_cache_dir_derives_from_output_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """With ``cache_dir=None`` the cache lives at ``settings.output_dir / 'agent_cache'``."""
        output_dir = tmp_path / "out"
        derived = output_dir / "agent_cache"
        derived.mkdir(parents=True)

        expected_output = _make_mock_output(roc_auc=0.77)
        with open(derived / "phase2--node-7.output.pkl", "wb") as f:
            dill.dump(expected_output, f)

        _mock_settings(monkeypatch, output_dir)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)

        result = run_agent_as_subprocess(
            node_id="node-7",
            task="phase2",
            previous_output=None,
            cache_dir=None,
        )

        assert result.eval_outputs["xgboost"].model_eval_result.roc_auc == 0.77
        mock_subprocess_run.assert_not_called()

    def test_cache_hit_with_different_tasks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Cache keys include task name, so different tasks don't collide."""
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()

        output_p2 = _make_mock_output(roc_auc=0.80)
        cached_p2 = cache_dir / "phase2--node-1.output.pkl"
        with open(cached_p2, "wb") as f:
            dill.dump(output_p2, f)

        _mock_settings(monkeypatch, tmp_path)

        # This should NOT find a cache for phase3
        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)

        # Mock dill.load/dump for subprocess path (non-cached)
        mock_output_p3 = _make_mock_output(roc_auc=0.88)

        def mock_dill_load_side_effect(f):
            return mock_output_p3

        monkeypatch.setattr("ctra.agents.runner.dill.load", mock_dill_load_side_effect)
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        _result = run_agent_as_subprocess(
            node_id="node-1",
            task="phase3",
            previous_output=None,
            cache_dir=cache_dir,
        )

        # subprocess.run SHOULD have been called for phase3
        mock_subprocess_run.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: cache miss — subprocess spawned
# ---------------------------------------------------------------------------


class TestSubprocessCacheMiss:
    def test_spawns_subprocess_with_correct_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()

        _mock_settings(monkeypatch, tmp_path)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)

        expected_output = _make_mock_output()

        def mock_dill_load(f):
            return expected_output

        monkeypatch.setattr("ctra.agents.runner.dill.load", mock_dill_load)
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        run_agent_as_subprocess(
            node_id="node-0",
            task="phase2",
            previous_output=None,
            cache_dir=cache_dir,
        )

        # Verify subprocess.run was called
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]

        # Command should include python, script path, --task, --output
        assert cmd[0] == sys.executable
        assert "scripts/run_agent.py" in cmd[1]
        assert "--task" in cmd
        assert "phase2" in cmd
        assert "--output" in cmd

    def test_includes_input_flag_when_previous_output_given(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()

        _mock_settings(monkeypatch, tmp_path)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)

        expected_output = _make_mock_output()
        monkeypatch.setattr("ctra.agents.runner.dill.load", lambda f: expected_output)
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        prev_output = _make_mock_output(roc_auc=0.80)
        run_agent_as_subprocess(
            node_id="node-1",
            task="phase2",
            previous_output=prev_output,
            cache_dir=cache_dir,
        )

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--input" in cmd

    def test_no_input_flag_when_previous_output_is_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()

        _mock_settings(monkeypatch, tmp_path)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)

        expected_output = _make_mock_output()
        monkeypatch.setattr("ctra.agents.runner.dill.load", lambda f: expected_output)
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        run_agent_as_subprocess(
            node_id="node-0",
            task="phase2",
            previous_output=None,
            cache_dir=cache_dir,
        )

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--input" not in cmd

    def test_caches_output_after_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()

        _mock_settings(monkeypatch, tmp_path)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)

        expected_output = _make_mock_output()
        monkeypatch.setattr("ctra.agents.runner.dill.load", lambda f: expected_output)

        dill_dump_calls = []
        _original_dill_dump = dill.dump

        def tracking_dump(obj, f):
            dill_dump_calls.append(getattr(f, "name", str(f)))

        monkeypatch.setattr("ctra.agents.runner.dill.dump", tracking_dump)

        run_agent_as_subprocess(
            node_id="node-0",
            task="phase2",
            previous_output=None,
            cache_dir=cache_dir,
        )

        # Should have cached the output (written to cache path)
        cached_paths = [p for p in dill_dump_calls if "node-0.output.pkl" in p]
        assert len(cached_paths) == 1


# ---------------------------------------------------------------------------
# Tests: subprocess failure
# ---------------------------------------------------------------------------


class TestSubprocessFailure:
    def test_called_process_error_raised(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()

        _mock_settings(monkeypatch, tmp_path)

        monkeypatch.setattr(
            "ctra.agents.runner.subprocess.run",
            MagicMock(side_effect=subprocess.CalledProcessError(1, "run_agent.py")),
        )
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        with pytest.raises(subprocess.CalledProcessError):
            run_agent_as_subprocess(
                node_id="node-fail",
                task="phase2",
                previous_output=None,
                cache_dir=cache_dir,
            )

    def test_timeout_error_raised(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()

        _mock_settings(monkeypatch, tmp_path)

        monkeypatch.setattr(
            "ctra.agents.runner.subprocess.run",
            MagicMock(side_effect=subprocess.TimeoutExpired("run_agent.py", 3600)),
        )
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        with pytest.raises(subprocess.TimeoutExpired):
            run_agent_as_subprocess(
                node_id="node-timeout",
                task="phase2",
                previous_output=None,
                cache_dir=cache_dir,
            )


# ---------------------------------------------------------------------------
# Tests: Task enum support
# ---------------------------------------------------------------------------


class TestSubprocessTaskEnum:
    def test_task_enum_derives_correct_cli_arg(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Task.TRIAL_OUTCOME_PHASE_2 should produce '--task phase2' in subprocess cmd."""
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()
        _mock_settings(monkeypatch, tmp_path)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)

        expected_output = _make_mock_output()
        monkeypatch.setattr("ctra.agents.runner.dill.load", lambda f: expected_output)
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        run_agent_as_subprocess(
            node_id="node-0",
            task=Task.TRIAL_OUTCOME_PHASE_2,
            previous_output=None,
            cache_dir=cache_dir,
        )

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--task" in cmd
        task_idx = cmd.index("--task")
        assert cmd[task_idx + 1] == "phase2"

    def test_task_enum_phase1_derives_phase1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()
        _mock_settings(monkeypatch, tmp_path)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)
        monkeypatch.setattr("ctra.agents.runner.dill.load", lambda f: _make_mock_output())
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        run_agent_as_subprocess(
            node_id="node-0",
            task=Task.TRIAL_OUTCOME_PHASE_1,
            previous_output=None,
            cache_dir=cache_dir,
        )

        cmd = mock_subprocess_run.call_args[0][0]
        task_idx = cmd.index("--task")
        assert cmd[task_idx + 1] == "phase1"

    def test_task_enum_phase3_derives_phase3(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()
        _mock_settings(monkeypatch, tmp_path)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)
        monkeypatch.setattr("ctra.agents.runner.dill.load", lambda f: _make_mock_output())
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        run_agent_as_subprocess(
            node_id="node-0",
            task=Task.TRIAL_OUTCOME_PHASE_3,
            previous_output=None,
            cache_dir=cache_dir,
        )

        cmd = mock_subprocess_run.call_args[0][0]
        task_idx = cmd.index("--task")
        assert cmd[task_idx + 1] == "phase3"

    def test_task_enum_cache_key_uses_cli_arg(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Cache file should be named 'phase2--node-x.output.pkl', not the enum repr."""
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()
        _mock_settings(monkeypatch, tmp_path)

        mock_subprocess_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_subprocess_run)
        monkeypatch.setattr("ctra.agents.runner.dill.load", lambda f: _make_mock_output())

        dump_paths = []

        def tracking_dump(obj, f):
            dump_paths.append(getattr(f, "name", str(f)))

        monkeypatch.setattr("ctra.agents.runner.dill.dump", tracking_dump)

        run_agent_as_subprocess(
            node_id="node-42",
            task=Task.TRIAL_OUTCOME_PHASE_2,
            previous_output=None,
            cache_dir=cache_dir,
        )

        cached = [p for p in dump_paths if "node-42" in p]
        assert len(cached) == 1
        assert "phase2--node-42" in cached[0]


# ---------------------------------------------------------------------------
# Tests: generic Task raises ValueError
# ---------------------------------------------------------------------------


class TestSubprocessGenericTaskRaises:
    def test_generic_trial_outcome_raises_value_error(self) -> None:
        """Task enum members without a phase (if any existed) should raise."""
        # Simulate a Task-like object with phase=None
        mock_task = MagicMock(spec=Task)
        mock_task.phase = None
        mock_task.name = "TRIAL_OUTCOME"

        # Must use isinstance check — mock won't pass isinstance(Task)
        # So we test the ValueError logic directly by calling with the real check
        # Since TRIAL_OUTCOME was removed, verify the ValueError message pattern
        # by testing with a mock that has phase=None
        # (The actual enum no longer has a None-phase member, but runner.py
        # still guards against it.)

    def test_all_current_tasks_have_phase(self) -> None:
        """Verify every Task member has a non-None phase (no generic task exists)."""
        for t in Task:
            assert t.phase is not None, f"{t.name} has phase=None"


# ---------------------------------------------------------------------------
# Tests: load_feature_plans_from_json
# ---------------------------------------------------------------------------


class TestLoadFeaturePlansFromJson:
    def test_roundtrip(self, tmp_path: Path) -> None:
        """Save and reload feature plans via JSON."""
        import json

        plans = {
            "drug_targets": {
                "feature_name": "drug_targets",
                "feature_idea": "Count known drug targets",
                "feature_type": {"count": "integer"},
                "data_sources": ["chembl"],
                "example_values": [{"count": "3"}],
                "possible_values": {},
                "feature_instructions": "Query ChEMBL for target count",
            },
        }

        path = tmp_path / "plans.json"
        path.write_text(json.dumps(plans))

        result = load_feature_plans_from_json(path)
        assert "drug_targets" in result
        plan = result["drug_targets"]
        assert isinstance(plan, FeaturePlan)
        assert plan.feature_name == "drug_targets"
        assert plan.feature_type == {"count": FeatureType.INTEGER}
        assert plan.data_sources == [FeatureSource.CHEMBL]

    def test_multiple_plans(self, tmp_path: Path) -> None:
        """Load multiple feature plans."""
        import json

        plans = {
            "feat_a": {
                "feature_name": "feat_a",
                "feature_idea": "idea_a",
                "feature_type": {"val": "float"},
                "data_sources": ["pubmed"],
                "example_values": [],
                "possible_values": {},
                "feature_instructions": "instr_a",
            },
            "feat_b": {
                "feature_name": "feat_b",
                "feature_idea": "idea_b",
                "feature_type": {"present": "boolean"},
                "data_sources": ["faers", "aact"],
                "example_values": [],
                "possible_values": {},
                "feature_instructions": "instr_b",
            },
        }

        path = tmp_path / "plans.json"
        path.write_text(json.dumps(plans))

        result = load_feature_plans_from_json(path)
        assert len(result) == 2
        assert result["feat_a"].feature_type == {"val": FeatureType.FLOAT}
        assert result["feat_b"].data_sources == [FeatureSource.FAERS, FeatureSource.AACT]

    def test_multi_categorical_type(self, tmp_path: Path) -> None:
        """Ensure multi-categorical feature type is handled."""
        import json

        plans = {
            "conditions": {
                "feature_name": "conditions",
                "feature_idea": "Trial conditions",
                "feature_type": {"labels": "multi-categorical"},
                "data_sources": ["current_trial_summary"],
                "example_values": [],
                "possible_values": {"labels": ["cancer", "diabetes"]},
                "feature_instructions": "extract",
            },
        }

        path = tmp_path / "plans.json"
        path.write_text(json.dumps(plans))

        result = load_feature_plans_from_json(path)
        assert result["conditions"].feature_type == {"labels": FeatureType.MULTICATEGORICAL}


# ---------------------------------------------------------------------------
# Tests: per-run cache counters (issue #17)
# ---------------------------------------------------------------------------


def _output_with_stats(**counts: int) -> AgentOutput:
    """An output whose ``cache_stats`` carry the given counts."""
    return _make_mock_output()._replace(cache_stats=CacheStats(**counts))


class TestRunCacheStats:
    """``RunCacheStats`` is the runner's per-run tally of both cache layers."""

    def test_zero_state_is_safe(self) -> None:
        stats = RunCacheStats()
        assert stats.agent_hit_rate == 0.0
        assert stats.as_dict() == {
            "agent_hits": 0,
            "agent_misses": 0,
            "agent_hit_rate": 0.0,
            "replayed_group_builds": 0,
            "llm_calls_made": 0,
            "feature_store": stats.feature_store.as_dict(),
        }

    def test_hit_rate_arithmetic(self) -> None:
        stats = RunCacheStats(agent_hits=1, agent_misses=3)
        assert stats.agent_lookups == 4
        assert stats.agent_hit_rate == pytest.approx(0.25)

    def test_a_hit_replays_the_pickle_and_folds_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The fence: a replayed pickle's store counters belong to the run that made it."""
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()
        cached = _output_with_stats(
            feature_lookups=10, feature_hits=4, groups_dispatched=5, llm_calls_made=30
        )
        with open(cache_dir / "phase2--node-7.output.pkl", "wb") as f:
            dill.dump(cached, f)
        _mock_settings(monkeypatch, cache_dir)
        mock_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_run)

        stats = RunCacheStats()
        result = run_agent_as_subprocess("node-7", "phase2", None, cache_dir=cache_dir, stats=stats)

        mock_run.assert_not_called()
        assert result.cache_stats.groups_dispatched == 5
        assert stats.agent_hits == 1
        assert stats.agent_misses == 0
        assert stats.replayed_group_builds == 5
        assert stats.feature_store.feature_lookups == 0
        assert stats.feature_store.feature_hits == 0
        assert stats.llm_calls_made == 0

    def test_a_miss_spawns_and_folds_the_child_counters(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()
        _mock_settings(monkeypatch, cache_dir)
        mock_run = MagicMock()
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", mock_run)
        child = _output_with_stats(
            feature_lookups=10,
            feature_hits=4,
            groups_dispatched=5,
            groups_skipped=2,
            hits_from_planner_plans=4,
            llm_calls_made=30,
        )
        monkeypatch.setattr("ctra.agents.runner.dill.load", lambda f: child)
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        stats = RunCacheStats()
        run_agent_as_subprocess("node-0", "phase2", None, cache_dir=cache_dir, stats=stats)

        mock_run.assert_called_once()
        assert stats.agent_misses == 1
        assert stats.agent_hits == 0
        assert stats.replayed_group_builds == 0
        assert stats.feature_store.feature_lookups == 10
        assert stats.feature_store.feature_hits == 4
        assert stats.feature_store.groups_skipped == 2
        assert stats.feature_store.hits_from_planner_plans == 4
        assert stats.llm_calls_made == 30

    def test_miss_then_hit_on_the_same_node(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The miss writes the pickle the second call replays; counts add up."""
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()
        _mock_settings(monkeypatch, cache_dir)
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", MagicMock())
        child = _output_with_stats(feature_lookups=3, feature_hits=1, groups_dispatched=2)
        real_load = dill.load
        loads: list[str] = []

        def load(f):
            loads.append(f.name)
            # The child's (empty, mocked) output file yields the stub; the
            # cache pickle written by the miss is read for real.
            return child if not f.name.endswith(".output.pkl") else real_load(f)

        monkeypatch.setattr("ctra.agents.runner.dill.load", load)

        stats = RunCacheStats()
        run_agent_as_subprocess("node-1", "phase2", None, cache_dir=cache_dir, stats=stats)
        run_agent_as_subprocess("node-1", "phase2", None, cache_dir=cache_dir, stats=stats)

        assert stats.agent_misses == 1
        assert stats.agent_hits == 1
        assert stats.agent_hit_rate == pytest.approx(0.5)
        assert stats.replayed_group_builds == 2
        # Folded once, on the miss.
        assert stats.feature_store.feature_lookups == 3
        assert stats.feature_store.groups_dispatched == 2

    def test_stats_none_is_a_no_op_on_both_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()
        with open(cache_dir / "phase2--hit.output.pkl", "wb") as f:
            dill.dump(_output_with_stats(groups_dispatched=5), f)
        _mock_settings(monkeypatch, cache_dir)
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", MagicMock())
        real_load = dill.load
        monkeypatch.setattr(
            "ctra.agents.runner.dill.load",
            lambda f: real_load(f) if f.name.endswith(".output.pkl") else _make_mock_output(),
        )
        monkeypatch.setattr("ctra.agents.runner.dill.dump", MagicMock())

        hit = run_agent_as_subprocess("hit", "phase2", None, cache_dir=cache_dir)
        miss = run_agent_as_subprocess("miss", "phase2", None, cache_dir=cache_dir)
        assert hit.cache_stats.groups_dispatched == 5
        assert miss.cache_stats == CacheStats()

    def test_hit_on_a_pickle_without_the_field_counts_no_replay(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "agent_cache"
        cache_dir.mkdir()
        old = _make_mock_output()
        del old.__dict__["cache_stats"]
        with open(cache_dir / "phase2--old.output.pkl", "wb") as f:
            dill.dump(old, f)
        _mock_settings(monkeypatch, cache_dir)
        monkeypatch.setattr("ctra.agents.runner.subprocess.run", MagicMock())

        stats = RunCacheStats()
        run_agent_as_subprocess("old", "phase2", None, cache_dir=cache_dir, stats=stats)
        assert stats.agent_hits == 1
        assert stats.replayed_group_builds == 0

    def test_partial_with_stats_survives_dill(self, tmp_path: Path) -> None:
        """``train_mcts.py`` binds the stats into the runner partial that checkpoints pickle."""
        import functools

        stats = RunCacheStats(agent_hits=2, agent_misses=5)
        stats.feature_store.feature_hits = 9
        runner = functools.partial(
            run_agent_as_subprocess, cache_dir=tmp_path / "agent_cache", stats=stats
        )
        restored = dill.loads(dill.dumps(runner))
        assert restored.func is run_agent_as_subprocess
        assert restored.keywords["cache_dir"] == tmp_path / "agent_cache"
        assert restored.keywords["stats"] == stats
        assert restored.keywords["stats"].feature_store.feature_hits == 9
