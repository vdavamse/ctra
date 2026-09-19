"""Tests for scripts/train_mcts.py — MCTS training entry point.

Pure unit tests with all external dependencies mocked (LLM, RAG,
CTGLoader, Agent, MCTSSearch, dill, file I/O).
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# We import the script module directly; all heavy dependencies are
# lazy-imported inside functions, so import itself is safe.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_benchmark_df(id_col: str = "nctid", n: int = 10):
    """Create a minimal polars-like DataFrame that has .to_pandas()."""
    pdf = pd.DataFrame(
        {
            id_col: [f"NCT{i:08d}" for i in range(n)],
            "label": np.random.randint(0, 2, size=n),
        }
    )
    mock_df = MagicMock()
    mock_df.to_pandas.return_value = pdf
    return mock_df


def _make_mock_settings(num_rollouts=3, max_depth=2):
    """Create a mock settings object matching MCTSConfig interface."""
    settings = MagicMock()
    settings.mcts.num_rollouts = num_rollouts
    settings.mcts.max_depth = max_depth
    settings.mcts.objectives = ["accuracy", "parsimony"]
    settings.mcts.reference_point = [0.5, 0.0]
    settings.mcts.backprop = "mean"
    return settings


def _make_best_node(features=None):
    """Create a mock MCTSNode returned by search()."""
    node = MagicMock()
    node.features = features or ["feat_a", "feat_b"]
    node.mean_reward = np.array([0.85, 0.75])
    return node


def _make_mock_mcts(
    best_node,
    all_nodes=None,
    own_objectives=(0.95, 0.70),
    reference_point=(0.5, 0.0),
    backprop="mean",
):
    """Create a mock MCTSSearch.

    ``best_own_objectives`` has to return a real array: ``train_mcts`` writes
    it to ``results.json`` as ``best_objectives`` and prints it, so a bare
    ``MagicMock`` attribute would make the file unserialisable (issue #15).
    ``config.reference_point`` has to be a real list for the same reason: a
    resume compares it with the current settings (issue #18), as it does
    ``config.backprop`` (issue #16) — a bare ``MagicMock`` attribute would
    never equal the settings' string and every resume would warn.
    ``value_estimate`` mirrors the real method (it returns the node's
    ``mean_reward``): ``train_mcts`` writes it as ``best_mean_objectives``.
    """
    mcts = MagicMock()
    mcts.search.return_value = best_node
    mcts.all_nodes = list(all_nodes) if all_nodes is not None else [best_node]
    mcts.best_own_objectives.return_value = np.array(own_objectives)
    mcts.value_estimate.side_effect = lambda node: node.mean_reward
    mcts.config.reference_point = list(reference_point)
    mcts.config.backprop = backprop
    return mcts


def _make_mock_output():
    """Create a mock AgentOutput with eval_outputs."""
    mock_eval = MagicMock()
    mock_eval.model_eval_result.pipeline = MagicMock()
    mock_eval.model_eval_result.roc_auc = 0.88

    output = MagicMock()
    output.feature_plans = {"feat_a": MagicMock(), "feat_b": MagicMock()}
    output.eval_outputs = {"xgboost": mock_eval}
    output.get_best_eval_output.return_value = (mock_eval, MagicMock())
    return output


# ---------------------------------------------------------------------------
# Tests: parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--task is required; omitting it should cause SystemExit."""
        monkeypatch.setattr(sys, "argv", ["train_mcts.py"])
        from train_mcts import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_valid_task_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["train_mcts.py", "--task", "phase2"])
        from train_mcts import parse_args

        args = parse_args()
        assert args.task == "phase2"

    def test_invalid_task_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["train_mcts.py", "--task", "phase4"])
        from train_mcts import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["train_mcts.py", "--task", "phase1"])
        from train_mcts import parse_args

        args = parse_args()
        assert args.rollouts is None
        assert args.depth is None
        assert args.resume is None
        assert args.output_dir == ".output"
        assert args.checkpoint_every == 5

    def test_all_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mcts.py",
                "--task",
                "phase3",
                "--rollouts",
                "50",
                "--depth",
                "15",
                "--resume",
                "/tmp/ckpt.pkl",
                "--output-dir",
                "/tmp/out",
                "--checkpoint-every",
                "10",
            ],
        )
        from train_mcts import parse_args

        args = parse_args()
        assert args.task == "phase3"
        assert args.rollouts == 50
        assert args.depth == 15
        assert args.resume == "/tmp/ckpt.pkl"
        assert args.output_dir == "/tmp/out"
        assert args.checkpoint_every == 10


# ---------------------------------------------------------------------------
# Tests: main() integration (fully mocked)
# ---------------------------------------------------------------------------


class TestMainFreshStart:
    """Test main() in fresh-start mode (no --resume)."""

    def test_main_full_flow(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mcts.py",
                "--task",
                "phase2",
                "--rollouts",
                "3",
                "--output-dir",
                str(output_dir),
                "--checkpoint-every",
                "2",
            ],
        )

        # Mock configure_lm
        mock_configure_lm = MagicMock()
        monkeypatch.setattr("ctra.agents.lm_config.configure_lm", mock_configure_lm)

        # Mock load_benchmark (via CTGLoader)
        train_df = _make_benchmark_df("nctid", 20)
        val_df = _make_benchmark_df("nctid", 5)
        test_df = _make_benchmark_df("nctid", 5)
        mock_loader = MagicMock()
        mock_loader.load_benchmark_splits.return_value = (train_df, val_df, test_df)
        monkeypatch.setattr("ctra.data.ctg_loader.CTGLoader", MagicMock(return_value=mock_loader))

        # Mock Agent
        mock_agent = MagicMock()
        mock_agent_cls = MagicMock(return_value=mock_agent)
        monkeypatch.setattr("ctra.agents.orchestrator.Agent", mock_agent_cls)

        # Mock MCTSSearch — train_mcts.py now constructs it directly
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node)

        monkeypatch.setattr(
            "ctra.search.mcts.MCTSSearch",
            MagicMock(return_value=mock_mcts),
        )

        # Mock get_settings
        mock_settings = _make_mock_settings(num_rollouts=3, max_depth=2)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )

        # Mock dump_as_json
        monkeypatch.setattr(
            "ctra.agents.feature_utils.dump_as_json",
            MagicMock(return_value='{"feat_a": {}}'),
        )

        # Mock dill.dump (for checkpoint/model saving)
        mock_dill_dump = MagicMock()
        monkeypatch.setattr("dill.dump", mock_dill_dump)

        # Mock tqdm to avoid progress bar issues
        monkeypatch.setattr("builtins.__import__", _import_without_tqdm(monkeypatch))

        from train_mcts import main

        main()

        # Verify search was called
        mock_mcts.search.assert_called_once()
        call_kwargs = mock_mcts.search.call_args
        assert call_kwargs.kwargs.get("start_rollout", call_kwargs[1].get("start_rollout", 0)) == 0

        # Verify output files exist (phase-namespaced under output_dir/phase2/)
        phase_dir = output_dir / "phase2"
        assert (phase_dir / "results.json").exists()
        assert (phase_dir / "feature_plans.json").exists()

        # Verify dill.dump was called (for model + final checkpoint)
        assert mock_dill_dump.call_count >= 2

    def test_main_sets_rollout_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """CLI --rollouts overrides settings.mcts.num_rollouts."""
        output_dir = tmp_path / "output"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mcts.py",
                "--task",
                "phase1",
                "--rollouts",
                "42",
                "--output-dir",
                str(output_dir),
            ],
        )

        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node)

        monkeypatch.setattr(
            "ctra.search.mcts.MCTSSearch",
            MagicMock(return_value=mock_mcts),
        )

        mock_settings = _make_mock_settings(num_rollouts=20)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))
        monkeypatch.setattr("dill.dump", MagicMock())

        from train_mcts import main

        main()

        # num_rollouts should have been overridden to 42
        assert mock_settings.mcts.num_rollouts == 42


class TestMainResume:
    """Test main() in resume mode (--resume)."""

    def test_resume_loads_checkpoint(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        ckpt_path = tmp_path / "checkpoint.pkl"
        ckpt_path.touch()  # just needs to exist for open()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mcts.py",
                "--task",
                "phase2",
                "--resume",
                str(ckpt_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        # Checkpoint data
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node)

        checkpoint = {
            "mcts": mock_mcts,
            "rollout": 5,
            "args": {"task": "phase2"},
        }

        mock_dill_load = MagicMock(return_value=checkpoint)
        monkeypatch.setattr("dill.load", mock_dill_load)
        monkeypatch.setattr("dill.dump", MagicMock())

        mock_settings = _make_mock_settings(num_rollouts=10)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))

        from train_mcts import main

        main()

        # Search should be called with start_rollout=6 (checkpoint rollout 5 + 1)
        search_call = mock_mcts.search.call_args
        assert search_call.kwargs.get("start_rollout", search_call[1].get("start_rollout")) == 6

    @staticmethod
    def _resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mock_mcts) -> MagicMock:
        """Run ``main`` in resume mode with ``mock_mcts`` in the checkpoint; return the settings."""
        output_dir = tmp_path / "output"
        ckpt_path = tmp_path / "checkpoint.pkl"
        ckpt_path.touch()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                *("train_mcts.py", "--task", "phase2"),
                *("--resume", str(ckpt_path), "--output-dir", str(output_dir)),
            ],
        )
        checkpoint = {
            "mcts": mock_mcts,
            "rollout": 5,
            "args": {"task": "phase2", "output_dir": str(output_dir)},
        }
        monkeypatch.setattr("dill.load", MagicMock(return_value=checkpoint))
        monkeypatch.setattr("dill.dump", MagicMock())
        mock_settings = _make_mock_settings(num_rollouts=10)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))

        from train_mcts import main

        main()
        return mock_settings

    def test_resume_warns_when_the_checkpoint_keeps_an_older_reference_point(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A checkpoint from before issue #18 carries [0.0, 0.0]; it is kept and named."""
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node, reference_point=(0.0, 0.0))

        with caplog.at_level("WARNING", logger="ctra.train"):
            mock_settings = self._resume(monkeypatch, tmp_path, mock_mcts)

        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, warnings
        assert "reference_point [0.0, 0.0]" in warnings[0]
        assert "current settings say [0.5, 0.0]" in warnings[0]
        # Neither side is rewritten: the checkpoint's geometry stays mid-run.
        assert mock_mcts.config.reference_point == [0.0, 0.0]
        assert mock_settings.mcts.reference_point == [0.5, 0.0]

    def test_resume_is_silent_when_the_reference_points_agree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node, reference_point=(0.5, 0.0))

        with caplog.at_level("WARNING", logger="ctra.train"):
            self._resume(monkeypatch, tmp_path, mock_mcts)

        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_resume_warns_when_the_checkpoint_keeps_another_backprop_rule(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A checkpoint started under ``max`` keeps it and says so (issue #16)."""
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node, backprop="max")

        with caplog.at_level("WARNING", logger="ctra.train"):
            mock_settings = self._resume(monkeypatch, tmp_path, mock_mcts)

        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, warnings
        assert "backprop 'max'" in warnings[0]
        assert "current settings say 'mean'" in warnings[0]
        assert mock_mcts.config.backprop == "max"
        assert mock_settings.mcts.backprop == "mean"

    def test_resume_reads_a_checkpoint_without_the_backprop_field_as_mean(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A checkpoint pickled before the field existed ran under ``mean``: no warning."""
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node)
        del mock_mcts.config.backprop  # MagicMock: attribute access now raises AttributeError

        with caplog.at_level("WARNING", logger="ctra.train"):
            self._resume(monkeypatch, tmp_path, mock_mcts)

        assert not [r for r in caplog.records if r.levelname == "WARNING"]


# ---------------------------------------------------------------------------
# Tests: on_rollout callback
# ---------------------------------------------------------------------------


class TestOnRolloutCallback:
    def test_checkpoint_saved_at_interval(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """on_rollout saves checkpoint when (rollout_idx + 1) % checkpoint_every == 0."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mcts.py",
                "--task",
                "phase2",
                "--output-dir",
                str(output_dir),
                "--checkpoint-every",
                "2",
            ],
        )

        # Capture on_rollout from search() call
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()

        def mock_search(initial_features, on_rollout=None, start_rollout=0):
            # Simulate calling on_rollout for rollouts 0, 1, 2
            if on_rollout is not None:
                on_rollout(0, best_node, np.array([0.80, 0.70]))  # idx 0 -> no checkpoint (1%2!=0)
                on_rollout(1, best_node, np.array([0.82, 0.72]))  # idx 1 -> checkpoint (2%2==0)
                on_rollout(2, best_node, np.array([0.85, 0.75]))  # idx 2 -> no checkpoint (3%2!=0)
            return best_node

        mock_mcts = _make_mock_mcts(best_node)
        mock_mcts.search.side_effect = mock_search

        monkeypatch.setattr(
            "ctra.search.mcts.MCTSSearch",
            MagicMock(return_value=mock_mcts),
        )

        mock_settings = _make_mock_settings(num_rollouts=3)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))

        dill_dump_calls = []
        checkpoint_args = []

        def tracking_dill_dump(obj, f):
            if isinstance(obj, dict) and "rollout" in obj:
                dill_dump_calls.append(obj["rollout"])
                checkpoint_args.append(obj["args"])

        monkeypatch.setattr("dill.dump", tracking_dill_dump)

        from train_mcts import main

        main()

        # Checkpoint should have been saved at rollout index 1 (since (1+1)%2==0)
        assert 1 in dill_dump_calls
        # Every checkpoint carries what rebuilds the per-run agent cache dir
        # (output_dir / task subdir / "agent_cache" / run_id) on resume.
        for ckpt_args in checkpoint_args:
            assert ckpt_args["output_dir"] == str(output_dir)
            assert ckpt_args["task"] == "phase2"
            assert ckpt_args["run_id"]


# ---------------------------------------------------------------------------
# Tests: per-run agent cache (issue #12 review)
# ---------------------------------------------------------------------------


def _assert_per_run_runner(runner, expected_cache_dir: Path) -> None:
    """``runner`` is ``run_agent_as_subprocess`` bound to this run's cache dir and stats."""
    from ctra.agents.runner import RunCacheStats, run_agent_as_subprocess

    assert isinstance(runner, functools.partial), runner
    assert runner.func is run_agent_as_subprocess
    assert runner.args == ()
    assert set(runner.keywords) == {"cache_dir", "stats"}
    assert runner.keywords["cache_dir"] == expected_cache_dir
    assert isinstance(runner.keywords["stats"], RunCacheStats)


class TestPerRunAgentCache:
    """The node id is stable across runs, so the agent cache must be per run.

    ``run_agent_as_subprocess`` defaults ``cache_dir`` to the global
    ``output/agent_cache``; with a run-stable key a fresh run would load the
    previous run's root pickle (same ``initial_features=[]``, rollout 0, no
    parent) and replay rollout 0 wholesale instead of running the agent.
    The same holds for two fresh runs over one output directory, so the
    cache lives under ``<output_dir>/<task>/agent_cache/<run_id>`` with the
    run id persisted in every checkpoint's ``args``.
    """

    @staticmethod
    def _capture_checkpoints(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        checkpoints: list[dict] = []

        def tracking_dill_dump(obj, f):
            if isinstance(obj, dict) and "rollout" in obj:
                checkpoints.append(obj)

        monkeypatch.setattr("dill.dump", tracking_dill_dump)
        return checkpoints

    @staticmethod
    def _fresh_main(monkeypatch: pytest.MonkeyPatch, output_dir: Path) -> MagicMock:
        """Run a fresh ``main()`` with ``MCTSSearch`` mocked; return the mocked class."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["train_mcts.py", "--task", "phase2", "--output-dir", str(output_dir)],
        )
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node)
        mock_mcts_cls = MagicMock(return_value=mock_mcts)
        monkeypatch.setattr("ctra.search.mcts.MCTSSearch", mock_mcts_cls)
        mock_settings = _make_mock_settings(num_rollouts=3)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))

        from train_mcts import main

        main()
        mock_mcts_cls.assert_called_once()
        return mock_mcts_cls

    @staticmethod
    def _resume_main(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        cli_task: str,
        ckpt_args: dict,
    ) -> tuple[MagicMock, Path]:
        """Run ``main()`` in resume mode against a mocked checkpoint.

        Returns the restored ``mcts`` mock and the checkpoint path.  The
        checkpoint's ``args`` are what a previous run's ``vars(args)`` dumped.
        """
        output_dir = tmp_path / "output"
        ckpt_path = tmp_path / "checkpoint.pkl"
        ckpt_path.touch()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mcts.py",
                "--task",
                cli_task,
                "--resume",
                str(ckpt_path),
                "--output-dir",
                str(output_dir),
            ],
        )
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node)
        checkpoint = {"mcts": mock_mcts, "rollout": 4, "args": ckpt_args}
        monkeypatch.setattr("dill.load", MagicMock(return_value=checkpoint))

        mock_mcts_cls = MagicMock()
        monkeypatch.setattr("ctra.search.mcts.MCTSSearch", mock_mcts_cls)
        mock_settings = _make_mock_settings(num_rollouts=10)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))

        from train_mcts import main

        main()
        mock_mcts_cls.assert_not_called()
        return mock_mcts, ckpt_path

    def test_fresh_start_scopes_cache_to_run_id_under_output_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        checkpoints = self._capture_checkpoints(monkeypatch)
        mock_mcts_cls = self._fresh_main(monkeypatch, output_dir)

        runner = mock_mcts_cls.call_args.kwargs["runner"]
        cache_dir = runner.keywords["cache_dir"]
        assert cache_dir.parent == output_dir / "phase2" / "agent_cache"
        _assert_per_run_runner(runner, cache_dir)
        # The run id is persisted in every checkpoint so a resume rebuilds the dir.
        assert checkpoints, "the final checkpoint was not dumped"
        for checkpoint in checkpoints:
            assert checkpoint["args"]["run_id"] == cache_dir.name
            assert checkpoint["args"]["output_dir"] == str(output_dir)
            assert checkpoint["args"]["task"] == "phase2"

    def test_two_fresh_starts_get_different_cache_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Same ``--output-dir`` twice: the second run must not replay the first."""
        output_dir = tmp_path / "output"
        monkeypatch.setattr("dill.dump", MagicMock())
        first = self._fresh_main(monkeypatch, output_dir).call_args.kwargs["runner"]
        second = self._fresh_main(monkeypatch, output_dir).call_args.kwargs["runner"]
        first_dir, second_dir = first.keywords["cache_dir"], second.keywords["cache_dir"]
        assert first_dir != second_dir
        assert first_dir.parent == second_dir.parent == output_dir / "phase2" / "agent_cache"

    def test_resume_reuses_persisted_run_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A dill-loaded search object keeps hitting *its own* pre-crash cache."""
        checkpoints = self._capture_checkpoints(monkeypatch)
        output_dir = tmp_path / "output"
        mock_mcts, _ckpt_path = self._resume_main(
            monkeypatch,
            tmp_path,
            cli_task="phase3",
            ckpt_args={
                "task": "phase3",
                "output_dir": str(output_dir),
                "run_id": "20260101T000000-deadbeef",
            },
        )
        mock_mcts.set_runner.assert_called_once()
        runner = mock_mcts.set_runner.call_args.args[0]
        _assert_per_run_runner(
            runner, output_dir / "phase3" / "agent_cache" / "20260101T000000-deadbeef"
        )
        assert checkpoints
        assert all(c["args"]["run_id"] == "20260101T000000-deadbeef" for c in checkpoints)

    def test_resume_without_run_id_falls_back_to_checkpoint_stem(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checkpoints written before run ids existed still get a stable, private dir."""
        monkeypatch.setattr("dill.dump", MagicMock())
        output_dir = tmp_path / "output"
        mock_mcts, ckpt_path = self._resume_main(
            monkeypatch,
            tmp_path,
            cli_task="phase3",
            ckpt_args={"task": "phase3", "output_dir": str(output_dir)},
        )
        runner = mock_mcts.set_runner.call_args.args[0]
        _assert_per_run_runner(runner, output_dir / "phase3" / "agent_cache" / ckpt_path.stem)

    def test_resume_rejects_task_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``--task phase2 --resume <phase-3 checkpoint>`` is an error, not a phase-3 tree in phase-2 dirs."""
        monkeypatch.setattr("dill.dump", MagicMock())
        with pytest.raises(SystemExit, match=r"--task phase2 does not match .*'phase3'"):
            self._resume_main(
                monkeypatch,
                tmp_path,
                cli_task="phase2",
                ckpt_args={"task": "phase3", "output_dir": str(tmp_path / "output")},
            )

    def test_resume_warns_on_output_dir_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("dill.dump", MagicMock())
        with caplog.at_level("WARNING", logger="ctra.train"):
            self._resume_main(
                monkeypatch,
                tmp_path,
                cli_task="phase3",
                ckpt_args={"task": "phase3", "output_dir": "/elsewhere", "run_id": "x"},
            )
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("differs from the checkpoint's /elsewhere" in w for w in warnings), warnings

    def test_runner_partial_survives_dill_checkpoint(self, tmp_path: Path) -> None:
        """The partial is what the checkpoint pickles; dill must round-trip it intact."""
        import dill

        from ctra.agents.runner import RunCacheStats, run_agent_as_subprocess

        cache_dir = tmp_path / "phase2" / "agent_cache"
        runner = functools.partial(
            run_agent_as_subprocess, cache_dir=cache_dir, stats=RunCacheStats()
        )
        restored = dill.loads(dill.dumps(runner))
        _assert_per_run_runner(restored, cache_dir)


# ---------------------------------------------------------------------------
# Tests: output file creation
# ---------------------------------------------------------------------------


class TestOutputFiles:
    def test_results_json_structure(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mcts.py",
                "--task",
                "phase3",
                "--output-dir",
                str(output_dir),
            ],
        )

        best_node = _make_best_node(features=["drug_mechanism", "trial_size"])
        best_node.mean_reward = np.array([0.90, 0.80])
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(
            best_node, all_nodes=[best_node, MagicMock()], own_objectives=(0.95, 0.70)
        )  # 2 nodes total

        monkeypatch.setattr(
            "ctra.search.mcts.MCTSSearch",
            MagicMock(return_value=mock_mcts),
        )

        mock_settings = _make_mock_settings(num_rollouts=3, max_depth=5)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))
        monkeypatch.setattr("dill.dump", MagicMock())

        from train_mcts import main

        main()

        results_path = output_dir / "phase3" / "results.json"
        assert results_path.exists()
        results = json.loads(results_path.read_text())
        assert results["task"] == "phase3"
        assert results["rollouts"] == 3
        assert results["depth"] == 5
        assert results["best_features"] == ["drug_mechanism", "trial_size"]
        assert results["total_nodes"] == 2
        assert results["backprop"] == "mean"

    def test_results_json_reports_own_and_mean_objectives(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``best_objectives`` is the best node's own score, not its subtree mean.

        Before issue #15 this field held ``mean_reward`` — the average over the
        node's whole subtree, which is a different (usually lower) number than
        the score of the feature set in ``best_features``.  The mean is kept
        beside it as ``best_mean_objectives``.
        """
        output_dir = tmp_path / "output"
        monkeypatch.setattr(
            sys,
            "argv",
            ["train_mcts.py", "--task", "phase3", "--output-dir", str(output_dir)],
        )

        best_node = _make_best_node(features=["drug_mechanism", "trial_size"])
        best_node.mean_reward = np.array([0.62, 0.88])  # diluted by the subtree
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node, own_objectives=(0.91, 0.96))

        monkeypatch.setattr(
            "ctra.search.mcts.MCTSSearch",
            MagicMock(return_value=mock_mcts),
        )
        mock_settings = _make_mock_settings(num_rollouts=3, max_depth=5)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))
        monkeypatch.setattr("dill.dump", MagicMock())

        from train_mcts import main

        main()

        results = json.loads((output_dir / "phase3" / "results.json").read_text())
        mock_mcts.best_own_objectives.assert_called_once_with(best_node)
        mock_mcts.value_estimate.assert_called_once_with(best_node)
        assert results["best_objectives"] == [0.91, 0.96]
        assert results["best_mean_objectives"] == [0.62, 0.88]

    @pytest.mark.parametrize(
        ("backprop", "label"),
        [("mean", "Subtree mean"), ("max", "Subtree max"), ("max_hv", "Best subtree point")],
    )
    def test_best_mean_objectives_is_the_value_estimate_under_the_run_rule(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        backprop: str,
        label: str,
    ) -> None:
        """Under a non-``mean`` rule the field is not a mean (issue #16).

        ``best_mean_objectives`` keeps its key for continuity but holds
        ``value_estimate(best_node)`` — the vector UCB exploited under the
        run's rule — and ``results.json`` records the rule as ``backprop``;
        the printed label follows the rule.
        """
        output_dir = tmp_path / "output"
        monkeypatch.setattr(
            sys,
            "argv",
            ["train_mcts.py", "--task", "phase3", "--output-dir", str(output_dir)],
        )

        best_node = _make_best_node(features=["drug_mechanism", "trial_size"])
        best_node.mean_reward = np.array([0.62, 0.88])
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node, own_objectives=(0.91, 0.96), backprop=backprop)
        # Distinct from ``mean_reward`` so the script cannot pass by reading the node.
        mock_mcts.value_estimate.side_effect = None
        mock_mcts.value_estimate.return_value = np.array([0.70, 0.90])

        monkeypatch.setattr(
            "ctra.search.mcts.MCTSSearch",
            MagicMock(return_value=mock_mcts),
        )
        mock_settings = _make_mock_settings(num_rollouts=3, max_depth=5)
        mock_settings.mcts.backprop = backprop
        monkeypatch.setattr(
            "ctra.config.settings.get_settings", MagicMock(return_value=mock_settings)
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))
        monkeypatch.setattr("dill.dump", MagicMock())

        from train_mcts import main

        main()

        results = json.loads((output_dir / "phase3" / "results.json").read_text())
        mock_mcts.value_estimate.assert_called_once_with(best_node)
        assert results["backprop"] == backprop
        assert results["best_objectives"] == [0.91, 0.96]
        assert results["best_mean_objectives"] == [0.70, 0.90]
        out = capsys.readouterr().out
        assert f"{label}:" in out
        assert "[0.7 0.9]" in out


# ---------------------------------------------------------------------------
# Helper: suppress tqdm import
# ---------------------------------------------------------------------------


def _import_without_tqdm(monkeypatch):
    """Return a patched __import__ that makes tqdm unavailable."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def patched_import(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("tqdm not available in test")
        return original_import(name, *args, **kwargs)

    return patched_import


# ---------------------------------------------------------------------------
# Tests: cache hit-rate reporting (issue #17)
# ---------------------------------------------------------------------------


class TestCacheReporting:
    """Per-rollout log line, ``results.json["cache"]`` and per-run stats binding."""

    @staticmethod
    def _main_with_search(monkeypatch: pytest.MonkeyPatch, output_dir: Path, mock_search):
        """Run a fresh ``main()`` whose mocked ``search`` is ``mock_search``; return the class mock."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mcts.py",
                "--task",
                "phase2",
                "--rollouts",
                "2",
                "--output-dir",
                str(output_dir),
            ],
        )
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node)
        mock_mcts.search.side_effect = mock_search
        mock_mcts_cls = MagicMock(return_value=mock_mcts)
        monkeypatch.setattr("ctra.search.mcts.MCTSSearch", mock_mcts_cls)
        monkeypatch.setattr(
            "ctra.config.settings.get_settings",
            MagicMock(return_value=_make_mock_settings(num_rollouts=2)),
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))
        monkeypatch.setattr("dill.dump", MagicMock())

        from train_mcts import main

        main()
        return mock_mcts_cls, best_node

    def test_rollout_line_and_results_report_the_running_totals(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        output_dir = tmp_path / "output"
        holder: dict = {}

        def mock_search(initial_features, on_rollout=None, start_rollout=0):
            # The runner would update the bound stats in place; do it here.
            stats = holder["cls"].call_args.kwargs["runner"].keywords["stats"]
            stats.agent_misses += 1
            stats.feature_store.feature_lookups += 4
            stats.feature_store.record_hits(1, "planner")
            stats.feature_store.groups_skipped += 1
            stats.feature_store.groups_dispatched += 3
            stats.llm_calls_made += 20
            on_rollout(0, holder["node"], np.array([0.80, 0.70]))
            stats.agent_hits += 1
            stats.replayed_group_builds += 3
            on_rollout(1, holder["node"], np.array([0.82, 0.72]))
            return holder["node"]

        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        holder["node"] = best_node
        mock_mcts = _make_mock_mcts(best_node)
        mock_mcts.search.side_effect = mock_search
        holder["cls"] = MagicMock(return_value=mock_mcts)
        monkeypatch.setattr("ctra.search.mcts.MCTSSearch", holder["cls"])
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mcts.py",
                "--task",
                "phase2",
                "--rollouts",
                "2",
                "--output-dir",
                str(output_dir),
            ],
        )
        monkeypatch.setattr(
            "ctra.config.settings.get_settings",
            MagicMock(return_value=_make_mock_settings(num_rollouts=2)),
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))
        monkeypatch.setattr("dill.dump", MagicMock())

        from train_mcts import main

        with caplog.at_level("INFO", logger="ctra.train"):
            main()

        lines = [r.getMessage() for r in caplog.records if "cache —" in r.getMessage()]
        assert len(lines) == 2, lines
        assert lines[0] == (
            "Rollout 1/2 cache — agent hits/misses=0/1, feature-store hit rate=25.0% (1/4), "
            "groups skipped=1, LLM calls avoided~6, LLM calls made=20"
        )
        assert "Rollout 2/2 cache — agent hits/misses=1/1" in lines[1]

        results = json.loads((output_dir / "phase2" / "results.json").read_text())
        cache = results["cache"]
        assert cache["agent_hits"] == 1
        assert cache["agent_misses"] == 1
        assert cache["agent_hit_rate"] == pytest.approx(0.5)
        assert cache["replayed_group_builds"] == 3
        assert cache["llm_calls_made"] == 20
        assert cache["feature_store"] == {
            "feature_lookups": 4,
            "feature_hits": 1,
            "groups_dispatched": 3,
            "groups_skipped": 1,
            "hits_from_initializer_plans": 0,
            "hits_from_planner_plans": 1,
            "store_writes": 0,
            "hit_rate": 0.25,
            "llm_calls_avoided_estimate": 6,
        }

    def test_results_carry_a_zero_cache_block_without_evaluations(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        monkeypatch.setattr("dill.dump", MagicMock())
        TestPerRunAgentCache._fresh_main(monkeypatch, output_dir)
        results = json.loads((output_dir / "phase2" / "results.json").read_text())
        assert results["cache"]["agent_hits"] == 0
        assert results["cache"]["feature_store"]["hit_rate"] == 0.0

    def test_each_fresh_run_binds_its_own_stats(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from ctra.agents.runner import RunCacheStats

        output_dir = tmp_path / "output"
        monkeypatch.setattr("dill.dump", MagicMock())
        first = TestPerRunAgentCache._fresh_main(monkeypatch, output_dir)
        second = TestPerRunAgentCache._fresh_main(monkeypatch, output_dir)
        stats_a = first.call_args.kwargs["runner"].keywords["stats"]
        stats_b = second.call_args.kwargs["runner"].keywords["stats"]
        assert isinstance(stats_a, RunCacheStats)
        assert stats_a is not stats_b
        assert stats_a == RunCacheStats()

    def test_resume_rebinds_a_fresh_stats_object(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from ctra.agents.runner import RunCacheStats

        monkeypatch.setattr("dill.dump", MagicMock())
        mock_mcts, _ = TestPerRunAgentCache._resume_main(
            monkeypatch,
            tmp_path,
            cli_task="phase3",
            ckpt_args={"task": "phase3", "output_dir": str(tmp_path / "output"), "run_id": "r"},
        )
        runner = mock_mcts.set_runner.call_args.args[0]
        assert isinstance(runner.keywords["stats"], RunCacheStats)
        assert runner.keywords["stats"] == RunCacheStats()


# ---------------------------------------------------------------------------
# Tests: --mlflow (issue #17)
# ---------------------------------------------------------------------------


class TestMlflowFlag:
    """The tracker is built and fed only when ``--mlflow`` is given."""

    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch, output_dir: Path, *extra: str) -> MagicMock:
        monkeypatch.setattr(
            sys,
            "argv",
            ["train_mcts.py", "--task", "phase2", "--output-dir", str(output_dir), *extra],
        )
        best_node = _make_best_node()
        best_node.eval_output = _make_mock_output()
        mock_mcts = _make_mock_mcts(best_node)

        def mock_search(initial_features, on_rollout=None, start_rollout=0):
            on_rollout(0, best_node, np.array([0.80, 0.70]))
            on_rollout(1, best_node, np.array([0.82, 0.72]))
            return best_node

        mock_mcts.search.side_effect = mock_search
        monkeypatch.setattr("ctra.search.mcts.MCTSSearch", MagicMock(return_value=mock_mcts))
        monkeypatch.setattr(
            "ctra.config.settings.get_settings",
            MagicMock(return_value=_make_mock_settings(num_rollouts=2)),
        )
        monkeypatch.setattr("ctra.agents.feature_utils.dump_as_json", MagicMock(return_value="{}"))
        monkeypatch.setattr("dill.dump", MagicMock())
        tracker_cls = MagicMock()
        monkeypatch.setattr("ctra.mlops.experiment_tracker.ExperimentTracker", tracker_cls)

        from train_mcts import main

        main()
        return tracker_cls

    def test_default_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["train_mcts.py", "--task", "phase2"])
        from train_mcts import parse_args

        assert parse_args().mlflow is False

    def test_flag_off_constructs_no_tracker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        tracker_cls = self._run(monkeypatch, tmp_path / "output")
        tracker_cls.assert_not_called()

    def test_flag_on_logs_per_rollout_and_final_totals(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from ctra.agents.runner import RunCacheStats

        tracker_cls = self._run(monkeypatch, tmp_path / "output", "--mlflow")
        tracker_cls.assert_called_once_with()
        tracker = tracker_cls.return_value
        tracker.start_mcts_run.assert_called_once()
        assert tracker.start_mcts_run.call_args.args[1] == "phase2"
        steps = [c.kwargs.get("step") for c in tracker.log_cache_stats.call_args_list]
        assert steps == [0, 1, None]
        assert all(
            isinstance(c.args[0], RunCacheStats) for c in tracker.log_cache_stats.call_args_list
        )
        tracker.end_run.assert_called_once()
