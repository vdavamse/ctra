"""Tests for scripts/train_mcts.py — MCTS training entry point.

Pure unit tests with all external dependencies mocked (LLM, RAG,
CTGLoader, Agent, MCTSSearch, dill, file I/O).
"""

from __future__ import annotations

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
    settings.mcts.reference_point = [0.0, 0.0]
    return settings


def _make_best_node(features=None):
    """Create a mock MCTSNode returned by search()."""
    node = MagicMock()
    node.features = features or ["feat_a", "feat_b"]
    node.mean_reward = np.array([0.85, 0.75])
    return node


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
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = best_node
        mock_mcts.all_nodes = [best_node]

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
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = best_node
        mock_mcts.all_nodes = [best_node]

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
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = best_node
        mock_mcts.all_nodes = [best_node]

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

        mock_mcts = MagicMock()
        mock_mcts.search.side_effect = mock_search
        mock_mcts.all_nodes = [best_node]

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

        def tracking_dill_dump(obj, f):
            if isinstance(obj, dict) and "rollout" in obj:
                dill_dump_calls.append(obj["rollout"])

        monkeypatch.setattr("dill.dump", tracking_dill_dump)

        from train_mcts import main

        main()

        # Checkpoint should have been saved at rollout index 1 (since (1+1)%2==0)
        assert 1 in dill_dump_calls


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
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = best_node
        mock_mcts.all_nodes = [best_node, MagicMock()]  # 2 nodes total

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
