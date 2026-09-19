"""Tests for scripts/run_agent.py — single Agent iteration entry point.

Pure unit tests with all external dependencies mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_benchmark_df(id_col: str = "nctid", n: int = 10):
    """Create a mock polars-like DataFrame with .to_pandas()."""
    pdf = pd.DataFrame(
        {
            id_col: [f"NCT{i:08d}" for i in range(n)],
            "label": np.random.randint(0, 2, size=n),
        }
    )
    mock_df = MagicMock()
    mock_df.to_pandas.return_value = pdf
    return mock_df


def _make_mock_output(roc_auc: float = 0.85, has_evals: bool = True):
    """Create a mock AgentOutput."""
    if has_evals:
        mock_eval = MagicMock()
        mock_eval.model_eval_result.roc_auc = roc_auc
        mock_eval.suggestions = ["try adding more features"]

        output = MagicMock()
        output.eval_outputs = {"xgboost": mock_eval}
        output.feature_plans = {"feat_a": MagicMock()}
        output.operation = "add"
        output.get_best_eval_output.return_value = (mock_eval, MagicMock())
    else:
        output = MagicMock()
        output.eval_outputs = {}
        output.feature_plans = {}
        output.operation = None
    return output


def _setup_common_mocks(monkeypatch):
    """Set up mocks common to all main() tests. Returns mock objects dict."""
    mock_configure_lm = MagicMock()
    monkeypatch.setattr("ctra.agents.lm_config.configure_lm", mock_configure_lm)

    train_df = _make_benchmark_df("nctid", 10)
    val_df = _make_benchmark_df("nctid", 3)
    test_df = _make_benchmark_df("nctid", 3)
    mock_loader = MagicMock()
    mock_loader.load_benchmark_splits.return_value = (train_df, val_df, test_df)
    monkeypatch.setattr("ctra.data.ctg_loader.CTGLoader", MagicMock(return_value=mock_loader))

    mock_agent = MagicMock()
    mock_agent_cls = MagicMock(return_value=mock_agent)
    monkeypatch.setattr("ctra.agents.orchestrator.Agent", mock_agent_cls)

    # Keep dspy's global callbacks untouched across tests: hand ``main`` an
    # unregistered counter instead of letting it configure dspy (issue #17).
    import run_agent

    monkeypatch.setattr(run_agent, "install_llm_call_counter", run_agent.LMCallCounter)

    return {
        "configure_lm": mock_configure_lm,
        "loader": mock_loader,
        "agent": mock_agent,
        "agent_cls": mock_agent_cls,
    }


# ---------------------------------------------------------------------------
# Tests: parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--task and --output are required."""
        monkeypatch.setattr(sys, "argv", ["run_agent.py"])
        from run_agent import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_task_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["run_agent.py", "--output", "out.pkl"])
        from run_agent import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_output_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["run_agent.py", "--task", "phase2"])
        from run_agent import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_valid_args_no_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_agent.py",
                "--task",
                "phase2",
                "--output",
                "result.pkl",
            ],
        )
        from run_agent import parse_args

        args = parse_args()
        assert args.task == "phase2"
        assert args.output == "result.pkl"
        assert args.input is None

    def test_valid_args_with_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_agent.py",
                "--task",
                "phase1",
                "--input",
                "prev.pkl",
                "--output",
                "result.pkl",
            ],
        )
        from run_agent import parse_args

        args = parse_args()
        assert args.task == "phase1"
        assert args.input == "prev.pkl"

    def test_invalid_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_agent.py",
                "--task",
                "phase99",
                "--output",
                "out.pkl",
            ],
        )
        from run_agent import parse_args

        with pytest.raises(SystemExit):
            parse_args()


# ---------------------------------------------------------------------------
# Tests: main() — iteration 0 (no --input)
# ---------------------------------------------------------------------------


class TestMainIteration0:
    def test_calls_forward_with_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "result.pkl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_agent.py",
                "--task",
                "phase2",
                "--output",
                str(output_path),
            ],
        )

        mocks = _setup_common_mocks(monkeypatch)
        mock_output = _make_mock_output()
        mocks["agent"].forward.return_value = mock_output

        mock_dill_dump = MagicMock()
        monkeypatch.setattr("dill.dump", mock_dill_dump)

        from run_agent import main

        main()

        # agent.forward should be called with previous_output=None
        mocks["agent"].forward.assert_called_once_with(previous_output=None)

    def test_writes_output_via_dill(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "result.pkl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_agent.py",
                "--task",
                "phase2",
                "--output",
                str(output_path),
            ],
        )

        mocks = _setup_common_mocks(monkeypatch)
        mock_output = _make_mock_output()
        mocks["agent"].forward.return_value = mock_output

        mock_dill_dump = MagicMock()
        monkeypatch.setattr("dill.dump", mock_dill_dump)

        from run_agent import main

        main()

        # dill.dump should be called with the output object
        mock_dill_dump.assert_called_once()
        dumped_obj = mock_dill_dump.call_args[0][0]
        assert dumped_obj is mock_output


# ---------------------------------------------------------------------------
# Tests: main() — iteration N (with --input)
# ---------------------------------------------------------------------------


class TestMainIterationN:
    def test_loads_previous_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "prev.pkl"
        input_path.touch()
        output_path = tmp_path / "result.pkl"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_agent.py",
                "--task",
                "phase2",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
        )

        mocks = _setup_common_mocks(monkeypatch)
        mock_prev_output = _make_mock_output(0.80)
        mock_curr_output = _make_mock_output(0.88)
        mocks["agent"].forward.return_value = mock_curr_output

        mock_dill_load = MagicMock(return_value=mock_prev_output)
        monkeypatch.setattr("dill.load", mock_dill_load)
        monkeypatch.setattr("dill.dump", MagicMock())

        from run_agent import main

        main()

        # dill.load should have been called to load prev.pkl
        mock_dill_load.assert_called_once()

        # agent.forward should be called with the loaded previous output
        mocks["agent"].forward.assert_called_once_with(previous_output=mock_prev_output)

    def test_forward_passes_loaded_previous(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "prev.pkl"
        input_path.touch()
        output_path = tmp_path / "result.pkl"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_agent.py",
                "--task",
                "phase3",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
        )

        mocks = _setup_common_mocks(monkeypatch)
        sentinel = object()  # unique previous output
        mock_curr_output = _make_mock_output()
        mocks["agent"].forward.return_value = mock_curr_output

        monkeypatch.setattr("dill.load", MagicMock(return_value=sentinel))
        monkeypatch.setattr("dill.dump", MagicMock())

        from run_agent import main

        main()

        mocks["agent"].forward.assert_called_once_with(previous_output=sentinel)


# ---------------------------------------------------------------------------
# Tests: summary output
# ---------------------------------------------------------------------------


class TestSummaryPrinting:
    def test_prints_summary_with_eval_outputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys,
    ) -> None:
        output_path = tmp_path / "result.pkl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_agent.py",
                "--task",
                "phase2",
                "--output",
                str(output_path),
            ],
        )

        mocks = _setup_common_mocks(monkeypatch)
        mock_output = _make_mock_output(roc_auc=0.9123)
        mocks["agent"].forward.return_value = mock_output

        monkeypatch.setattr("dill.dump", MagicMock())

        from run_agent import main

        main()

        captured = capsys.readouterr()
        # Summary is printed to stderr
        assert "ROC-AUC" in captured.err

    def test_prints_no_eval_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys,
    ) -> None:
        output_path = tmp_path / "result.pkl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_agent.py",
                "--task",
                "phase2",
                "--output",
                str(output_path),
            ],
        )

        mocks = _setup_common_mocks(monkeypatch)
        mock_output = _make_mock_output(has_evals=False)
        mocks["agent"].forward.return_value = mock_output

        monkeypatch.setattr("dill.dump", MagicMock())

        from run_agent import main

        main()

        captured = capsys.readouterr()
        assert "no eval outputs" in captured.err
