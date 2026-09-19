"""``scripts/run_agent.py`` records the iteration's LLM call count (issue #17).

The count is ``len(dspy.clients.base_lm.GLOBAL_HISTORY)`` at the end of the
child process: the calibration figure for ``LLM_CALLS_PER_GROUP_BUILD``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from tests.test_agents.test_orchestrator import _make_output
from tests.test_scripts.test_run_agent import _setup_common_mocks

if TYPE_CHECKING:
    import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _run_main(monkeypatch: pytest.MonkeyPatch, tmp_path, output):
    monkeypatch.setattr(
        sys, "argv", ["run_agent.py", "--task", "phase2", "--output", str(tmp_path / "out.pkl")]
    )
    mocks = _setup_common_mocks(monkeypatch)
    mocks["agent"].forward.return_value = output
    monkeypatch.setattr("dill.dump", MagicMock())
    from run_agent import main

    main()


class TestLlmCallsMade:
    def test_records_the_global_history_length(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import dspy.clients.base_lm as base_lm

        monkeypatch.setattr(base_lm, "GLOBAL_HISTORY", [object()] * 4)
        output = _make_output()
        _run_main(monkeypatch, tmp_path, output)
        assert output.cache_stats.llm_calls_made == 4

    def test_reports_zero_when_dspy_has_no_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import builtins

        import run_agent

        real_import = builtins.__import__

        def no_history(name, *args, **kwargs):
            if name == "dspy.clients.base_lm":
                raise ImportError("no GLOBAL_HISTORY")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_history)
        assert run_agent._llm_calls_made() == 0

    def test_an_output_without_the_field_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        output = _make_output()
        del output.__dict__["cache_stats"]
        _run_main(monkeypatch, tmp_path, output)
        assert not hasattr(output, "cache_stats")
