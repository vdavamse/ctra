"""``scripts/run_agent.py`` records the iteration's LLM call count (issue #17).

The count comes from ``LMCallCounter``, a dspy callback whose ``on_lm_start``
fires once per ``LM.__call__``: the calibration figure for
``LLM_CALLS_PER_GROUP_BUILD``.  It is unbounded, unlike
``len(dspy.clients.base_lm.GLOBAL_HISTORY)``, which dspy caps at
``MAX_HISTORY_SIZE`` entries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from tests.test_agents.test_orchestrator import _make_output
from tests.test_scripts.test_run_agent import _setup_common_mocks

if TYPE_CHECKING:
    from collections.abc import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_agent


def _run_main(monkeypatch: pytest.MonkeyPatch, tmp_path, output, on_forward=None, counter=None):
    """Run ``main`` with the agent mocked; ``on_forward`` runs inside ``Agent.forward``.

    ``counter`` replaces the installed ``LMCallCounter`` (``_setup_common_mocks``
    already keeps dspy's global callbacks untouched).
    """
    monkeypatch.setattr(
        sys, "argv", ["run_agent.py", "--task", "phase2", "--output", str(tmp_path / "out.pkl")]
    )
    mocks = _setup_common_mocks(monkeypatch)
    if counter is not None:
        monkeypatch.setattr(run_agent, "install_llm_call_counter", lambda: counter)

    def forward(previous_output=None):
        if on_forward is not None:
            on_forward()
        return output

    mocks["agent"].forward.side_effect = forward
    monkeypatch.setattr("dill.dump", MagicMock())
    run_agent.main()


def _fire(counter: run_agent.LMCallCounter, n: int, instance=None) -> None:
    for i in range(n):
        counter.on_lm_start(call_id=str(i), instance=instance, inputs={})


@pytest.fixture
def restore_dspy_callbacks() -> Iterator[None]:
    """Snapshot dspy's global callbacks and put them back after the test."""
    import dspy

    before = list(dspy.settings.get("callbacks", []))
    yield
    dspy.configure(callbacks=before)


class TestLlmCallsMade:
    def test_records_every_lm_call_the_iteration_made(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        counter = run_agent.LMCallCounter()
        output = _make_output()
        _run_main(
            monkeypatch, tmp_path, output, on_forward=lambda: _fire(counter, 4), counter=counter
        )
        assert output.cache_stats.llm_calls_made == 4

    def test_the_count_is_not_capped_by_dspy_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The figure keeps counting past ``MAX_HISTORY_SIZE``; the history does not.

        Each simulated call does what dspy does on ``LM.__call__``: fires the
        callback and appends to ``GLOBAL_HISTORY`` (which pops its oldest entry
        once full).  With the cap lowered to 5 and 7 calls made, a history-length
        implementation reports 5; the counter reports 7.
        """
        import dspy.clients.base_lm as base_lm

        monkeypatch.setattr(base_lm, "MAX_HISTORY_SIZE", 5)
        monkeypatch.setattr(base_lm, "GLOBAL_HISTORY", [])
        lm = base_lm.BaseLM(model="stub")
        counter = run_agent.LMCallCounter()

        def seven_calls() -> None:
            for i in range(7):
                counter.on_lm_start(call_id=str(i), instance=lm, inputs={})
                lm.update_history({"call": i})

        output = _make_output()
        _run_main(monkeypatch, tmp_path, output, on_forward=seven_calls, counter=counter)
        assert len(base_lm.GLOBAL_HISTORY) == 5
        assert output.cache_stats.llm_calls_made == 7

    def test_counter_exceeds_the_real_history_cap(self) -> None:
        import dspy.clients.base_lm as base_lm

        counter = run_agent.LMCallCounter()
        _fire(counter, base_lm.MAX_HISTORY_SIZE + 1)
        assert counter.calls == base_lm.MAX_HISTORY_SIZE + 1 == 10_001

    @pytest.mark.usefixtures("restore_dspy_callbacks")
    def test_installed_counter_sees_real_lm_calls(self) -> None:
        """Registered on dspy's global callbacks, the counter sees ``LM.__call__``."""
        import dspy
        from dspy.utils import DummyLM

        before = list(dspy.settings.get("callbacks", []))
        counter = run_agent.install_llm_call_counter()
        assert counter is not None
        assert dspy.settings.get("callbacks") == [*before, counter]

        lm = DummyLM([{"answer": "a"}, {"answer": "b"}, {"answer": "c"}])
        for _ in range(3):
            lm(messages=[{"role": "user", "content": "q"}])
        assert counter.calls == 3

    def test_reports_zero_when_the_counter_cannot_be_registered(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import dspy

        def refuse(**kwargs):
            raise RuntimeError(
                "dspy.settings can only be changed by the thread that initially configured it."
            )

        monkeypatch.setattr(dspy, "configure", refuse)
        assert run_agent.install_llm_call_counter() is None

        output = _make_output()
        monkeypatch.setattr(
            sys, "argv", ["run_agent.py", "--task", "phase2", "--output", str(tmp_path / "o.pkl")]
        )
        mocks = _setup_common_mocks(monkeypatch)
        monkeypatch.setattr(run_agent, "install_llm_call_counter", lambda: None)
        mocks["agent"].forward.return_value = output
        monkeypatch.setattr("dill.dump", MagicMock())
        run_agent.main()
        assert output.cache_stats.llm_calls_made == 0

    def test_reports_none_when_dspy_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def no_dspy(name, *args, **kwargs):
            if name == "dspy":
                raise ImportError("no dspy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_dspy)
        assert run_agent.install_llm_call_counter() is None

    def test_an_output_without_the_field_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        output = _make_output()
        del output.__dict__["cache_stats"]
        _run_main(monkeypatch, tmp_path, output, counter=run_agent.LMCallCounter())
        assert not hasattr(output, "cache_stats")
