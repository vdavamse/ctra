"""Tests for MCTSSearch.search() on_rollout callback and start_rollout."""

from __future__ import annotations

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSSearch
from ctra.search.objectives import ObjectiveResult
from tests.test_search.conftest import make_stub_runner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace get_settings with a minimal config."""
    settings = type(
        "Settings",
        (),
        {
            "mcts": MCTSConfig(
                num_rollouts=3,
                max_depth=2,
                objectives=["accuracy", "parsimony"],
                reference_point=[0.0, 0.0],
                deep_simulation=False,
                adaptive_branching=False,
                min_branch_factor=2,
                max_branch_factor=2,
            ),
        },
    )()
    monkeypatch.setattr("ctra.search.mcts.get_settings", lambda: settings)


def _make_evaluate_fn():
    """Simple evaluate_fn that returns incrementing objectives."""
    call_count = [0]

    def evaluate_fn(features, fidelity=1.0):
        call_count[0] += 1
        return ObjectiveResult(
            values=np.array([0.5 + call_count[0] * 0.05, 0.8]),
            names=["accuracy", "parsimony"],
            details={},
        )

    return evaluate_fn, call_count


def _make_expand_fn():
    """Simple expand_fn that adds one feature."""

    def expand_fn(node, max_children=2):
        if len(node.features) >= 3:
            return []
        new_feat = f"feat_{len(node.features)}"
        return [
            ([*node.features, new_feat], "add", new_feat),
        ]

    return expand_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOnRolloutCallback:
    def test_callback_called_per_rollout(self, mock_settings) -> None:
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        callback_calls = []

        def on_rollout(idx, node, obj):
            callback_calls.append((idx, len(node.features), obj.copy()))

        mcts.search(initial_features=["base"], on_rollout=on_rollout)

        # 3 rollouts configured
        assert len(callback_calls) == 3
        # Rollout indices are 0, 1, 2
        assert [c[0] for c in callback_calls] == [0, 1, 2]

    def test_callback_receives_objective_vector(self, mock_settings) -> None:
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        received = []

        def on_rollout(idx, node, obj):
            received.append(obj)

        mcts.search(initial_features=["base"], on_rollout=on_rollout)

        for obj in received:
            assert isinstance(obj, np.ndarray)
            assert obj.shape == (2,)

    def test_no_callback_is_fine(self, mock_settings) -> None:
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        # Should not raise
        best = mcts.search(initial_features=["base"])
        assert best is not None


class TestStartRollout:
    def test_start_rollout_skips_root_eval(self, mock_settings) -> None:
        evaluate_fn, call_count = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        # First do a normal search to establish root
        mcts.search(initial_features=["base"])
        calls_after_full = call_count[0]

        # Now resume from rollout 2 (only 1 rollout left: index 2)
        call_count[0] = 0
        mcts.search(
            initial_features=["base"],
            start_rollout=2,
        )

        # Should have done fewer evaluations than a full search
        assert call_count[0] < calls_after_full

    def test_start_rollout_zero_is_fresh(self, mock_settings) -> None:
        evaluate_fn, _call_count = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        best = mcts.search(initial_features=["base"], start_rollout=0)
        assert best is not None
        assert mcts.root is not None

    def test_start_rollout_without_root_raises(self, mock_settings) -> None:
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        with pytest.raises(AssertionError, match="Cannot resume"):
            mcts.search(initial_features=["base"], start_rollout=1)
