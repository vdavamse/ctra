"""Edge-case tests for MCTSSearch callback and start_rollout behavior.

Extends test_mcts_callback.py with additional edge cases:
- on_rollout receives the correct (simulated) node, not the root
- start_rollout == num_rollouts produces zero rollouts
- Callback exceptions propagate (current behavior verification)
"""

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


@pytest.fixture()
def mock_settings_zero_rollouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config with 0 rollouts."""
    settings = type(
        "Settings",
        (),
        {
            "mcts": MCTSConfig(
                num_rollouts=0,
                max_depth=2,
                objectives=["accuracy", "parsimony"],
                reference_point=[0.0, 0.0],
                deep_simulation=False,
                adaptive_branching=False,
            ),
        },
    )()
    monkeypatch.setattr("ctra.search.mcts.get_settings", lambda: settings)


@pytest.fixture()
def mock_settings_single_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config with exactly 1 rollout."""
    settings = type(
        "Settings",
        (),
        {
            "mcts": MCTSConfig(
                num_rollouts=1,
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
    """Simple expand_fn that adds one feature per call."""

    def expand_fn(node, max_children=2):
        if len(node.features) >= 3:
            return []
        new_feat = f"feat_{len(node.features)}"
        return [
            ([*node.features, new_feat], "add", new_feat),
        ]

    return expand_fn


# ---------------------------------------------------------------------------
# Tests: on_rollout receives correct node
# ---------------------------------------------------------------------------


class TestCallbackReceivesCorrectNode:
    def test_callback_node_is_not_root(self, mock_settings) -> None:
        """The node passed to on_rollout should be the simulated node,
        which may differ from root after expansion."""
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        received_nodes = []

        def on_rollout(idx, node, obj):
            received_nodes.append(node)

        mcts.search(initial_features=["base"], on_rollout=on_rollout)

        # At least one node should be a child (not root) after expansion
        root = mcts.root
        non_root_nodes = [n for n in received_nodes if n is not root]
        assert len(non_root_nodes) > 0, (
            "At least one callback node should be a non-root node (indicating expansion happened)"
        )

    def test_callback_node_has_features(self, mock_settings) -> None:
        """Every node passed to on_rollout should have a features list."""
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        received_features = []

        def on_rollout(idx, node, obj):
            received_features.append(list(node.features))

        mcts.search(initial_features=["base"], on_rollout=on_rollout)

        for features in received_features:
            assert isinstance(features, list)
            assert len(features) >= 1  # at least the base feature

    def test_expanded_node_has_more_features_than_root(self, mock_settings) -> None:
        """After expansion, child nodes should have additional features."""
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        max_features_seen = [0]

        def on_rollout(idx, node, obj):
            max_features_seen[0] = max(max_features_seen[0], len(node.features))

        mcts.search(initial_features=["base"], on_rollout=on_rollout)

        # At least one node should have more than the initial 1 feature
        assert max_features_seen[0] > 1


# ---------------------------------------------------------------------------
# Tests: start_rollout == num_rollouts (zero rollouts)
# ---------------------------------------------------------------------------


class TestStartRolloutEqualsNumRollouts:
    def test_no_rollouts_no_crash(self, mock_settings) -> None:
        """When start_rollout == num_rollouts, no rollouts execute and
        the search should return the root node without crashing."""
        evaluate_fn, call_count = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        # First do a normal search so root is established
        mcts.search(initial_features=["base"])
        _calls_after_full = call_count[0]

        # Now try with start_rollout == num_rollouts (3 == 3)
        call_count[0] = 0
        callback_calls = []

        def on_rollout(idx, node, obj):
            callback_calls.append(idx)

        best = mcts.search(
            initial_features=["base"],
            start_rollout=3,  # equals num_rollouts
            on_rollout=on_rollout,
        )

        # No rollouts should have executed
        assert len(callback_calls) == 0
        # No evaluations should have been called
        assert call_count[0] == 0
        # Should still return a valid node
        assert best is not None

    def test_start_beyond_num_rollouts(self, mock_settings) -> None:
        """When start_rollout > num_rollouts, no rollouts execute."""
        evaluate_fn, call_count = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        # Establish root
        mcts.search(initial_features=["base"])
        call_count[0] = 0

        best = mcts.search(
            initial_features=["base"],
            start_rollout=100,  # way beyond num_rollouts=3
        )

        assert call_count[0] == 0
        assert best is not None


# ---------------------------------------------------------------------------
# Tests: zero-rollout config
# ---------------------------------------------------------------------------


class TestZeroRolloutConfig:
    def test_zero_rollouts_fresh_start(self, mock_settings_zero_rollouts) -> None:
        """With num_rollouts=0 and start_rollout=0, only root eval happens."""
        evaluate_fn, call_count = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        callback_calls = []

        def on_rollout(idx, node, obj):
            callback_calls.append(idx)

        best = mcts.search(
            initial_features=["base"],
            on_rollout=on_rollout,
        )

        # Root evaluation happens, but no rollouts
        assert call_count[0] == 1  # only root eval
        assert len(callback_calls) == 0
        assert best is not None


# ---------------------------------------------------------------------------
# Tests: callback exception behavior
# ---------------------------------------------------------------------------


class TestCallbackExceptions:
    def test_callback_exception_propagates(self, mock_settings_single_rollout) -> None:
        """Exceptions in the on_rollout callback should propagate to the
        caller (current behavior: no exception suppression)."""
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        def exploding_callback(idx, node, obj):
            raise ValueError("callback boom")

        with pytest.raises(ValueError, match="callback boom"):
            mcts.search(
                initial_features=["base"],
                on_rollout=exploding_callback,
            )

    def test_callback_exception_after_backpropagation(self, mock_settings_single_rollout) -> None:
        """The callback is invoked AFTER backpropagation, so node stats
        should be updated even if the callback raises."""
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        def exploding_callback(idx, node, obj):
            # Verify backpropagation already happened by checking root visit count
            root = mcts.root
            # Root should have visit_count > 1 at this point
            # (1 from root eval + 1 from rollout backprop)
            assert root.visit_count >= 2, (
                f"Expected root visit_count >= 2 after backprop, got {root.visit_count}"
            )
            raise ValueError("callback boom")

        with pytest.raises(ValueError, match="callback boom"):
            mcts.search(
                initial_features=["base"],
                on_rollout=exploding_callback,
            )


# ---------------------------------------------------------------------------
# Tests: objective vector in callback
# ---------------------------------------------------------------------------


class TestCallbackObjectiveVector:
    def test_objective_vector_shape(self, mock_settings) -> None:
        """Objective vector passed to callback should have correct shape."""
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        received = []

        def on_rollout(idx, node, obj):
            received.append(obj.copy())

        mcts.search(initial_features=["base"], on_rollout=on_rollout)

        for obj in received:
            assert isinstance(obj, np.ndarray)
            assert obj.shape == (2,)  # 2 objectives configured

    def test_objective_values_are_finite(self, mock_settings) -> None:
        """All objective values should be finite numbers."""
        evaluate_fn, _ = _make_evaluate_fn()
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(runner=runner, task="test", expand_fn=expand_fn)

        received = []

        def on_rollout(idx, node, obj):
            received.append(obj.copy())

        mcts.search(initial_features=["base"], on_rollout=on_rollout)

        for obj in received:
            assert np.all(np.isfinite(obj))
