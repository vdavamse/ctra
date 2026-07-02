"""Tests for ctra.search.mcts — MCTSNode properties and MCTSSearch helpers.

Pure math tests: no mocks, no external services.  Uses monkeypatch to
isolate from the global settings singleton.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig, Settings, get_settings
from ctra.search.mcts import MCTSNode, MCTSSearch
from tests.test_search.conftest import make_stub_runner

# ======================================================================
# Settings fixture — isolate from environment / .env
# ======================================================================


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace get_settings with a deterministic test configuration."""
    get_settings.cache_clear()
    settings = Settings(
        mcts=MCTSConfig(
            max_features=50,
            objectives=["accuracy", "parsimony"],
            num_rollouts=20,
            reference_point=[0.0, 0.0],
        ),
    )
    monkeypatch.setattr("ctra.search.mcts.get_settings", lambda: settings)
    yield
    get_settings.cache_clear()


# ======================================================================
# MCTSNode.mean_reward
# ======================================================================


class TestMCTSNodeMeanReward:
    """Tests for the mean_reward property."""

    def test_normal_mean(self) -> None:
        node = MCTSNode(
            features=["a"],
            total_reward=np.array([2.0, 4.0]),
            visit_count=2,
        )
        expected = np.array([1.0, 2.0])
        np.testing.assert_allclose(node.mean_reward, expected, atol=1e-6)

    def test_zero_visits_returns_zeros(self) -> None:
        node = MCTSNode(
            features=["a"],
            total_reward=np.array([5.0, 5.0]),
            visit_count=0,
        )
        np.testing.assert_allclose(node.mean_reward, [0.0, 0.0], atol=1e-6)

    def test_single_visit(self) -> None:
        node = MCTSNode(
            features=["a"],
            total_reward=np.array([0.8, 0.6]),
            visit_count=1,
        )
        np.testing.assert_allclose(node.mean_reward, [0.8, 0.6], atol=1e-6)


# ======================================================================
# MCTSNode.is_leaf
# ======================================================================


class TestMCTSNodeIsLeaf:
    """Tests for the is_leaf property."""

    def test_no_children_is_leaf(self) -> None:
        node = MCTSNode(features=["a"])
        assert node.is_leaf is True

    def test_with_children_not_leaf(self) -> None:
        parent = MCTSNode(features=["a"])
        child = MCTSNode(features=["a", "b"], parent=parent)
        parent.children.append(child)
        assert parent.is_leaf is False

    def test_child_is_leaf(self) -> None:
        parent = MCTSNode(features=["a"])
        child = MCTSNode(features=["a", "b"], parent=parent)
        parent.children.append(child)
        assert child.is_leaf is True


# ======================================================================
# MCTSNode.ucb_scores
# ======================================================================


class TestMCTSNodeUCBScores:
    """Tests for the UCB1 score computation."""

    def test_unvisited_returns_inf(self) -> None:
        node = MCTSNode(features=["a"], visit_count=0)
        ucb = node.ucb_scores()
        assert np.all(np.isinf(ucb))

    def test_visited_returns_finite(self) -> None:
        parent = MCTSNode(features=[], visit_count=10)
        child = MCTSNode(
            features=["a"],
            total_reward=np.array([0.6, 0.4]),
            visit_count=5,
            parent=parent,
        )
        ucb = child.ucb_scores(exploration_constant=1.414)
        assert np.all(np.isfinite(ucb))
        # Each UCB value should be >= mean_reward (exploration term >= 0)
        assert np.all(ucb >= child.mean_reward - 1e-9)


# ======================================================================
# MCTSNode.value
# ======================================================================


class TestMCTSNodeValue:
    """Tests for the scalar hypervolume value."""

    def test_value_default_reference(self) -> None:
        node = MCTSNode(
            features=["a"],
            total_reward=np.array([0.8, 0.6]),
            visit_count=1,
        )
        val = node.value()
        expected = 0.8 * 0.6
        np.testing.assert_allclose(val, expected, atol=1e-6)

    def test_value_with_reference(self) -> None:
        node = MCTSNode(
            features=["a"],
            total_reward=np.array([0.8, 0.6]),
            visit_count=1,
        )
        ref = np.array([0.1, 0.1])
        val = node.value(reference=ref)
        expected = 0.7 * 0.5
        np.testing.assert_allclose(val, expected, atol=1e-6)

    def test_value_below_reference_is_zero(self) -> None:
        node = MCTSNode(
            features=["a"],
            total_reward=np.array([0.0, 0.0]),
            visit_count=1,
        )
        ref = np.array([0.5, 0.5])
        val = node.value(reference=ref)
        np.testing.assert_allclose(val, 0.0, atol=1e-6)


# ======================================================================
# Single-objective backward compatibility
# ======================================================================


class TestSingleObjective:
    """Verify that a single-objective config degenerates correctly."""

    def test_single_objective_search_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = Settings(
            mcts=MCTSConfig(
                max_features=50,
                objectives=["accuracy"],
                num_rollouts=2,
                reference_point=[0.0],
            ),
        )
        monkeypatch.setattr("ctra.search.mcts.get_settings", lambda: settings)

        def eval_fn(features, **kw):
            return np.array([0.85])

        def expand_fn(node, max_children):
            return []

        runner = make_stub_runner(eval_fn, expand_fn)
        search = MCTSSearch(
            runner=runner,
            task="test",
            expand_fn=expand_fn,
        )
        assert search._n_objectives == 1

    def test_single_objective_node_mean(self) -> None:
        node = MCTSNode(
            features=["a"],
            total_reward=np.array([3.0]),
            visit_count=3,
        )
        np.testing.assert_allclose(node.mean_reward, [1.0], atol=1e-6)
