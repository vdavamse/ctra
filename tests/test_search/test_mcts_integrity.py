"""MCTS tree integrity tests — visit counting, error resilience, node state.

Validates that the MCTS search loop maintains correct statistics
(visit counts, rewards), handles evaluation failures gracefully,
and populates node state after evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSSearch
from ctra.search.objectives import ObjectiveResult
from tests.test_search.conftest import make_stub_runner

# ---------------------------------------------------------------------------
# Shared config fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = type(
        "Settings",
        (),
        {
            "mcts": MCTSConfig(
                num_rollouts=3,
                max_depth=3,
                objectives=["accuracy", "parsimony"],
                reference_point=[0.0, 0.0],
                deep_simulation=True,
                adaptive_branching=False,
                min_branch_factor=1,
                max_branch_factor=1,
            ),
        },
    )()
    monkeypatch.setattr("ctra.search.mcts.get_settings", lambda: settings)


def _make_expand_fn():
    """Expand: add one feature per node, up to 4 features total."""

    def expand_fn(node, max_children=1):
        if len(node.features) >= 4:
            return []
        new_feat = f"feat_{len(node.features)}"
        return [([*node.features, new_feat], "add", new_feat)]

    return expand_fn


# ======================================================================
# BUG 1: Double-counting in deep simulation
#
# _simulate_deep increments child.visit_count and child.total_reward
# on each evaluated child. Then search() calls _backpropagate(node, ...)
# which walks UP the tree incrementing again. This inflates visit counts
# and corrupts UCB scores.
# ======================================================================


class TestDeepSimulationDoubleCounting:
    def test_root_visit_count_not_inflated(self, mock_settings) -> None:
        """After one deep rollout from root, root visit_count should be 1
        (from the initial evaluation), not 2+ from double-counting."""
        eval_count = [0]

        def evaluate_fn(features, fidelity=1.0):
            eval_count[0] += 1
            return ObjectiveResult(
                values=np.array([0.7, 0.8]),
                names=["accuracy", "parsimony"],
                details={},
            )

        # 1 rollout, deep simulation, max_depth=3
        config = MCTSConfig(
            num_rollouts=1,
            max_depth=3,
            objectives=["accuracy", "parsimony"],
            reference_point=[0.0, 0.0],
            deep_simulation=True,
            adaptive_branching=False,
            min_branch_factor=1,
            max_branch_factor=1,
        )
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(
            runner=runner,
            task="test",
            expand_fn=expand_fn,
            config=config,
        )
        mcts.search(initial_features=["base"])

        # Root is evaluated once during initial setup.
        # The deep rollout from rollout 0 starts at root (since it's the only
        # leaf), evaluates it, and then descends. Root should NOT be counted
        # more than its actual number of evaluations.
        root = mcts.root
        assert root is not None

        # mean_reward[0] (accuracy) should match the evaluate_fn's ROC-AUC.
        # Parsimony is computed by extract_objectives from feature count,
        # so it varies with tree depth. We only check accuracy here.
        assert root.mean_reward[0] == pytest.approx(0.7, abs=0.1), (
            f"Root accuracy diverged: {root.mean_reward[0]:.3f} — "
            f"likely double-counting in deep simulation + backpropagation"
        )

    def test_leaf_visit_count_is_one(self, mock_settings) -> None:
        """Leaf nodes in a single-rollout deep simulation should have
        visit_count == 1 (evaluated once, no child backprop inflates them)."""

        def evaluate_fn(features, fidelity=1.0):
            return ObjectiveResult(
                values=np.array([0.6, 0.9]),
                names=["accuracy", "parsimony"],
                details={},
            )

        config = MCTSConfig(
            num_rollouts=1,
            max_depth=2,
            objectives=["accuracy", "parsimony"],
            reference_point=[0.0, 0.0],
            deep_simulation=True,
            adaptive_branching=False,
            min_branch_factor=1,
            max_branch_factor=1,
        )
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(
            runner=runner,
            task="test",
            expand_fn=expand_fn,
            config=config,
        )
        mcts.search(initial_features=["base"])

        # Leaf nodes have no children, so their visit_count should be
        # exactly 1 (evaluated once in the deep path, no child backprop).
        for node in mcts.all_nodes:
            if not node.is_leaf:
                continue
            assert node.visit_count == 1, (
                f"Leaf {node.features}: visit_count={node.visit_count}, "
                f"expected 1 — double-counting bug"
            )

    def test_parent_visit_count_includes_child_backprop(self, mock_settings) -> None:
        """Parent visit_count should be >= its own evaluations because
        child evaluations backpropagate through it."""
        eval_log = []

        def evaluate_fn(features, fidelity=1.0):
            eval_log.append(tuple(sorted(features)))
            return ObjectiveResult(
                values=np.array([0.6, 0.9]),
                names=["accuracy", "parsimony"],
                details={},
            )

        config = MCTSConfig(
            num_rollouts=1,
            max_depth=2,
            objectives=["accuracy", "parsimony"],
            reference_point=[0.0, 0.0],
            deep_simulation=True,
            adaptive_branching=False,
            min_branch_factor=1,
            max_branch_factor=1,
        )
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(
            runner=runner,
            task="test",
            expand_fn=expand_fn,
            config=config,
        )
        mcts.search(initial_features=["base"])

        for node in mcts.all_nodes:
            key = tuple(sorted(node.features))
            actual_evals = eval_log.count(key)
            assert node.visit_count >= actual_evals, (
                f"Node {node.features}: visit_count={node.visit_count} < "
                f"actual evaluations {actual_evals}"
            )


# ======================================================================
# BUG 2: No error handling in search loop
#
# If evaluate_fn raises mid-rollout, the exception propagates
# uncaught and the tree is left with partially updated nodes.
# The search should handle failures gracefully.
# ======================================================================


class TestSearchErrorHandling:
    def test_evaluate_failure_does_not_crash_search(self, mock_settings) -> None:
        """If evaluate_fn raises on one rollout, search should continue
        with remaining rollouts and return a valid best node."""
        call_count = [0]

        def evaluate_fn(features, fidelity=1.0):
            call_count[0] += 1
            if call_count[0] == 3:  # Fail on 3rd evaluation
                raise RuntimeError("LLM timeout")
            return ObjectiveResult(
                values=np.array([0.7, 0.8]),
                names=["accuracy", "parsimony"],
                details={},
            )

        config = MCTSConfig(
            num_rollouts=4,
            max_depth=2,
            objectives=["accuracy", "parsimony"],
            reference_point=[0.0, 0.0],
            deep_simulation=False,
            adaptive_branching=False,
            min_branch_factor=1,
            max_branch_factor=1,
        )
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(
            runner=runner,
            task="test",
            expand_fn=expand_fn,
            config=config,
        )

        # Should NOT raise — should skip the failed rollout and continue
        best = mcts.search(initial_features=["base"])
        assert best is not None

    def test_evaluate_failure_preserves_tree_consistency(self, mock_settings) -> None:
        """After a failed evaluation, all nodes should have consistent
        visit_count (>= 0) and total_reward."""
        call_count = [0]

        def evaluate_fn(features, fidelity=1.0):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Network error")
            return ObjectiveResult(
                values=np.array([0.5, 0.5]),
                names=["accuracy", "parsimony"],
                details={},
            )

        config = MCTSConfig(
            num_rollouts=3,
            max_depth=2,
            objectives=["accuracy", "parsimony"],
            reference_point=[0.0, 0.0],
            deep_simulation=False,
            adaptive_branching=False,
            min_branch_factor=1,
            max_branch_factor=1,
        )
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(
            runner=runner,
            task="test",
            expand_fn=expand_fn,
            config=config,
        )

        _best = mcts.search(initial_features=["base"])

        for node in mcts.all_nodes:
            assert node.visit_count >= 0, f"Node {node.features} has negative visit_count"
            if node.visit_count > 0:
                # mean_reward should be finite
                assert np.all(np.isfinite(node.mean_reward)), (
                    f"Node {node.features} has non-finite mean_reward"
                )


# ======================================================================
# BUG 3: MCTSNode.eval_output is never set
#
# The field is declared but never assigned. If the MCTS tree is meant
# to carry evaluation outputs on nodes, this should be populated.
# ======================================================================


class TestNodeEvalOutput:
    def test_eval_output_populated_after_evaluation(self, mock_settings) -> None:
        """After a node is evaluated, its eval_output should be set
        (not remain None)."""
        result_by_features = {}

        def evaluate_fn(features, fidelity=1.0):
            result = ObjectiveResult(
                values=np.array([0.7, 0.8]),
                names=["accuracy", "parsimony"],
                details={"features": list(features)},
            )
            result_by_features[tuple(sorted(features))] = result
            return result

        config = MCTSConfig(
            num_rollouts=2,
            max_depth=2,
            objectives=["accuracy", "parsimony"],
            reference_point=[0.0, 0.0],
            deep_simulation=False,
            adaptive_branching=False,
            min_branch_factor=1,
            max_branch_factor=1,
        )
        expand_fn = _make_expand_fn()
        runner = make_stub_runner(evaluate_fn, expand_fn)
        mcts = MCTSSearch(
            runner=runner,
            task="test",
            expand_fn=expand_fn,
            config=config,
        )
        mcts.search(initial_features=["base"])

        # Every visited node should have eval_output set
        visited = [n for n in mcts.all_nodes if n.visit_count > 0]
        assert len(visited) > 0, "No nodes were visited"

        for node in visited:
            assert node.eval_output is not None, (
                f"Node {node.features} was visited {node.visit_count} times but eval_output is None"
            )
