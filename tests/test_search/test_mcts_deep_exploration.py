"""Tests verifying MCTS explores deep feature combinations.

AutoCT's key insight: the best feature SET is found by iteratively
refining combinations — not by evaluating individual features in
isolation. A feature that is useless alone may be critical in
combination with others (feature interaction / epistasis).

The MCTS tree should go deep: each level represents a modification
(add/remove/refine) to the parent's feature set. Depth 10 means
10 sequential refinements. The search should discover that deep
paths produce better feature combinations than shallow ones.

These tests verify:
    1. The tree actually reaches meaningful depth (not stuck shallow)
    2. Deep nodes with good feature combinations are preferred
    3. A feature that is bad alone but good in combination is discovered
    4. The search explores depth vs breadth appropriately
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig, Settings, get_settings
from ctra.search.mcts import MCTSNode, MCTSSearch
from tests.test_search.conftest import make_stub_runner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Settings tuned for deep exploration testing."""
    get_settings.cache_clear()
    settings = Settings(
        mcts=MCTSConfig(
            num_rollouts=60,  # more rollouts to give depth a chance
            max_depth=15,
            objectives=["accuracy", "parsimony"],
            max_features=50,
            exploration_constant=1.414,
            adaptive_branching=True,
            min_branch_factor=2,
            max_branch_factor=3,  # keep branching narrow to encourage depth
            reference_point=[0.0, 0.0],
        ),
    )
    monkeypatch.setattr("ctra.search.mcts.get_settings", lambda: settings)
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_depth(node: MCTSNode) -> int:
    """Compute depth of a node (root = 0)."""
    d = 0
    current = node
    while current.parent is not None:
        d += 1
        current = current.parent
    return d


def _tree_max_depth(node: MCTSNode) -> int:
    """Compute maximum depth of the subtree rooted at node."""
    if not node.children:
        return 0
    return 1 + max(_tree_max_depth(c) for c in node.children)


def _depth_distribution(all_nodes: list[MCTSNode]) -> dict[int, int]:
    """Count how many nodes exist at each depth level."""
    dist: dict[int, int] = {}
    for n in all_nodes:
        d = _node_depth(n)
        dist[d] = dist.get(d, 0) + 1
    return dict(sorted(dist.items()))


# ---------------------------------------------------------------------------
# Tests: Deep tree exploration
# ---------------------------------------------------------------------------


class TestDeepExploration:
    """Verify that MCTS builds a deep tree, not just a wide shallow one."""

    def test_tree_reaches_depth_at_least_5(self):
        """With 40 rollouts and narrow branching, tree should reach depth >= 5.

        AutoCT uses depth 10 with 10 rollouts. Our Pareto MCTS with 40 rollouts
        and branching factor 2-3 should at minimum reach depth 5, meaning
        5 sequential feature refinements from the root.
        """
        rng = np.random.default_rng(42)

        def evaluate(features, fidelity=1.0):
            n = len(features)
            acc = 0.5 + 0.12 * math.log2(max(n, 1)) + rng.normal(0, 0.02)
            return np.array([np.clip(acc, 0, 1), max(0, 1 - n / 50)])

        def expand(node, max_children=3):
            candidates = []
            for _ in range(max_children):
                new = [*list(node.features), f"f{rng.integers(200)}"]
                candidates.append((new, "add", "add:random"))
            return candidates

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        _best = search.search(initial_features=["f0", "f1"])

        depth = _tree_max_depth(search.root)
        assert depth >= 5, (
            f"Tree max depth is {depth}, expected >= 5. "
            f"Depth distribution: {_depth_distribution(search.all_nodes)}"
        )

    def test_best_node_is_not_always_root(self):
        """The best node should be a refined version of the root, not the root itself.

        If MCTS is working correctly, iterative feature refinement should
        produce a node that outperforms the initial feature set.
        """
        rng = np.random.default_rng(42)

        def evaluate(features, fidelity=1.0):
            n = len(features)
            # Accuracy jumps sharply at n=3 to create a clear sweet spot
            # that dominates root even after backpropagation averaging
            acc = 0.9 + rng.normal(0, 0.005) if n >= 3 else 0.3 + 0.05 * n + rng.normal(0, 0.005)
            return np.array([np.clip(acc, 0, 1), max(0, 1 - n / 50)])

        def expand(node, max_children=3):
            return [
                ([*node.features, f"f{rng.integers(200)}"], "add", "add:x")
                for _ in range(max_children)
            ]

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        best = search.search(initial_features=["f0"])

        best_depth = _node_depth(best)
        assert best_depth > 0, (
            "Best node is the root — MCTS failed to find any improvement through feature refinement"
        )

    def test_deep_combination_discovered(self):
        """MCTS should discover that features A+B+C together score higher
        than any subset, even though individual features score poorly.

        This simulates feature epistasis: the interaction effect is
        non-linear (synergistic).
        """
        rng = np.random.default_rng(42)

        # Synergistic feature trio: each alone scores ~0.55, but all three
        # together score 0.85. The MCTS must go deep enough to discover this.
        synergy_features = {"alpha", "beta", "gamma"}

        def evaluate(features, fidelity=1.0):
            fset = set(features)
            synergy_count = len(fset & synergy_features)

            if synergy_count == 3:
                # All three synergistic features present — high accuracy
                acc = 0.85 + rng.normal(0, 0.01)
            elif synergy_count == 2:
                acc = 0.65 + rng.normal(0, 0.01)
            elif synergy_count == 1:
                acc = 0.55 + rng.normal(0, 0.01)
            else:
                acc = 0.50 + rng.normal(0, 0.01)

            n = len(features)
            return np.array([np.clip(acc, 0, 1), max(0, 1 - n / 50)])

        # Expander that can add the synergy features
        feature_pool = ["alpha", "beta", "gamma", "noise1", "noise2", "noise3"]

        def expand(node, max_children=3):
            available = [f for f in feature_pool if f not in node.features]
            candidates = []
            for _ in range(min(max_children, len(available))):
                feat = available[rng.integers(len(available))]
                available = [f for f in available if f != feat]
                candidates.append(([*node.features, feat], "add", f"add:{feat}"))
            return candidates

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        best = search.search(initial_features=["noise1"])

        best_features = set(best.features)
        synergy_found = len(best_features & synergy_features)

        # The search should discover at least 2 of the 3 synergy features
        assert synergy_found >= 2, (
            f"Expected MCTS to discover synergistic feature combination, "
            f"but best features {best.features} only contain "
            f"{synergy_found}/3 synergy features (alpha, beta, gamma). "
            f"Best mean accuracy: {best.mean_reward[0]:.3f}"
        )

    def test_remove_operations_prune_bad_features(self):
        """MCTS should discover that removing a bad feature improves accuracy.

        Start with a poisoned feature set: one feature actively hurts accuracy.
        The search should find a deeper node where the bad feature is removed.
        """
        rng = np.random.default_rng(42)
        poison_feature = "poison_feat"

        def evaluate(features, fidelity=1.0):
            n = len(features)
            has_poison = poison_feature in features
            base_acc = 0.7 + 0.02 * n + rng.normal(0, 0.01)
            if has_poison:
                base_acc -= 0.15  # poison reduces accuracy by 15%
            return np.array([np.clip(base_acc, 0, 1), max(0, 1 - n / 50)])

        def expand(node, max_children=3):
            candidates = []
            # Try removing features
            for i in range(min(max_children, len(node.features))):
                if i < len(node.features):
                    removed = node.features[i]
                    new = [f for f in node.features if f != removed]
                    if new:  # don't create empty feature set
                        candidates.append((new, "remove", f"remove:{removed}"))
            # Also try adding
            if len(candidates) < max_children:
                candidates.append(([*node.features, f"new_{rng.integers(100)}"], "add", "add:x"))
            return candidates

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        best = search.search(initial_features=["good_a", "good_b", poison_feature, "good_c"])

        assert poison_feature not in best.features, (
            f"MCTS should have removed the poison feature '{poison_feature}', "
            f"but best features are: {best.features}"
        )


class TestDepthVsBreadthTradeoff:
    """Verify the tree explores both deep and wide paths."""

    def test_multiple_depths_explored(self):
        """The tree should have nodes at multiple depth levels, not just depth 0-1."""
        rng = np.random.default_rng(42)

        def evaluate(features, fidelity=1.0):
            n = len(features)
            return np.array(
                [
                    0.5 + 0.1 * math.log2(max(n, 1)) + rng.normal(0, 0.02),
                    max(0, 1 - n / 50),
                    0.9 + rng.normal(0, 0.01),
                ]
            )

        def expand(node, max_children=3):
            return [
                ([*node.features, f"f{rng.integers(200)}"], "add", "add:x")
                for _ in range(max_children)
            ]

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        search.search(initial_features=["f0", "f1"])

        dist = _depth_distribution(search.all_nodes)

        # Should have nodes at depth 0, 1, 2, and at least one more
        assert len(dist) >= 4, f"Expected nodes at >= 4 depth levels, got {len(dist)}: {dist}"

    def test_visited_nodes_at_multiple_depths(self):
        """Evaluated (visited) nodes should span multiple depths,
        meaning the search actually evaluates deep feature combinations."""
        rng = np.random.default_rng(42)

        def evaluate(features, fidelity=1.0):
            n = len(features)
            return np.array([0.6 + rng.normal(0, 0.03), max(0, 1 - n / 50)])

        def expand(node, max_children=2):
            return [
                ([*node.features, f"f{rng.integers(200)}"], "add", "add:x")
                for _ in range(max_children)
            ]

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        search.search(initial_features=["f0"])

        visited_depths = set()
        for n in search.all_nodes:
            if n.visit_count > 0:
                visited_depths.add(_node_depth(n))

        assert max(visited_depths) >= 3, (
            f"Expected evaluated nodes at depth >= 3, "
            f"but deepest evaluated node is at depth {max(visited_depths)}. "
            f"Visited depths: {sorted(visited_depths)}"
        )


class TestFeatureRefinementChain:
    """Verify that the best node represents a chain of meaningful refinements."""

    def test_best_node_path_shows_refinement(self):
        """Trace the path from root to best — each step should be a feature modification."""
        rng = np.random.default_rng(42)

        def evaluate(features, fidelity=1.0):
            n = len(features)
            acc = 0.5 + 0.04 * min(n, 15) + rng.normal(0, 0.01)
            return np.array([np.clip(acc, 0, 1), max(0, 1 - n / 50)])

        def expand(node, max_children=3):
            return [
                (
                    [*node.features, f"feat_{rng.integers(100)}"],
                    "add",
                    f"add:feat_{rng.integers(100)}",
                )
                for _ in range(max_children)
            ]

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        best = search.search(initial_features=["seed_feat"])

        # Trace path from best back to root
        path = []
        current = best
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()  # root → ... → best

        assert len(path) >= 2, "Best should be at least 1 step from root"

        # Each step should change the feature list (length or content)
        for i in range(1, len(path)):
            parent_features = path[i - 1].features
            child_features = path[i].features
            assert parent_features != child_features, (
                f"Step {i}: features unchanged from parent. "
                f"Parent: {parent_features}, Child: {child_features}"
            )

        # The operation should be recorded
        for i in range(1, len(path)):
            assert path[i].operation != "root", (
                f"Step {i}: expected a non-root operation, got '{path[i].operation}'"
            )
