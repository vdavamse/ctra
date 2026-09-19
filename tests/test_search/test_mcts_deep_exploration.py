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
from tests.test_search.conftest import make_stub_output, make_stub_runner

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
            reference_point=[0.5, 0.0],  # accuracy at the ROC-AUC chance baseline (#18)
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


class _FeatureAwareRunner:
    """A runner that evaluates each node's **own** feature set.

    ``make_stub_runner`` (conftest) rebuilds its output from
    ``previous_output.feature_plans`` — the *parent's* plans — and never
    applies the suggestion the child is following, so every node in the tree
    is evaluated with the root's feature set: ``evaluate_fn`` only ever sees
    one feature and the accuracy landscape these tests describe is never
    reached.  That made the three tests below vacuous (issue #15, R2); fixing
    the shared helper has a suite-wide blast radius and is tracked separately,
    so this module carries its own runner.

    This one behaves like ``Agent.forward``'s proposer: suggestion ``i`` ADDs
    the ``i``-th feature of ``feature_pool`` that the parent does not already
    carry.  The returned ``AgentOutput``'s plans are therefore the *evaluated*
    node's own feature set, which ``_call_evaluate`` syncs back onto
    ``node.features``, and its suggestions advertise what is still addable —
    an empty list when the pool is spent, which stops expansion there.

    ``seen`` records every feature set handed to ``evaluate_fn``, so a test can
    assert the sets really grow instead of trusting the runner.
    """

    def __init__(
        self,
        evaluate_fn,
        initial_features: list[str],
        feature_pool: list[str],
        max_suggestions: int = 3,
    ) -> None:
        self.evaluate_fn = evaluate_fn
        self.initial_features = list(initial_features)
        self.feature_pool = list(feature_pool)
        self.max_suggestions = max_suggestions
        self.seen: list[list[str]] = []

    def __call__(self, node_id, task, previous_output):
        if previous_output is None or not previous_output.feature_plans:
            features = list(self.initial_features)
        else:
            parent = list(previous_output.feature_plans.keys())
            available = [f for f in self.feature_pool if f not in parent]
            index = previous_output.suggestion_index
            features = [*parent, available[index]] if 0 <= index < len(available) else parent

        self.seen.append(list(features))
        values = np.asarray(self.evaluate_fn(features), dtype=float).ravel()
        remaining = [f for f in self.feature_pool if f not in features][: self.max_suggestions]
        output = make_stub_output(features, float(values[0]), [f"add {f}" for f in remaining])
        if not remaining:
            # ``make_stub_output`` substitutes five generic suggestions for an
            # empty list; a proposer with nothing left to add offers none, and
            # ``_suggestion_expand`` then stops expanding this node.
            output.eval_outputs = {
                name: ev._replace(suggestions=[]) for name, ev in output.eval_outputs.items()
            }
        return output

    @property
    def sizes_seen(self) -> list[int]:
        """Size of every feature set that reached ``evaluate_fn``."""
        return [len(f) for f in self.seen]


def _make_feature_aware_runner(evaluate_fn, initial_features, feature_pool, max_suggestions=3):
    """Build a ``_FeatureAwareRunner`` (see its docstring)."""
    return _FeatureAwareRunner(evaluate_fn, initial_features, feature_pool, max_suggestions)


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
        produce a node that outperforms the initial feature set.  Selection
        ranks nodes by their own evaluation (issue #15), so the sweet spot at
        n=3 has to actually be *evaluated* at n=3 — hence the local runner.
        """

        def evaluate(features):
            n = len(features)
            # Accuracy jumps sharply at n=3 to create a clear sweet spot,
            # and stays flat after it so the extra features only cost parsimony.
            # No jitter, for the same reason as the synergy test below: the
            # sweet spot is a deterministic step, and noise only spreads the
            # n>=3 sets into a cluster whose exclusive widths shrink towards
            # the root's.  (With sigma 0.005 the seeded run still passed, n=3
            # set 0.0304 vs root 0.0070, a seeded-deterministic margin; without
            # jitter it is 0.47 vs 0.007.)
            acc = 0.9 if n >= 3 else 0.3 + 0.05 * n
            return np.array([acc, max(0, 1 - n / 50)])

        runner = _make_feature_aware_runner(evaluate, ["f0"], [f"f{i}" for i in range(1, 13)])
        search = MCTSSearch(runner=runner, task="test")
        best = search.search(initial_features=["f0"])

        # The fixture is only meaningful if the feature sets actually grew.
        assert max(runner.sizes_seen) >= 3, (
            f"the runner never evaluated a 3-feature set: sizes {sorted(set(runner.sizes_seen))}"
        )

        best_depth = _node_depth(best)
        assert best_depth > 0, (
            "Best node is the root — MCTS failed to find any improvement through feature refinement"
        )
        assert len(best.features) >= 3, f"best node is below the n=3 sweet spot: {best.features}"
        assert search.best_own_objectives(best)[0] > 0.8, (
            f"best node's own accuracy {search.best_own_objectives(best)[0]:.3f} "
            f"is not from the sweet spot"
        )

    def test_deep_combination_discovered(self):
        """MCTS should discover that features A+B+C together score higher
        than any subset, even though individual features score poorly.

        This simulates feature epistasis: the interaction effect is
        non-linear (synergistic).
        """
        # Synergistic feature trio: each alone scores 0.55, but all three
        # together score 0.85. The MCTS must go deep enough to discover this.
        #
        # The tiers are exact, with no jitter: the search evaluates dozens of
        # feature sets in the top tier and a jitter spreads them into a dense
        # cluster on the Pareto front, where no single point has a meaningful
        # exclusive hypervolume contribution and the lone 1-feature root wins
        # the ranking on width alone.  The epistasis this test is about is a
        # deterministic interaction effect, so the noise only hid it.
        # The noisy variant (sigma 0.01) selected the root under
        # ``reference_point=[0, 0]`` — root 0.01006 vs synergy set 0.00818 —
        # and selects the synergy set at the chance baseline ``[0.5, 0]``
        # (root 0.00006 vs 0.00818, issue #18); see
        # ``test_deep_combination_survives_accuracy_noise``.
        synergy_features = {"alpha", "beta", "gamma"}
        tiers = {3: 0.85, 2: 0.65, 1: 0.55, 0: 0.50}

        def evaluate(features):
            synergy_count = len(set(features) & synergy_features)
            n = len(features)
            return np.array([tiers[synergy_count], max(0, 1 - n / 50)])

        feature_pool = ["alpha", "beta", "gamma", "noise1", "noise2", "noise3"]

        runner = _make_feature_aware_runner(evaluate, ["noise1"], feature_pool)
        search = MCTSSearch(runner=runner, task="test")
        best = search.search(initial_features=["noise1"])

        # The fixture is only meaningful if the feature sets actually grew:
        # a runner that re-evaluates the root's single feature can never
        # produce a synergy count above 1.
        synergy_counts = {len(set(f) & synergy_features) for f in runner.seen}
        assert 3 in synergy_counts, (
            f"the runner never evaluated all three synergy features together; "
            f"synergy counts seen: {sorted(synergy_counts)}"
        )

        best_features = set(best.features)
        synergy_found = len(best_features & synergy_features)

        # The search should discover at least 2 of the 3 synergy features
        assert synergy_found >= 2, (
            f"Expected MCTS to discover synergistic feature combination, "
            f"but best features {best.features} only contain "
            f"{synergy_found}/3 synergy features (alpha, beta, gamma). "
            f"Best own accuracy: {search.best_own_objectives(best)[0]:.3f}"
        )

    def test_deep_combination_survives_accuracy_noise(self):
        """The synergy fixture with ``rng.normal(0, 0.01)`` accuracy jitter.

        Same tiers as ``test_deep_combination_discovered``; the noise spreads
        the top tier into a dense cluster on the front where no point keeps a
        meaningful exclusive width.  Under ``reference_point=[0, 0]`` the lone
        1-feature root won on width alone (root 0.01006 vs synergy set
        0.00818): its width was the 0.5 of ROC-AUC every classifier gets for
        free.  With the reference at the chance baseline ``[0.5, 0]`` that
        width is dead volume (root 0.00006 vs 0.00818) and the 4-feature
        synergy set is selected (issue #18).
        """
        rng = np.random.default_rng(42)
        synergy_features = {"alpha", "beta", "gamma"}
        tiers = {3: 0.85, 2: 0.65, 1: 0.55, 0: 0.50}

        def evaluate(features):
            synergy_count = len(set(features) & synergy_features)
            n = len(features)
            acc = tiers[synergy_count] + rng.normal(0, 0.01)
            return np.array([np.clip(acc, 0, 1), max(0, 1 - n / 50)])

        feature_pool = ["alpha", "beta", "gamma", "noise1", "noise2", "noise3"]

        runner = _make_feature_aware_runner(evaluate, ["noise1"], feature_pool)
        search = MCTSSearch(runner=runner, task="test")
        best = search.search(initial_features=["noise1"])

        synergy_counts = {len(set(f) & synergy_features) for f in runner.seen}
        assert 3 in synergy_counts, (
            f"the runner never evaluated all three synergy features together; "
            f"synergy counts seen: {sorted(synergy_counts)}"
        )

        assert synergy_features <= set(best.features), (
            f"best features {best.features} do not contain all of alpha, beta, gamma; "
            f"best own accuracy: {search.best_own_objectives(best)[0]:.3f}"
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

        def evaluate(features):
            n = len(features)
            acc = 0.5 + 0.04 * min(n, 15) + rng.normal(0, 0.01)
            return np.array([np.clip(acc, 0, 1), max(0, 1 - n / 50)])

        # A pool of 20 lets the sets grow to n=16, past the point where
        # ``np.clip`` flattens accuracy at 1.0 (n >= 13): dozens of nodes
        # share the front coordinate and the ranking there is by parsimony
        # alone.  Under ``reference_point=[0, 0]`` the 1-feature root won on
        # width (root 0.01086 vs 0.00189 for the best deep set) and PR #30
        # shrank the pool to 10 to dodge it; at the chance baseline
        # ``[0.5, 0]`` the root keeps 0.00086 and the smallest saturated set
        # (13 features; 14-16 are dominated by it) is selected (issue #18).
        runner = _make_feature_aware_runner(
            evaluate, ["seed_feat"], [f"feat_{i}" for i in range(20)]
        )
        search = MCTSSearch(runner=runner, task="test")
        best = search.search(initial_features=["seed_feat"])

        # The fixture is only meaningful if the feature sets actually grew.
        assert max(runner.sizes_seen) > 1, (
            f"the runner only ever evaluated the root's feature set: "
            f"sizes {sorted(set(runner.sizes_seen))}"
        )
        assert len(best.features) == 13, (
            f"expected the smallest saturated set (13 features), got "
            f"{len(best.features)}: {best.features}; own accuracy "
            f"{search.best_own_objectives(best)[0]:.3f}"
        )

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
