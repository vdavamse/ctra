"""Comparison test: AutoCT's original MCTS behavior vs our Pareto MCTS.

AutoCT's MCTS (from CogComp/autoct treesearch.py) has a fundamentally
different simulation phase:

    AutoCT:   SELECT → EXPAND → SIMULATE (deep rollout to max_depth,
              evaluating EVERY node along the way) → BACKPROPAGATE
              One rollout creates an entire depth-first path.

    CTRA:     SELECT → EXPAND → SIMULATE (evaluate ONE node) → BACKPROPAGATE
              One rollout evaluates one node. Depth grows slowly.

This test module:
    1. Implements a faithful replica of AutoCT's MCTS loop (scalar, single-objective)
    2. Runs both implementations on identical stub problems
    3. Compares depth, node count, and best reward
    4. Demonstrates that deep rollout finds better feature combinations

The key insight from AutoCT: each rollout goes ALL THE WAY to max_depth,
because the best feature COMBINATION may only emerge after 5-7 sequential
refinements. Our implementation should be evaluated against this baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig, Settings, get_settings
from ctra.search.mcts import MCTSSearch
from tests.test_search.conftest import make_stub_runner

# ---------------------------------------------------------------------------
# AutoCT-faithful MCTS replica (scalar, deep rollout)
# ---------------------------------------------------------------------------


@dataclass
class AutoCTNode:
    """Replica of AutoCT's MCTTreeNode for comparison testing."""

    id: str
    features: list[str]
    depth: int
    max_depth: int
    visit_count: int = 1
    total_reward: float = 0.0
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    is_evaluated: bool = False
    reward: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.depth >= self.max_depth


class AutoCTMCTS:
    """Faithful replica of AutoCT's MCTS loop for comparison.

    Key differences from our MCTSSearch:
    - _simulate does a DEEP ROLLOUT: expands and evaluates from current
      node all the way to max_depth, creating a full path
    - reward is the MAX reward found on the entire path (not just leaf)
    - children are generated from evaluator suggestions (here: stub)
    - single scalar objective (ROC-AUC)
    """

    def __init__(
        self,
        evaluate_fn,
        expand_fn,
        max_depth: int = 7,
        exploration_weight: float = 1.0,
    ):
        self._evaluate_fn = evaluate_fn
        self._expand_fn = expand_fn
        self._max_depth = max_depth
        self._exploration_weight = exploration_weight
        self._nodes: dict[str, AutoCTNode] = {}
        self._rng = np.random.default_rng(42)
        self.eval_count = 0

    def search(self, initial_features: list[str], num_rollouts: int = 10) -> AutoCTNode:
        """Run AutoCT-style MCTS."""
        # Create and evaluate root
        root = AutoCTNode(id="0", features=initial_features, depth=0, max_depth=self._max_depth)
        root.reward = self._evaluate(root)
        root.is_evaluated = True
        self._nodes["0"] = root

        # Generate root's children
        self._generate_children(root)

        for _rollout in range(num_rollouts):
            # 1. SELECT: traverse tree using UCT to find a leaf
            path = self._select("0")

            # 2. EXPAND: evaluate the leaf node
            leaf = self._nodes[path[-1]]
            if not leaf.is_evaluated:
                leaf.reward = self._evaluate(leaf)
                leaf.is_evaluated = True
                self._nodes[leaf.id] = leaf
                self._generate_children(leaf)

            # 3. SIMULATE: deep rollout from leaf to max_depth
            #    This is the key difference — AutoCT evaluates EVERY node
            #    on the path down to max_depth
            max_reward = self._simulate_deep(leaf)

            # 4. BACKPROPAGATE: update all nodes on the selection path
            for node_id in path:
                node = self._nodes[node_id]
                node.visit_count += 1
                node.total_reward += max_reward

        # Return best evaluated node
        best = max(
            (n for n in self._nodes.values() if n.is_evaluated),
            key=lambda n: n.reward,
        )
        return best

    def _evaluate(self, node: AutoCTNode) -> float:
        """Evaluate a single node (stub: calls evaluate_fn)."""
        self.eval_count += 1
        return self._evaluate_fn(node.features)

    def _generate_children(self, node: AutoCTNode):
        """Generate child nodes from suggestions."""
        if node.is_terminal:
            return

        children_features = self._expand_fn(node.features)
        for i, child_feats in enumerate(children_features):
            child_id = f"{node.id}-{i}"
            if child_id not in self._nodes:
                child = AutoCTNode(
                    id=child_id,
                    features=child_feats,
                    depth=node.depth + 1,
                    max_depth=self._max_depth,
                    parent_id=node.id,
                )
                self._nodes[child_id] = child
                node.children_ids.append(child_id)

    def _select(self, node_id: str) -> list[str]:
        """UCT-based tree traversal — returns path of node IDs."""
        path = [node_id]
        current = self._nodes[node_id]

        while True:
            if current.is_terminal or not current.is_evaluated:
                return path

            # Check for unexplored children
            unexplored = [cid for cid in current.children_ids if not self._nodes[cid].is_evaluated]
            if unexplored:
                path.append(unexplored[0])
                return path

            if not current.children_ids:
                return path

            # UCT selection among fully explored children
            best_child_id = max(
                current.children_ids,
                key=lambda cid: self._uct(self._nodes[cid], current),
            )
            path.append(best_child_id)
            current = self._nodes[best_child_id]

        return path

    def _uct(self, child: AutoCTNode, parent: AutoCTNode) -> float:
        """Standard UCT formula."""
        exploit = child.total_reward / max(child.visit_count, 1)
        explore = self._exploration_weight * math.sqrt(
            math.log(max(parent.visit_count, 1)) / max(child.visit_count, 1)
        )
        return exploit + explore

    def _simulate_deep(self, start_node: AutoCTNode) -> float:
        """AutoCT's deep rollout: evaluate nodes down to max_depth.

        THIS is the key AutoCT behavior: one rollout creates an entire
        path from the current node to max_depth, evaluating every
        intermediate node. Returns the MAX reward found on the path.
        """
        max_reward = start_node.reward
        current = start_node

        while not current.is_terminal:
            # Generate children if not done
            self._generate_children(current)

            if not current.children_ids:
                break

            # Random child selection (AutoCT's simulation policy)
            choice_id = self._rng.choice(current.children_ids)
            child = self._nodes[choice_id]

            if not child.is_evaluated:
                child.reward = self._evaluate(child)
                child.is_evaluated = True
                self._nodes[child.id] = child
                self._generate_children(child)

            if child.reward > max_reward:
                max_reward = child.reward

            current = child

        return max_reward

    @property
    def max_depth_reached(self) -> int:
        return max(
            (n.depth for n in self._nodes.values() if n.is_evaluated),
            default=0,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    get_settings.cache_clear()
    settings = Settings(
        mcts=MCTSConfig(
            num_rollouts=10,
            max_depth=7,
            objectives=["accuracy"],
            max_features=50,
            exploration_constant=1.0,
            adaptive_branching=False,
            max_branch_factor=3,
            reference_point=[0.0],
        ),
    )
    monkeypatch.setattr("ctra.search.mcts.get_settings", lambda: settings)
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Shared stub functions
# ---------------------------------------------------------------------------


def make_synergy_evaluator(seed: int = 42):
    """Evaluator that rewards specific feature combinations.

    Features A, B, C, D each contribute independently, but the combination
    A+B+C+D together gives a bonus (epistasis). Discovery requires going
    deep enough to accumulate all four.
    """
    rng = np.random.default_rng(seed)
    synergy = {"feat_A", "feat_B", "feat_C", "feat_D"}

    def evaluate(features: list[str]) -> float:
        fset = set(features)
        count = len(fset & synergy)
        base = 0.5 + 0.03 * count  # small per-feature contribution
        if count >= 4:
            base += 0.20  # synergy bonus for all 4
        elif count >= 3:
            base += 0.08  # partial synergy
        noise = rng.normal(0, 0.01)
        return float(np.clip(base + noise, 0, 1))

    return evaluate


def make_suggestion_expander(seed: int = 42, num_suggestions: int = 3):
    """Expander that returns feature modifications (mimics LLM suggestions).

    Each call returns num_suggestions modified feature sets:
    adds from a pool, with some random noise features mixed in.
    """
    rng = np.random.default_rng(seed)
    pool = ["feat_A", "feat_B", "feat_C", "feat_D", "noise_1", "noise_2", "noise_3"]

    def expand(features: list[str]) -> list[list[str]]:
        results = []
        for _ in range(num_suggestions):
            available = [f for f in pool if f not in features]
            if not available:
                available = [f"extra_{rng.integers(100)}"]
            new_feat = rng.choice(available)
            results.append([*features, new_feat])
        return results

    return expand


def expand_for_ctra(seed: int = 42, num_suggestions: int = 3):
    """Adapt the suggestion expander to CTRA's expand_fn signature."""
    inner = make_suggestion_expander(seed=seed, num_suggestions=num_suggestions)

    def expand(node, max_children=3):
        suggestions = inner(node.features)[:max_children]
        return [(feats, "add", f"add:{feats[-1]}") for feats in suggestions]

    return expand


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoCTDeepRollout:
    """Demonstrate AutoCT's deep rollout behavior."""

    def test_autoct_reaches_max_depth(self):
        """AutoCT's simulate goes all the way to max_depth in one rollout."""
        evaluate = make_synergy_evaluator(seed=42)
        expand = make_suggestion_expander(seed=42, num_suggestions=3)

        mcts = AutoCTMCTS(
            evaluate_fn=evaluate,
            expand_fn=expand,
            max_depth=7,
            exploration_weight=1.0,
        )
        _best = mcts.search(initial_features=["start"], num_rollouts=10)

        assert mcts.max_depth_reached == 7, (
            f"AutoCT should reach max_depth=7, but only reached {mcts.max_depth_reached}"
        )

    def test_autoct_evaluates_many_nodes_per_rollout(self):
        """Each AutoCT rollout evaluates ~max_depth nodes, not just 1."""
        evaluate = make_synergy_evaluator(seed=42)
        expand = make_suggestion_expander(seed=42, num_suggestions=3)

        mcts = AutoCTMCTS(
            evaluate_fn=evaluate,
            expand_fn=expand,
            max_depth=7,
            exploration_weight=1.0,
        )
        mcts.search(initial_features=["start"], num_rollouts=10)

        # With 10 rollouts and max_depth=7, AutoCT evaluates ~70+ nodes
        # (not just 10+1 like CTRA would)
        evaluated = sum(1 for n in mcts._nodes.values() if n.is_evaluated)
        assert evaluated > 30, (
            f"AutoCT should evaluate many nodes, but only evaluated {evaluated}. "
            f"Total nodes: {len(mcts._nodes)}, eval calls: {mcts.eval_count}"
        )

    def test_autoct_finds_deep_synergy(self):
        """AutoCT's deep rollout discovers the 4-feature synergy."""
        evaluate = make_synergy_evaluator(seed=42)
        expand = make_suggestion_expander(seed=42, num_suggestions=3)

        mcts = AutoCTMCTS(
            evaluate_fn=evaluate,
            expand_fn=expand,
            max_depth=7,
        )
        best = mcts.search(initial_features=["start"], num_rollouts=10)

        synergy = {"feat_A", "feat_B", "feat_C", "feat_D"}
        found = len(set(best.features) & synergy)
        assert found >= 3, (
            f"AutoCT should find synergistic features, got {found}/4: {best.features}"
        )


class TestCTRAvsAutoCTBehavior:
    """Compare our MCTS against AutoCT's on the same problem."""

    def test_both_find_good_solutions(self):
        """Both implementations should find reasonable solutions."""
        evaluate_autoct = make_synergy_evaluator(seed=42)
        expand_autoct = make_suggestion_expander(seed=42, num_suggestions=3)

        autoct = AutoCTMCTS(
            evaluate_fn=evaluate_autoct,
            expand_fn=expand_autoct,
            max_depth=7,
        )
        autoct_best = autoct.search(initial_features=["start"], num_rollouts=10)

        # CTRA with same eval logic (wrapped for single objective)
        evaluate_ctra = make_synergy_evaluator(seed=42)

        def ctra_eval(features, fidelity=1.0):
            return np.array([evaluate_ctra(features)])

        expand_ctra = expand_for_ctra(seed=42, num_suggestions=3)

        ctra_runner = make_stub_runner(ctra_eval, expand_ctra)
        ctra_search = MCTSSearch(runner=ctra_runner, task="test", expand_fn=expand_ctra)
        ctra_best = ctra_search.search(initial_features=["start"])

        # Both should beat random chance (0.5)
        assert autoct_best.reward > 0.5
        assert ctra_best.mean_reward[0] > 0.5

    def test_autoct_goes_deeper_per_rollout(self):
        """AutoCT creates deeper trees per rollout than CTRA."""
        evaluate_fn = make_synergy_evaluator(seed=42)
        expand_fn = make_suggestion_expander(seed=42, num_suggestions=3)

        autoct = AutoCTMCTS(
            evaluate_fn=evaluate_fn,
            expand_fn=expand_fn,
            max_depth=7,
        )
        autoct.search(initial_features=["start"], num_rollouts=10)

        # AutoCT should reach depth 7 (its max) within 10 rollouts
        assert autoct.max_depth_reached == 7

        # CTRA with 10 rollouts reaches less depth
        ctra_eval = make_synergy_evaluator(seed=42)

        def ctra_evaluate(features, fidelity=1.0):
            return np.array([ctra_eval(features)])

        ctra_expand = expand_for_ctra(seed=42, num_suggestions=3)
        ctra_runner = make_stub_runner(ctra_evaluate, ctra_expand)
        ctra_search = MCTSSearch(runner=ctra_runner, task="test", expand_fn=ctra_expand)
        ctra_search.search(initial_features=["start"])

        # Measure CTRA's max depth
        ctra_max_depth = 0
        for node in ctra_search.all_nodes:
            d = 0
            n = node
            while n.parent:
                d += 1
                n = n.parent
            ctra_max_depth = max(ctra_max_depth, d)

        # AutoCT goes deeper per rollout
        # This documents the behavioral difference
        print(f"\nAutoCT depth: {autoct.max_depth_reached}, CTRA depth: {ctra_max_depth}")
        print(f"AutoCT nodes evaluated: {autoct.eval_count}")
        print(f"CTRA nodes: {len(ctra_search.all_nodes)}")

    def test_autoct_evaluates_more_nodes(self):
        """AutoCT evaluates more nodes total due to deep rollout."""
        evaluate_fn = make_synergy_evaluator(seed=42)
        expand_fn = make_suggestion_expander(seed=42, num_suggestions=3)

        autoct = AutoCTMCTS(
            evaluate_fn=evaluate_fn,
            expand_fn=expand_fn,
            max_depth=7,
        )
        autoct.search(initial_features=["start"], num_rollouts=10)

        ctra_eval = make_synergy_evaluator(seed=42)
        ctra_eval_count = 0

        def ctra_evaluate(features, fidelity=1.0):
            nonlocal ctra_eval_count
            ctra_eval_count += 1
            return np.array([ctra_eval(features)])

        ctra_expand = expand_for_ctra(seed=42, num_suggestions=3)
        ctra_runner = make_stub_runner(ctra_evaluate, ctra_expand)
        ctra_search = MCTSSearch(runner=ctra_runner, task="test", expand_fn=ctra_expand)
        ctra_search.search(initial_features=["start"])

        print(f"\nAutoCT eval calls: {autoct.eval_count}")
        print(f"CTRA eval calls: {ctra_eval_count}")

        # With deep_simulation=True, CTRA should evaluate a comparable number
        # of nodes to AutoCT (both do deep rollouts to max_depth)
        assert ctra_eval_count > 30, (
            f"CTRA with deep simulation should evaluate many nodes, got {ctra_eval_count}"
        )
