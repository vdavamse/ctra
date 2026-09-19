"""Unit tests for ``MCTSSearch._select_best`` — final ranking by own evaluation.

``_select_best`` used to rank by ``mean_reward``, the average over a node's
*subtree*: evaluating mediocre children of the best feature set dragged its
score down and the search could return a worse, unexpanded node (issue #15).
It now ranks by each node's own ``objective_history``.  These tests build the
tree by hand so the ranking rule is exercised directly, with no rollouts.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSNode, MCTSSearch
from ctra.search.objectives import ObjectiveResult


def _search(objectives: list[str], reference: list[float]) -> MCTSSearch:
    """An ``MCTSSearch`` with no tree — the tests attach one by hand."""
    config = MCTSConfig(
        num_rollouts=1,
        max_depth=3,
        objectives=objectives,
        reference_point=reference,
        deep_simulation=False,
    )
    return MCTSSearch(runner=lambda *_a: None, task="phase2", config=config)


def _node(
    features: list[str],
    *,
    own: list[list[float]] | None = None,
    total_reward: list[float] | None = None,
    visit_count: int = 0,
    parent: MCTSNode | None = None,
) -> MCTSNode:
    """A detached node with a hand-written own-evaluation history."""
    node = MCTSNode(
        features=list(features),
        parent=parent,
        total_reward=np.array(total_reward if total_reward is not None else [0.0]),
    )
    node.visit_count = visit_count
    node.objective_history = [
        ObjectiveResult(values=np.array(v), names=[f"obj_{i}" for i in range(len(v))], details={})
        for v in (own or [])
    ]
    return node


def _wire(search: MCTSSearch, root: MCTSNode, *rest: MCTSNode) -> None:
    search._root = root
    search._all_nodes = [root, *rest]


class TestSingleObjective:
    def test_diluted_ancestor_beats_undiluted_child(self):
        """The 0.90 feature set wins even though its subtree mean is 0.70.

        Mirrors scenario A4 of ``notes/repro_mcts_bugs.py``: node X scored
        0.90 itself and was then expanded; two 0.60 children backpropagated
        through it, leaving ``mean_reward`` 0.70 — below node Y's 0.80.
        """
        search = _search(["accuracy"], [0.0])
        x = _node(["good_set"], own=[[0.90]], total_reward=[0.90 + 0.60 + 0.60], visit_count=3)
        y = _node(["mediocre_set"], own=[[0.80]], total_reward=[0.80], visit_count=1, parent=x)
        _wire(search, x, y)

        assert search._select_best() is x
        assert x.mean_reward[0] == pytest.approx(0.70)
        assert search.best_own_objectives(x) == pytest.approx(np.array([0.90]))

    def test_highest_own_score_wins_regardless_of_visits(self):
        search = _search(["accuracy"], [0.0])
        root = _node(["a"], own=[[0.40]], total_reward=[0.40], visit_count=1)
        mid = _node(["a", "b"], own=[[0.95]], total_reward=[0.95], visit_count=1, parent=root)
        deep = _node(["a", "b", "c"], own=[[0.55]], total_reward=[0.55], visit_count=1, parent=mid)
        _wire(search, root, mid, deep)

        assert search._select_best() is mid

    def test_selection_is_deterministic(self):
        """Repeated calls on the same tree return the same node."""
        search = _search(["accuracy"], [0.0])
        root = _node(["a"], own=[[0.70]], visit_count=1)
        peers = [_node(["a", f"b{i}"], own=[[0.70]], visit_count=1, parent=root) for i in range(5)]
        _wire(search, root, *peers)

        picks = {id(search._select_best()) for _ in range(5)}
        assert len(picks) == 1


class TestMultiObjective:
    def test_diluted_ancestor_holding_the_best_vector_is_selected(self):
        """The parent's own (0.95, 0.94) wins although its subtree mean is poor."""
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a"], own=[[0.30, 0.98]], total_reward=[0.30, 0.98], visit_count=1)
        good = _node(
            ["a", "b", "c"],
            own=[[0.95, 0.94]],
            # own 0.95 plus three 0.40 children -> mean accuracy 0.5375
            total_reward=[0.95 + 0.40 * 3, 0.94 * 4],
            visit_count=4,
            parent=root,
        )
        children = [
            _node(["a", "b", "c", f"d{i}"], own=[[0.40, 0.92]], visit_count=1, parent=good)
            for i in range(3)
        ]
        _wire(search, root, good, *children)

        assert good.mean_reward[0] == pytest.approx(0.5375)
        assert search._select_best() is good

    def test_node_without_own_evaluation_is_excluded(self):
        """A visited-but-never-evaluated node is not a candidate (issue #7 skips)."""
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a"], own=[[0.60, 0.98]], total_reward=[0.60, 0.98], visit_count=1)
        # Expanded, backpropagated through, never evaluated itself.
        skipped = _node(["a", "b"], own=[], total_reward=[9.0, 9.0], visit_count=9, parent=root)
        _wire(search, root, skipped)

        assert skipped.mean_reward[0] == pytest.approx(1.0)
        assert search._select_best() is root

    def test_falls_back_to_root_when_nothing_was_evaluated(self):
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a"], own=[])
        child = _node(["a", "b"], own=[], parent=root)
        _wire(search, root, child)

        assert search._select_best() is root

    def test_duplicate_front_points_do_not_cancel(self):
        """Two nodes holding the same vector must not both score zero.

        ``hypervolume_contribution`` is leave-one-out, so coincident points
        cover nothing exclusively.  Own-eval ranking makes coincidence
        ordinary (one feature set, several paths), and without the
        distinct-point handling in ``_select_best`` the lone root would win.
        """
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a"], own=[[0.50, 0.98]], visit_count=1)
        twins = [
            _node(["a", "b", "c"], own=[[0.85, 0.94]], visit_count=1, parent=root) for _ in range(2)
        ]
        _wire(search, root, *twins)

        assert search._select_best() is twins[0]


class TestMultipleOwnEvaluations:
    def test_best_entry_of_the_history_is_used(self):
        """A re-evaluated node is ranked by its best entry, not its last."""
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a"], own=[[0.60, 0.98]], visit_count=1)
        rerun = _node(
            ["a", "b"],
            own=[[0.55, 0.96], [0.97, 0.96], [0.58, 0.96]],
            visit_count=3,
            parent=root,
        )
        _wire(search, root, rerun)

        assert search.best_own_objectives(rerun) == pytest.approx(np.array([0.97, 0.96]))
        assert search._select_best() is rerun

    def test_entries_differing_only_in_accuracy_pick_the_higher(self):
        """Parsimony is a function of ``len(features)``, so re-runs differ in accuracy only."""
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        node = _node(["a", "b"], own=[[0.40, 0.96], [0.72, 0.96]], visit_count=2)
        _wire(search, node)

        assert search.best_own_objectives(node) == pytest.approx(np.array([0.72, 0.96]))

    def test_accuracy_breaks_ties_below_the_reference_point(self):
        """All vectors have hypervolume 0 under the reference; accuracy decides."""
        search = _search(["accuracy", "parsimony"], [0.5, 0.5])
        node = _node(["a"], own=[[0.10, 0.20], [0.30, 0.20]], visit_count=2)
        _wire(search, node)

        assert search.best_own_objectives(node) == pytest.approx(np.array([0.30, 0.20]))

    def test_accessor_returns_a_copy(self):
        search = _search(["accuracy"], [0.0])
        node = _node(["a"], own=[[0.80]], visit_count=1)
        _wire(search, node)

        returned = search.best_own_objectives(node)
        returned[0] = 0.0
        assert node.objective_history[0].values[0] == pytest.approx(0.80)

    def test_accessor_falls_back_to_mean_reward_without_history(self):
        search = _search(["accuracy"], [0.0])
        node = _node(["a"], own=[], total_reward=[1.2], visit_count=2)
        _wire(search, node)

        assert search.best_own_objectives(node) == pytest.approx(np.array([0.6]))


class TestTieBreaking:
    def test_exact_tie_prefers_the_smaller_feature_set(self):
        search = _search(["accuracy"], [0.0])
        root = _node(["a"], own=[[0.50]], visit_count=1)
        wide = _node(["a", "b", "c", "d"], own=[[0.90]], visit_count=1, parent=root)
        narrow = _node(["a", "b"], own=[[0.90]], visit_count=1, parent=root)
        # ``wide`` comes first in _all_nodes, so only the feature count can pick ``narrow``.
        _wire(search, root, wide, narrow)

        assert search._select_best() is narrow

    def test_exact_tie_on_features_prefers_the_first_found(self):
        search = _search(["accuracy"], [0.0])
        root = _node(["a"], own=[[0.50]], visit_count=1)
        first = _node(["a", "b"], own=[[0.90]], visit_count=1, parent=root)
        second = _node(["a", "c"], own=[[0.90]], visit_count=1, parent=root)
        _wire(search, root, first, second)

        assert search._select_best() is first

    def test_multi_objective_tie_prefers_the_smaller_feature_set(self):
        """Identical front vectors: the tie-break, not ``argmax``, decides."""
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a"], own=[[0.20, 0.30]], visit_count=1)
        wide = _node(["a", "b", "c"], own=[[0.90, 0.90]], visit_count=1, parent=root)
        narrow = _node(["a", "b"], own=[[0.90, 0.90]], visit_count=1, parent=root)
        _wire(search, root, wide, narrow)

        assert search._select_best() is narrow

    def test_all_contributions_zero_is_a_tie_not_an_argmax(self):
        """Every point below the reference: contributions are all 0 (issue #18)."""
        search = _search(["accuracy", "parsimony"], [0.5, 0.5])
        root = _node(["a"], own=[[0.10, 0.40]], visit_count=1)
        wide = _node(["a", "b", "c"], own=[[0.40, 0.10]], visit_count=1, parent=root)
        narrow = _node(["a", "b"], own=[[0.30, 0.20]], visit_count=1, parent=root)
        _wire(search, root, wide, narrow)

        # All three are mutually non-dominated and all contribute zero volume.
        assert search._select_best() is root


class TestNonFiniteScores:
    """A NaN evaluation must neither win nor crash the final ranking."""

    def test_nan_child_does_not_crash_single_objective_selection(self):
        """Root 0.70 plus a NaN child: the root is selected, nothing raises."""
        search = _search(["accuracy"], [0.0])
        root = _node(["a"], own=[[0.70]], visit_count=1)
        child = _node(["a", "b"], own=[[float("nan")]], visit_count=1, parent=root)
        _wire(search, root, child)

        assert search._select_best() is root

    def test_accessor_skips_entries_with_a_nan_component(self):
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        node = _node(["a", "b"], own=[[float("nan"), 0.96], [0.9, 0.96]], visit_count=2)
        _wire(search, node)

        assert search.best_own_objectives(node) == pytest.approx(np.array([0.9, 0.96]))

    def test_all_nan_single_objective_still_returns_a_node(self):
        search = _search(["accuracy"], [0.0])
        root = _node(["a"], own=[[float("nan")]], visit_count=1)
        child = _node(["a", "b"], own=[[float("nan")]], visit_count=1, parent=root)
        _wire(search, root, child)

        assert isinstance(search._select_best(), MCTSNode)
