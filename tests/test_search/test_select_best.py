"""Unit tests for ``MCTSSearch._select_best`` — final ranking by own evaluation.

``_select_best`` used to rank by ``mean_reward``, the average over a node's
*subtree*: evaluating mediocre children of the best feature set dragged its
score down and the search could return a worse, unexpanded node (issue #15).
It now ranks by each node's own ``objective_history``.  These tests build the
tree by hand so the ranking rule is exercised directly, with no rollouts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSNode, MCTSSearch
from ctra.search.objectives import ObjectiveResult

from .conftest import make_stub_output

if TYPE_CHECKING:
    from ctra.agents.data_models import AgentOutput


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


def _entry(values: list[float], features: list[str] | tuple[str, ...] | None) -> ObjectiveResult:
    """One history entry, snapshotting ``features`` the way ``_wrap_result`` does.

    ``None`` writes no snapshot at all — the shape of an entry restored from
    a checkpoint written before the snapshot existed.
    """
    names = [f"obj_{i}" for i in range(len(values))]
    details: dict[str, Any] = dict(zip(names, map(float, values), strict=True))
    if features is not None:
        details["features"] = list(features)
    return ObjectiveResult(values=np.array(values), names=names, details=details)


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
        """Re-runs of one set usually share a parsimony and differ in accuracy.

        Parsimony is a function of ``len(features)``; a re-evaluation that
        changed the plans is filtered out by the feature snapshot
        (``TestFeatureSnapshot``), so what remains differs in accuracy.
        """
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


class _PlanChangingRunner:
    """Non-root call *n* returns the parent's plans plus ``x<n>``, scored ``rocs[n-1]``.

    A proposer that never returns the same plan twice: a node's second
    evaluation lands on a different feature set.  ``scores`` maps every
    feature set the runner returned to the ROC-AUC it gave it.
    """

    def __init__(self, rocs: list[float]) -> None:
        self._rocs = list(rocs)
        self.calls = 0
        self.scores: dict[tuple[str, ...], float] = {}

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        if previous_output is None:
            features, roc = ["f0"], 0.50
        else:
            self.calls += 1
            features = [*previous_output.feature_plans, f"x{self.calls}"]
            roc = self._rocs[self.calls - 1]
        self.scores[tuple(features)] = roc
        return make_stub_output(features, roc)


class TestFeatureSnapshot:
    """A node re-evaluated with changed plans is ranked by the set it now holds.

    ``_call_evaluate`` rewrites ``node.features`` when the runner's plans differ
    from the parent's, so a re-picked node can carry a history entry scored
    for a set it no longer has.  Each entry snapshots its set
    (``details["features"]``) and ``_best_own_objectives`` skips the ones
    that do not match ``node.features``.
    """

    def test_entry_for_a_previous_feature_set_is_ignored(self):
        """History ``{a,b,c}: 0.91`` then ``{a,b,d}: 0.72``; the node now holds ``[a,b,d]``.

        Ranked by the stale 0.91 the node wins the front outright; ranked by
        its own 0.72 it is dominated by ``other`` (0.80 on two features).
        """
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a"], own=[[0.60, 0.98]], visit_count=1)
        rerun = _node(["a", "b", "d"], visit_count=2, parent=root)
        rerun.objective_history = [
            _entry([0.91, 0.94], ["a", "b", "c"]),
            _entry([0.72, 0.94], ["a", "b", "d"]),
        ]
        other = _node(["a", "b"], own=[[0.80, 0.96]], visit_count=1, parent=root)
        _wire(search, root, rerun, other)

        assert search.best_own_objectives(rerun) == pytest.approx(np.array([0.72, 0.94]))
        assert search._select_best() is other

    def test_entry_without_a_snapshot_still_counts(self):
        """A checkpoint written before the snapshot existed: its entries all count."""
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        node = _node(["a", "b", "d"], visit_count=2)
        node.objective_history = [
            _entry([0.91, 0.94], None),
            _entry([0.72, 0.94], ["a", "b", "d"]),
        ]
        _wire(search, node)

        assert search.best_own_objectives(node) == pytest.approx(np.array([0.91, 0.94]))

    def test_snapshot_is_compared_as_a_list_in_stored_order(self):
        """A tuple (a JSON/dill round-trip) matches; a reordered set does not."""
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        node = _node(["a", "b"], visit_count=2)
        node.objective_history = [
            _entry([0.95, 0.96], ["b", "a"]),
            _entry([0.70, 0.96], ("a", "b")),
        ]
        _wire(search, node)

        assert search.best_own_objectives(node) == pytest.approx(np.array([0.70, 0.96]))

    def test_search_ranks_a_re_evaluated_node_by_its_current_set(self):
        """Through ``search()``: the re-picked child is scored for a new set and ranked by it.

        Shallow mode, ``max_depth=1``, two children of the root and no
        expansion below them: rollouts 0 and 1 evaluate the two children
        (0.91, 0.80); rollout 2 re-selects one of them, which cannot expand
        and is evaluated again under a new node id — the runner hands it a
        new plan scored 0.72 and ``_call_evaluate`` rewrites its features.
        Ranked by the stale entry, the search would report the old score
        beside ``best_features`` it never earned.
        """
        runner = _PlanChangingRunner([0.91, 0.80, 0.72])
        config = MCTSConfig(
            num_rollouts=3,
            max_depth=1,
            objectives=["accuracy", "parsimony"],
            reference_point=[0.0, 0.0],
            deep_simulation=False,
            adaptive_branching=False,
            min_branch_factor=2,
            max_branch_factor=2,
            max_features=20,
        )

        def expand_fn(node: MCTSNode, max_children: int = 2) -> list[tuple[list[str], str, str]]:
            if node.parent is not None:
                return []
            return [(list(node.features), "add", f"s{i}") for i in range(max_children)]

        search = MCTSSearch(runner=runner, task="phase2", config=config, expand_fn=expand_fn)
        best = search.search(["f0"])

        evaluated = [n for n in search.all_nodes if n.objective_history]
        rerun = [n for n in evaluated if len(n.objective_history) == 2]
        assert len(rerun) == 1
        assert runner.scores[tuple(rerun[0].features)] == pytest.approx(0.72)

        # Every node is ranked by the score of the set it holds now ...
        for node in evaluated:
            own = search.best_own_objectives(node)
            assert own[0] == pytest.approx(runner.scores[tuple(node.features)])
        # ... and the pick is the node whose *current* set scored highest.
        expected = max(evaluated, key=lambda n: runner.scores[tuple(n.features)])
        assert best is expected
        assert search.best_own_objectives(best)[0] == pytest.approx(
            runner.scores[tuple(best.features)]
        )
        # The history records both sets, and only the second is the node's.
        snapshots = [e.details.get("features") for e in rerun[0].objective_history]
        assert snapshots[0] != snapshots[1] and snapshots[1] == rerun[0].features


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
        """Every point below the reference: contributions are all 0 (issue #18).

        The smallest set is the *last* candidate, so ``argmax`` (front index
        0, the root) and the tie-break disagree — the test can see which
        one decided.
        """
        search = _search(["accuracy", "parsimony"], [0.5, 0.5])
        root = _node(["a", "b", "c"], own=[[0.10, 0.40]], visit_count=1)
        wide = _node(["a", "b", "c", "d"], own=[[0.40, 0.10]], visit_count=1, parent=root)
        narrow = _node(["a"], own=[[0.30, 0.20]], visit_count=1, parent=root)
        _wire(search, root, wide, narrow)

        # All three are mutually non-dominated and all contribute zero volume.
        assert search._select_best() is narrow

    def test_contributions_equal_on_paper_but_ulps_apart_are_a_tie(self):
        """(0.9, 0.2) and (0.6, 0.3) each add 0.06 exclusively — a few ULPs apart in float.

        Exact equality would hand the pick to rounding noise (``wide``);
        ``np.isclose`` makes it a tie and the smaller set wins.
        """
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a", "b"], own=[[0.50, 0.10]], visit_count=1)  # dominated by wide
        wide = _node(["a", "b", "c"], own=[[0.90, 0.20]], visit_count=1, parent=root)
        narrow = _node(["a"], own=[[0.60, 0.30]], visit_count=1, parent=root)
        _wire(search, root, wide, narrow)

        contributions = search._front_contributions(np.array([[0.90, 0.20], [0.60, 0.30]]))
        assert contributions[0] != contributions[1]  # the ULP gap is real ...
        assert np.isclose(contributions[0], contributions[1])  # ... and is not a ranking
        assert search._select_best() is narrow

    def test_near_coincident_front_points_are_deduped(self):
        """Own vectors 1e-13 apart (in opposite directions, so both stay on the front) are one point.

        Without the rounding in ``_front_contributions`` each twin's exclusive
        volume is ~1e-13 and the lone root (0.02) would win.
        """
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a"], own=[[0.50, 0.98]], visit_count=1)
        twins = [
            _node(["a", "b", "c"], own=[[0.85, 0.94 + 1e-13]], visit_count=1, parent=root),
            _node(["a", "b", "d"], own=[[0.85 + 1e-13, 0.94]], visit_count=1, parent=root),
        ]
        _wire(search, root, *twins)

        assert search._select_best() is twins[0]


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

    def test_every_score_non_finite_falls_back_to_the_root(self):
        """No entry survives the finiteness filter: zero candidates, so the root fallback answers."""
        search = _search(["accuracy"], [0.0])
        root = _node(["a"], own=[[float("nan")]], visit_count=1)
        child = _node(["a", "b"], own=[[float("nan")]], visit_count=1, parent=root)
        _wire(search, root, child)

        assert search._select_best() is root


class TestTolerances:
    """Ties and near-duplicates are decided by ``_TIE_ATOL`` (1e-9), not by exact equality or a grid."""

    def test_twins_straddling_a_decimal_grid_boundary_are_deduped(self):
        """Merging is by tolerance, not by rounding to a decimal grid.

        These twins are 2e-15 apart but round to different 12-decimal values,
        so a round-then-unique dedupe would keep both and let the root win.
        """
        search = _search(["accuracy", "parsimony"], [0.0, 0.0])
        root = _node(["a"], own=[[0.50, 0.98]], visit_count=1)
        twins = [
            _node(["a", "b", "c"], own=[[0.85, 0.940000000000499]], visit_count=1, parent=root),
            _node(["a", "b", "d"], own=[[0.85, 0.940000000000501]], visit_count=1, parent=root),
        ]
        _wire(search, root, *twins)

        assert search._select_best() is twins[0]

    def test_a_measurably_better_accuracy_is_not_tied_away(self):
        """One ordered pair on a 500/500 split (4e-6) is far above the tolerance and must win."""
        search = _search(["accuracy"], [0.0])
        root = _node(["a"], own=[[0.900000]], visit_count=1)
        better = _node(["a", "b", "c"], own=[[0.900005]], visit_count=1, parent=root)
        _wire(search, root, better)

        assert search._select_best() is better
