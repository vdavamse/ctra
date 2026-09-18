"""Acceptance tests for issue #7: suggestion exhaustion is visible to the search layer.

``AgentOutput.suggestions_exhausted`` (PR #8) made the state *nameable*, but
``MCTSSearch`` never asked: children could be born with a ``suggestion_index``
past their parent's suggestion list (custom ``expand_fn``s), and a node whose
counter had run past the end was re-evaluated every time it was selected — a
full subprocess round-trip for a guaranteed no-op whose duplicate reward was
then backpropagated.

Every test here uses ``make_orchestrator_like_runner``; ``make_stub_runner``
hard-codes ``suggestion_index=0`` and therefore cannot reach exhaustion.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from ctra.agents.data_models import AgentOutput, EvalOutput
from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSNode, MCTSSearch

from .conftest import OrchestratorLikeRunner, make_orchestrator_like_runner, make_stub_output


def _config(**overrides: Any) -> MCTSConfig:
    base: dict[str, Any] = {
        "num_rollouts": 6,
        "max_depth": 3,
        "objectives": ["accuracy", "parsimony"],
        "reference_point": [0.0, 0.0],
        "deep_simulation": True,
        "adaptive_branching": False,
        "min_branch_factor": 2,
        "max_branch_factor": 2,
        "max_features": 20,
    }
    base.update(overrides)
    return MCTSConfig(**base)


def _wide_expand(node: MCTSNode, max_children: int = 4) -> list[tuple[list[str], str, str]]:
    """Retraining-style expander: ``max_children`` candidates, blind to suggestions."""
    return [(list(node.features), "suggestion", f"cand_{i}") for i in range(max_children)]


def _with_suggestions(output: AgentOutput, suggestions: list[str]) -> AgentOutput:
    """Return ``output`` with the best model's suggestion list replaced (EvalOutput is frozen)."""
    eval_result = output.eval_outputs["stub"].model_eval_result
    replaced = EvalOutput(model_eval_result=eval_result, suggestions=list(suggestions))
    return output._replace(eval_outputs={"stub": replaced})


class _DeadEndRunner(OrchestratorLikeRunner):
    """Orchestrator-faithful runner whose *children* come back without suggestions.

    ``FeatureProposer`` deliberately degrades to an empty suggestion, and an
    output with no suggestions cannot be expanded (``_suggestion_expand``
    returns ``[]``).  ``search()`` then keeps the visited leaf as the node to
    simulate, which is the one path that re-evaluates a node whose
    ``suggestion_index`` the write-back has already advanced past its parent's
    list — i.e. the path issue #7's evaluation guard exists for.
    """

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        output = super().__call__(node_id, task, previous_output)
        if previous_output is None:
            return output
        return _with_suggestions(output, [])


class _SkipCountingSearch(MCTSSearch):
    """Records every ``_call_evaluate`` result, in order and per rollout.

    ``skips`` proves the guard fired; ``returned_by_rollout`` lets a test
    derive which rollouts evaluated nothing from the evaluation results
    themselves, independently of the ``on_rollout`` callback under test (one
    rollout can contain several skips, or a skip followed by a real
    evaluation, so a skip count alone says nothing about the callback).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skips = 0
        self.returned: list[Any] = []
        self.returned_by_rollout: dict[int, list[Any]] = {}

    def _call_evaluate(self, node: MCTSNode, rollout: int) -> Any:
        result = super()._call_evaluate(node, rollout)
        if result is None:
            self.skips += 1
        self.returned.append(result)
        self.returned_by_rollout.setdefault(rollout, []).append(result)
        return result


def _subtree_history(node: MCTSNode) -> int:
    return len(node.objective_history) + sum(_subtree_history(c) for c in node.children)


class _BirthRecordingSearch(MCTSSearch):
    """Records every child's exhaustion state at the moment it is created.

    End-of-search state cannot be used for this invariant: a successfully
    evaluated node has its counter advanced by the write-back in
    ``_call_evaluate`` (issue #14), so evaluated last-index children are
    legitimately past the end afterwards.  Birth is the moment R2 governs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.born: list[tuple[int, int | None, bool]] = []

    def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        children = super()._expand(node)
        count = self._suggestion_count(node)
        self.born.extend((c.suggestion_index, count, self._is_exhausted(c)) for c in children)
        return children


# ---------------------------------------------------------------------------
# Expansion-time bound (R2)
# ---------------------------------------------------------------------------


def test_children_are_never_born_exhausted() -> None:
    """A 4-candidate custom expander against a 2-suggestion parent yields 2 children."""
    runner = make_orchestrator_like_runner(["s0", "s1"], initial_features=["f0"])
    search = _BirthRecordingSearch(
        runner=runner,
        task="phase2",
        expand_fn=_wide_expand,
        config=_config(num_rollouts=3, min_branch_factor=4, max_branch_factor=4),
    )
    search.search(initial_features=["f0"])

    assert search.born, "the search never expanded anything"
    indices = [idx for idx, _count, _ in search.born]
    assert max(indices) < 2, f"child indices past the 2 suggestions: {sorted(indices)}"
    assert all(count == 2 for _, count, _ in search.born)
    assert not any(exhausted for _, _, exhausted in search.born), (
        f"children born exhausted: {sum(1 for _, _, e in search.born if e)}/{len(search.born)}"
    )
    assert runner.exhausted_replays == 0
    # Every child was born under an expanded node, and each expansion was
    # capped at the suggestion count, never at the expander's 4.
    for node in search.all_nodes:
        assert len(node.children) <= 2


def test_default_expander_branch_factor_unchanged() -> None:
    """The default expander already bounds itself; the cap must not shrink it further."""
    runner = make_orchestrator_like_runner(
        [f"s{i}" for i in range(5)],
        initial_features=["f0"],
    )
    search = MCTSSearch(
        runner=runner,
        task="phase2",
        config=_config(min_branch_factor=5, max_branch_factor=5),
    )
    root = MCTSNode(features=["f0"], total_reward=np.zeros(2))
    root.eval_output = runner("root", "phase2", None)
    root.visit_count = 1
    search._root, search._all_nodes = root, [root]

    children = search._expand(root)

    assert len(children) == 5
    assert [c.suggestion_index for c in children] == [0, 1, 2, 3, 4]


def test_empty_suggestions_do_not_truncate_expansion() -> None:
    """Empty suggestions are not exhaustion (``data_models.py``): the cap must not apply."""

    class _NoSuggestionRunner:
        def __call__(self, node_id: str, task: Any, previous_output: Any) -> Any:
            output = make_stub_output(["f0", "f1", "f2"], 0.7, ["placeholder"])
            eval_result = output.eval_outputs["stub"].model_eval_result
            empty = EvalOutput(model_eval_result=eval_result, suggestions=[])
            return output._replace(eval_outputs={"stub": empty})

    def _remove_one(node: MCTSNode, max_children: int = 4) -> list[tuple[list[str], str, str]]:
        return [
            ([f for f in node.features if f != feat], "remove", f"remove:{feat}")
            for feat in node.features[:max_children]
        ]

    search = MCTSSearch(
        runner=_NoSuggestionRunner(),
        task="phase2",
        expand_fn=_remove_one,
        config=_config(min_branch_factor=3, max_branch_factor=3),
    )
    root = MCTSNode(features=["f0", "f1", "f2"], total_reward=np.zeros(2))
    root.eval_output = search._runner("root", "phase2", None)
    root.visit_count = 1
    search._root, search._all_nodes = root, [root]

    assert search._suggestion_count(root) is None
    assert len(search._expand(root)) == 3


def test_non_agentoutput_expander_keeps_branch_factor() -> None:
    """A runner returning a plain object (no ``get_best_eval_output``) keeps its branch factor.

    ``mlops/retraining.py`` returns an ``ObjectiveResult`` and expands by
    removing one feature at a time; the cap must be a no-op there.
    """

    class _PlainResult:
        def __init__(self, features: list[str]) -> None:
            self.features = features

    class _PlainRunner:
        def __call__(self, node_id: str, task: Any, previous_output: Any) -> _PlainResult:
            return _PlainResult(["f0", "f1", "f2", "f3", "f4", "f5"])

    def _remove_one(node: MCTSNode, max_children: int = 4) -> list[tuple[list[str], str, str]]:
        return [
            ([f for f in node.features if f != feat], "remove", f"remove:{feat}")
            for feat in node.features[:max_children]
        ]

    features = ["f0", "f1", "f2", "f3", "f4", "f5"]
    search = MCTSSearch(
        runner=_PlainRunner(),
        task="retrain",
        expand_fn=_remove_one,
        config=_config(min_branch_factor=4, max_branch_factor=4),
    )
    root = MCTSNode(features=list(features), total_reward=np.zeros(2))
    root.eval_output = _PlainResult(list(features))
    root.visit_count = 1
    search._root, search._all_nodes = root, [root]

    children = search._expand(root)

    assert search._suggestion_count(root) is None
    assert len(children) == min(len(features), 4) == 4
    assert all(not search._is_exhausted(c) for c in children)


# ---------------------------------------------------------------------------
# Evaluation-time guard (R1, R6, R7, R9)
# ---------------------------------------------------------------------------


def test_exhausted_node_is_not_re_evaluated() -> None:
    """A last-index child is evaluated once; every later attempt is a ``None`` skip.

    Before the guard this cost four runner invocations (three of them no-op
    replays that the orchestrator early-skipped) and backpropagated three
    duplicate reward vectors.
    """
    runner = make_orchestrator_like_runner(["s0", "s1"], initial_features=["f0"])
    search = MCTSSearch(
        runner=runner,
        task="phase2",
        config=_config(num_rollouts=1, deep_simulation=False),
    )
    root = MCTSNode(features=["f0"], total_reward=np.zeros(2))
    root.eval_output = runner("root", "phase2", None)
    root.visit_count = 1
    search._root, search._all_nodes = root, [root]
    child = MCTSNode(features=["f0"], parent=root, suggestion_index=1, total_reward=np.zeros(2))
    root.children.append(child)
    search._all_nodes.append(child)

    results = [search._call_evaluate(child, rollout=k) for k in range(4)]

    assert runner.evaluations == 2  # root seed + exactly one child evaluation
    first = results[0]
    assert isinstance(first, np.ndarray) and first.shape == (2,)
    assert [r is None for r in results] == [False, True, True, True]
    assert runner.seen_indices == [1]
    assert runner.lm_calls == 3
    assert runner.exhausted_replays == 0
    # The write-back advanced the counter past the two suggestions, which is
    # what the guard reads; the node itself carries no new state.
    assert child.suggestion_index == 2
    assert search._is_exhausted(child)


@pytest.mark.parametrize("deep", [True, False], ids=["deep", "shallow"])
def test_skip_does_not_backpropagate_or_count(deep: bool) -> None:
    """A skipped evaluation leaves no trace: no history entry, no visit, no reward."""
    runner = _DeadEndRunner(["s0", "s1"], initial_features=["f0"])
    search = _SkipCountingSearch(
        runner=runner,
        task="phase2",
        config=_config(num_rollouts=8, deep_simulation=deep),
    )
    search.search(initial_features=["f0"])

    assert search.skips > 0, "scenario did not exercise the guard"
    assert runner.exhausted_replays == 0
    # One runner invocation <-> one history entry, everywhere in the tree.
    assert sum(len(n.objective_history) for n in search.all_nodes) == runner.evaluations
    # visit_count is exactly the number of evaluations in the node's subtree
    # (the root's setup evaluation counts as one); a phantom backprop would
    # push some node above that.
    for node in search.all_nodes:
        assert node.visit_count == _subtree_history(node), node.operation_detail
    # Rewards are constant, so total_reward must be visit_count * one vector.
    for node in search.all_nodes:
        if node.visit_count:
            np.testing.assert_allclose(node.mean_reward, node.objective_history[0].values)


@pytest.mark.parametrize("deep", [True, False], ids=["deep", "shallow"])
def test_on_rollout_never_receives_none(deep: bool) -> None:
    """The callback sees one ``(2,)`` vector per evaluated rollout and nothing for skips."""
    runner = _DeadEndRunner(["s0", "s1"], initial_features=["f0"])
    search = _SkipCountingSearch(
        runner=runner,
        task="phase2",
        config=_config(num_rollouts=8, deep_simulation=deep),
    )
    calls: list[tuple[int, MCTSNode, Any]] = []

    def on_rollout(rollout: int, node: MCTSNode, objective_vector: Any) -> None:
        calls.append((rollout, node, objective_vector))
        # What scripts/train_mcts.py does with it.
        assert float(objective_vector[0]) >= 0.0 and float(objective_vector[1]) >= 0.0

    search.search(initial_features=["f0"], on_rollout=on_rollout)

    assert search.skips > 0, "scenario did not exercise the guard"
    # Derived from the evaluation results, not from the callback under test.
    assert set(search.returned_by_rollout) == set(range(8))
    noop = {r for r, res in search.returned_by_rollout.items() if all(x is None for x in res)}
    assert noop, "scenario produced no rollout that evaluated nothing"
    assert {r for r, _, _ in calls} == set(range(8)) - noop
    for _rollout, _node, vector in calls:
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (2,)
    assert [r for r, _, _ in calls] == sorted({r for r, _, _ in calls})


@pytest.mark.parametrize("deep", [True, False], ids=["deep", "shallow"])
def test_search_terminates_with_usable_best_when_frontier_exhausts(deep: bool) -> None:
    """A 1-suggestion tree exhausts after one child; the remaining rollouts must be cheap."""
    runner = _DeadEndRunner(["only"], initial_features=["f0"])
    search = _SkipCountingSearch(
        runner=runner,
        task="phase2",
        config=_config(
            num_rollouts=10, max_branch_factor=1, min_branch_factor=1, deep_simulation=deep
        ),
    )

    best = search.search(initial_features=["f0"])

    assert best.visit_count > 0
    assert best in search.all_nodes
    assert search.skips > 0
    # Root setup + the single child, and nothing for the exhausted rollouts
    # (a per-rollout evaluation would have cost eleven).
    assert runner.evaluations == 2
    assert runner.lm_calls == 3
    assert runner.exhausted_replays == 0
    assert all(not isinstance(r, np.ndarray) or r.shape == (2,) for r in search.returned)


def test_skip_logged_once_per_node(caplog: pytest.LogCaptureFixture) -> None:
    """One INFO record per skipped node, DEBUG thereafter, however often it is skipped."""
    runner = _DeadEndRunner(["only"], initial_features=["f0"])
    search = _SkipCountingSearch(
        runner=runner,
        task="phase2",
        config=_config(
            num_rollouts=10, max_branch_factor=1, min_branch_factor=1, deep_simulation=False
        ),
    )
    with caplog.at_level(logging.DEBUG, logger="ctra.search.mcts"):
        search.search(initial_features=["f0"])

    skip_records = [r for r in caplog.records if "Skipping evaluation of node" in r.getMessage()]
    assert search.skips > 1, "need repeated skips of the same node to test the dedupe"
    assert len(skip_records) == search.skips
    assert sum(1 for r in skip_records if r.levelno == logging.INFO) == 1
    assert sum(1 for r in skip_records if r.levelno == logging.DEBUG) == search.skips - 1
    # Dedupe state lives on the search object (R8).
    assert len(search._skip_logged) == 1


def test_log_skip_survives_checkpoint_without_skip_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A search unpickled from before this change has no ``_skip_logged``; it must still log."""
    runner = make_orchestrator_like_runner(["s0"], initial_features=["f0"])
    search = MCTSSearch(runner=runner, task="phase2", config=_config())
    del search._skip_logged
    node = MCTSNode(features=["f0"], suggestion_index=1)

    with caplog.at_level(logging.DEBUG, logger="ctra.search.mcts"):
        search._log_skip(node)
        search._log_skip(node)

    levels = [r.levelno for r in caplog.records if "Skipping evaluation" in r.getMessage()]
    assert levels == [logging.INFO, logging.DEBUG]
    assert search._skip_logged == {id(node)}


def test_resume_resets_skip_logged_so_first_skip_is_info(caplog: pytest.LogCaptureFixture) -> None:
    """A resumed search discards the pickled dedupe set: stale ``id()`` values must not silence a node.

    The checkpoint's set holds ids from the previous process; a fresh node
    can collide with one, and without the reset its first skip would go to
    DEBUG — a missing INFO line, not an extra one.
    """
    runner = make_orchestrator_like_runner(["s0"], initial_features=["f0"])
    config = _config(num_rollouts=3, deep_simulation=False)
    search = MCTSSearch(runner=runner, task="phase2", config=config)
    root = MCTSNode(features=["f0"], total_reward=np.zeros(2))
    root.eval_output = runner("root", "phase2", None)
    root.visit_count = 1
    root.objective_history.append(search._wrap_result(np.array([0.7, 0.9])))
    search._root, search._all_nodes = root, [root]
    fresh = MCTSNode(features=["f0"], parent=root, suggestion_index=1, operation_detail="fresh")
    # Simulate the stale checkpoint: the set already "knows" this node's id.
    search._skip_logged = {id(fresh), 12345}

    # Resume with no rollouts left to run: only the resume branch executes.
    search.search(initial_features=["f0"], start_rollout=config.num_rollouts)
    assert search._skip_logged == set()

    with caplog.at_level(logging.DEBUG, logger="ctra.search.mcts"):
        search._log_skip(fresh)
        search._log_skip(fresh)

    levels = [r.levelno for r in caplog.records if "Skipping evaluation" in r.getMessage()]
    assert levels == [logging.INFO, logging.DEBUG]
    assert search._skip_logged == {id(fresh)}
    assert runner.evaluations == 1, "the resume must not re-evaluate the root"


def test_deep_simulation_filters_exhausted_children_before_picking() -> None:
    """``_simulate_deep`` never asks the rng about an exhausted child.

    The unvisited-first policy would otherwise pick the exhausted (never
    visited) sibling on every rollout, skip it, and stop the path before the
    viable sibling is ever evaluated.
    """
    invocations: list[list[str]] = []

    def evaluate(features: list[str]) -> np.ndarray:
        invocations.append(list(features))
        return np.array([0.7, 0.9])

    from .conftest import make_stub_runner

    search = _SkipCountingSearch(
        runner=make_stub_runner(evaluate, initial_features=["f0"]),
        task="phase2",
        config=_config(num_rollouts=1, max_depth=1),
    )
    root = MCTSNode(features=["f0"], total_reward=np.zeros(2))
    root.eval_output = search._runner("root", "phase2", None)  # 5 suggestions, index 0
    root.visit_count = 1
    viable = MCTSNode(features=["f0"], parent=root, suggestion_index=0, total_reward=np.zeros(2))
    exhausted = MCTSNode(
        features=["f0"], parent=root, suggestion_index=99, total_reward=np.zeros(2)
    )
    root.children.extend([viable, exhausted])
    search._root, search._all_nodes = root, [root, viable, exhausted]
    assert search._is_exhausted(exhausted) and not search._is_exhausted(viable)
    invocations.clear()

    results = [search._simulate_deep(root, rollout=r) for r in range(4)]

    assert exhausted.visit_count == 0 and exhausted.objective_history == []
    assert viable.visit_count == 4
    assert len(invocations) == 8  # root + viable child, four times over
    assert search.skips == 0
    assert all(obj is not None and obj.shape == (2,) for obj, _node in results)


def test_skipped_unevaluated_start_node_is_not_expanded() -> None:
    """A never-evaluated node skipped as exhausted must not be expanded (review M1).

    Its ``eval_output`` is ``None``, so any child's ``_call_evaluate`` would
    build ``parent_output=None`` — a root-style fresh initialization mid-tree.
    R6 scenario: the parent is re-evaluated to fewer suggestions after its
    children were born, retroactively exhausting the second child.
    """
    runner = make_orchestrator_like_runner(["s0", "s1"], initial_features=["f0"])
    search = _SkipCountingSearch(
        runner=runner,
        task="phase2",
        expand_fn=_wide_expand,
        config=_config(num_rollouts=1, max_depth=3, min_branch_factor=4, max_branch_factor=4),
    )
    root = MCTSNode(features=["f0"], total_reward=np.zeros(2))
    root.eval_output = runner("root", "phase2", None)
    root.visit_count = 1
    search._root, search._all_nodes = root, [root]
    children = search._expand(root)
    assert len(children) == 2
    root.eval_output = _with_suggestions(root.eval_output, ["only"])
    exhausted = children[1]
    assert search._is_exhausted(exhausted) and exhausted.eval_output is None
    invocations_before = runner.evaluations

    best_obj, best_node = search._simulate_deep(exhausted, rollout=0)

    assert best_obj is None and best_node is exhausted
    assert search.skips == 1
    assert runner.evaluations == invocations_before, "the runner was called for a skipped path"
    assert runner.seen_indices == [], "a child was evaluated with previous_output=None"
    assert exhausted.children == []
    assert search.all_nodes == [root, *children]
    assert exhausted.visit_count == 0 and exhausted.objective_history == []


def test_shallow_rollout_with_all_fresh_children_exhausted_evaluates_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Shallow ``search()``: if every fresh child is exhausted, the rollout evaluates nothing.

    Unreachable with the real predicate (children are born capped in the
    same rollout), so the predicate is forced: every non-root node counts
    as exhausted.  The visited leaf must not be re-evaluated in their place.
    """
    runner = make_orchestrator_like_runner(["s0", "s1"], initial_features=["f0"])
    search = _SkipCountingSearch(
        runner=runner,
        task="phase2",
        config=_config(num_rollouts=1, deep_simulation=False),
    )
    # One rollout: the root (visited) is selected and expanded.  A second
    # rollout would legitimately select an unvisited child and evaluate it.
    monkeypatch.setattr(search, "_is_exhausted", lambda node: node.parent is not None)
    calls: list[int] = []

    with caplog.at_level(logging.DEBUG, logger="ctra.search.mcts"):
        best = search.search(initial_features=["f0"], on_rollout=lambda r, n, v: calls.append(r))

    root = search.root
    assert root is not None and best is root
    assert runner.evaluations == 1, "only the root setup evaluation may run"
    assert search.returned_by_rollout == {0: [search.returned[0]]}  # root setup only
    assert search.skips == 0  # nothing reached the guard; nothing was evaluated
    assert calls == []
    assert root.visit_count == 1 and len(root.objective_history) == 1
    assert len(root.children) == 2
    for child in root.children:
        assert child.visit_count == 0 and child.objective_history == []
    exhausted_msgs = [
        r
        for r in caplog.records
        if "every child of the selected leaf is exhausted" in r.getMessage()
    ]
    assert [r.levelno for r in exhausted_msgs] == [logging.INFO]
