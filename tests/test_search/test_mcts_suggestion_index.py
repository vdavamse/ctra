"""Acceptance tests for issue #14: the write-back only honours an *advance*.

``_call_evaluate`` stores the runner's ``AgentOutput`` on the node and then
reconciles the node's ``suggestion_index`` with the one the output carries.
``Agent.forward`` returns three shapes: ``sent + 1`` on a skipped iteration
(proposer failure, unhandled operation), ``sent`` on the exhausted early-skip,
and a hard-coded ``0`` on every *successful* iteration.  Adopting that ``0``
reset the index assigned at expansion, so a re-evaluation (deep-mode revisit,
or a leaf that could not be expanded) asked the proposer for suggestion 0
instead of the one this node stands for.

Most tests drive ``_call_evaluate`` directly on a hand-built two-node tree:
the clobber is per-evaluation state, and a full ``search()`` hides which
suggestion each runner call actually received.  The last section pins the
behaviour change through ``search()`` itself, with a subclass that records
what each node was sent.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSNode, MCTSSearch

from .conftest import make_orchestrator_like_runner, make_refining_agent, make_stub_output

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ctra.agents.data_models import AgentOutput


def _config(**overrides: Any) -> MCTSConfig:
    base: dict[str, Any] = {
        "num_rollouts": 1,
        "max_depth": 3,
        "objectives": ["accuracy", "parsimony"],
        "reference_point": [0.0, 0.0],
        "deep_simulation": False,
        "adaptive_branching": False,
        "min_branch_factor": 2,
        "max_branch_factor": 2,
        "max_features": 20,
    }
    base.update(overrides)
    return MCTSConfig(**base)


def _seeded(runner: Any, *, child_index: int) -> tuple[MCTSSearch, MCTSNode, MCTSNode]:
    """A search whose root is already evaluated, plus one child at ``child_index``."""
    search = MCTSSearch(runner=runner, task="phase2", config=_config())
    root = MCTSNode(features=["f0"], total_reward=np.zeros(2))
    root.eval_output = runner("root", "phase2", None)
    root.visit_count = 1
    search._root, search._all_nodes = root, [root]
    child = MCTSNode(
        features=["f0"], parent=root, suggestion_index=child_index, total_reward=np.zeros(2)
    )
    root.children.append(child)
    search._all_nodes.append(child)
    return search, root, child


class _ScriptedRunner:
    """Replays a scripted sequence of ``Agent.forward`` return shapes.

    One entry per non-root call: ``"skip"`` -> ``sent + 1``
    (``orchestrator.py:396-399``/``:522-525``), ``"success"`` -> the hard-coded
    ``0`` (``:619-633``, which ``make_stub_output`` already produces),
    ``"echo"`` -> ``sent`` unchanged (the exhausted early-skip, ``:370``),
    ``"jump"`` -> ``sent + 2`` (a multi-step advance), ``"numpy"`` ->
    ``np.int64(sent + 1)``.  ``seen`` records the index of every non-root
    input, in call order.
    """

    def __init__(self, script: list[str], suggestions: list[str]) -> None:
        self.script = list(script)
        self.suggestions = list(suggestions)
        self.seen: list[int] = []

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        if previous_output is None:
            return make_stub_output(["f0"], 0.7, list(self.suggestions))
        if len(self.seen) >= len(self.script):
            pytest.fail("runner called past its script")
        shape = self.script[len(self.seen)]
        sent = previous_output.suggestion_index
        self.seen.append(sent)
        output = make_stub_output(list(previous_output.feature_plans), 0.7, list(self.suggestions))
        if shape == "success":
            return output
        returned: Any = {
            "skip": sent + 1,
            "echo": sent,
            "jump": sent + 2,
            "numpy": np.int64(sent + 1),
        }[shape]
        return output._replace(suggestion_index=returned)


# ---------------------------------------------------------------------------
# AC1/AC2: a successful evaluation leaves the assigned index alone
# ---------------------------------------------------------------------------


def test_successful_evaluation_preserves_the_assigned_index() -> None:
    """AC1: the child keeps the index expansion gave it (the output says ``0``)."""
    agent = make_refining_agent()
    search, _root, child = _seeded(agent, child_index=2)

    result = search._call_evaluate(child, rollout=0)

    assert isinstance(result, np.ndarray) and result.shape == (2,)
    assert child.suggestion_index == 2
    assert agent.seen_suggestion_indices == [2]


def test_re_evaluation_sends_the_same_suggestion_twice() -> None:
    """AC2: two evaluations of one node ask the proposer for the *same* suggestion."""
    agent = make_refining_agent()
    search, _root, child = _seeded(agent, child_index=2)

    first = search._call_evaluate(child, rollout=0)
    second = search._call_evaluate(child, rollout=1)

    assert first is not None and second is not None
    assert agent.seen_suggestion_indices == [2, 2]
    assert child.suggestion_index == 2
    assert agent.invocations == 3  # root seed + both evaluations really ran


def test_root_evaluation_leaves_the_index_at_zero() -> None:
    """The root has no parent output, so nothing is sent and nothing is adopted."""
    agent = make_refining_agent()
    search = MCTSSearch(runner=agent, task="phase2", config=_config())
    root = MCTSNode(features=["f0"], total_reward=np.zeros(2))
    search._root, search._all_nodes = root, [root]

    result = search._call_evaluate(root, rollout=0)

    assert isinstance(result, np.ndarray)
    assert root.suggestion_index == 0
    assert agent.seen_suggestion_indices == []


# ---------------------------------------------------------------------------
# The skip-advance half of the contract (R3) still holds
# ---------------------------------------------------------------------------


def test_skipped_iteration_still_advances_the_index() -> None:
    """A skip returns ``sent + 1``; the node takes it and reads exhausted (issue #7)."""
    runner = make_orchestrator_like_runner(["s0", "s1"], initial_features=["f0"])
    search, _root, child = _seeded(runner, child_index=1)

    result = search._call_evaluate(child, rollout=0)

    assert isinstance(result, np.ndarray)
    assert child.suggestion_index == 2
    assert runner.seen_indices == [1]
    assert search._is_exhausted(child)
    assert search._call_evaluate(child, rollout=1) is None
    assert runner.evaluations == 2  # root seed + one child call; the guard held


def test_a_success_after_a_skip_does_not_rewind_the_index() -> None:
    """Skip -> success -> skip: the counter is monotonic, never back to 0."""
    runner = _ScriptedRunner(["skip", "success", "skip"], ["s0", "s1"])
    search, _root, child = _seeded(runner, child_index=0)

    search._call_evaluate(child, rollout=0)
    assert child.suggestion_index == 1
    search._call_evaluate(child, rollout=1)
    assert child.suggestion_index == 1  # the success must not rewind to 0
    search._call_evaluate(child, rollout=2)
    assert child.suggestion_index == 2

    assert runner.seen == [0, 1, 1]
    assert search._is_exhausted(child)
    assert search._call_evaluate(child, rollout=3) is None


def test_a_multi_step_advance_is_honoured() -> None:
    """``>`` not ``== sent + 1``: a counter that jumps stays jumped."""
    runner = _ScriptedRunner(["jump"], ["s0", "s1", "s2"])
    search, _root, child = _seeded(runner, child_index=0)

    search._call_evaluate(child, rollout=0)

    assert child.suggestion_index == 2
    assert runner.seen == [0]


def test_an_unchanged_returned_index_is_a_no_op() -> None:
    """The exhausted early-skip returns the input index; the node is left as it was."""
    runner = _ScriptedRunner(["echo"], ["s0", "s1"])
    search, _root, child = _seeded(runner, child_index=1)

    search._call_evaluate(child, rollout=0)

    assert child.suggestion_index == 1
    assert runner.seen == [1]


def test_numpy_integer_advance_is_stored_as_a_plain_int() -> None:
    """A numpy index is honoured but normalised: the node's field stays a builtin int."""
    runner = _ScriptedRunner(["numpy"], ["s0", "s1", "s2"])
    search, _root, child = _seeded(runner, child_index=0)

    search._call_evaluate(child, rollout=0)

    assert child.suggestion_index == 1
    assert type(child.suggestion_index) is int


# ---------------------------------------------------------------------------
# R5: stand-in outputs never land on the node and never raise
# ---------------------------------------------------------------------------

_STAND_INS = [
    pytest.param(SimpleNamespace(eval_outputs={}), id="attribute-missing"),
    pytest.param(SimpleNamespace(eval_outputs={}, suggestion_index=Mock()), id="mock"),
    # ``isinstance(Mock(spec=int), int)`` is True but ``int()`` of it raises.
    pytest.param(
        SimpleNamespace(eval_outputs={}, suggestion_index=Mock(spec=int)), id="mock-spec-int"
    ),
    pytest.param(SimpleNamespace(eval_outputs={}, suggestion_index=True), id="bool"),
    pytest.param(SimpleNamespace(eval_outputs={}, suggestion_index="1"), id="string"),
]


@pytest.mark.parametrize("output", _STAND_INS)
def test_non_integer_returned_index_leaves_the_node_untouched(output: Any) -> None:
    """R5 on a child at index 0 (load-bearing for ``bool``: ``True == 1`` would advance it).

    The guard is a pure predicate on the two values, so what holds here holds
    for the root too; the root path, which is the one outside ``search()``'s
    rollout try/except, gets its own test below.
    """

    def runner(node_id: str, task: Any, previous_output: Any) -> Any:
        if previous_output is None:
            return make_stub_output(["f0"], 0.7, ["s0", "s1"])
        return output

    search, _root, child = _seeded(runner, child_index=0)

    result = search._call_evaluate(child, rollout=0)

    assert isinstance(result, np.ndarray)
    assert child.suggestion_index == 0
    assert type(child.suggestion_index) is int


@pytest.mark.parametrize("output", _STAND_INS)
def test_root_evaluation_never_raises_on_a_stand_in(output: Any) -> None:
    """R5 where it bites: the rollout-0 root evaluation runs outside ``search()``'s try/except."""

    def runner(node_id: str, task: Any, previous_output: Any) -> Any:
        return output

    search = MCTSSearch(runner=runner, task="phase2", config=_config())
    root = MCTSNode(features=["f0"], total_reward=np.zeros(2))
    search._root, search._all_nodes = root, [root]

    result = search._call_evaluate(root, rollout=0)

    assert isinstance(result, np.ndarray)
    assert root.suggestion_index == 0
    assert type(root.suggestion_index) is int


# ---------------------------------------------------------------------------
# S1: the behaviour change seen through a whole search()
# ---------------------------------------------------------------------------


class _RecordingSearch(MCTSSearch):
    """Records every ``_call_evaluate`` as ``(node, sent, stored, evaluated)``.

    The runner only ever sees a ``node_id``; the node is known here.  ``sent``
    is ``node.suggestion_index`` as the call starts — exactly what
    ``_call_evaluate`` puts on the runner input, which the tests cross-check
    against the runner's own ``seen`` — ``stored`` is the field after the
    write-back, and ``evaluated`` is False when the exhaustion guard
    (issue #7) returned ``None`` without calling the runner.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[MCTSNode, int, int, bool]] = []

    def _call_evaluate(self, node: MCTSNode, rollout: int) -> NDArray[Any] | None:
        sent = node.suggestion_index
        result = super()._call_evaluate(node, rollout)
        self.calls.append((node, sent, node.suggestion_index, result is not None))
        return result

    def evaluations_by_node(self) -> list[tuple[MCTSNode, list[tuple[int, int]]]]:
        """``(sent, stored)`` per real evaluation, grouped by non-root node in first-seen order."""
        grouped: dict[int, tuple[MCTSNode, list[tuple[int, int]]]] = {}
        for node, sent, stored, evaluated in self.calls:
            if node is self._root or not evaluated:
                continue
            grouped.setdefault(id(node), (node, []))[1].append((sent, stored))
        return list(grouped.values())

    def sent_to_runner(self) -> list[int]:
        """The index of every non-root input the runner actually received, in call order."""
        return [
            sent for node, sent, _, evaluated in self.calls if evaluated and node is not self._root
        ]


def _with_suggestions(output: AgentOutput, suggestions: list[str]) -> AgentOutput:
    """Set the suggestion list explicitly (``make_stub_output`` swaps ``[]`` for five defaults)."""
    best = output.eval_outputs["stub"]
    return output._replace(eval_outputs={"stub": best._replace(suggestions=list(suggestions))})


class _ShrinkingTreeRunner:
    """Deep-mode runner: interleaves skips and successes over a tree that stops growing.

    Every output carries two suggestions fewer than its input, so with four at
    the root the children (branch factor 2) still expand and the grandchildren
    are leaves that cannot; from then on ``search()`` re-selects them and
    ``_simulate_deep`` re-evaluates its start node.  Every ``skip_every``-th
    non-root call is a skip (``sent + 1``), the rest succeed (``0``), so single
    nodes see both shapes across rollouts.  ``exhausted_replays`` counts inputs
    the search layer should never have sent (issue #7).
    """

    def __init__(self, root_suggestions: list[str], *, skip_every: int) -> None:
        self.root_suggestions = list(root_suggestions)
        self.skip_every = skip_every
        self.seen: list[int] = []
        self.exhausted_replays = 0

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        if previous_output is None:
            return make_stub_output(["f0"], 0.7, list(self.root_suggestions))
        sent = previous_output.suggestion_index
        self.seen.append(sent)
        if previous_output.suggestions_exhausted:
            self.exhausted_replays += 1
        inherited = previous_output.eval_outputs["stub"].suggestions
        output = _with_suggestions(
            make_stub_output(list(previous_output.feature_plans), 0.7, ["placeholder"]),
            inherited[: max(len(inherited) - 2, 0)],
        )
        if len(self.seen) % self.skip_every == 0:
            return output._replace(suggestion_index=sent + 1)
        return output


class _LeafOnlyRunner:
    """Shallow-mode runner whose non-root outputs carry no suggestions.

    The root expands to two children; neither can be expanded, so from the
    third rollout on ``search()`` re-evaluates whichever child it selected
    (its shallow branch).  Every evaluation succeeds (``0``).
    """

    def __init__(self) -> None:
        self.seen: list[int] = []

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        if previous_output is None:
            return make_stub_output(["f0"], 0.7, ["s0", "s1"])
        self.seen.append(previous_output.suggestion_index)
        return _with_suggestions(
            make_stub_output(list(previous_output.feature_plans), 0.7, ["placeholder"]), []
        )


def test_deep_search_never_rewinds_a_node_below_its_birth_index() -> None:
    """Deep mode, whole run: what a node is sent only ever advances, and exhaustion sticks.

    Under the unconditional write-back a second child dropped to 0 on its
    first success (below its birth index) and a skip-then-success node
    oscillated 0 -> 1 -> 0 instead of keeping the burned suggestion.
    """
    runner = _ShrinkingTreeRunner(["s0", "s1", "s2", "s3"], skip_every=2)
    search = _RecordingSearch(
        runner=runner,
        task="phase2",
        config=_config(deep_simulation=True, num_rollouts=10, max_depth=3),
    )

    search.search(initial_features=["f0"])

    assert runner.seen == search.sent_to_runner()
    assert runner.exhausted_replays == 0
    by_node = search.evaluations_by_node()
    for node, evals in by_node:
        birth = search._lineage(node)[-1]
        received = [sent for sent, _ in evals]
        assert received == sorted(received), (search._lineage(node), received)
        assert all(stored >= sent for sent, stored in evals), (search._lineage(node), evals)
        assert node.suggestion_index >= birth, (search._lineage(node), node.suggestion_index)

    # Not vacuous: a node was re-evaluated and succeeded, and one node saw a
    # skip followed by a success (the case that used to oscillate).
    reevaluated = [evals for _, evals in by_node if len(evals) > 1]
    assert any(stored == sent for evals in reevaluated for sent, stored in evals[1:])
    assert any(
        evals[i - 1][1] > evals[i - 1][0] and evals[i][1] == evals[i][0]
        for evals in reevaluated
        for i in range(1, len(evals))
    )

    # A node that burned through its parent's suggestions was re-selected but
    # not evaluated again: only guard skips follow its last evaluation.
    exhausted = [n for n in search._all_nodes if search._is_exhausted(n)]
    assert exhausted
    guarded = 0
    for node in exhausted:
        flags = [evaluated for n, _, _, evaluated in search.calls if n is node]
        assert flags == sorted(flags, reverse=True), (search._lineage(node), flags)
        guarded += flags.count(False)
    assert guarded >= 1


def test_shallow_search_sends_a_leaf_the_same_index_on_every_re_evaluation() -> None:
    """Shallow mode: a leaf that cannot expand is re-evaluated with the index it was born at."""
    runner = _LeafOnlyRunner()
    search = _RecordingSearch(
        runner=runner,
        task="phase2",
        config=_config(deep_simulation=False, num_rollouts=6, max_depth=3),
    )

    search.search(initial_features=["f0"])

    assert runner.seen == search.sent_to_runner()
    reevaluated = [(node, evals) for node, evals in search.evaluations_by_node() if len(evals) > 1]
    assert reevaluated
    # A second child is the one a rewind to 0 would change.
    assert any(search._lineage(node)[-1] == 1 for node, _ in reevaluated)
    for node, evals in reevaluated:
        birth = search._lineage(node)[-1]
        assert [sent for sent, _ in evals] == [birth] * len(evals), (search._lineage(node), evals)
