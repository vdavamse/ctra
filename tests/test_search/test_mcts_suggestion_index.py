"""Acceptance tests for issue #14: the write-back only honours an *advance*.

``_call_evaluate`` stores the runner's ``AgentOutput`` on the node and then
reconciles the node's ``suggestion_index`` with the one the output carries.
``Agent.forward`` returns three shapes: ``sent + 1`` on a skipped iteration
(proposer failure, unhandled operation), ``sent`` on the exhausted early-skip,
and a hard-coded ``0`` on every *successful* iteration.  Adopting that ``0``
reset the index assigned at expansion, so a re-evaluation (deep-mode revisit,
or a leaf that could not be expanded) asked the proposer for suggestion 0
instead of the one this node stands for.

These tests drive ``_call_evaluate`` directly on a hand-built two-node tree:
the clobber is per-evaluation state, and a full ``search()`` hides which
suggestion each runner call actually received.
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
    pytest.param(SimpleNamespace(eval_outputs={}, suggestion_index=True), id="bool"),
    pytest.param(SimpleNamespace(eval_outputs={}, suggestion_index="1"), id="string"),
]


@pytest.mark.parametrize("output", _STAND_INS)
def test_non_integer_returned_index_leaves_the_node_untouched(output: Any) -> None:
    """R5: ``_call_evaluate`` runs outside ``search()``'s try/except for the root."""

    def runner(node_id: str, task: Any, previous_output: Any) -> Any:
        if previous_output is None:
            return make_stub_output(["f0"], 0.7, ["s0", "s1"])
        return output

    search, _root, child = _seeded(runner, child_index=0)

    result = search._call_evaluate(child, rollout=0)

    assert isinstance(result, np.ndarray)
    assert child.suggestion_index == 0
    assert type(child.suggestion_index) is int
