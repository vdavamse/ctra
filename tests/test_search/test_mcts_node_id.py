"""Acceptance and unit tests for issues #12 and #13: the node id is the cache key.

``MCTSSearch._make_node_id`` names the pickle ``run_agent_as_subprocess``
writes to ``output/agent_cache/{task}--{node_id}.output.pkl`` and returns on
a repeated id *without running the agent*.  Keyed on the feature names, it
gave a REFINE child (same names, refined idea) and the orchestrator's
proposer-failure skip (same plans, index advanced) their parent's id, so the
cache replayed the ancestor's output down the deep path (#12).  Built with
the per-process-salted builtin ``hash()`` it could never hit across a resume,
and a negative hash produced ``r3--...`` filenames (#13).

Contract under test: the id identifies the *evaluation* -- rollout, depth,
sibling index, parent plan content, feature names -- is byte-identical across
processes, and never raises for the stand-in outputs the suite hands the
runner.  It is a crash-recovery cache, not memoisation: the same node in a
different rollout must miss, because deep revisits are intentional.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import dill
import numpy as np
import pytest

from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSNode, MCTSSearch
from ctra.search.objectives import ObjectiveResult

from .conftest import (
    make_caching_runner,
    make_orchestrator_like_runner,
    make_refining_agent,
    make_stub_output,
)

if TYPE_CHECKING:
    from ctra.agents.data_models import AgentOutput

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHAPE = re.compile(r"r\d+-[0-9a-f]{16}")
_SUGGESTIONS = [f"Suggestion {i}" for i in range(6)]


def _config(**overrides: Any) -> MCTSConfig:
    """The deep-rollout configuration of the issue's repro (4 rollouts, depth 4, branch 3)."""
    base: dict[str, Any] = {
        "num_rollouts": 4,
        "max_depth": 4,
        "objectives": ["accuracy", "parsimony"],
        "reference_point": [0.0, 0.0],
        "deep_simulation": True,
        "adaptive_branching": False,
        "min_branch_factor": 3,
        "max_branch_factor": 3,
    }
    base.update(overrides)
    return MCTSConfig(**base)


def _search(runner: Any = None, config: MCTSConfig | None = None) -> MCTSSearch:
    return MCTSSearch(
        runner=runner if runner is not None else (lambda *_args: None),
        task="phase2",
        config=config if config is not None else _config(deep_simulation=False),
    )


def _lineage() -> tuple[MCTSNode, MCTSNode]:
    """An evaluated root and its REFINE child: same feature names, sibling index 1."""
    root = MCTSNode(features=["f0"])
    root.eval_output = make_stub_output(["f0"], 0.6, list(_SUGGESTIONS))
    child = MCTSNode(features=["f0"], parent=root, suggestion_index=1, operation="refine")
    root.children.append(child)
    return root, child


def _parent_input(node: MCTSNode) -> AgentOutput:
    """The runner input ``_call_evaluate`` builds for ``node`` (parent output + child index)."""
    assert node.parent is not None and node.parent.eval_output is not None
    return node.parent.eval_output._replace(suggestion_index=node.suggestion_index)


def _refined(output: AgentOutput) -> AgentOutput:
    """``output`` after the orchestrator's REFINE: same names, ``feature_idea`` extended."""
    plans = dict(output.feature_plans)
    name = sorted(plans)[0]
    plans[name] = plans[name]._replace(feature_idea=plans[name].feature_idea + "\n---\nrefined")
    return output._replace(feature_plans=plans)


def _digest(node_id: str) -> str:
    return node_id.split("-", 1)[1]


def _depth(node: MCTSNode) -> int:
    depth = 0
    while node.parent is not None:
        depth += 1
        node = node.parent
    return depth


def _describe(node: MCTSNode) -> str:
    return f"depth={_depth(node)} idx={node.suggestion_index} op={node.operation_detail[:32]!r}"


def _aliased_pairs(search: MCTSSearch) -> list[str]:
    """Distinct tree nodes holding the *same* ``AgentOutput`` object: a replayed evaluation."""
    nodes = [n for n in search.all_nodes if n.eval_output is not None]
    return [
        f"{_describe(a)} ~ {_describe(b)}"
        for i, a in enumerate(nodes)
        for b in nodes[i + 1 :]
        if a.eval_output is b.eval_output
    ]


# ---------------------------------------------------------------------------
# Unit: what the key must and must not distinguish
# ---------------------------------------------------------------------------


def test_refine_child_key_differs_from_parent() -> None:
    search = _search()
    root, child = _lineage()
    assert search._make_node_id(child, 0, _parent_input(child)) != search._make_node_id(
        root, 0, None
    )


def test_refined_parent_plans_change_key() -> None:
    """Same child node, same rollout; only the parent's plan *content* differs (REFINE chain)."""
    search = _search()
    _root, child = _lineage()
    parent_input = _parent_input(child)
    assert search._make_node_id(child, 0, parent_input) != search._make_node_id(
        child, 0, _refined(parent_input)
    )


def test_failure_skip_child_does_not_reuse_parent_key() -> None:
    """Proposer-failure skip: plans rebuilt unchanged, index advanced, same sibling index.

    ``OrchestratorLikeRunner`` returns content-identical plans for every
    input, so parent and child agree on rollout, sibling index, parent plan
    digest and feature names.  Only their depth tells them apart.
    """
    runner = make_caching_runner(
        make_orchestrator_like_runner(_SUGGESTIONS, initial_features=["f0"])
    )
    search = _search(runner)
    root = MCTSNode(features=["f0"])
    search._root, search._all_nodes = root, [root]
    search._call_evaluate(root, rollout=0)

    parent = MCTSNode(features=["f0"], parent=root, suggestion_index=1)
    root.children.append(parent)
    search._all_nodes.append(parent)
    search._call_evaluate(parent, rollout=0)

    child = MCTSNode(features=["f0"], parent=parent, suggestion_index=1)
    parent.children.append(child)
    search._all_nodes.append(child)
    search._call_evaluate(child, rollout=0)

    assert runner.hits == [], f"cache hits: {runner.hits}; aliased pairs: {_aliased_pairs(search)}"
    assert len(set(runner.keys)) == 3, runner.keys
    assert child.eval_output is not parent.eval_output


def test_sibling_keys_differ() -> None:
    search = _search()
    root, _child = _lineage()
    first = MCTSNode(features=["f0"], parent=root, suggestion_index=0)
    second = MCTSNode(features=["f0"], parent=root, suggestion_index=1)
    assert search._make_node_id(first, 0, _parent_input(first)) != search._make_node_id(
        second, 0, _parent_input(second)
    )


def test_same_node_different_rollout_differs() -> None:
    """R6, the anti-over-fix: revisiting a node in a later rollout is intentional exploration.

    The digest itself must carry the rollout; the ``r{n}-`` prefix is a
    readable label, not the identity, so a prefix change can never quietly
    turn the cache into cross-rollout memoisation.
    """
    search = _search()
    _root, child = _lineage()
    parent_input = _parent_input(child)
    key_r0 = search._make_node_id(child, 0, parent_input)
    key_r1 = search._make_node_id(child, 1, parent_input)
    assert key_r0 != key_r1
    assert _digest(key_r0) != _digest(key_r1), (key_r0, key_r1)


def test_key_is_feature_order_independent() -> None:
    search = _search()
    assert search._make_node_id(MCTSNode(features=["b", "a"]), 0, None) == search._make_node_id(
        MCTSNode(features=["a", "b"]), 0, None
    )


def test_root_key_is_stable() -> None:
    """Same root, same rollout, no parent: identical across calls and search instances."""
    root = MCTSNode(features=["f0", "f1"])
    first, second = _search(), _search()
    keys = {
        first._make_node_id(root, 0, None),
        first._make_node_id(root, 0, None),
        second._make_node_id(root, 0, None),
    }
    assert len(keys) == 1, keys


def test_key_shape_has_no_double_dash() -> None:
    """``r{rollout}-{16 hex}``: a negative ``hash()`` used to yield ``r3--...`` filenames (#13)."""
    search = _search()
    root, child = _lineage()
    keys = [
        search._make_node_id(root, 0, None),
        search._make_node_id(child, 0, _parent_input(child)),
        search._make_node_id(child, 12, _parent_input(child)),
        search._make_node_id(MCTSNode(features=[]), 7, None),
    ]
    for key in keys:
        assert _SHAPE.fullmatch(key), key
        assert "--" not in key, key


_STAND_INS = [
    pytest.param(lambda: None, id="none"),
    pytest.param(Mock, id="mock"),
    pytest.param(lambda: SimpleNamespace(features=["f0"]), id="plain-object"),
    pytest.param(
        lambda: ObjectiveResult(values=np.array([0.5, 0.5]), names=["a", "p"], details={}),
        id="objective-result",
    ),
    pytest.param(lambda: make_stub_output(["f0"])._replace(feature_plans={}), id="plans-empty"),
    pytest.param(lambda: make_stub_output(["f0"])._replace(feature_plans=None), id="plans-none"),
    pytest.param(
        lambda: make_stub_output(["f0"])._replace(feature_plans={"f0": "not-a-plan"}),
        id="plans-garbage",
    ),
]


@pytest.mark.parametrize("stand_in", _STAND_INS)
def test_stand_in_parent_outputs_never_raise(stand_in: Any) -> None:
    """R9: a raise here lands in ``search()``'s rollout try/except and silently skips the rollout."""
    search = _search()
    _root, child = _lineage()
    key = search._make_node_id(child, 0, stand_in())
    assert _SHAPE.fullmatch(key), key


def test_mock_suggestion_index_never_raises() -> None:
    """Issue #14's write-back leaves a Mock runner's ``suggestion_index`` on the node."""
    search = _search()
    _root, child = _lineage()
    child.suggestion_index = Mock()
    key = search._make_node_id(child, 0, Mock())
    assert _SHAPE.fullmatch(key), key


def test_root_and_opaque_parent_child_differ() -> None:
    """No parent and an unhashable parent are different situations, not one sentinel."""
    search = _search()
    no_parent = MCTSNode(features=["f0"])
    opaque_parent = MCTSNode(features=["f0"])
    assert search._make_node_id(no_parent, 0, None) != search._make_node_id(
        opaque_parent, 0, make_stub_output(["f0"])._replace(feature_plans={})
    )


# ---------------------------------------------------------------------------
# Cross-process: #13 and the resume the cache exists for
# ---------------------------------------------------------------------------

_SUBPROCESS_PRELUDE = """
import sys
sys.path.insert(0, {root!r})
from tests.test_search.conftest import make_stub_output
from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSNode, MCTSSearch
cfg = MCTSConfig(
    num_rollouts=1, max_depth=2, objectives=["accuracy", "parsimony"],
    reference_point=[0.0, 0.0], deep_simulation=False,
)
"""

_KEY_CODE = (
    _SUBPROCESS_PRELUDE
    + """
search = MCTSSearch(runner=lambda *a: None, task="phase2", config=cfg)
root = MCTSNode(features=["f0"])
root.eval_output = make_stub_output(["f0"])
child = MCTSNode(features=["f1", "f0"], parent=root, suggestion_index=2)
print(search._make_node_id(child, 3, root.eval_output._replace(suggestion_index=2)))
"""
)

_RESUME_CODE = (
    _SUBPROCESS_PRELUDE
    + """
import dill, json
from pathlib import Path
from tests.test_search.conftest import make_caching_runner
runner = make_caching_runner(cache_dir=Path({cache_dir!r}))
search = MCTSSearch(runner=runner, task="phase2", config=cfg)
root = dill.loads(Path({snapshot!r}).read_bytes())
child = root.children[0]
search._root, search._all_nodes = root, [root, child]
search._call_evaluate(child, rollout=5)
print(json.dumps({{
    "events": runner.events,
    "agent_invocations": runner.inner.invocations,
    "f0_idea": child.eval_output.feature_plans["f0"].feature_idea,
}}))
"""
)


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.skipif(
    "PYTHONHASHSEED" in os.environ,
    reason="a fixed PYTHONHASHSEED hides the per-process salt of builtin hash() (#13)",
)
def test_key_is_stable_across_processes() -> None:
    code = _KEY_CODE.format(root=str(_REPO_ROOT))
    first = _run_python(code).stdout.strip()
    second = _run_python(code).stdout.strip()
    assert _SHAPE.fullmatch(first), first
    assert first == second, (first, second)


def test_resume_hits_pre_crash_entry_for_same_rollout(tmp_path: Path) -> None:
    """R7: a checkpoint is written *before* the evaluation a crash loses.

    The resumed process re-runs the same rollout index over the same
    on-disk cache and must find the pre-crash pickle instead of paying for
    the agent again.
    """
    cache_dir = tmp_path / "agent_cache"
    agent = make_refining_agent()
    runner = make_caching_runner(agent, cache_dir=cache_dir)
    search = _search(runner, _config(deep_simulation=False, num_rollouts=6))
    root = MCTSNode(features=["f0"])
    root.eval_output = agent.run(None)
    root.visit_count = 1
    child = MCTSNode(features=["f0"], parent=root, suggestion_index=1)
    root.children.append(child)
    search._root, search._all_nodes = root, [root, child]

    snapshot = tmp_path / "tree.dill"
    snapshot.write_bytes(dill.dumps(root))
    search._call_evaluate(child, rollout=5)
    assert runner.events == [(runner.keys[0], "MISS")], runner.events
    pre_crash_key = runner.keys[0]
    pre_crash_idea = child.eval_output.feature_plans["f0"].feature_idea

    code = _RESUME_CODE.format(
        root=str(_REPO_ROOT), cache_dir=str(cache_dir), snapshot=str(snapshot)
    )
    resumed = json.loads(_run_python(code).stdout)
    assert resumed == {
        "events": [[pre_crash_key, "HIT"]],
        "agent_invocations": 0,
        "f0_idea": pre_crash_idea,
    }, resumed


# ---------------------------------------------------------------------------
# Acceptance: the issue's deep-rollout scenario
# ---------------------------------------------------------------------------


def test_deep_search_with_refining_agent_never_aliases() -> None:
    """The issue's repro: deep rollouts, a name-preserving REFINE, a write-through cache."""
    runner = make_caching_runner()
    agent = runner.inner
    search = _search(runner, _config())
    search.search(initial_features=["f0"])

    aliased = _aliased_pairs(search)
    assert runner.hits == [], (
        f"cache hits within the search: {runner.hits}; aliased pairs: {aliased}"
    )
    assert aliased == [], f"distinct nodes sharing one AgentOutput: {aliased}"
    assert agent.invocations >= _config().num_rollouts, agent.invocations
    assert agent.invocations == runner.invocations == len(set(runner.keys)), (
        agent.invocations,
        runner.invocations,
        runner.keys,
    )


def test_identical_plan_rebuild_does_not_alias_parent_and_child() -> None:
    """Content-identical plans at every depth (the orchestrator's proposer-failure skip)."""
    inner = make_orchestrator_like_runner(_SUGGESTIONS, initial_features=["f0"])
    runner = make_caching_runner(inner)
    search = _search(runner, _config())
    search.search(initial_features=["f0"])

    aliased = _aliased_pairs(search)
    assert runner.hits == [], (
        f"cache hits within the search: {runner.hits}; aliased pairs: {aliased}"
    )
    assert aliased == [], f"distinct nodes sharing one AgentOutput: {aliased}"
    assert inner.evaluations >= _config().num_rollouts, inner.evaluations
    assert inner.evaluations == runner.invocations == len(set(runner.keys)), (
        inner.evaluations,
        runner.invocations,
        runner.keys,
    )
