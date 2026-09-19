"""Acceptance and unit tests for issues #12 and #13: the node id is the cache key.

``MCTSSearch._make_node_id`` names the pickle ``run_agent_as_subprocess``
writes to ``output/agent_cache/{task}--{node_id}.output.pkl`` and returns on
a repeated id *without running the agent*.  Keyed on the feature names, it
gave a REFINE child (same names, refined idea) and the orchestrator's
proposer-failure skip (same plans, index advanced) their parent's id, so the
cache replayed the ancestor's output down the deep path (#12).  Built with
the per-process-salted builtin ``hash()`` it could never hit across a resume,
and a negative hash produced ``r3--...`` filenames (#13).

Contract under test: the id identifies the *evaluation* -- rollout, lineage
(the child-position path from the root: depth and sibling index in one),
parent plan content, feature names -- is byte-identical across processes,
and never raises for the stand-in outputs the suite hands the runner.  It is
a crash-recovery cache, not memoisation: the same node in a different rollout
must miss, because deep revisits are intentional.  SELECT is seeded per
rollout so a resumed process walks the pre-crash path and finds its entries.
"""

from __future__ import annotations

import json
import logging
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
    digest and feature names.  Only their lineage (its length is the depth)
    tells them apart.
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
    """Siblings differ by their position among the parent's children (the lineage's last entry)."""
    search = _search()
    root, _child = _lineage()
    first = MCTSNode(features=["f0"], parent=root, suggestion_index=0)
    second = MCTSNode(features=["f0"], parent=root, suggestion_index=1)
    root.children.extend([first, second])
    assert search._make_node_id(first, 0, _parent_input(first)) != search._make_node_id(
        second, 0, _parent_input(second)
    )


def test_numpy_suggestion_indices_do_not_collapse_siblings() -> None:
    """The key reads the tree position, not ``suggestion_index``, so a numpy int is harmless."""
    search = _search()
    root, _child = _lineage()
    first = MCTSNode(features=["f0"], parent=root, suggestion_index=np.int64(0))  # type: ignore[arg-type]
    second = MCTSNode(features=["f0"], parent=root, suggestion_index=np.int64(1))  # type: ignore[arg-type]
    root.children.extend([first, second])
    parent_output = root.eval_output
    assert search._make_node_id(first, 0, parent_output) != search._make_node_id(
        second, 0, parent_output
    )


def test_cousins_with_identical_parent_plans_differ() -> None:
    """Cousins at the same depth and sibling index, under content-identical parents.

    Siblings A (index 0) and B (index 1) both take the proposer-failure skip,
    so their outputs carry the same plans; their first children A0 and B0
    share rollout, depth, sibling index, parent plan digest and feature
    names.  Only the lineage (``[0, 0]`` vs ``[1, 0]``) tells them apart —
    without it the cache hands B0 A0's object across a resume.
    """
    runner = make_caching_runner(
        make_orchestrator_like_runner(_SUGGESTIONS, initial_features=["f0"])
    )
    search = _search(runner)
    root = MCTSNode(features=["f0"])
    search._root, search._all_nodes = root, [root]
    search._call_evaluate(root, rollout=7)

    a = MCTSNode(features=["f0"], parent=root, suggestion_index=0)
    b = MCTSNode(features=["f0"], parent=root, suggestion_index=1)
    root.children.extend([a, b])
    search._all_nodes.extend([a, b])
    search._call_evaluate(a, rollout=7)
    search._call_evaluate(b, rollout=7)
    assert a.eval_output.feature_plans == b.eval_output.feature_plans

    a0 = MCTSNode(features=["f0"], parent=a, suggestion_index=0)
    b0 = MCTSNode(features=["f0"], parent=b, suggestion_index=0)
    a.children.append(a0)
    b.children.append(b0)
    search._all_nodes.extend([a0, b0])
    search._call_evaluate(a0, rollout=7)
    search._call_evaluate(b0, rollout=7)

    assert search._lineage(a0) == [0, 0] and search._lineage(b0) == [1, 0]
    assert search._make_node_id(a0, 7, _parent_input(a0)) != search._make_node_id(
        b0, 7, _parent_input(b0)
    )
    assert runner.hits == [], f"cache hits: {runner.hits}; aliased pairs: {_aliased_pairs(search)}"
    assert len(set(runner.keys)) == 5, runner.keys
    assert a0.eval_output is not b0.eval_output


def test_detached_node_uses_opaque_position_without_raising() -> None:
    """R9: a node its parent does not list (a detached stand-in) gets a sentinel, not a raise."""
    search = _search()
    _root, child = _lineage()
    detached = MCTSNode(features=["f0"], parent=child)  # not in child.children
    assert search._lineage(detached) == [0, "opaque-position"]
    key = search._make_node_id(detached, 0, make_stub_output(["f0"]))
    assert _SHAPE.fullmatch(key), key


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
    """A node carrying a ``Mock`` in ``suggestion_index`` still yields a well-formed id.

    Issue #14's write-back no longer copies a stand-in runner's
    ``suggestion_index`` onto the node, but a hand-built node or a stubbed
    expansion can still put one there.  The lineage reads the tree position
    instead, so the Mock is never touched.
    """
    search = _search()
    _root, child = _lineage()
    child.suggestion_index = Mock()
    key = search._make_node_id(child, 0, Mock())
    assert _SHAPE.fullmatch(key), key


def test_feature_plan_hash_failure_warns_with_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Real ``FeaturePlan`` values that fail to hash weaken the key: say so loudly."""
    import ctra.search.mcts as mcts_module

    def boom(_plan: Any) -> str:
        raise RuntimeError("hashing exploded")

    monkeypatch.setattr(mcts_module, "plan_content_hash", boom)
    search = _search()
    _root, child = _lineage()
    with caplog.at_level(logging.DEBUG, logger="ctra.search.mcts"):
        key = search._make_node_id(child, 0, _parent_input(child))

    assert _SHAPE.fullmatch(key), key
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, caplog.records
    assert "FeaturePlan" in warnings[0].getMessage()
    assert warnings[0].exc_info is not None
    assert warnings[0].exc_info[0] is RuntimeError


def test_mock_parent_hash_failure_logs_only_debug(caplog: pytest.LogCaptureFixture) -> None:
    """Test stand-ins are expected to be opaque: DEBUG, never WARNING."""
    search = _search()
    _root, child = _lineage()
    with caplog.at_level(logging.DEBUG, logger="ctra.search.mcts"):
        key = search._make_node_id(child, 0, Mock())

    assert _SHAPE.fullmatch(key), key
    levels = [r.levelno for r in caplog.records if r.name == "ctra.search.mcts"]
    assert logging.WARNING not in levels, caplog.records
    assert logging.DEBUG in levels, caplog.records


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
# ``tests.*`` needs the repo root.  ``ctra`` would resolve through the
# shared venv's editable install (as it does under pytest here), but that
# install follows whichever checkout last ran uv, so ``src`` goes first to
# keep the interpreter on this checkout's code.
sys.path.insert(0, {root!r})
sys.path.insert(0, {src!r})
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
root.children.append(child)
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


def _start_python(code: str, hash_seed: str | None = None) -> subprocess.Popen[str]:
    env = os.environ if hash_seed is None else {**os.environ, "PYTHONHASHSEED": hash_seed}
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
    )


def _finish_python(proc: subprocess.Popen[str]) -> str:
    stdout, stderr = proc.communicate()
    assert proc.returncode == 0, stderr
    return stdout


def _run_python(code: str, hash_seed: str | None = None) -> str:
    return _finish_python(_start_python(code, hash_seed))


def test_key_is_stable_across_processes() -> None:
    """Two interpreters with *different* hash seeds agree on the key (#13).

    Explicit seeds make the test independent of the outer environment: a
    fixed ``PYTHONHASHSEED`` there would otherwise hide builtin ``hash()``'s
    per-process salt and let the old key pass.
    """
    code = _KEY_CODE.format(root=str(_REPO_ROOT), src=str(_REPO_ROOT / "src"))
    # Both interpreters spend their time importing ``ctra``; run them side by side.
    procs = [_start_python(code, hash_seed="1"), _start_python(code, hash_seed="2")]
    first, second = (_finish_python(proc).strip() for proc in procs)
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
        root=str(_REPO_ROOT),
        src=str(_REPO_ROOT / "src"),
        cache_dir=str(cache_dir),
        snapshot=str(snapshot),
    )
    resumed = json.loads(_run_python(code))
    assert resumed == {
        "events": [[pre_crash_key, "HIT"]],
        "agent_invocations": 0,
        "f0_idea": pre_crash_idea,
    }, resumed


# ---------------------------------------------------------------------------
# Resume determinism: SELECT is seeded per rollout
# ---------------------------------------------------------------------------


def _select_through_search(rollout: int) -> int:
    """Which of six unvisited root children ``search()`` evaluates in ``rollout``.

    The tree is identical for every call; only the rollout index (>= 1, so
    ``search()`` resumes over it instead of building a fresh root) varies.
    The runner reports the pick through the ``suggestion_index`` on its input.
    """
    picked: list[int] = []

    def runner(_node_id: str, _task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        assert previous_output is not None
        picked.append(previous_output.suggestion_index)
        return make_stub_output(["f0"], 0.6, list(_SUGGESTIONS))

    search = _search(runner, _config(deep_simulation=False, num_rollouts=rollout + 1))
    root = MCTSNode(features=["f0"], visit_count=1)
    root.eval_output = make_stub_output(["f0"], 0.6, list(_SUGGESTIONS))
    for index in range(6):
        root.children.append(MCTSNode(features=["f0"], parent=root, suggestion_index=index))
    search._root, search._all_nodes = root, [root, *root.children]
    search.search(initial_features=["f0"], start_rollout=rollout)
    assert picked and len(picked) == 1, picked
    return picked[0]


def test_select_is_seeded_per_rollout() -> None:
    """Two searches over identical trees pick the same child in the same rollout.

    ``pareto_select`` picks an *unvisited* child at random; unseeded, a
    resumed process would repeat the pre-crash path only by chance.  The
    picks still vary across rollouts, so exploration is not flattened.
    """
    rollouts = range(1, 9)
    first = [_select_through_search(rollout) for rollout in rollouts]
    second = [_select_through_search(rollout) for rollout in rollouts]
    assert first == second, (first, second)
    assert len(set(first)) > 1, first


def test_resume_through_search_replays_pre_crash_path(tmp_path: Path) -> None:
    """The crash-recovery contract end to end: resume from a checkpoint, hit every entry.

    A deep search with six children per node is checkpointed after rollout
    0 and runs to the end, filling the cache.  A fresh search object
    restored from that checkpoint re-runs rollouts 1-5 over the same cache
    directory and must walk the same path — every evaluation a hit, the
    agent never invoked.  Rollouts 1-5 each start with a random pick among
    the root's unvisited children, so an unseeded SELECT replays the whole
    path with probability 1/120.
    """
    cache_dir = tmp_path / "agent_cache"
    runner = make_caching_runner(cache_dir=cache_dir)
    config = _config(num_rollouts=6, min_branch_factor=6, max_branch_factor=6)
    search = _search(runner, config)
    snapshots: dict[str, bytes] = {}

    def on_rollout(rollout: int, _node: MCTSNode, _objectives: Any) -> None:
        if rollout == 0:
            snapshots["after-rollout-0"] = dill.dumps(search)

    search.search(initial_features=["f0"], on_rollout=on_rollout)
    assert runner.hits == [], runner.hits
    pre_crash = [key for key in runner.keys if not key.split("--", 1)[1].startswith("r0-")]
    assert len(pre_crash) >= 5, runner.keys

    resumed: MCTSSearch = dill.loads(snapshots["after-rollout-0"])
    resumed_runner = make_caching_runner(cache_dir=cache_dir)
    resumed.set_runner(resumed_runner)
    resumed.search(initial_features=["f0"], start_rollout=1)

    assert resumed_runner.misses == [], resumed_runner.events
    assert resumed_runner.inner.invocations == 0
    assert resumed_runner.keys == pre_crash, (resumed_runner.keys, pre_crash)


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
