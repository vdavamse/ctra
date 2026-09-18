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

from typing import Any

import numpy as np

from ctra.agents.data_models import EvalOutput
from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSNode, MCTSSearch

from .conftest import make_orchestrator_like_runner, make_stub_output


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
