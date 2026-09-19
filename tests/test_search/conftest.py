"""Shared test fixtures for MCTS search tests.

Provides ``make_stub_runner`` to wrap simple ``(features -> objectives)``
functions into the runner protocol expected by ``MCTSSearch``, and
``make_orchestrator_like_runner`` for tests that need the runner to honour
``suggestion_index`` the way ``Agent.forward`` does (issue #7).
``make_refining_agent`` / ``make_caching_runner`` serve tests that need a
name-preserving REFINE and the runner's disk-cache semantics (issue #12).
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import dill
import numpy as np
import pandas as pd

from ctra.agents.data_models import (
    AgentOutput,
    EvalOutput,
    FeaturePlan,
    FeatureSource,
    FeatureType,
    ModelEvalResult,
)
from ctra.search.objectives import ObjectiveResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _make_plan(name: str) -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[],
        possible_values={},
        feature_instructions=f"Extract {name}.",
    )


def make_stub_output(
    features: list[str],
    roc_auc: float = 0.5,
    suggestions: list[str] | None = None,
) -> AgentOutput:
    """Build a minimal AgentOutput for testing."""
    plans = {f: _make_plan(f) for f in features}
    suggestions = suggestions or [f"Suggestion {i}" for i in range(5)]
    er = ModelEvalResult(
        roc_auc=roc_auc,
        f1=0.5,
        pr_auc=0.5,
        interaction_values={},
        wrong_idxs=[],
        wrong_preds=[],
        wrong_df=pd.DataFrame(),
        pipeline=None,
    )
    return AgentOutput(
        eval_outputs={"stub": EvalOutput(model_eval_result=er, suggestions=suggestions)},
        test_eval_outputs={"stub": er},
        operation=None,
        feature_plans=plans,
        df=pd.DataFrame(),
        val_df=pd.DataFrame(),
        suggestion_index=0,
        raw_features={},
        raw_val_features={},
        raw_test_features={},
        none_explanations={},
        builder_meta={},
    )


def make_stub_runner(
    evaluate_fn: Callable[[list[str]], np.ndarray | ObjectiveResult],
    expand_fn: Callable[..., list[tuple[list[str], str, str]]] | None = None,
    initial_features: list[str] | None = None,
) -> Callable[..., AgentOutput]:
    """Wrap a simple ``(features -> objectives)`` function as a runner.

    The returned runner builds a minimal ``AgentOutput`` whose ROC-AUC
    matches the first objective value from ``evaluate_fn``.  If
    ``expand_fn`` is provided, its output is stored as suggestions
    on the ``AgentOutput`` so the default ``_suggestion_expand`` can
    generate children.

    Parameters:
        evaluate_fn: ``(features) -> ndarray[accuracy, parsimony]``
            or ``(features, fidelity=) -> ...``
        expand_fn: Optional legacy ``(node, max_children=) -> candidates``
            function.  Used to generate suggestion text for the output.
        initial_features: Features for the root node (when previous_output is None).
    """
    _initial = initial_features or ["base"]
    _last_features: dict[str, list[str]] = {"current": list(_initial)}
    _expand_results: dict[str, list] = {}

    def runner(node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        # Determine features: from previous_output plans, or initial
        if previous_output is not None and previous_output.feature_plans:
            features = list(previous_output.feature_plans.keys())
        else:
            features = list(_initial)
        _last_features["current"] = features

        # Call the evaluate function
        try:
            result = evaluate_fn(features, fidelity=1.0)
        except TypeError:
            result = evaluate_fn(features)

        if isinstance(result, ObjectiveResult):
            roc_auc = float(result.values[0])
        else:
            arr = np.asarray(result).ravel()
            roc_auc = float(arr[0]) if len(arr) > 0 else 0.0

        # Generate suggestions from expand_fn if available
        suggestions = [f"Suggestion {i}" for i in range(5)]
        if expand_fn is not None:
            from ctra.search.mcts import MCTSNode

            dummy_node = MCTSNode(features=features)
            # Store AgentOutput on the dummy node so expand_fn can read it
            dummy_node.eval_output = make_stub_output(features, roc_auc, suggestions)
            try:
                candidates = expand_fn(dummy_node, max_children=5)
                suggestions = [f"{op}: {detail}" for _feats, op, detail in candidates]
                # Store for later reference
                key = "|".join(sorted(features))
                _expand_results[key] = candidates
            except Exception:
                pass

        return make_stub_output(features, roc_auc, suggestions)

    return runner


class OrchestratorLikeRunner:
    """Runner that mimics ``Agent.forward``'s iter-N contract.

    ``make_stub_runner`` hard-codes ``suggestion_index=0`` on every output and
    ignores the index it is handed, so no test built on it can ever *reach*
    suggestion exhaustion.  This runner keeps the counter honest:

    - ``previous_output is None`` -> fresh root output at ``suggestion_index=0``.
    - exhausted input -> early skip (``orchestrator.py:365``): returns
      ``deepcopy(previous_output)`` unchanged, zero LM calls.
    - otherwise -> the always-invalid-proposer path: ``lm_calls += 3`` (the
      N=3 ``Refine`` attempts) and the returned output advances
      ``suggestion_index`` by one, as ``forward`` does on a skipped iteration.

    Counters: ``evaluations`` (runner invocations), ``lm_calls``,
    ``exhausted_replays`` (invocations whose input was already exhausted —
    work the search layer should never have sent), ``seen_indices`` (the
    ``suggestion_index`` of every non-root input, in call order).
    """

    def __init__(
        self,
        suggestions: list[str],
        *,
        roc_auc: float = 0.7,
        initial_features: list[str],
    ) -> None:
        self.suggestions = list(suggestions)
        self.roc_auc = roc_auc
        self.initial_features = list(initial_features)
        self.evaluations = 0
        self.lm_calls = 0
        self.exhausted_replays = 0
        self.seen_indices: list[int] = []

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        self.evaluations += 1
        if previous_output is None:
            return make_stub_output(self.initial_features, self.roc_auc, list(self.suggestions))

        self.seen_indices.append(previous_output.suggestion_index)
        if previous_output.suggestions_exhausted:
            self.exhausted_replays += 1
            return deepcopy(previous_output)

        self.lm_calls += 3
        features = list(previous_output.feature_plans.keys()) or list(self.initial_features)
        output = make_stub_output(features, self.roc_auc, list(self.suggestions))
        return output._replace(suggestion_index=previous_output.suggestion_index + 1)


def make_orchestrator_like_runner(
    suggestions: list[str] | None = None,
    *,
    roc_auc: float = 0.7,
    initial_features: list[str] | None = None,
) -> OrchestratorLikeRunner:
    """Build an ``OrchestratorLikeRunner`` (see its docstring).

    Parameters:
        suggestions: The evaluator's suggestion list carried by every output.
            Defaults to two entries so exhaustion is reachable in a handful
            of rollouts.
        roc_auc: ROC-AUC reported by every output (constant, so the reward
            vector cannot hide a duplicate backpropagation behind noise).
        initial_features: Features for the root node.
    """
    return OrchestratorLikeRunner(
        suggestions if suggestions is not None else ["Suggestion 0", "Suggestion 1"],
        roc_auc=roc_auc,
        initial_features=initial_features or ["base"],
    )


# ---------------------------------------------------------------------------
# Issue #12: name-preserving REFINE + the runner's cache semantics
# ---------------------------------------------------------------------------


class RefiningAgent:
    """Fake orchestrator whose REFINE keeps the feature *names* (issue #12).

    Applies the suggestion at ``previous_output.suggestion_index`` the way
    ``Agent.forward`` does:

    - ``previous_output is None`` -> root output with a single feature ``f0``.
    - suggestion 0 -> ADD: a new feature ``f{n}`` (the feature-name set changes).
    - suggestion 1, 2 -> REFINE: same feature names, only ``feature_idea`` grows
      by ``"\n---\nrefined"`` (the orchestrator's real REFINE path concatenates
      the idea and keeps ``feature_name``).

    Every output carries the same three suggestions, so a deep rollout can
    keep descending.  Counters: ``invocations`` (real agent calls, i.e. the
    work a cache hit saves), ``seen_suggestion_indices`` (the index of every
    non-root input, in call order).

    Usable directly as a runner (``__call__`` ignores ``node_id`` and ``task``)
    or as the ``inner`` of a ``CachingRunner``.
    """

    SUGGESTIONS = (
        "ADD a new feature",
        "REFINE existing feature (name unchanged)",
        "REFINE existing feature again (name unchanged)",
    )

    def __init__(self) -> None:
        self.invocations = 0
        self.seen_suggestion_indices: list[int] = []

    def run(self, previous_output: AgentOutput | None) -> AgentOutput:
        self.invocations += 1
        if previous_output is None:
            return make_stub_output(["f0"], 0.60, list(self.SUGGESTIONS))
        self.seen_suggestion_indices.append(previous_output.suggestion_index)
        plans = dict(previous_output.feature_plans)
        if previous_output.suggestion_index == 0:  # ADD
            name = f"f{len(plans)}"
            plans[name] = _make_plan(name)
            roc_auc = min(0.60 + 0.02 * len(plans), 0.95)
        else:  # REFINE: same names, refined idea
            name = sorted(plans)[0]
            old = plans[name]
            plans[name] = old._replace(feature_idea=old.feature_idea + "\n---\nrefined")
            roc_auc = 0.60 + 0.001 * self.invocations
        output = make_stub_output(list(plans), roc_auc, list(self.SUGGESTIONS))
        return output._replace(feature_plans=plans)

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        return self.run(previous_output)


def make_refining_agent() -> RefiningAgent:
    """Build a ``RefiningAgent`` (see its docstring)."""
    return RefiningAgent()


class CachingRunner:
    """Reproduces ``run_agent_as_subprocess``'s cache semantics (``runner.py``).

    Cache key ``f"{task}--{node_id}"``; write-through; a repeated key returns
    the stored output *without* calling ``inner`` — for an in-memory hit the
    very same object, which is what lets a test detect two tree nodes sharing
    one ``AgentOutput``.  With ``cache_dir`` the outputs are also persisted as
    ``{key}.output.pkl`` (dill) and a fresh process over the same directory
    hits them, which is the crash-recovery scenario ``node_id`` exists for.

    Counters: ``invocations``, ``hits`` / ``misses`` (keys, in call order),
    ``keys`` (every key seen, in call order), ``events`` (``(key, "HIT"|"MISS")``),
    ``cache`` (the in-memory store).
    """

    def __init__(
        self,
        inner: Callable[[str, Any, AgentOutput | None], AgentOutput],
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        self.cache: dict[str, AgentOutput] = {}
        self.invocations = 0
        self.hits: list[str] = []
        self.misses: list[str] = []
        self.keys: list[str] = []
        self.events: list[tuple[str, str]] = []

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        key = f"{task}--{node_id}"
        self.invocations += 1
        self.keys.append(key)
        cached = self._load(key)
        if cached is not None:
            self.hits.append(key)
            self.events.append((key, "HIT"))
            return cached
        self.misses.append(key)
        self.events.append((key, "MISS"))
        output = self.inner(node_id, task, previous_output)
        self._store(key, output)
        return output

    def _load(self, key: str) -> AgentOutput | None:
        if key in self.cache:
            return self.cache[key]
        if self.cache_dir is not None:
            path = self.cache_dir / f"{key}.output.pkl"
            if path.exists():
                with path.open("rb") as fh:
                    loaded: AgentOutput = dill.load(fh)
                self.cache[key] = loaded
                return loaded
        return None

    def _store(self, key: str, output: AgentOutput) -> None:
        self.cache[key] = output
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with (self.cache_dir / f"{key}.output.pkl").open("wb") as fh:
                dill.dump(output, fh)


def make_caching_runner(
    inner: Callable[[str, Any, AgentOutput | None], AgentOutput] | None = None,
    *,
    cache_dir: Path | None = None,
) -> CachingRunner:
    """Build a ``CachingRunner`` (see its docstring).

    Parameters:
        inner: The runner whose outputs are cached.  Defaults to a fresh
            ``RefiningAgent`` (reachable afterwards as ``runner.inner``).
        cache_dir: Persist outputs here as well, so another process can hit them.
    """
    return CachingRunner(inner if inner is not None else make_refining_agent(), cache_dir=cache_dir)
