#!/usr/bin/env python
"""Ablate the MCTS backpropagation rule (``MCTSConfig.backprop``, issue #16).

Runs the production ``MCTSSearch`` under each rule — ``mean`` (subtree mean,
PMMG-style, the default), ``max`` (elementwise running maximum, the
vectorised reading of AutoCT's max-reward backpropagation) and ``max_hv``
(best realised vector by hypervolume) — on the synergy landscape of
``tests/test_search/test_autoct_mcts_comparison.py``: ``feat_A..D`` add
+0.03 ROC-AUC each, +0.08 at three of them, +0.20 at all four, with
N(0, 0.01) noise, beside ``noise_1..3`` that add nothing.  The landscape is
driven through a *feature-aware* runner (each node is evaluated on its own
feature set) because ``make_stub_runner`` rebuilds every output from the
parent's plans and never leaves the root's 1-feature set — a copy, not an
import, so the harness does not depend on the test suite.

Backprop state reaches the search only through ``ucb_scores`` ->
``pareto_select``, which picks an unvisited child before it computes UCB, so
the rules can only differ where nodes are *revisited*.  Every row therefore
carries the number of UCB-decided selections; the ``saturated`` regime is
where that number is large.  Seeds vary the landscape (noise stream and pool
order) only: CTRA's own RNGs are rollout-seeded.

Usage:
    python scripts/eval/ablate_backprop.py --regime issue --seeds 3
    python scripts/eval/ablate_backprop.py --regime all --seeds 20 \\
        --json output/backprop_ablation/ablation.json
    python scripts/eval/ablate_backprop.py --regime saturated --variants mean,max --seeds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import ctra.search.mcts as mcts_mod
from ctra.agents.data_models import (
    AgentOutput,
    EvalOutput,
    FeaturePlan,
    FeatureSource,
    FeatureType,
    ModelEvalResult,
)
from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSSearch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

VARIANTS = ("mean", "max", "max_hv")
SYNERGY = ("feat_A", "feat_B", "feat_C", "feat_D")
NOISE = ("noise_1", "noise_2", "noise_3")
INITIAL = ["start"]
MAX_DEPTH = 7
MAX_FEATURES = 50
REFERENCE = [0.5, 0.0]

COLUMNS = (
    "regime",
    "variant",
    "deep",
    "adaptive",
    "rollouts",
    "branch",
    "note",
    "seed",
    "evaluations",
    "best_auc",
    "best_depth",
    "synergy_found",
    "best_synergy_count",
    "evals_to_first_synergy",
    "nodes",
    "depth_reached",
    "ucb_decided",
    "max_size_seen",
    "max_synergy_seen",
    "seconds",
)


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One factorial cell: a search configuration every variant and seed runs."""

    regime: str
    deep: bool
    adaptive: bool
    rollouts: int
    branch: int  # max_branch_factor; min_branch_factor is 2 in every cell
    note: str = ""  # labels a control cell; part of the cell key in the summary


REGIMES: dict[str, tuple[Cell, ...]] = {
    # The issue's factorial (deep x adaptive at 10 rollouts, branch 3) plus a
    # shallow control at the deep cell's evaluation budget (~63 evaluations).
    "issue": (
        Cell("issue", deep=True, adaptive=False, rollouts=10, branch=3),
        Cell("issue", deep=True, adaptive=True, rollouts=10, branch=3),
        Cell("issue", deep=False, adaptive=False, rollouts=10, branch=3),
        Cell("issue", deep=False, adaptive=True, rollouts=10, branch=3),
        Cell("issue", deep=False, adaptive=False, rollouts=63, branch=3, note="equal evals"),
    ),
    # Enough revisits for UCB to decide most selections.
    "saturated": (
        Cell("saturated", deep=False, adaptive=False, rollouts=120, branch=2),
        Cell("saturated", deep=False, adaptive=True, rollouts=120, branch=2, note="min==max"),
        Cell("saturated", deep=False, adaptive=False, rollouts=120, branch=3),
        Cell("saturated", deep=True, adaptive=False, rollouts=40, branch=2),
    ),
    # Adaptive branching with a real range (min 2, max 8 = production default).
    "deep-wide": (
        Cell("deep-wide", deep=True, adaptive=False, rollouts=40, branch=8),
        Cell("deep-wide", deep=True, adaptive=True, rollouts=40, branch=8),
    ),
}


# ---------------------------------------------------------------------------
# Landscape and runner (copies of the test fixtures; see the module docstring)
# ---------------------------------------------------------------------------


def make_synergy_evaluator(seed: int) -> Callable[[list[str]], float]:
    """ROC-AUC landscape with an epistatic bonus for ``feat_A..D`` together."""
    rng = np.random.default_rng(seed)
    synergy = set(SYNERGY)

    def evaluate(features: list[str]) -> float:
        count = len(set(features) & synergy)
        base = 0.5 + 0.03 * count
        if count >= 4:
            base += 0.20
        elif count >= 3:
            base += 0.08
        return float(np.clip(base + rng.normal(0, 0.01), 0, 1))

    return evaluate


def make_pool(seed: int) -> list[str]:
    """The addable features in a seed-specific order (the runner adds in pool order)."""
    pool = [*SYNERGY, *NOISE]
    np.random.default_rng(seed).shuffle(pool)
    return pool


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


def stub_output(features: list[str], roc_auc: float, suggestions: list[str]) -> AgentOutput:
    """A minimal ``AgentOutput`` scoring ``features`` at ``roc_auc``.

    An empty ``suggestions`` list is kept empty (a proposer with nothing
    left to add offers none), which stops ``_suggestion_expand`` there.
    """
    result = ModelEvalResult(
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
        eval_outputs={"stub": EvalOutput(model_eval_result=result, suggestions=suggestions)},
        test_eval_outputs={"stub": result},
        operation=None,
        feature_plans={f: _make_plan(f) for f in features},
        df=pd.DataFrame(),
        val_df=pd.DataFrame(),
        suggestion_index=0,
        raw_features={},
        raw_val_features={},
        raw_test_features={},
        none_explanations={},
        builder_meta={},
    )


class FeatureAwareRunner:
    """A runner that evaluates each node's **own** feature set.

    Suggestion ``i`` ADDs the ``i``-th feature of ``pool`` the parent does
    not already carry (``Agent.forward``'s proposer shape).  ``seen`` records
    every feature set handed to the evaluator, so the harness can assert the
    sets really grow instead of trusting the runner.
    """

    def __init__(
        self, evaluate: Callable[[list[str]], float], pool: list[str], max_suggestions: int
    ) -> None:
        self.evaluate = evaluate
        self.pool = list(pool)
        self.max_suggestions = max_suggestions
        self.seen: list[list[str]] = []

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        if previous_output is None or not previous_output.feature_plans:
            features = list(INITIAL)
        else:
            parent = list(previous_output.feature_plans)
            available = [f for f in self.pool if f not in parent]
            index = previous_output.suggestion_index
            features = [*parent, available[index]] if 0 <= index < len(available) else parent
        self.seen.append(list(features))
        auc = self.evaluate(features)
        remaining = [f for f in self.pool if f not in features][: self.max_suggestions]
        return stub_output(features, auc, [f"add {f}" for f in remaining])


# ---------------------------------------------------------------------------
# One search
# ---------------------------------------------------------------------------


class _SelectionCounter:
    """Count how many ``pareto_select`` calls were decided by UCB.

    A call with one candidate or with an unvisited candidate never reads
    backprop state (an unvisited child wins outright), so only calls where
    every candidate has been visited can tell the variants apart.
    """

    def __init__(self) -> None:
        self.ucb_decided = 0
        self._original = mcts_mod.pareto_select

    def __enter__(self) -> _SelectionCounter:
        def counting(nodes: Sequence[Any], **kwargs: Any) -> Any:
            if len(nodes) > 1 and all(n.visit_count > 0 for n in nodes):
                self.ucb_decided += 1
            return self._original(nodes, **kwargs)

        mcts_mod.pareto_select = counting  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        mcts_mod.pareto_select = self._original  # type: ignore[assignment]


def run_one(variant: str, seed: int, cell: Cell) -> dict[str, Any]:
    """Run one search and measure it; returns a row with every ``COLUMNS`` key."""
    runner = FeatureAwareRunner(make_synergy_evaluator(seed), make_pool(seed), cell.branch)
    config = MCTSConfig(
        backprop=variant,  # type: ignore[arg-type]
        num_rollouts=cell.rollouts,
        max_depth=MAX_DEPTH,
        objectives=["accuracy", "parsimony"],
        reference_point=REFERENCE,
        max_features=MAX_FEATURES,
        exploration_constant=1.0,
        adaptive_branching=cell.adaptive,
        min_branch_factor=2,
        max_branch_factor=cell.branch,
        deep_simulation=cell.deep,
    )
    search = MCTSSearch(runner=runner, task="ablation", config=config)
    started = time.perf_counter()
    with _SelectionCounter() as counter:
        best = search.search(initial_features=list(INITIAL))
    seconds = time.perf_counter() - started

    synergy = set(SYNERGY)
    counts = [len(set(f) & synergy) for f in runner.seen]
    first = next((i for i, c in enumerate(counts, 1) if c == 4), None)
    return {
        "regime": cell.regime,
        "variant": variant,
        "deep": cell.deep,
        "adaptive": cell.adaptive,
        "rollouts": cell.rollouts,
        "branch": cell.branch,
        "note": cell.note,
        "seed": seed,
        "evaluations": len(runner.seen),
        "best_auc": float(search.best_own_objectives(best)[0]),
        "best_depth": MCTSSearch._node_depth(best),
        "synergy_found": first is not None,
        "best_synergy_count": len(set(best.features) & synergy),
        "evals_to_first_synergy": float("nan") if first is None else float(first),
        "nodes": len(search.all_nodes),
        "depth_reached": max(MCTSSearch._node_depth(n) for n in search.all_nodes),
        "ucb_decided": counter.ucb_decided,
        "max_size_seen": max(len(f) for f in runner.seen),
        "max_synergy_seen": max(counts),
        "seconds": seconds,
    }


# ---------------------------------------------------------------------------
# Sweep, fixture assertions, summary
# ---------------------------------------------------------------------------


def check_fixture(rows: Iterable[dict[str, Any]], regime: str) -> None:
    """Refuse a table the fixture cannot have produced.

    A runner that rebuilds outputs from the parent's plans scores the root's
    set forever (``max_size_seen == 1``); a landscape the search never
    reaches the bonus of cannot separate the variants.  Both would give a
    confident, wrong table, so each regime must reach a feature set larger
    than the root and all four synergy features at least once.
    """
    rows = [r for r in rows if r["regime"] == regime]
    if not rows:
        raise RuntimeError(f"regime {regime!r} produced no rows")
    if max(r["max_size_seen"] for r in rows) <= 1:
        raise RuntimeError(
            f"regime {regime!r}: feature-set sizes > 1 were never evaluated — the runner "
            "is scoring the root's set for every node (parent-plans regression)"
        )
    if max(r["max_synergy_seen"] for r in rows) < 4:
        raise RuntimeError(
            f"regime {regime!r}: synergy count 4 was never evaluated — the landscape's "
            "bonus is out of reach at this budget"
        )


def run_sweep(
    regimes: Sequence[str],
    variants: Sequence[str],
    seeds: Sequence[int],
    rollouts: int | None = None,
) -> list[dict[str, Any]]:
    """Every (cell, variant, seed) row for ``regimes``; ``rollouts`` overrides each cell's."""
    rows: list[dict[str, Any]] = []
    for regime in regimes:
        for cell in REGIMES[regime]:
            if rollouts is not None:
                cell = Cell(**{**asdict(cell), "rollouts": rollouts})
            for variant in variants:
                for seed in seeds:
                    rows.append(run_one(variant, seed, cell))
        check_fixture(rows, regime)
    return rows


_CELL_KEYS = ("regime", "deep", "adaptive", "rollouts", "branch", "note")
_MEAN_KEYS = (
    "evaluations",
    "best_auc",
    "best_depth",
    "synergy_found",
    "best_synergy_count",
    "evals_to_first_synergy",
    "nodes",
    "depth_reached",
    "ucb_decided",
)
_IDENTITY_KEYS = ("evaluations", "best_auc", "best_depth", "best_synergy_count", "ucb_decided")


def summarise(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-cell, per-variant means (``evals_to_first_synergy`` over the seeds that found it).

    ``same_as_mean`` counts the seeds on which the variant's search ended
    exactly where ``mean``'s did (same evaluations, best, depth and UCB
    count): the variants are byte-identical wherever UCB never decided.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in (*_CELL_KEYS, "variant"))
        groups.setdefault(key, []).append(row)

    def mean_rows(cell: tuple[Any, ...]) -> dict[int, dict[str, Any]]:
        return {r["seed"]: r for r in groups.get((*cell, "mean"), [])}

    summary: list[dict[str, Any]] = []
    for key, group in groups.items():
        cell, variant = key[:-1], key[-1]
        entry: dict[str, Any] = dict(zip(_CELL_KEYS, cell, strict=True))
        entry["variant"] = variant
        entry["n"] = len(group)
        for metric in _MEAN_KEYS:
            values = np.asarray([float(r[metric]) for r in group])
            finite = values[np.isfinite(values)]
            entry[metric] = float(finite.mean()) if finite.size else float("nan")
        reference = mean_rows(cell)
        entry["same_as_mean"] = sum(
            1
            for r in group
            if r["seed"] in reference
            and all(r[k] == reference[r["seed"]][k] for k in _IDENTITY_KEYS)
        )
        summary.append(entry)
    return summary


# ---------------------------------------------------------------------------
# Rendering and I/O
# ---------------------------------------------------------------------------

_SUMMARY_HEADER = (
    f"{'regime':10} {'variant':7} {'deep':5} {'adapt':5} {'roll':>4} {'br':>2} {'n':>3} "
    f"{'evals':>7} {'AUC':>7} {'depth':>5} {'found':>5} {'syn':>4} {'first':>6} "
    f"{'nodes':>6} {'reached':>7} {'UCB':>6} {'=mean':>5}"
)
_ROW_HEADER = (
    f"{'regime':10} {'variant':7} {'deep':5} {'adapt':5} {'roll':>4} {'br':>2} {'seed':>4} "
    f"{'evals':>5} {'AUC':>7} {'depth':>5} {'found':>5} {'syn':>4} {'first':>6} "
    f"{'nodes':>6} {'reached':>7} {'UCB':>6}"
)


def _fmt(value: float, width: int, digits: int) -> str:
    return f"{'nan':>{width}}" if math.isnan(value) else f"{value:{width}.{digits}f}"


def format_table(summary: Sequence[dict[str, Any]]) -> str:
    """The per-cell means as a fixed-width table (``nan`` where no seed found the synergy)."""
    lines = [_SUMMARY_HEADER]
    for e in summary:
        lines.append(
            f"{e['regime']:10} {e['variant']:7} {e['deep']!s:5} {e['adaptive']!s:5} "
            f"{e['rollouts']:4d} {e['branch']:2d} {e['n']:3d} "
            f"{_fmt(e['evaluations'], 7, 1)} {_fmt(e['best_auc'], 7, 4)} "
            f"{_fmt(e['best_depth'], 5, 2)} {_fmt(e['synergy_found'], 5, 2)} "
            f"{_fmt(e['best_synergy_count'], 4, 1)} {_fmt(e['evals_to_first_synergy'], 6, 1)} "
            f"{_fmt(e['nodes'], 6, 1)} {_fmt(e['depth_reached'], 7, 1)} "
            f"{_fmt(e['ucb_decided'], 6, 1)} {e['same_as_mean']:5d}"
            + (f"  ({e['note']})" if e["note"] else "")
        )
    return "\n".join(lines)


def format_rows(rows: Sequence[dict[str, Any]]) -> str:
    """Every seed's row (the per-seed view the research note keeps for saturated regimes)."""
    lines = [_ROW_HEADER]
    for r in rows:
        lines.append(
            f"{r['regime']:10} {r['variant']:7} {r['deep']!s:5} {r['adaptive']!s:5} "
            f"{r['rollouts']:4d} {r['branch']:2d} {r['seed']:4d} {r['evaluations']:5d} "
            f"{r['best_auc']:7.4f} {r['best_depth']:5d} {r['synergy_found']!s:>5} "
            f"{r['best_synergy_count']:4d} {_fmt(r['evals_to_first_synergy'], 6, 0)} "
            f"{r['nodes']:6d} {r['depth_reached']:7d} {r['ucb_decided']:6d}"
        )
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(
    path: Path, rows: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]], **meta: Any
) -> None:
    """Dump rows and summary; NaN becomes ``null`` so any JSON reader can load it."""
    payload = {
        **meta,
        "rows": [{k: _jsonable(v) for k, v in r.items()} for r in rows],
        "summary": [{k: _jsonable(v) for k, v in e.items()} for e in summary],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate MCTSConfig.backprop on the synergy stub landscape (issue #16)."
    )
    parser.add_argument(
        "--regime",
        choices=[*REGIMES, "all"],
        default="all",
        help="issue: the issue's factorial at its budget; saturated: enough revisits for "
        "UCB to decide; deep-wide: adaptive branching with a real range; all: every regime",
    )
    parser.add_argument(
        "--variants", default=",".join(VARIANTS), help="comma-separated subset of mean,max,max_hv"
    )
    parser.add_argument("--seeds", type=int, default=20, help="run seeds 0..N-1")
    parser.add_argument(
        "--rollouts", type=int, default=None, help="override every cell's rollout count"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/backprop_ablation"), help="for the dumps"
    )
    parser.add_argument("--json", type=Path, default=None, help="JSON path (default in output-dir)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    regimes = list(REGIMES) if args.regime == "all" else [args.regime]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        print(f"unknown variants {unknown}; choose from {VARIANTS}", file=sys.stderr)
        return 2
    seeds = list(range(args.seeds))

    # The search logs every rollout at INFO and the expansion cap at WARNING;
    # hundreds of searches of that would bury the table.  Restore the caller's
    # disable level afterwards rather than resetting it: a test session or
    # notebook with its own ``logging.disable(...)`` keeps it.
    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    started = time.perf_counter()
    try:
        rows = run_sweep(regimes, variants, seeds, rollouts=args.rollouts)
    finally:
        logging.disable(previous_disable)
    elapsed = time.perf_counter() - started
    summary = summarise(rows)

    print(f"backprop ablation: {len(rows)} searches in {elapsed:.1f}s\n")
    print(format_table(summary))
    print()
    print(format_rows(rows))

    json_path = args.json or args.output_dir / f"ablation_{args.regime}.json"
    write_json(
        json_path,
        rows,
        summary,
        regimes=regimes,
        variants=variants,
        seeds=seeds,
        rollouts_override=args.rollouts,
        elapsed_seconds=elapsed,
    )
    print(f"\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
