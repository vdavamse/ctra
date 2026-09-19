#!/usr/bin/env python
"""Measure feature-store and agent-cache reuse offline (issue #17).

Drives the production ``MCTSSearch`` and the production ``Agent.forward``
(orchestrator, ``compute_features``, ``WrappedFeatureBuilder``, a real
feature store in a temporary directory) with the five dspy modules replaced
by deterministic stubs -- the patch set of
``tests/test_agents/test_orchestrator.py`` -- and the model block reduced to
a dummy classifier with a synthetic ROC-AUC landscape.  What is measured is
the **mechanism**: how the counters behave when branches of one search send
plans to a shared store.  What is *not* measured is how often Claude emits
byte-identical plan text for the same feature idea on two branches; that
rate is an input here (``--collision-rate``), so every number below is a
function of it, not a production figure.

Plans: the initializer always mints the same three plans (iteration 0, one
evaluation per search).  On iteration N the planner stub returns, for a
feature name, either the canonical plan text (with probability
``--collision-rate``) or a uniquely worded one.  Two branches that both add
``feat_C`` therefore collide in the store only when both drew the canonical
text -- the offline stand-in for "the LLM wrote the same plan twice".

Agent cache: ``run_agent_as_subprocess`` is not called (it spawns
``scripts/run_agent.py``); ``InProcessRunner`` mirrors its semantics with a
dict in place of the pickle directory and the same
``RunCacheStats.record_hit`` / ``record_miss`` accounting.  Within one search
every node id carries its rollout, so no evaluation repeats and the agent
hit rate is 0 by construction; ``--replay`` runs the same search a second
time against the same cache to exercise the hit path (the resume scenario).

Usage:
    python scripts/eval/measure_cache_reuse.py --seeds 3 --rollouts 5
    python scripts/eval/measure_cache_reuse.py --collision-rate 0,0.25,0.5,1 \\
        --seeds 10 --rollouts 10 --json output/cache_reuse/measurement.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ctra.agents.data_models import (
    EvalOutput,
    FeatureOp,
    FeaturePlan,
    FeatureSource,
    FeatureType,
    ModelEvalResult,
    ProposerOutput,
)
from ctra.agents.orchestrator import Agent
from ctra.agents.runner import RunCacheStats
from ctra.config.settings import ClassifierType, MCTSConfig
from ctra.search.mcts import MCTSSearch

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ctra.agents.data_models import AgentOutput

logger = logging.getLogger("ctra.measure_cache_reuse")

INITIAL = ("init_1", "init_2", "init_3")
POOL = ("feat_A", "feat_B", "feat_C", "feat_D", "feat_E", "feat_F", "feat_G", "feat_H")
N_TRAIN, N_VAL, N_TEST = 6, 2, 2
N_TRIALS = N_TRAIN + N_VAL + N_TEST
MAX_FEATURES = 50
REFERENCE = [0.5, 0.0]
TASK = "measure cache reuse"

COLUMNS = (
    "collision_rate",
    "seed",
    "rollouts",
    "evaluations",
    "agent_hits",
    "agent_hit_rate",
    "feature_lookups",
    "feature_hits",
    "feature_store_hit_rate",
    "hits_from_initializer_plans",
    "hits_from_planner_plans",
    "planner_calls",
    "planner_plan_hit_rate",
    "groups_dispatched",
    "groups_skipped",
    "builds",
    "store_writes",
    "llm_calls_avoided_estimate",
    "replay_evaluations",
    "replay_agent_hits",
    "replay_group_builds",
    "seconds",
)


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------


def canonical_plan(name: str, wording: str = "") -> FeaturePlan:
    """The plan text for ``name``; ``wording`` is the salt that makes it unique."""
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[],
        possible_values={},
        feature_instructions=f"Extract {name}.{wording}",
    )


class World:
    """One (collision rate, seed) configuration shared by every stub."""

    def __init__(self, collision_rate: float, seed: int, branch: int) -> None:
        self.collision_rate = collision_rate
        self.seed = seed
        self.branch = branch
        self.rng = np.random.default_rng(seed)
        self.planner_calls = 0
        self.builds = 0

    def plan_for(self, name: str) -> FeaturePlan:
        """Canonical text with probability ``collision_rate``, else a unique wording.

        The salt carries the seed as well as the call serial, so a wording is
        unique across runs too: with ``--shared-store`` only the canonical
        texts (and the initializer's plans) can hit from an earlier seed.
        """
        self.planner_calls += 1
        if self.rng.random() < self.collision_rate:
            return canonical_plan(name)
        return canonical_plan(name, wording=f" Wording s{self.seed}-{self.planner_calls}.")

    def roc_auc(self, features: Sequence[str]) -> float:
        count = len(set(features) & set(POOL))
        return float(np.clip(0.5 + 0.03 * count + self.rng.normal(0, 0.01), 0, 1))


def _stub_classes(world: World) -> dict[str, type]:
    """The five dspy modules ``Agent.__init__`` constructs, as deterministic stubs."""

    class Initializer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __call__(self) -> dict[str, FeaturePlan]:
            return {name: canonical_plan(name) for name in INITIAL}

    class FeatureProposer:
        def __init__(self, task_description: str) -> None:
            pass

        def __call__(self, previous_output: AgentOutput) -> ProposerOutput:
            suggestion = previous_output.get_next_suggestion()
            name = suggestion.removeprefix("add ")
            return ProposerOutput(FeatureOp.ADD, name, f"{name} idea")

    class FeaturePlanner:
        def __init__(self, task_description: str) -> None:
            pass

        def __call__(self, feature_name: str, feature_idea: str) -> tuple[FeaturePlan, str]:
            return world.plan_for(feature_name), "raw"

    class Evaluator:
        def __init__(self, task_description: str) -> None:
            pass

        def __call__(self, feature_plans: dict[str, FeaturePlan], **kwargs: Any) -> EvalOutput:
            remaining = [f for f in POOL if f not in feature_plans][: world.branch]
            return EvalOutput(
                model_eval_result=kwargs["model_eval_result"],
                suggestions=[f"add {f}" for f in remaining],
            )

    class FeatureGrouper:
        def __init__(self, task_description: str) -> None:
            pass

        def __call__(
            self, feature_plans: dict[str, FeaturePlan], task: str
        ) -> list[dict[str, FeaturePlan]]:
            items = list(feature_plans.items())
            return [dict(items[i : i + 5]) for i in range(0, len(items), 5)]

    return {
        "Initializer": Initializer,
        "FeatureProposer": FeatureProposer,
        "FeaturePlanner": FeaturePlanner,
        "Evaluator": Evaluator,
        "FeatureGrouper": FeatureGrouper,
    }


class XGBClassifier(DummyClassifier):  # type: ignore[misc]
    """Named to satisfy the orchestrator's classifier-class guard; a dummy underneath."""


def _fake_eval_model(world: World) -> Any:
    def eval_model(pipeline: Any, df: pd.DataFrame, y_true: Any) -> ModelEvalResult:
        features = sorted({c.split("--", 1)[0] for c in df.columns if c != "id"})
        return ModelEvalResult(
            roc_auc=world.roc_auc(features),
            f1=0.5,
            pr_auc=0.5,
            interaction_values={},
            wrong_idxs=[],
            wrong_preds=[],
            wrong_df=pd.DataFrame(),
            pipeline=None,
        )

    return eval_model


def _fake_builder(world: World) -> Any:
    def build(
        nctid: str, feature_plan_group: dict[str, FeaturePlan]
    ) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
        world.builds += 1
        return (
            {fn: {"value": float(world.rng.random())} for fn in feature_plan_group},
            {"builder_reasoning": "stub", "research_results": "stub"},
        )

    return build


@contextlib.contextmanager
def stubbed_pipeline(world: World, store_dir: Path) -> Iterator[None]:
    """Patch the dspy modules, the builder and the model block around a search."""
    settings = SimpleNamespace(
        mcts=SimpleNamespace(feature_store_dir=store_dir, feature_store_enabled=True),
        model=SimpleNamespace(
            classifiers=[ClassifierType.XGBOOST],
            shapiq_max_order=2,
            shapiq_max_samples=8,
            shapiq_budget=8,
        ),
    )
    registry = SimpleNamespace(create_classifier=lambda kind: XGBClassifier(strategy="prior"))
    stubs = _stub_classes(world)
    with contextlib.ExitStack() as stack:
        for name, cls in stubs.items():
            stack.enter_context(patch(f"ctra.agents.orchestrator.{name}", cls))
        stack.enter_context(
            patch("ctra.agents.orchestrator.ResettingRefine", lambda module, **kw: module)
        )
        stack.enter_context(patch("ctra.agents.orchestrator.get_settings", lambda: settings))
        stack.enter_context(patch("ctra.agents.orchestrator.ModelRegistry", registry))
        stack.enter_context(patch("ctra.agents.orchestrator.eval_model", _fake_eval_model(world)))
        stack.enter_context(
            patch("ctra.agents.orchestrator.compute_shapiq_for_pipeline", lambda **kw: {})
        )
        builder = stack.enter_context(patch("ctra.agents.feature_builder.FeatureBuilder"))
        builder.return_value = _fake_builder(world)
        stack.enter_context(
            patch("ctra.agents.feature_builder.ResettingRefine", lambda module, **kw: module)
        )
        yield


# ---------------------------------------------------------------------------
# In-process runner (the agent-cache layer)
# ---------------------------------------------------------------------------


class InProcessRunner:
    """``run_agent_as_subprocess`` without the process.

    Same cache key (``f"{task}--{node_id}"``), same early return on a hit,
    same ``RunCacheStats`` methods on both paths; a dict stands in for the
    pickle directory and ``Agent.forward`` runs in this process.
    """

    def __init__(self, agent: Agent, stats: RunCacheStats) -> None:
        self.agent = agent
        self.stats = stats
        self.cache: dict[str, AgentOutput] = {}

    def __call__(self, node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        key = f"{task}--{node_id}"
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.record_hit(cached)
            return cached
        output = self.agent.forward(previous_output=previous_output)
        self.stats.record_miss(output)
        self.cache[key] = output
        return output


def make_agent() -> Agent:
    ids = [f"NCT{i:08d}" for i in range(N_TRIALS)]
    labels = np.array([i % 2 for i in range(N_TRIALS)])
    return Agent(
        task=TASK,
        X_train=pd.Series(ids[:N_TRAIN]),
        y_train=labels[:N_TRAIN],
        X_val=pd.Series(ids[N_TRAIN : N_TRAIN + N_VAL]),
        y_val=labels[N_TRAIN : N_TRAIN + N_VAL],
        X_test=pd.Series(ids[N_TRAIN + N_VAL :]),
        y_test=labels[N_TRAIN + N_VAL :],
    )


# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------


def _config(rollouts: int, depth: int, branch: int, deep: bool) -> MCTSConfig:
    return MCTSConfig(
        num_rollouts=rollouts,
        max_depth=depth,
        objectives=["accuracy", "parsimony"],
        reference_point=REFERENCE,
        max_features=MAX_FEATURES,
        exploration_constant=1.0,
        adaptive_branching=False,
        min_branch_factor=2,
        max_branch_factor=branch,
        deep_simulation=deep,
    )


def run_one(
    collision_rate: float,
    seed: int,
    *,
    rollouts: int,
    depth: int,
    branch: int,
    deep: bool,
    store_dir: Path,
    replay: bool,
) -> dict[str, Any]:
    """Run one search (and optionally its replay) and return a ``COLUMNS`` row."""
    world = World(collision_rate, seed, branch)
    stats = RunCacheStats()
    started = time.perf_counter()
    with stubbed_pipeline(world, store_dir):
        runner = InProcessRunner(make_agent(), stats)
        config = _config(rollouts, depth, branch, deep)
        MCTSSearch(runner=runner, task="measure", config=config).search(initial_features=[])
        evaluations = stats.agent_misses
        agent_hits = stats.agent_hits
        replay_stats = RunCacheStats()
        if replay:
            runner.stats = replay_stats
            MCTSSearch(runner=runner, task="measure", config=config).search(initial_features=[])
    seconds = time.perf_counter() - started

    fs = stats.feature_store
    planner_lookups = world.planner_calls * N_TRIALS
    return {
        "collision_rate": collision_rate,
        "seed": seed,
        "rollouts": rollouts,
        "evaluations": evaluations,
        "agent_hits": agent_hits,
        "agent_hit_rate": stats.agent_hit_rate,
        "feature_lookups": fs.feature_lookups,
        "feature_hits": fs.feature_hits,
        "feature_store_hit_rate": fs.hit_rate,
        "hits_from_initializer_plans": fs.hits_from_initializer_plans,
        "hits_from_planner_plans": fs.hits_from_planner_plans,
        "planner_calls": world.planner_calls,
        "planner_plan_hit_rate": (
            fs.hits_from_planner_plans / planner_lookups if planner_lookups else 0.0
        ),
        "groups_dispatched": fs.groups_dispatched,
        "groups_skipped": fs.groups_skipped,
        "builds": world.builds,
        "store_writes": fs.store_writes,
        "llm_calls_avoided_estimate": fs.llm_calls_avoided_estimate,
        "replay_evaluations": replay_stats.agent_lookups,
        "replay_agent_hits": replay_stats.agent_hits,
        "replay_group_builds": replay_stats.replayed_group_builds,
        "seconds": seconds,
    }


def check_row(row: dict[str, Any]) -> None:
    """Refuse a row the mechanism cannot have produced."""
    if row["builds"] != row["groups_dispatched"]:
        raise RuntimeError(
            f"builder ran {row['builds']} times but {row['groups_dispatched']} groups were dispatched"
        )
    if row["feature_lookups"] < row["feature_hits"]:
        raise RuntimeError("more hits than lookups")
    if row["evaluations"] == 0:
        raise RuntimeError("the search evaluated nothing")


def run_sweep(
    rates: Sequence[float],
    seeds: Sequence[int],
    *,
    rollouts: int,
    depth: int,
    branch: int,
    deep: bool,
    shared_store: bool,
    replay: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ctra-cache-reuse-") as tmp:
        for rate in rates:
            for seed in seeds:
                store = Path(tmp) / f"rate{rate}" / ("shared" if shared_store else f"seed{seed}")
                row = run_one(
                    rate,
                    seed,
                    rollouts=rollouts,
                    depth=depth,
                    branch=branch,
                    deep=deep,
                    store_dir=store,
                    replay=replay,
                )
                check_row(row)
                rows.append(row)
                logger.info(
                    "rate=%.2f seed=%d: %d evaluations, store hit rate %.1f%%, planner-plan hit "
                    "rate %.1f%%, %d groups skipped (%.1fs)",
                    rate,
                    seed,
                    row["evaluations"],
                    row["feature_store_hit_rate"] * 100,
                    row["planner_plan_hit_rate"] * 100,
                    row["groups_skipped"],
                    row["seconds"],
                )
    return rows


# ---------------------------------------------------------------------------
# Summary and output
# ---------------------------------------------------------------------------

MEAN_COLUMNS = (
    "evaluations",
    "agent_hit_rate",
    "feature_store_hit_rate",
    "hits_from_initializer_plans",
    "hits_from_planner_plans",
    "planner_plan_hit_rate",
    "groups_dispatched",
    "groups_skipped",
    "llm_calls_avoided_estimate",
    "replay_agent_hits",
    "seconds",
)


def summarise(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean of every ``MEAN_COLUMNS`` entry per collision rate."""
    summary = []
    for rate in sorted({r["collision_rate"] for r in rows}):
        group = [r for r in rows if r["collision_rate"] == rate]
        entry: dict[str, Any] = {"collision_rate": rate, "seeds": len(group)}
        for col in MEAN_COLUMNS:
            entry[col] = float(np.mean([r[col] for r in group]))
        summary.append(entry)
    return summary


def format_table(summary: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'rate':>5} {'seeds':>5} {'evals':>6} {'store hit%':>10} {'init hits':>9} "
        f"{'plan hits':>9} {'plan hit%':>9} {'dispatched':>10} {'skipped':>8} "
        f"{'avoided~':>8} {'replay hits':>11}"
    )
    lines = [header, "-" * len(header)]
    for s in summary:
        lines.append(
            f"{s['collision_rate']:>5.2f} {s['seeds']:>5d} {s['evaluations']:>6.1f} "
            f"{s['feature_store_hit_rate'] * 100:>10.1f} {s['hits_from_initializer_plans']:>9.1f} "
            f"{s['hits_from_planner_plans']:>9.1f} {s['planner_plan_hit_rate'] * 100:>9.1f} "
            f"{s['groups_dispatched']:>10.1f} {s['groups_skipped']:>8.1f} "
            f"{s['llm_calls_avoided_estimate']:>8.1f} {s['replay_agent_hits']:>11.1f}"
        )
    return "\n".join(lines)


def write_json(
    path: Path, rows: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]], args: Any
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "collision_rates": args.collision_rate,
            "seeds": args.seeds,
            "rollouts": args.rollouts,
            "depth": args.depth,
            "branch": args.branch,
            "deep": args.deep,
            "shared_store": args.shared_store,
            "replay": args.replay,
            "trials": N_TRIALS,
            "initial_features": list(INITIAL),
            "pool": list(POOL),
        },
        "rows": list(rows),
        "summary": list(summary),
    }
    path.write_text(json.dumps(payload, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "--collision-rate",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[0.0, 0.25, 0.5, 1.0],
        help="comma-separated plan-collision rates to sweep (default 0,0.25,0.5,1)",
    )
    parser.add_argument("--seeds", type=int, default=10, help="run seeds 0..N-1")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--branch", type=int, default=3, help="suggestions per node")
    parser.add_argument(
        "--no-deep", dest="deep", action="store_false", help="disable deep simulation"
    )
    parser.add_argument(
        "--shared-store",
        action="store_true",
        help="share one feature store across the seeds of a rate (cross-run reuse)",
    )
    parser.add_argument(
        "--replay", action="store_true", help="re-run each search against its agent cache"
    )
    parser.add_argument("--json", type=Path, default=None, help="write rows and summary here")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logging.getLogger("ctra").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    rows = run_sweep(
        args.collision_rate,
        range(args.seeds),
        rollouts=args.rollouts,
        depth=args.depth,
        branch=args.branch,
        deep=args.deep,
        shared_store=args.shared_store,
        replay=args.replay,
    )
    summary = summarise(rows)
    print(format_table(summary))
    if args.json is not None:
        write_json(args.json, rows, summary, args)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
