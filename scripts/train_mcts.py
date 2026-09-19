#!/usr/bin/env python
"""MCTS training script for clinical trial outcome prediction.

Runs the full MCTS search loop: each evaluation spawns an isolated
subprocess (via ``run_agent_as_subprocess``) that creates an Agent,
loads benchmark data, and runs a single iteration.

Usage::

    # Train Phase 2 model with 20 rollouts
    python scripts/train_mcts.py --task phase2 --rollouts 20 --depth 10

    # Resume from checkpoint
    python scripts/train_mcts.py --task phase2 --resume .output/phase2/checkpoint.pkl

    # Custom output directory
    python scripts/train_mcts.py --task phase2 --output-dir .output/phase2_v1/
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dill
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ctra.train")

# Suppress noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MCTS training.

    Returns:
        Namespace with task, rollouts, depth, resume, output_dir, and
        checkpoint_every arguments.  ``main`` adds ``run_id`` before the
        first checkpoint is written, so ``vars(args)`` persists it.
    """
    parser = argparse.ArgumentParser(
        description="Train a clinical trial outcome prediction model via MCTS feature search.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["phase1", "phase2", "phase3"],
        help="Clinical trial phase to train on.",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=None,
        help="Number of MCTS rollouts (default: from config, typically 20).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Maximum MCTS tree depth (default: from config, typically 10).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint pickle to resume from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".output",
        help="Directory for outputs and checkpoints (default: .output/).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N rollouts (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    """Run MCTS search for optimal feature set and trained model.

    Initializes or resumes MCTSSearch, runs the configured number of rollouts
    with per-rollout callbacks for progress tracking and checkpointing, and saves
    outputs (feature_plans.json, best_model.pkl, results.json, and final checkpoint).

    Each rollout spawns an isolated subprocess running Agent to ensure clean
    feature engineering iterations with no state leakage between evaluations.
    """
    args = parse_args()

    from ctra.agents.data_models import Task
    from ctra.agents.runner import RunCacheStats, run_agent_as_subprocess
    from ctra.config.settings import get_settings
    from ctra.search.mcts import MCTSSearch

    task = Task.from_cli_arg(args.task)
    output_dir = Path(args.output_dir) / task.output_subdir

    settings = get_settings()

    # Override config if CLI args provided
    if args.rollouts is not None:
        settings.mcts.num_rollouts = args.rollouts
    if args.depth is not None:
        settings.mcts.max_depth = args.depth

    start_rollout = 0
    mcts = None
    checkpoint: dict[str, Any] | None = None

    if args.resume:
        # ---- Resume from checkpoint ----
        logger.info("Resuming from checkpoint: %s", args.resume)
        with open(args.resume, "rb") as f:
            checkpoint = dill.load(f)
        ckpt_args: dict[str, Any] = checkpoint.get("args") or {}
        if ckpt_args.get("task") != args.task:
            raise SystemExit(
                f"--task {args.task} does not match the checkpoint's task "
                f"{ckpt_args.get('task')!r} ({args.resume}); a checkpoint resumes "
                "the phase it was trained on."
            )
        ckpt_output_dir = Path(ckpt_args.get("output_dir") or "").resolve()
        if ckpt_output_dir != Path(args.output_dir).resolve():
            logger.warning(
                "--output-dir %s differs from the checkpoint's %s: outputs and the "
                "agent cache land under the new directory",
                args.output_dir,
                ckpt_args.get("output_dir"),
            )
        # The run id persisted in the checkpoint keeps the resumed process on
        # its own agent cache; checkpoints written before run ids existed
        # fall back to the checkpoint file's stem.
        run_id = ckpt_args.get("run_id") or Path(args.resume).stem
    else:
        run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    # Set before any checkpoint is dumped: ``vars(args)`` persists it.
    args.run_id = run_id

    # The runner's cache is keyed on a run-stable node id (rollout, lineage,
    # parent plans, features), so it must be scoped per *run*, not per
    # output directory: a second fresh run over the same ``--output-dir``
    # would otherwise replay the first run's pickles (starting with the
    # root evaluation at rollout 0) instead of running the agent.  A resume
    # reuses its run id and keeps hitting its own pre-crash entries.
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_cache_dir = output_dir / "agent_cache" / run_id
    # Cache counters for *this process* (issue #17): agent-cache hits and
    # misses at the runner, and the feature-store counters each child sends
    # back.  Bound into the partial so the runner updates them in place; a
    # resume starts a fresh set (its hits on the pre-crash pickles are the
    # figure of interest), so results.json describes the process that wrote it.
    stats = RunCacheStats()
    runner = functools.partial(run_agent_as_subprocess, cache_dir=agent_cache_dir, stats=stats)
    logger.info("Run id %s — agent cache at %s", run_id, agent_cache_dir)

    if checkpoint is not None:
        mcts = checkpoint["mcts"]
        # Re-point the pickled runner at this run's cache directory.
        mcts.set_runner(runner)
        # The pickled search keeps the reference point it was started under.
        # That geometry is deliberately kept — the hypervolume ranking in
        # ``_select_best`` must not change mid-run — but a checkpoint from
        # before issue #18 moved the default to [0.5, 0.0] ranks by the
        # origin, and the operator should know.
        ckpt_reference = list(mcts.config.reference_point)
        current_reference = list(settings.mcts.reference_point)
        if ckpt_reference != current_reference:
            logger.warning(
                "Checkpoint %s was started with reference_point %s; the current "
                "settings say %s. The resumed search keeps the checkpoint's "
                "reference point.",
                args.resume,
                ckpt_reference,
                current_reference,
            )
        # Same for the backpropagation rule (issue #16): the tree's UCB state
        # was folded under the checkpoint's rule, and switching rules mid-run
        # would mix aggregates.  A checkpoint pickled before the field existed
        # ran under ``mean``.
        ckpt_backprop = getattr(mcts.config, "backprop", "mean")
        current_backprop = settings.mcts.backprop
        if ckpt_backprop != current_backprop:
            logger.warning(
                "Checkpoint %s was started with backprop %r; the current settings "
                "say %r. The resumed search keeps the checkpoint's rule.",
                args.resume,
                ckpt_backprop,
                current_backprop,
            )
        start_rollout = checkpoint["rollout"] + 1
        logger.info("Resumed at rollout %d", start_rollout)
    else:
        # ---- Fresh start ----
        mcts = MCTSSearch(
            runner=runner,
            task=task,
        )

    # ---- Define rollout callback ----
    checkpoint_every = args.checkpoint_every
    total_rollouts = settings.mcts.num_rollouts
    search_start = time.monotonic()

    try:
        from tqdm import tqdm

        pbar = tqdm(total=total_rollouts, initial=start_rollout, desc="MCTS Rollouts")
    except ImportError:
        pbar = None

    def on_rollout(
        rollout_idx: int, node: Any, objective_vector: NDArray[np.floating[Any]]
    ) -> None:
        elapsed = time.monotonic() - search_start
        logger.info(
            "Rollout %d/%d complete — objectives=%s, elapsed=%.0fs",
            rollout_idx + 1,
            total_rollouts,
            np.round(objective_vector, 4),
            elapsed,
        )

        # Running totals: ``on_rollout`` sees only the last node of a deep
        # rollout, so the counters live at the runner boundary, not on nodes.
        fs = stats.feature_store
        logger.info(
            "Rollout %d/%d cache — agent hits/misses=%d/%d, feature-store hit rate=%.1f%% "
            "(%d/%d), groups skipped=%d, LLM calls avoided~%d, LLM calls made=%d",
            rollout_idx + 1,
            total_rollouts,
            stats.agent_hits,
            stats.agent_misses,
            fs.hit_rate * 100,
            fs.feature_hits,
            fs.feature_lookups,
            fs.groups_skipped,
            fs.llm_calls_avoided_estimate,
            stats.llm_calls_made,
        )

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "acc": f"{objective_vector[0]:.3f}",
                    "pars": f"{objective_vector[1]:.3f}" if len(objective_vector) > 1 else "?",
                }
            )

        # Periodic checkpoint
        if (rollout_idx + 1) % checkpoint_every == 0:
            ckpt_path = output_dir / "checkpoint.pkl"
            logger.info("Saving checkpoint to %s", ckpt_path)
            with open(ckpt_path, "wb") as fw:
                dill.dump(
                    {
                        "mcts": mcts,
                        "rollout": rollout_idx,
                        "args": vars(args),
                    },
                    fw,
                )

    # ---- Run MCTS search ----
    logger.info(
        "Starting MCTS search: %d rollouts, depth %d, %d objectives",
        total_rollouts,
        settings.mcts.max_depth,
        len(settings.mcts.objectives),
    )

    best_node = mcts.search(
        initial_features=[],
        on_rollout=on_rollout,
        start_rollout=start_rollout,
    )

    if pbar is not None:
        pbar.close()

    total_elapsed = time.monotonic() - search_start
    logger.info("Search complete in %.0fs", total_elapsed)

    # ---- Save outputs ----
    from ctra.agents.feature_utils import dump_as_json

    # 1. Feature plans — from best node's eval_output
    best_output = best_node.eval_output

    if best_output is not None and best_output.feature_plans:
        plans_path = output_dir / "feature_plans.json"
        plans_path.write_text(dump_as_json(best_output.feature_plans))
        logger.info("Saved feature plans to %s", plans_path)

        # 2. Best model pipeline
        if best_output.eval_outputs:
            best_eval, _ = best_output.get_best_eval_output()
            model_path = output_dir / "best_model.pkl"
            with open(model_path, "wb") as fw:
                dill.dump(best_eval.model_eval_result.pipeline, fw)
            logger.info("Saved best model to %s", model_path)

    # 3. Full checkpoint for resume
    final_ckpt = output_dir / "mcts_state.pkl"
    with open(final_ckpt, "wb") as fw:
        dill.dump(
            {
                "mcts": mcts,
                "rollout": total_rollouts - 1,
                "args": vars(args),
            },
            fw,
        )
    logger.info("Saved final checkpoint to %s", final_ckpt)

    # 4. Results summary
    # ``best_objectives`` is the best node's *own* evaluation (issue #15): the
    # score of ``best_features``.  Each history entry snapshots the feature
    # set it scored and the accessor ranks only the entries matching the
    # node's current set, so a re-evaluation that changed the plans cannot
    # leave this field describing an earlier set than the ``eval_output``
    # shipped in ``feature_plans.json`` (a checkpoint from before the snapshot
    # existed is the one exception: its entries carry no set and all count).
    # Before that fix this field held ``mean_reward``, the average over the
    # node's subtree, which is a different (usually lower) number; it is kept
    # alongside as ``best_mean_objectives`` for continuity with older runs.
    # That value is ``value_estimate(best_node)``, the vector UCB exploited
    # under the run's backprop rule (issue #16): a subtree mean under
    # ``mean``, the elementwise maximum under ``max``, the best realised
    # vector under ``max_hv``.  The key is kept for continuity and
    # ``backprop`` records the rule; a checkpoint from before the field
    # existed ran under ``mean``.
    best_own = mcts.best_own_objectives(best_node)
    backprop = getattr(mcts.config, "backprop", "mean")
    best_estimate = mcts.value_estimate(best_node)
    results = {
        "task": args.task,
        "rollouts": total_rollouts,
        "depth": settings.mcts.max_depth,
        "backprop": backprop,
        "best_features": best_node.features,
        "best_objectives": best_own.tolist(),
        "best_mean_objectives": best_estimate.tolist(),
        "total_nodes": len(mcts.all_nodes),
        "elapsed_seconds": round(total_elapsed, 1),
        # Both cache layers for this process (issue #17); see RunCacheStats.
        "cache": stats.as_dict(),
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved results to %s", results_path)

    # ---- Print summary ----
    print(f"\n{'=' * 60}")
    print(f"MCTS Search Complete — {args.task}")
    print(f"{'=' * 60}")
    print(f"  Rollouts:    {total_rollouts}")
    print(f"  Total nodes: {len(mcts.all_nodes)}")
    print(f"  Best features ({len(best_node.features)}):")
    for feat in best_node.features:
        print(f"    - {feat}")
    print(f"  Best objectives: {np.round(best_own, 4)}")
    estimate_label = {"mean": "Subtree mean", "max": "Subtree max"}.get(
        backprop, "Best subtree point"
    )
    print(f"  {estimate_label + ':':<16} {np.round(best_estimate, 4)}")
    fs = stats.feature_store
    print(
        f"  Agent cache: {stats.agent_hits} hits / {stats.agent_misses} misses "
        f"({stats.agent_hit_rate:.1%}), {stats.replayed_group_builds} group builds replayed"
    )
    print(
        f"  Feature store: {fs.feature_hits}/{fs.feature_lookups} hits ({fs.hit_rate:.1%}), "
        f"{fs.groups_skipped} groups skipped, ~{fs.llm_calls_avoided_estimate} LLM calls "
        f"avoided, {stats.llm_calls_made} made"
    )
    print(f"  Time: {total_elapsed:.0f}s")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
