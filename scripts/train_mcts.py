#!/usr/bin/env python
"""MCTS training script for clinical trial outcome prediction.

Runs the full MCTS search loop: each evaluation spawns an isolated
subprocess (via ``run_agent_as_subprocess``) that creates an Agent,
loads benchmark data, and runs a single iteration.

Usage::

    # Train Phase 2 model with 20 rollouts
    python scripts/train_mcts.py --task phase2 --rollouts 20 --depth 10

    # Resume from checkpoint
    python scripts/train_mcts.py --task phase2 --resume .output/checkpoint.pkl

    # Custom output directory
    python scripts/train_mcts.py --task phase2 --output-dir runs/phase2_v1/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
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
        checkpoint_every arguments.
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
    from ctra.agents.runner import run_agent_as_subprocess
    from ctra.config.settings import get_settings
    from ctra.search.mcts import MCTSSearch

    task = Task.from_cli_arg(args.task)
    output_dir = Path(args.output_dir) / task.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()

    # Override config if CLI args provided
    if args.rollouts is not None:
        settings.mcts.num_rollouts = args.rollouts
    if args.depth is not None:
        settings.mcts.max_depth = args.depth

    start_rollout = 0
    mcts = None

    if args.resume:
        # ---- Resume from checkpoint ----
        logger.info("Resuming from checkpoint: %s", args.resume)
        with open(args.resume, "rb") as f:
            checkpoint = dill.load(f)
        mcts = checkpoint["mcts"]
        start_rollout = checkpoint["rollout"] + 1
        logger.info("Resumed at rollout %d", start_rollout)
    else:
        # ---- Fresh start ----
        mcts = MCTSSearch(
            runner=run_agent_as_subprocess,
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
    results = {
        "task": args.task,
        "rollouts": total_rollouts,
        "depth": settings.mcts.max_depth,
        "best_features": best_node.features,
        "best_objectives": best_node.mean_reward.tolist(),
        "total_nodes": len(mcts.all_nodes),
        "elapsed_seconds": round(total_elapsed, 1),
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
    print(f"  Best objectives: {np.round(best_node.mean_reward, 4)}")
    print(f"  Time: {total_elapsed:.0f}s")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
