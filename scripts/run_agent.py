#!/usr/bin/env python
"""Run a single Agent iteration.

Standalone script for debugging or subprocess invocation from
``train_mcts.py`` / ``run_agent_as_subprocess()``.

Usage::

    # Iteration 0 (no previous output)
    python scripts/run_agent.py --task phase2 --output result.pkl

    # Iteration N (resume from previous output)
    python scripts/run_agent.py --task phase2 --input prev.pkl --output result.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys

import dill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ctra.run_agent")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for agent iteration.

    Returns:
        Namespace with task, input, and output arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a single Agent iteration for a clinical trial task.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["phase1", "phase2", "phase3"],
        help="Clinical trial phase.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to dill-serialized AgentOutput from previous iteration. Omit for iteration 0.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the resulting AgentOutput (dill-serialized).",
    )
    return parser.parse_args()


def main() -> None:
    """Run a single Agent iteration for feature engineering.

    Loads benchmark trial data split by phase, instantiates Agent with
    train/val/test splits, and runs a forward pass to engineer features
    and train models. Outputs are dill-serialized for resumable MCTS search.

    If --input is provided, loads previous AgentOutput to resume feature
    engineering from the prior iteration's state.
    """
    args = parse_args()

    from ctra.agents.data_models import Task

    task = Task.from_cli_arg(args.task)
    phase = task.phase
    if phase is None:
        raise ValueError(f"Task {task.name} has no phase number")

    # ---- Configure LLM ----
    from ctra.agents.lm_config import configure_lm

    configure_lm()

    # ---- Load benchmark data ----
    from ctra.data.ctg_loader import CTGLoader

    logger.info("Loading benchmark data for %s...", args.task)
    loader = CTGLoader()
    train_df, val_df, test_df = loader.load_benchmark_splits(
        task="trial_approval",
        phase=phase,
    )

    train_pd = train_df.to_pandas()
    val_pd = val_df.to_pandas()
    test_pd = test_df.to_pandas()

    id_col = "nctid" if "nctid" in train_pd.columns else "trial_id"

    X_train = train_pd[id_col]
    y_train = train_pd["label"]
    X_val = val_pd[id_col]
    y_val = val_pd["label"]
    X_test = test_pd[id_col]
    y_test = test_pd["label"]

    # ---- Load previous output (if resuming) ----
    previous_output = None
    if args.input is not None:
        logger.info("Loading previous output from %s", args.input)
        with open(args.input, "rb") as f:
            previous_output = dill.load(f)

    # ---- Create and run agent ----
    from ctra.agents.orchestrator import Agent

    agent = Agent(
        task=task,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    iteration = "0" if previous_output is None else "N"
    logger.info("Running Agent iteration %s...", iteration)

    output = agent.forward(previous_output=previous_output)

    # ---- Serialize output ----
    with open(args.output, "wb") as f:
        dill.dump(output, f)
    logger.info("Output written to %s", args.output)

    # ---- Print summary ----
    if output.eval_outputs:
        best_eval, _ = output.get_best_eval_output()
        best_name = max(
            output.eval_outputs,
            key=lambda k: output.eval_outputs[k].model_eval_result.roc_auc,
        )
        print(
            f"Iteration {iteration}: "
            f"best_model={best_name}, "
            f"ROC-AUC={best_eval.model_eval_result.roc_auc:.4f}, "
            f"features={len(output.feature_plans)}, "
            f"operation={output.operation}",
            file=sys.stderr,
        )
    else:
        print(f"Iteration {iteration}: no eval outputs", file=sys.stderr)


if __name__ == "__main__":
    main()
