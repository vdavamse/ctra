#!/usr/bin/env python
"""Predict clinical trial outcome for a single NCT ID.

Loads the best model and feature plans from a completed MCTS training run,
builds features on-the-fly via RAG tools, and outputs P(success).

Usage::

    python scripts/predict.py --model-dir .output/ --nctid NCT00110279
    python scripts/predict.py --model-dir .output/ --nctid NCT00110279 --format json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import dill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ctra.predict")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for trial outcome prediction.

    Returns:
        Namespace with model_dir, nctid, and format arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict clinical trial outcome for a given NCT ID.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to MCTS output directory (from train_mcts.py).",
    )
    parser.add_argument(
        "--nctid",
        required=True,
        help="NCT ID of the trial to predict (e.g. NCT00110279).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    return parser.parse_args()


def main() -> None:
    """Predict clinical trial outcome for a single NCT ID.

    Loads trained MCTS model and feature plans from output directory,
    builds features on-the-fly via RAG agents, and outputs success probability
    with optional JSON or text formatting.

    Exits with status 1 if feature_plans.json or best_model.pkl not found
    in the model directory, or if feature building fails.
    """
    args = parse_args()
    model_dir = Path(args.model_dir)
    nctid = args.nctid

    # ---- Validate inputs ----
    plans_path = model_dir / "feature_plans.json"
    model_path = model_dir / "best_model.pkl"

    if not plans_path.exists():
        print(f"Error: feature_plans.json not found in {model_dir}", file=sys.stderr)
        print("Run train_mcts.py first to generate the model.", file=sys.stderr)
        sys.exit(1)

    if not model_path.exists():
        print(f"Error: best_model.pkl not found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- Configure LLM (needed for feature building) ----
    from ctra.agents.lm_config import configure_lm

    logger.info("Configuring LLM for feature building...")
    configure_lm()

    # ---- Load feature plans ----
    from ctra.agents.runner import load_feature_plans_from_json

    logger.info("Loading feature plans from %s", plans_path)
    feature_plans = load_feature_plans_from_json(plans_path)
    logger.info("Loaded %d feature plans", len(feature_plans))

    # ---- Load trained pipeline ----
    logger.info("Loading model from %s", model_path)
    with open(model_path, "rb") as f:
        pipeline = dill.load(f)

    # ---- Build features for the trial ----
    from ctra.agents.feature_builder import compute_features
    from ctra.agents.feature_grouper import FeatureGrouper
    from ctra.agents.feature_utils import features_to_df

    logger.info("Building features for %s...", nctid)

    # Determine task from results.json if available
    from ctra.agents.data_models import Task

    results_path = model_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        task_arg = results.get("task", "phase2")
    else:
        task_arg = "phase2"  # fallback

    task = Task.from_cli_arg(task_arg)
    task_description = task.description

    grouper = FeatureGrouper()

    raw_features, none_explanations, _builder_metadata = compute_features(
        grouper=grouper,
        nctids=[nctid],
        task_description=task_description,
        plans=feature_plans,
    )

    if nctid not in raw_features or not raw_features[nctid]:
        print(f"Error: Failed to build features for {nctid}", file=sys.stderr)
        if nctid in none_explanations:
            print(f"Reasons: {json.dumps(none_explanations[nctid], indent=2)}", file=sys.stderr)
        sys.exit(1)

    # ---- Convert to DataFrame and predict ----
    features_df = features_to_df(raw_features)
    feature_cols = [c for c in features_df.columns if c != "id"]

    logger.info("Running prediction with %d features...", len(feature_cols))
    proba = pipeline.predict_proba(features_df[feature_cols])
    p_success = float(proba[0, 1])
    prediction = "success" if p_success >= 0.5 else "failure"

    # ---- Output ----
    if args.format == "json":
        output = {
            "trial_id": nctid,
            "probability_success": round(p_success, 4),
            "prediction": prediction,
            "n_features": len(feature_cols),
            "features_used": feature_cols,
        }
        if none_explanations.get(nctid):
            output["none_explanations"] = none_explanations[nctid]
        print(json.dumps(output, indent=2))
    else:
        print(f"Trial {nctid}: P(success) = {p_success:.4f} → {prediction}")
        print(f"  Features used: {len(feature_cols)}")
        if none_explanations.get(nctid):
            print(f"  Features with missing data: {len(none_explanations[nctid])}")


if __name__ == "__main__":
    main()
