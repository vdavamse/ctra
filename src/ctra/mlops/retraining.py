"""Automated model retraining pipeline.

Runs a full MCTS feature search + model training cycle and saves the
resulting model to the model store.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ctra.config.settings import MCTSConfig, get_settings
from ctra.models.model_registry import ModelRegistry
from ctra.search.objectives import ObjectiveResult

if TYPE_CHECKING:
    from ctra.mlops.experiment_tracker import ExperimentTracker
    from ctra.mlops.model_store import ModelStore

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Automated model retraining when new data arrives or drift is detected.

    Args:
        tracker: ``ExperimentTracker`` for logging the retraining run.
        store: ``ModelStore`` for saving and loading model versions.
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        store: ModelStore,
    ) -> None:
        self._tracker = tracker
        self._store = store

    # ------------------------------------------------------------------
    # Full retraining
    # ------------------------------------------------------------------

    def retrain(
        self,
        task_splits_dir: str,
        mcts_config: MCTSConfig | None = None,
    ) -> str:
        """Run full retraining: MCTS feature search + model training.

        This method loads train/val splits from ``task_splits_dir``,
        runs the MCTS feature search to find optimal features, trains
        classifiers via ``ModelRegistry``, and saves the result to the
        model store.

        Expected directory layout for ``task_splits_dir``::

            task_splits_dir/
                X_train.parquet
                y_train.parquet
                X_val.parquet
                y_val.parquet
                feature_plans.json   (initial feature plans from proposer)

        Args:
            task_splits_dir: Path to the directory containing train/val splits and
                initial feature plans.
            mcts_config: MCTS configuration override. ``None`` uses defaults.

        Returns:
            The new model version string.
        """
        import json

        config = mcts_config or get_settings().mcts
        splits_path = Path(task_splits_dir)

        # Load data splits (polars for I/O, convert to pandas at ML boundary)
        X_train = pl.read_parquet(splits_path / "X_train.parquet").to_pandas()
        y_train = pl.read_parquet(splits_path / "y_train.parquet").to_pandas().iloc[:, 0]
        X_val = pl.read_parquet(splits_path / "X_val.parquet").to_pandas()
        y_val = pl.read_parquet(splits_path / "y_val.parquet").to_pandas().iloc[:, 0]

        # Load initial feature plans
        fp_path = splits_path / "feature_plans.json"
        with open(fp_path) as f:
            feature_plans: list[dict[str, Any]] = json.load(f)

        task_name = splits_path.name or "retrain"

        # Start experiment tracking
        run_id = self._tracker.start_mcts_run(config, task_name=task_name)

        try:
            # ---- MCTS feature search ----
            # Import here to avoid circular dependencies
            from ctra.search.mcts import MCTSSearch

            initial_features = [fp["name"] for fp in feature_plans]

            # Build evaluate_fn that trains models and returns objectives
            def evaluate_fn(
                features: list[str],
                fidelity: float = 1.0,
            ) -> ObjectiveResult:
                """Train classifiers on the given feature subset and evaluate."""
                # Filter columns to the requested features
                available = [f for f in features if f in X_train.columns]
                if not available:
                    # No features available -- return worst-case
                    return ObjectiveResult(
                        values=np.zeros(len(config.objectives)),
                        names=list(config.objectives),
                        details={},
                    )

                X_tr = X_train[available]
                X_v = X_val[available]

                # Subsample for multi-fidelity
                if fidelity < 1.0:
                    n = max(1, int(len(X_tr) * fidelity))
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(X_tr), size=n, replace=False)
                    X_tr = X_tr.iloc[sorted(idx)]
                    y_tr = y_train.iloc[sorted(idx)]  # type: ignore[call-overload]
                else:
                    y_tr = y_train

                registry = ModelRegistry()
                results = registry.train_and_evaluate(X_tr, y_tr, X_v, y_val)

                if not results:
                    return ObjectiveResult(
                        values=np.zeros(len(config.objectives)),
                        names=list(config.objectives),
                        details={},
                    )

                # Use best model's ROC-AUC
                best_name = registry.get_best_model(results, metric="roc_auc")
                roc_auc = results[best_name].metrics.get("roc_auc", 0.5)

                from ctra.mlops.objectives import MultiObjectiveEvaluator

                evaluator = MultiObjectiveEvaluator(config=config)
                return evaluator.evaluate(
                    feature_set=available,
                    roc_auc=roc_auc,
                )

            # Simple expansion function for retraining (no LLM calls)
            def expand_fn(
                node: Any,
                max_children: int = 4,
            ) -> list[tuple[list[str], str, str]]:
                """Generate child nodes by removing one feature at a time."""
                candidates: list[tuple[list[str], str, str]] = []
                for feat in node.features[:max_children]:
                    child_features = [f for f in node.features if f != feat]
                    if child_features:
                        candidates.append((child_features, "remove", f"remove:{feat}"))
                return candidates

            search = MCTSSearch(
                runner=evaluate_fn,
                task=task_name,
                expand_fn=expand_fn,
                config=config,
            )
            best_node = search.search(initial_features)

            # Log rollout results
            for i, obj_result in enumerate(best_node.objective_history):
                self._tracker.log_rollout(
                    rollout_idx=i,
                    node_id=str(id(best_node)),
                    objective_result=obj_result,
                )

            # ---- Final training with best features ----
            best_features = [f for f in best_node.features if f in X_train.columns]
            if not best_features:
                best_features = list(X_train.columns)

            final_registry = ModelRegistry()
            final_results = final_registry.train_and_evaluate(
                X_train[best_features],
                y_train,
                X_val[best_features],
                y_val,
            )

            # Log model results
            for model_name, result in final_results.items():
                self._tracker.log_model_result(model_name, result, split="val")

            # Determine best model
            best_model_name = final_registry.get_best_model(final_results, metric="roc_auc")

            # Build final objective result
            best_roc_auc = final_results[best_model_name].metrics.get("roc_auc", 0.5)
            from ctra.mlops.objectives import MultiObjectiveEvaluator

            evaluator = MultiObjectiveEvaluator(config=config)
            final_obj = evaluator.evaluate(
                feature_set=best_features,
                roc_auc=best_roc_auc,
            )

            # Filter feature plans to only the selected features
            best_feature_set = set(best_features)
            best_plans = [fp for fp in feature_plans if fp["name"] in best_feature_set]

            # Log best node
            self._tracker.log_best_node(
                node_id=str(id(best_node)),
                feature_plans=best_plans,
                objective_result=final_obj,
            )

            # Save to model store
            version = self._store.save_model(
                model_name=best_model_name,
                registry=final_registry,
                feature_plans=best_plans,
                objective_result=final_obj,
                metadata={
                    "task_splits_dir": str(task_splits_dir),
                    "mlflow_run_id": run_id,
                    "mcts_rollouts": config.num_rollouts,
                    "best_model": best_model_name,
                    "best_roc_auc": best_roc_auc,
                },
            )

            logger.info(
                "Retraining complete: version=%s model=%s roc_auc=%.4f",
                version,
                best_model_name,
                best_roc_auc,
            )
            return version

        finally:
            self._tracker.end_run()
