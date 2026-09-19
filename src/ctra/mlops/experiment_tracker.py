"""MLflow-based experiment tracking for CTRA.

Tracks MCTS feature-search runs, per-rollout objective values, model
prediction metrics (ROC-AUC, PR-AUC, F1, accuracy), SHAP feature
importance, and production predictions for drift monitoring.

Falls back to local file storage (``mlruns/``) when no MLflow tracking
server is reachable.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ctra.config.settings import MCTSConfig, get_settings

if TYPE_CHECKING:
    from ctra.agents.runner import RunCacheStats
    from ctra.models.tabpfn_classifier import PredictionResult
    from ctra.search.objectives import ObjectiveResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow-based experiment tracking for CTRA.

    Args:
        experiment_name: MLflow experiment name. Defaults to ``"ctra"``.
        tracking_uri: MLflow tracking URI. ``None`` falls back to
            ``Settings.mlflow_tracking_uri`` (default ``"mlruns"`` -- local
            file store).
    """

    def __init__(
        self,
        experiment_name: str = "ctra",
        tracking_uri: str | None = None,
    ) -> None:
        import mlflow

        self._mlflow = mlflow

        uri = tracking_uri or get_settings().mlops.mlflow_tracking_uri
        try:
            mlflow.set_tracking_uri(uri)
            logger.info("MLflow tracking URI set to %s", uri)
        except Exception:
            # If the URI is unreachable (e.g. remote server down),
            # fall back to local file store.
            fallback = "mlruns"
            logger.warning(
                "Could not set MLflow tracking URI %s; falling back to %s",
                uri,
                fallback,
            )
            mlflow.set_tracking_uri(fallback)

        mlflow.set_experiment(experiment_name)
        self._experiment_name = experiment_name
        self._active_run: Any | None = None

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_mcts_run(self, config: MCTSConfig, task_name: str) -> str:
        """Start tracking a new MCTS run.

        Args:
            config: MCTS configuration used for this run.
            task_name: Human-readable task identifier (e.g. ``"phase2_oncology"``).

        Returns:
            The MLflow ``run_id``.
        """
        self._active_run = self._mlflow.start_run(
            run_name=f"mcts_{task_name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}",
        )
        run_id = self._active_run.info.run_id

        # Log MCTS configuration as params
        self._mlflow.log_params(
            {
                "task_name": task_name,
                "num_rollouts": config.num_rollouts,
                "max_depth": config.max_depth,
                "exploration_constant": config.exploration_constant,
                "adaptive_branching": config.adaptive_branching,
                "min_branch_factor": config.min_branch_factor,
                "max_branch_factor": config.max_branch_factor,
                "objectives": ",".join(config.objectives),
                "max_features": config.max_features,
            }
        )

        logger.info("Started MLflow run %s for task %s", run_id, task_name)
        return run_id  # type: ignore[no-any-return]

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run.

        Args:
            status: The terminal run status recorded in MLflow --
                ``"FINISHED"`` (default) or ``"FAILED"`` when the caller is
                unwinding from an exception.
        """
        if self._active_run is not None:
            self._mlflow.end_run(status=status)
            logger.info("Ended MLflow run %s (%s)", self._active_run.info.run_id, status)
            self._active_run = None

    # ------------------------------------------------------------------
    # MCTS rollout logging
    # ------------------------------------------------------------------

    def log_rollout(
        self,
        rollout_idx: int,
        node_id: str,
        objective_result: ObjectiveResult,
    ) -> None:
        """Log a single MCTS rollout's results.

        Args:
            rollout_idx: Zero-based rollout index.
            node_id: Identifier for the MCTS node evaluated.
            objective_result: Combined objective values from the evaluation.
        """
        step = rollout_idx

        # Log per-objective values as metrics with step
        for name, value in zip(objective_result.names, objective_result.values, strict=True):
            self._mlflow.log_metric(f"rollout_{name}", float(value), step=step)

        # Log feature set size from details if available
        if "feature_count" in objective_result.details:
            self._mlflow.log_metric(
                "rollout_feature_count",
                float(objective_result.details["feature_count"]),
                step=step,
            )

        # Log composite info
        self._mlflow.log_metric(
            "rollout_hypervolume",
            float(np.prod(np.maximum(objective_result.values, 0.0))),
            step=step,
        )

        logger.debug(
            "Logged rollout %d (node=%s): %s",
            rollout_idx,
            node_id,
            dict(zip(objective_result.names, objective_result.values, strict=True)),
        )

    # ------------------------------------------------------------------
    # Cache instrumentation (issue #17)
    # ------------------------------------------------------------------

    def log_cache_stats(self, stats: RunCacheStats, step: int | None = None) -> None:
        """Log both cache layers' counters from a training run.

        Rates first (``agent_cache_hit_rate``, ``feature_store_hit_rate``),
        then the headline counts (``groups_skipped``,
        ``llm_calls_avoided_estimate``, ``llm_calls_made``) and every raw
        counter so the rates can be recomputed.  Each call adds a point at
        ``step`` to the metric's history; ``train_mcts.py`` uses the rollout
        index per rollout and ``num_rollouts`` for the run totals.  Note that
        ``step=None`` is recorded by MLflow at step 0.

        Args:
            stats: The ``RunCacheStats`` the runner partial accumulated.
            step: Optional rollout index for per-rollout history.
        """
        fs = stats.feature_store
        metrics: dict[str, float] = {
            "agent_cache_hit_rate": float(stats.agent_hit_rate),
            "feature_store_hit_rate": float(fs.hit_rate),
            "groups_skipped": float(fs.groups_skipped),
            "llm_calls_avoided_estimate": float(fs.llm_calls_avoided_estimate),
            "llm_calls_made": float(stats.llm_calls_made),
            "agent_hits": float(stats.agent_hits),
            "agent_misses": float(stats.agent_misses),
            "replayed_group_builds": float(stats.replayed_group_builds),
            "feature_lookups": float(fs.feature_lookups),
            "feature_hits": float(fs.feature_hits),
            "groups_dispatched": float(fs.groups_dispatched),
            "hits_from_initializer_plans": float(fs.hits_from_initializer_plans),
            "hits_from_planner_plans": float(fs.hits_from_planner_plans),
            "store_writes": float(fs.store_writes),
        }
        self._mlflow.log_metrics(metrics, step=step)
        logger.debug("Logged cache stats (step=%s): %s", step, metrics)

    # ------------------------------------------------------------------
    # Model result logging
    # ------------------------------------------------------------------

    def log_model_result(
        self,
        model_name: str,
        result: PredictionResult,
        split: str = "val",
    ) -> None:
        """Log model prediction results (metrics + SHAP summary).

        Args:
            model_name: Classifier name (e.g. ``"xgboost"``, ``"tabpfn"``).
            result: The ``PredictionResult`` from the model wrapper.
            split: Data split label (``"val"``, ``"test"``).
        """
        # Log scalar metrics
        for metric_name, metric_value in result.metrics.items():
            key = f"{split}_{metric_name}_{model_name}"
            self._mlflow.log_metric(key, metric_value)

        # Also log model-agnostic metrics (best across models will be tracked)
        for metric_name, metric_value in result.metrics.items():
            self._mlflow.log_metric(f"{split}_{metric_name}", metric_value)

        # Log top SHAP feature names as a param (if SHAP values exist)
        if result.shap_values is not None and len(result.feature_names) > 0:
            mean_abs_shap = np.mean(np.abs(result.shap_values), axis=0)
            top_k = min(10, len(result.feature_names))
            top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
            top_features = [result.feature_names[i] for i in top_indices]
            self._mlflow.log_param(
                f"{split}_top_shap_features_{model_name}",
                ",".join(top_features),
            )

        logger.info(
            "Logged %s results for %s: %s",
            split,
            model_name,
            result.metrics,
        )

    # ------------------------------------------------------------------
    # Best node logging
    # ------------------------------------------------------------------

    def log_best_node(
        self,
        node_id: str,
        feature_plans: list[dict[str, Any]],
        objective_result: ObjectiveResult,
    ) -> None:
        """Log the selected best MCTS node and its feature plans.

        Args:
            node_id: Identifier for the best node.
            feature_plans: List of feature plan dictionaries (each containing name, dtype,
                description, extraction_prompt, sources).
            objective_result: Final objective values for the best node.
        """
        # Log final objective values
        for name, value in zip(objective_result.names, objective_result.values, strict=True):
            self._mlflow.log_metric(f"best_{name}", float(value))

        self._mlflow.log_param("best_node_id", node_id)
        self._mlflow.log_param("best_feature_count", len(feature_plans))

        # Log feature plans as a JSON artifact
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="feature_plans_",
            delete=False,
        ) as f:
            json.dump(feature_plans, f, indent=2, default=str)
            tmp_path = f.name

        try:
            self._mlflow.log_artifact(tmp_path, artifact_path="feature_plans")
        finally:
            os.unlink(tmp_path)

        logger.info(
            "Logged best node %s with %d feature plans: %s",
            node_id,
            len(feature_plans),
            dict(zip(objective_result.names, objective_result.values, strict=True)),
        )

    # ------------------------------------------------------------------
    # Production prediction logging
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        trial_id: str,
        prediction: str,
        probability: float,
        model: str,
    ) -> None:
        """Log a production prediction for drift monitoring.

        Args:
            trial_id: ClinicalTrials.gov NCT ID or internal trial ID.
            prediction: ``"success"`` or ``"failure"``.
            probability: Predicted probability of success.
            model: Model name (e.g. ``"xgboost"``, ``"tabpfn"``).
        """
        self._mlflow.log_metrics(
            {
                "prediction_probability": probability,
            }
        )
        # Tag with trial metadata for later querying
        self._mlflow.set_tags(
            {
                "prediction_trial_id": trial_id,
                "prediction_result": prediction,
                "prediction_model": model,
                "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        logger.debug(
            "Logged prediction: trial=%s prediction=%s prob=%.4f model=%s",
            trial_id,
            prediction,
            probability,
            model,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_best_run(self, metric: str = "val_roc_auc") -> dict[str, Any]:
        """Query MLflow for the best run by a given metric.

        Args:
            metric: Metric name to rank by (e.g. ``"val_roc_auc"``).

        Returns:
            Dictionary with keys ``"run_id"``, ``"metrics"``, ``"params"``.
            Returns an empty dict if no runs are found.
        """
        import mlflow

        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            logger.warning("Experiment %s not found", self._experiment_name)
            return {}

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )

        assert isinstance(runs, pd.DataFrame)

        if runs.empty:
            logger.info("No runs found for experiment %s", self._experiment_name)
            return {}

        best = runs.iloc[0]
        run_id = best["run_id"]

        # Extract metrics and params columns
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        param_cols = [c for c in runs.columns if c.startswith("params.")]

        metrics_dict = {
            c.replace("metrics.", ""): best[c]
            for c in metric_cols
            if best[c] is not None and not (isinstance(best[c], float) and np.isnan(best[c]))
        }
        params_dict = {c.replace("params.", ""): best[c] for c in param_cols if best[c] is not None}

        logger.info(
            "Best run for %s: %s (value=%.4f)",
            metric,
            run_id,
            metrics_dict.get(metric, 0.0),
        )

        return {
            "run_id": run_id,
            "metrics": metrics_dict,
            "params": params_dict,
        }
