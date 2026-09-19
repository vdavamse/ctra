"""Model selection, comparison, and versioning via MLflow.

Provides a unified interface for training both classifiers, comparing results,
and selecting the best model for downstream consumption.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray
    from sklearn.base import BaseEstimator

from ctra.config.settings import ClassifierType, ModelConfig, get_settings
from ctra.models.tabpfn_classifier import PredictionResult, is_tabpfn_available
from ctra.models.xgboost_classifier import XGBoostWrapper

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Trains all configured classifiers, compares them, selects the best.

    Parameters:
        classifiers: List of classifier names to train.  Valid values are
            ``"xgboost"`` and ``"tabpfn"``.  Defaults to ``["xgboost", "tabpfn"]``.
        config: Optional ``ModelConfig`` override.
    """

    def __init__(
        self,
        classifiers: list[str] | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self._config = config or get_settings().model

        if classifiers is not None:
            self._classifier_names = classifiers
        else:
            self._classifier_names = [c.value for c in self._config.classifiers]

        self._wrappers: dict[str, Any] = {}
        self._results: dict[str, PredictionResult] = {}

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | NDArray[Any],
        X_val: pd.DataFrame,
        y_val: pd.Series | NDArray[Any],
    ) -> dict[str, PredictionResult]:
        """Train all configured classifiers and evaluate on validation set.

        Parameters:
            X_train: Training features.
            y_train: Training labels (binary 0/1).
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dict mapping classifier name (``"xgboost"``, ``"tabpfn"``) to
            :class:`PredictionResult` with metrics computed against ``y_val``.
        """
        y_train_arr = np.asarray(y_train)
        y_val_arr = np.asarray(y_val)
        results: dict[str, PredictionResult] = {}

        for clf_name in self._classifier_names:
            logger.info("Training %s ...", clf_name)

            try:
                if clf_name == ClassifierType.XGBOOST.value:
                    wrapper: Any = XGBoostWrapper(config=self._config)
                    wrapper.fit(X_train, y_train_arr, X_val=X_val, y_val=y_val_arr)
                    result = wrapper.predict(X_val, y_true=y_val_arr)

                elif clf_name == ClassifierType.TABPFN.value:
                    if not is_tabpfn_available():
                        logger.warning(
                            "TabPFN is not installed — skipping.  "
                            "Install with: pip install tabpfn 'tabpfn-extensions[interpretability]'"
                        )
                        continue

                    from ctra.models.tabpfn_classifier import TabPFNWrapper

                    wrapper = TabPFNWrapper(config=self._config)
                    wrapper.fit(X_train, y_train_arr)
                    result = wrapper.predict(X_val, y_true=y_val_arr)

                else:
                    logger.warning("Unknown classifier: %s — skipping", clf_name)
                    continue

            except Exception:
                logger.exception("Training failed for %s", clf_name)
                continue

            self._wrappers[clf_name] = wrapper
            results[clf_name] = result
            self._results[clf_name] = result

            logger.info(
                "%s -> ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  Accuracy=%.4f",
                clf_name,
                result.metrics.get("roc_auc", 0.0),
                result.metrics.get("pr_auc", 0.0),
                result.metrics.get("f1", 0.0),
                result.metrics.get("accuracy", 0.0),
            )

        return results

    def get_best_model(
        self,
        results: dict[str, PredictionResult],
        metric: str = "roc_auc",
    ) -> str:
        """Select the best model by the given metric.

        Parameters:
            results: Dict of classifier name -> PredictionResult (as returned
                by :meth:`train_and_evaluate`).
            metric: Metric to compare.  One of ``"roc_auc"``, ``"pr_auc"``,
                ``"f1"``, ``"accuracy"``.  Highest value wins.

        Returns:
            Name of the best classifier.

        Raises:
            ValueError: If ``results`` is empty or no model has the given metric.
        """
        if not results:
            raise ValueError("No results to compare — results dict is empty.")

        valid = {
            name: res.metrics.get(metric, float("-inf"))
            for name, res in results.items()
            if metric in res.metrics
        }

        if not valid:
            raise ValueError(
                f"No model has metric '{metric}'.  Available metrics: "
                f"{set().union(*(set(r.metrics.keys()) for r in results.values()))}"
            )

        best_name = max(valid, key=lambda k: valid[k])
        logger.info(
            "Best model by %s: %s (%.4f)",
            metric,
            best_name,
            valid[best_name],
        )
        return best_name

    def predict_with_best(
        self,
        X_test: pd.DataFrame,
        results: dict[str, PredictionResult],
        metric: str = "roc_auc",
        y_true: pd.Series | NDArray[Any] | None = None,
    ) -> PredictionResult:
        """Run prediction using the best model from training.

        Parameters:
            X_test: Test features.
            results: Dict of classifier name -> PredictionResult from
                :meth:`train_and_evaluate`.
            metric: Metric used to select the best model.
            y_true: If provided, compute metrics against these ground-truth labels.

        Returns:
            :class:`PredictionResult` from the best model on ``X_test``.

        Raises:
            RuntimeError: If the best model's wrapper is not available (e.g.,
                it was not trained in this session).
        """
        best_name = self.get_best_model(results, metric=metric)
        wrapper = self._wrappers.get(best_name)

        if wrapper is None:
            raise RuntimeError(
                f"Wrapper for '{best_name}' is not available.  "
                f"Was it trained in this ModelRegistry session?"
            )

        return wrapper.predict(X_test, y_true=y_true)  # type: ignore[no-any-return]

    def get_wrapper(self, clf_name: str) -> Any:
        """Return the fitted wrapper for a specific classifier.

        Parameters:
            clf_name: Classifier name (``"xgboost"`` or ``"tabpfn"``).

        Raises:
            KeyError: If no fitted wrapper exists for that classifier.
        """
        if clf_name not in self._wrappers:
            raise KeyError(
                f"No fitted wrapper for '{clf_name}'.  Available: {list(self._wrappers.keys())}"
            )
        return self._wrappers[clf_name]

    def log_to_mlflow(self, experiment_name: str = "ctra") -> None:
        """Log all results to MLflow.

        TODO(phase-3): Implement MLflow logging with metrics, params, and
        SHAP artifacts.
        """
        logger.info("MLflow logging not yet implemented (phase-3 TODO)")

    @staticmethod
    def create_classifier(
        classifier_type: ClassifierType,
        config: ModelConfig | None = None,
    ) -> BaseEstimator:
        """Return a configured-but-unfitted sklearn-compatible classifier.

        This is the single point of truth for classifier instantiation.
        The orchestrator calls this instead of constructing classifiers inline,
        ensuring hyperparameters from settings are always honored and the
        correct classifier class is always used.

        Args:
            classifier_type: Which classifier to create.
            config: Optional ModelConfig override. Uses get_settings().model
                if not provided.

        Returns:
            An unfitted sklearn-compatible estimator (XGBClassifier or
            TabPFNClassifier).

        Raises:
            ImportError: If TabPFN is requested but not installed.
            ValueError: If classifier_type is unknown.
        """
        cfg = config or get_settings().model

        if classifier_type == ClassifierType.XGBOOST:
            import xgboost as xgb

            xgb_kwargs: dict[str, Any] = {
                "n_estimators": cfg.xgb_n_estimators,
                "max_depth": cfg.xgb_max_depth,
                "learning_rate": cfg.xgb_learning_rate,
                "eval_metric": "logloss",
                "random_state": 42,  # TODO: expose as cfg.xgb_random_state
            }
            try:
                from packaging.version import Version

                if Version(xgb.__version__) < Version("2.0.0"):
                    xgb_kwargs["use_label_encoder"] = False
            except (ImportError, AttributeError):
                pass
            return xgb.XGBClassifier(**xgb_kwargs)

        elif classifier_type == ClassifierType.TABPFN:
            if not is_tabpfn_available():
                raise ImportError(
                    "TabPFN is not installed. Install with: pip install tabpfn "
                    "'tabpfn-extensions[interpretability]'"
                )
            from tabpfn import TabPFNClassifier

            # Note: TabPFNWrapper also supports tabpfn_finetune (which uses
            # FinetunedTabPFNClassifier) and tabpfn_fit_mode (fit_with_cache).
            # Those are intentionally omitted here because this factory returns
            # a bare estimator for the orchestrator's sklearn Pipeline, which
            # handles fitting separately. Fine-tuning and fit_mode are only
            # relevant when using TabPFNWrapper.fit() directly via ModelRegistry.
            return TabPFNClassifier(
                device=cfg.tabpfn_device,
                n_estimators=cfg.tabpfn_n_estimators,
                random_state=cfg.tabpfn_random_state,
            )

        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
