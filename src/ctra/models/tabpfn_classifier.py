"""TabPFN integration wrapper with SHAP interpretability.

TabPFN v2.5 is a pre-trained tabular foundation model that predicts via
in-context learning (no gradient updates).  It handles missing values and
categorical features natively -- no ColumnTransformer needed.

Key constraints (from research/tabpfn-evaluation.md):
    - Max 100,000 rows, 2,000 features (v2.5)
    - High-cardinality categoricals (>50 unique) degrade performance
    - Set random_state for reproducible MCTS evaluations
    - Use fit_mode='fit_with_cache' during MCTS rollouts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from ctra.config.settings import ModelConfig, get_settings

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared result container
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Result of a model prediction."""

    probabilities: NDArray[Any]  # shape (n_samples, 2) — [P(fail), P(success)]
    predictions: NDArray[Any]  # shape (n_samples,) — binary 0/1
    shap_values: NDArray[Any] | None  # shape (n_samples, n_features) — SHAP for positive class
    feature_names: list[str]
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    # {"roc_auc": ..., "pr_auc": ..., "f1": ..., "accuracy": ...}


def compute_metrics(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    y_prob: NDArray[Any],
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_pred: Binary predictions (0/1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary with roc_auc, pr_auc, f1, and accuracy.
    """
    metrics: dict[str, float] = {}
    try:
        roc = float(roc_auc_score(y_true, y_prob))
        # Some sklearn versions return NaN instead of raising for single-class
        if np.isnan(roc):
            raise ValueError("ROC-AUC returned NaN")
        metrics["roc_auc"] = roc
    except ValueError:
        # Only one class present in y_true — ROC-AUC is undefined
        logger.warning("ROC-AUC undefined (single class in y_true); setting to 0.0")
        metrics["roc_auc"] = 0.0

    try:
        pr = float(average_precision_score(y_true, y_prob))
        if np.isnan(pr):
            raise ValueError("PR-AUC returned NaN")
        metrics["pr_auc"] = pr
    except ValueError:
        logger.warning("PR-AUC undefined; setting to 0.0")
        metrics["pr_auc"] = 0.0

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0.0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# TabPFN availability check
# ---------------------------------------------------------------------------

_TABPFN_AVAILABLE = True
_TABPFN_IMPORT_ERROR: str | None = None

try:
    from tabpfn import TabPFNClassifier as _TabPFNClassifier  # noqa: F401
except ImportError as _exc:  # pragma: no cover
    _TABPFN_AVAILABLE = False
    _TABPFN_IMPORT_ERROR = str(_exc)

_TABPFN_EXTENSIONS_AVAILABLE = True
_TABPFN_EXTENSIONS_IMPORT_ERROR: str | None = None

try:
    from tabpfn_extensions import interpretability as _interpretability  # noqa: F401
except ImportError as _exc:  # pragma: no cover
    _TABPFN_EXTENSIONS_AVAILABLE = False
    _TABPFN_EXTENSIONS_IMPORT_ERROR = str(_exc)


def is_tabpfn_available() -> bool:
    """Return whether the tabpfn package is importable."""
    return _TABPFN_AVAILABLE


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class TabPFNWrapper:
    """Wrapper around ``TabPFNClassifier`` with SHAP support for CTRA pipeline.

    Args:
        n_estimators: Number of ensemble estimators.  Defaults to the value in
            ``ModelConfig.tabpfn_n_estimators`` (8).
        random_state: Random seed for reproducibility.  Defaults to
            ``ModelConfig.tabpfn_random_state`` (42).
        config: Optional ``ModelConfig`` override.  When provided, ``n_estimators``
            and ``random_state`` are taken from the config and the constructor
            arguments are ignored.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        random_state: int = 42,
        config: ModelConfig | None = None,
    ) -> None:
        if not _TABPFN_AVAILABLE:
            raise ImportError(
                f"TabPFN is not installed. Install with: pip install tabpfn "
                f"'tabpfn-extensions[interpretability]'  (error: {_TABPFN_IMPORT_ERROR})"
            )

        self._config = config or get_settings().model
        # Allow explicit args to override config when no config is passed
        if config is not None:
            self._n_estimators = self._config.tabpfn_n_estimators
            self._random_state = self._config.tabpfn_random_state
        else:
            self._n_estimators = n_estimators
            self._random_state = random_state

        self._model: Any = None
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: pd.DataFrame, y: pd.Series | NDArray[Any]) -> TabPFNWrapper:
        """Fit TabPFN on the training data.

        TabPFN does not perform gradient-based training.  ``fit()`` stores the
        training data for in-context learning during ``predict()``.

        TabPFN handles missing values (``pd.NA``, ``np.nan``) and categorical
        features natively -- do NOT impute or one-hot encode.

        Args:
            X: Training features.  Raw DataFrame -- no preprocessing needed.
            y: Binary labels (0/1).

        Returns:
            self (for method chaining).
        """
        from tabpfn import TabPFNClassifier

        self._feature_names = list(X.columns)

        device = self._config.tabpfn_device
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device required for TabPFN. No CUDA-capable GPU detected. "
                "Install CUDA toolkit and a CUDA-enabled PyTorch build."
            )

        init_kwargs: dict[str, Any] = {
            "device": device,
            "n_estimators": self._n_estimators,
            "random_state": self._random_state,
        }

        if self._config.tabpfn_finetune:
            from tabpfn.finetuning import FinetunedTabPFNClassifier

            logger.info("Using fine-tuned TabPFN (requires GPU with >= 80GB VRAM)")
            # FinetunedTabPFNClassifier uses n_estimators_finetune and
            # n_estimators_final_inference instead of bare n_estimators.
            # Remove n_estimators from init_kwargs to avoid TypeError (C4 fix).
            finetune_kwargs = {k: v for k, v in init_kwargs.items() if k != "n_estimators"}
            self._model = FinetunedTabPFNClassifier(
                epochs=30,
                learning_rate=2e-5,
                n_estimators_finetune=2,
                n_estimators_final_inference=self._n_estimators,
                **finetune_kwargs,
            )
        else:
            self._model = TabPFNClassifier(**init_kwargs)

        logger.info(
            "Fitting TabPFN: %d rows x %d features, device=%s, finetune=%s",
            len(X),
            len(self._feature_names),
            device,
            self._config.tabpfn_finetune,
        )
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        y_true: pd.Series | NDArray[Any] | None = None,
    ) -> PredictionResult:
        """Predict with probabilities, optional SHAP values, and optional metrics.

        Args:
            X: Test features.
            y_true: If provided, compute classification metrics against these
                ground-truth labels.

        Returns:
            A :class:`PredictionResult` containing predictions, probabilities,
            SHAP values (if enabled and available), and metrics.
        """
        self._check_fitted()

        probabilities = self._model.predict_proba(X)
        predictions = self._model.predict(X)

        # SHAP values (best effort)
        shap_values: NDArray[Any] | None = None
        if self._config.shap_enabled:
            try:
                shap_values = self.get_shap_values(X)
            except Exception:
                logger.exception("SHAP computation failed for TabPFN")

        # Metrics (only when labels are available)
        metrics: dict[str, float] = {}
        if y_true is not None:
            y_true_arr = np.asarray(y_true)
            metrics = compute_metrics(y_true_arr, predictions, probabilities[:, 1])

        return PredictionResult(
            probabilities=probabilities,
            predictions=predictions,
            shap_values=shap_values,
            feature_names=self._feature_names,
            model_name="tabpfn",
            metrics=metrics,
        )

    def predict_proba(self, X: pd.DataFrame) -> NDArray[Any]:
        """Return class probabilities (shape: n_samples x 2)."""
        self._check_fitted()
        return self._model.predict_proba(X)  # type: ignore[no-any-return]

    def get_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int | None = None,
    ) -> NDArray[Any]:
        """Compute SHAP values using tabpfn-extensions.

        Uses permutation SHAP (the only method supported by TabPFN).
        Returns SHAP values for the positive class (class 1 = success).

        Args:
            X: Test features to explain.
            max_samples: Maximum rows to explain (for speed).  Defaults to
                ``ModelConfig.shap_max_samples`` (100).

        Returns:
            SHAP values array of shape ``(n_samples, n_features)`` for the
            positive class.
        """
        self._check_fitted()

        if not self._config.shap_enabled:
            raise RuntimeError("SHAP is disabled in configuration (model.shap_enabled=False)")

        if not _TABPFN_EXTENSIONS_AVAILABLE:
            raise ImportError(
                f"tabpfn-extensions is not installed. Install with: "
                f"pip install 'tabpfn-extensions[interpretability]'  "
                f"(error: {_TABPFN_EXTENSIONS_IMPORT_ERROR})"
            )

        from tabpfn_extensions import interpretability

        max_samples = max_samples or self._config.shap_max_samples
        X_subset = X.head(max_samples)

        logger.info("Computing TabPFN SHAP values for %d samples", len(X_subset))
        shap_values = interpretability.shap.get_shap_values(
            estimator=self._model,
            test_x=X_subset,
            attribute_names=self._feature_names,
            algorithm="permutation",
        )

        # get_shap_values may return values for both classes; extract positive class
        if isinstance(shap_values, list):
            # Older tabpfn-extensions: list of [class0_vals, class1_vals]
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Shape (n_samples, n_features, n_classes) — take class 1
            shap_values = shap_values[:, :, 1]

        return shap_values  # type: ignore[no-any-return]

    def save(self, path: Path) -> None:
        """Save the fitted model using TabPFN's dedicated persistence API."""
        self._check_fitted()
        from tabpfn.model_loading import save_fitted_tabpfn_model

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_fitted_tabpfn_model(self._model, str(path))
        logger.info("TabPFN model saved to %s", path)

    def load(self, path: Path, device: str = "cpu") -> TabPFNWrapper:
        """Load a previously saved model.

        Cross-device loading is supported (train on GPU, load on CPU).
        """
        from tabpfn.model_loading import load_fitted_tabpfn_model

        self._model = load_fitted_tabpfn_model(str(path), device=device)
        self._is_fitted = True
        logger.info("TabPFN model loaded from %s (device=%s)", path, device)
        return self

    def _check_fitted(self) -> None:
        """Verify that the model has been fitted before prediction/explanation.

        Raises:
            RuntimeError: If ``fit()`` or ``load()`` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("TabPFNWrapper is not fitted.  Call fit() or load() first.")
