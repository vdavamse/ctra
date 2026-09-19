"""XGBoost wrapper with ColumnTransformer preprocessing and SHAP (TreeSHAP).

Unlike TabPFN, XGBoost requires explicit handling of missing values and
categorical encoding via a sklearn ``ColumnTransformer``.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from ctra.config.settings import ModelConfig, get_settings
from ctra.models.tabpfn_classifier import PredictionResult, compute_metrics

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _bool_to_int(X: pd.DataFrame) -> pd.DataFrame:
    """Cast boolean columns to integers (0/1) for XGBoost tree construction.

    XGBoost's tree splits work better with numeric types. This function
    is used as a ``FunctionTransformer`` in the preprocessing pipeline
    to ensure boolean columns are treated as numeric.

    Args:
        X: DataFrame with boolean columns.

    Returns:
        DataFrame with boolean columns cast to int.
    """
    return X.astype(int)


class XGBoostWrapper:
    """Wrapper around ``xgboost.XGBClassifier`` with ColumnTransformer preprocessing
    and TreeSHAP explanations.

    Args:
        random_state: Random seed.
        config: Optional ``ModelConfig`` override.
        **xgb_params: Additional keyword arguments forwarded to
            ``xgb.XGBClassifier``.  These override config defaults.
    """

    def __init__(
        self,
        random_state: int = 42,
        config: ModelConfig | None = None,
        **xgb_params: Any,
    ) -> None:
        self._config = config or get_settings().model
        self._random_state = random_state
        self._xgb_overrides = xgb_params
        self._pipeline: Pipeline | None = None
        self._model: Any = None  # the raw XGBClassifier inside the pipeline
        self._feature_names: list[str] = []
        self._transformed_feature_names: list[str] | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | NDArray[Any],
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | NDArray[Any] | None = None,
    ) -> XGBoostWrapper:
        """Fit XGBoost with automatic preprocessing based on column dtypes.

        Preprocessing pipeline:
            - Numeric columns -> ``SimpleImputer(strategy='median')``
            - Categorical / object columns -> ``SimpleImputer(fill='__missing__')``
              then ``OneHotEncoder(handle_unknown='ignore')``
            - Boolean columns -> cast to ``int`` (0/1)

        Args:
            X: Training features.
            y: Binary labels (0/1).
            X_val: Validation features for early stopping.
            y_val: Validation labels for early stopping.

        Returns:
            self (for method chaining).
        """
        import xgboost as xgb

        self._feature_names = list(X.columns)

        # --- Detect column types ---
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()

        # --- Build ColumnTransformer ---
        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric_cols:
            transformers.append(
                (
                    "num",
                    SimpleImputer(strategy="median"),
                    numeric_cols,
                )
            )
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(strategy="constant", fill_value="__missing__"),
                            ),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical_cols,
                )
            )
        if bool_cols:
            transformers.append(
                (
                    "bool",
                    FunctionTransformer(_bool_to_int, validate=False),
                    bool_cols,
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
        )

        # --- Build XGBClassifier ---
        xgb_kwargs: dict[str, Any] = {
            "n_estimators": self._config.xgb_n_estimators,
            "max_depth": self._config.xgb_max_depth,
            "learning_rate": self._config.xgb_learning_rate,
            "eval_metric": "logloss",
            "random_state": self._random_state,
        }
        # use_label_encoder was removed in xgboost 2.0 (C5 fix)
        try:
            from packaging.version import Version

            if Version(xgb.__version__) < Version("2.0.0"):
                xgb_kwargs["use_label_encoder"] = False
        except (ImportError, AttributeError):
            # packaging not available or xgb.__version__ missing;
            # try setting it and catch any TypeError
            pass
        xgb_kwargs.update(self._xgb_overrides)
        xgb_model = xgb.XGBClassifier(**xgb_kwargs)

        self._pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", xgb_model),
            ]
        )

        # --- Early stopping via fit_params ---
        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            # We need the preprocessor to be fit on training data first so we
            # can transform the validation set for early stopping.
            preprocessor.fit(X)
            X_val_t = preprocessor.transform(X_val)
            fit_params["classifier__eval_set"] = [(X_val_t, np.asarray(y_val))]
            # xgboost >=2.0 requires early_stopping_rounds in the constructor,
            # not as a fit() parameter. Set it on the model directly.
            if self._config.xgb_early_stopping_rounds:
                xgb_model.set_params(
                    early_stopping_rounds=self._config.xgb_early_stopping_rounds,
                )

        logger.info(
            "Fitting XGBoost: %d rows x %d features (%d numeric, %d categorical, %d boolean)",
            len(X),
            len(self._feature_names),
            len(numeric_cols),
            len(categorical_cols),
            len(bool_cols),
        )

        self._pipeline.fit(X, y, **fit_params)
        self._model = self._pipeline.named_steps["classifier"]
        self._is_fitted = True

        # Cache transformed feature names for SHAP explanation mapping
        self._transformed_feature_names = self._get_transformed_feature_names(preprocessor)

        return self

    def predict(
        self,
        X: pd.DataFrame,
        y_true: pd.Series | NDArray[Any] | None = None,
    ) -> PredictionResult:
        """Predict with probabilities, SHAP values, and optional metrics.

        Args:
            X: Test features.
            y_true: If provided, compute classification metrics against these
                ground-truth labels.

        Returns:
            A :class:`PredictionResult` containing predictions, probabilities,
            SHAP values (if enabled), and metrics.
        """
        self._check_fitted()
        assert self._pipeline is not None

        probabilities = self._pipeline.predict_proba(X)
        predictions = self._pipeline.predict(X)

        # SHAP values (best effort)
        shap_values: NDArray[Any] | None = None
        if self._config.shap_enabled:
            try:
                shap_values = self.get_shap_values(X)
            except Exception:
                logger.exception("SHAP computation failed for XGBoost")

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
            model_name="xgboost",
            metrics=metrics,
        )

    def predict_proba(self, X: pd.DataFrame) -> NDArray[Any]:
        """Return class probabilities (shape: n_samples x 2)."""
        self._check_fitted()
        assert self._pipeline is not None
        return self._pipeline.predict_proba(X)  # type: ignore[no-any-return]

    def get_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int | None = None,
    ) -> NDArray[Any]:
        """Compute TreeSHAP values for the positive class.

        TreeSHAP is exact and fast for tree-based models.

        Args:
            X: Test features to explain.
            max_samples: Maximum rows to explain (for speed).

        Returns:
            SHAP values array of shape ``(n_samples, n_features)`` for the
            positive class.  Note: these correspond to the *original* feature
            space when possible, but due to one-hot encoding of categoricals
            the actual SHAP matrix shape may be
            ``(n_samples, n_transformed_features)``.
        """
        self._check_fitted()
        if not self._config.shap_enabled:
            raise RuntimeError("SHAP is disabled in configuration (model.shap_enabled=False)")

        import shap

        max_samples = max_samples or self._config.shap_max_samples
        X_subset = X.head(max_samples)

        # Transform through the preprocessor to get the representation XGBoost sees
        assert self._pipeline is not None
        preprocessor = self._pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X_subset)

        logger.info("Computing XGBoost TreeSHAP values for %d samples", len(X_subset))
        explainer = shap.TreeExplainer(self._model)
        shap_result = explainer.shap_values(X_transformed)

        # TreeExplainer may return a list [class_0, class_1] for binary classification
        # or a single array depending on the XGBoost version / objective
        if isinstance(shap_result, list):
            # Extract positive class (class 1 = success)
            shap_values = shap_result[1]
        elif isinstance(shap_result, np.ndarray) and shap_result.ndim == 3:
            # Shape (n_samples, n_features, n_classes)
            shap_values = shap_result[:, :, 1]
        else:
            shap_values = shap_result

        return shap_values  # type: ignore[no-any-return]

    def save(self, path: Path) -> None:
        """Save the full pipeline (preprocessor + model) via pickle."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "pipeline": self._pipeline,
                    "feature_names": self._feature_names,
                    "transformed_feature_names": self._transformed_feature_names,
                },
                f,
            )
        logger.info("XGBoost pipeline saved to %s", path)

    def load(self, path: Path) -> XGBoostWrapper:
        """Load a previously saved pipeline."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self._pipeline = data["pipeline"]
            self._feature_names = data.get("feature_names", [])
            self._transformed_feature_names = data.get("transformed_feature_names")
        else:
            # Backwards compatibility with old save format (bare pipeline)
            self._pipeline = data
            self._feature_names = []
            self._transformed_feature_names = None

        assert self._pipeline is not None
        self._model = self._pipeline.named_steps["classifier"]
        self._is_fitted = True
        logger.info("XGBoost pipeline loaded from %s", path)
        return self

    def _get_transformed_feature_names(
        self,
        preprocessor: ColumnTransformer,
    ) -> list[str]:
        """Extract transformed feature names from the fitted ColumnTransformer.

        After preprocessing (imputation, one-hot encoding, etc.), the input
        features are expanded. This method gets the post-transform feature
        names for SHAP plotting.

        Args:
            preprocessor: The fitted ``ColumnTransformer`` pipeline.

        Returns:
            List of transformed feature names, or empty list if extraction fails.
        """
        try:
            return list(preprocessor.get_feature_names_out())
        except AttributeError:
            return []

    def _check_fitted(self) -> None:
        """Verify that the model has been fitted before prediction/explanation.

        Raises:
            RuntimeError: If ``fit()`` or ``load()`` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("XGBoostWrapper is not fitted.  Call fit() or load() first.")
