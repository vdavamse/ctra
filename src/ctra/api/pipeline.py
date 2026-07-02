"""Prediction pipeline connecting the API layer to the ML backend.

Manages model loading, feature extraction, and prediction.  For the MVP
the pipeline loads pre-computed feature DataFrames (from MCTS output or
manual feature engineering) rather than running the full LLM agent pipeline
on every request.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from ctra import __version__
from ctra.api.schemas import (
    BatchMetadata,
    BatchPredictResponse,
    FeatureExplanation,
    ModelInfo,
    PredictionMetadata,
    PredictResponse,
)
from ctra.config.settings import DataSource, get_settings

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ctra.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Paths relative to the configured output directory
_FEATURE_CACHE_FILENAME = "cached_features.parquet"
_FEATURE_PLAN_FILENAME = "feature_plan.json"


class PredictionPipeline:
    """Manages model loading, feature extraction, and prediction for the API.

    Lifecycle:
        1. ``await pipeline.initialize()`` during FastAPI startup
        2. ``await pipeline.predict(...)`` per request
        3. ``await pipeline.shutdown()`` during FastAPI shutdown
    """

    def __init__(self) -> None:
        """Initialize the prediction pipeline.

        Prepares internal state for model loading, feature extraction, and
        prediction. Call ``await initialize()`` before making predictions.
        """
        self._registry: ModelRegistry | None = None
        # Per-phase model wrappers: {phase_int_or_None: {clf_name: wrapper}}
        self._phase_wrappers: dict[int | None, dict[str, Any]] = {}
        # Legacy flat wrappers (backward compat, loaded from unnamespaced dir)
        self._wrappers: dict[str, Any] = {}  # clf_name -> fitted wrapper
        self._results: dict[str, Any] = {}  # clf_name -> PredictionResult from training
        self._feature_plans: list[dict[str, Any]] | None = None
        self._feature_set_id: str = "default"
        self._feature_cache: pl.DataFrame | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load models, feature plans, and cached features on startup."""
        settings = get_settings()
        output_dir = settings.output_dir

        # 1. Try to load cached features (pre-computed by MCTS pipeline)
        feature_cache_path = output_dir / _FEATURE_CACHE_FILENAME
        if feature_cache_path.exists():
            logger.info("Loading cached feature DataFrame from %s", feature_cache_path)
            self._feature_cache = await asyncio.to_thread(pl.read_parquet, feature_cache_path)
            logger.info(
                "Feature cache loaded: %d rows x %d columns",
                len(self._feature_cache),
                len(self._feature_cache.columns),
            )
        else:
            logger.warning(
                "No cached features found at %s.  "
                "Predictions will fail until features are available.",
                feature_cache_path,
            )

        # 2. Try to load feature plan (JSON list of FeatureSpec dicts)
        feature_plan_path = output_dir / _FEATURE_PLAN_FILENAME
        if feature_plan_path.exists():
            import json

            raw = await asyncio.to_thread(feature_plan_path.read_text)
            self._feature_plans = json.loads(raw)
            self._feature_set_id = (
                self._feature_plans[0].get("set_id", "default")
                if self._feature_plans
                else "default"
            )
            logger.info(
                "Loaded feature plan with %d features (set_id=%s)",
                len(self._feature_plans),
                self._feature_set_id,
            )

        # 3. Try to load persisted model artifacts
        # First: scan for phase-namespaced subdirectories (output/phase1/, phase2/, phase3/)
        for phase_num in (1, 2, 3):
            phase_dir = output_dir / f"phase{phase_num}"
            if phase_dir.is_dir():
                logger.info("Loading phase %d model artifacts from %s", phase_num, phase_dir)
                phase_wrappers: dict[str, Any] = {}
                await self._load_model_artifacts_from_dir(phase_dir, phase_wrappers)
                if phase_wrappers:
                    self._phase_wrappers[phase_num] = phase_wrappers
                    logger.info(
                        "Phase %d models loaded: %s", phase_num, list(phase_wrappers.keys())
                    )

        # Fallback: load from flat output dir (backward compat with pre-phase-isolation)
        await self._load_model_artifacts(output_dir)

        all_loaded = list(self._wrappers.keys())
        for p, pw in sorted(self._phase_wrappers.items()):
            all_loaded.extend(f"phase{p}/{n}" for n in pw)

        self._initialized = True
        logger.info(
            "PredictionPipeline initialized.  Models loaded: %s",
            all_loaded or ["none"],
        )

    async def shutdown(self) -> None:
        """Release resources on application shutdown."""
        self._wrappers.clear()
        self._results.clear()
        self._feature_cache = None
        self._initialized = False
        logger.info("PredictionPipeline shut down")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    async def predict(
        self,
        trial_id: str,
        model: str | None = None,
        include_shap: bool = True,
        phase: int | None = None,
    ) -> PredictResponse:
        """Full prediction pipeline for a single trial.

        Steps:
            1. Load trial data (from CTG parquet / API or internal parquet)
            2. Build features (from cache or, in future, via LLM agents)
            3. Run model prediction
            4. Format response with SHAP explanations
        """
        self._check_initialized()
        start = time.perf_counter()

        # 1. Resolve trial features
        features_df = await self._get_features(trial_id)

        # 2. Select model wrapper (phase-specific if available)
        wrapper, clf_name = self._resolve_wrapper(model, phase=phase)

        # 3. Run prediction
        result = await asyncio.to_thread(wrapper.predict, features_df)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # 4. Build response
        prob_success = float(result.probabilities[0, 1])
        prob_failure = float(result.probabilities[0, 0])
        prediction = "success" if result.predictions[0] == 1 else "failure"

        # 5. SHAP explanations
        explanations: list[FeatureExplanation] = []
        if include_shap and result.shap_values is not None:
            explanations = self._build_explanations(
                shap_values=result.shap_values[0],
                feature_names=result.feature_names,
                feature_values=features_df.iloc[0],
            )

        return PredictResponse(
            trial_id=trial_id,
            phase=phase,
            prediction=prediction,
            probability_success=prob_success,
            probability_failure=prob_failure,
            model_used=clf_name,
            features=explanations,
            metadata=PredictionMetadata(
                model_version=__version__,
                feature_set_id=self._feature_set_id,
                n_features=len(features_df.columns),
                inference_time_ms=round(elapsed_ms, 2),
            ),
        )

    async def predict_batch(
        self,
        trial_ids: list[str],
        model: str | None = None,
        include_shap: bool = True,
        phase: int | None = None,
    ) -> BatchPredictResponse:
        """Batch prediction with concurrent feature extraction."""
        self._check_initialized()
        start = time.perf_counter()

        predictions: list[PredictResponse] = []
        failed = 0

        # Run predictions concurrently (with bounded concurrency)
        sem = asyncio.Semaphore(10)

        async def _predict_one(tid: str) -> PredictResponse | None:
            async with sem:
                try:
                    return await self.predict(
                        trial_id=tid,
                        model=model,
                        include_shap=include_shap,
                        phase=phase,
                    )
                except Exception:
                    logger.exception("Prediction failed for %s", tid)
                    return None

        tasks = [_predict_one(tid) for tid in trial_ids]
        results = await asyncio.gather(*tasks)

        for r in results:
            if r is not None:
                predictions.append(r)
            else:
                failed += 1

        total_ms = (time.perf_counter() - start) * 1000.0

        return BatchPredictResponse(
            predictions=predictions,
            batch_metadata=BatchMetadata(
                total_trials=len(trial_ids),
                successful_predictions=len(predictions),
                failed_predictions=failed,
                total_time_ms=round(total_ms, 2),
            ),
        )

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def get_model_info(self) -> list[ModelInfo]:
        """Return info about all loaded models."""
        infos: list[ModelInfo] = []
        for clf_name, wrapper in self._wrappers.items():
            n_features = len(getattr(wrapper, "_feature_names", []))
            training_metrics = self._results.get(clf_name, {})
            if hasattr(training_metrics, "metrics"):
                training_metrics = training_metrics.metrics

            infos.append(
                ModelInfo(
                    name=clf_name,
                    type=clf_name,
                    version=__version__,
                    feature_set_id=self._feature_set_id,
                    n_features=n_features,
                    training_metrics=training_metrics if isinstance(training_metrics, dict) else {},
                )
            )
        return infos

    def get_loaded_model_names(self) -> list[str]:
        """Return names of loaded classifiers."""
        return list(self._wrappers.keys())

    def get_index_status(self) -> dict[str, str]:
        """Check readiness of RAG data sources."""
        settings = get_settings()
        status: dict[str, str] = {}

        source_paths = {
            DataSource.CTG.value: settings.data.ctg_parquet,
            DataSource.PUBMED.value: settings.data.pubmed_parquet,
            DataSource.CHEMBL.value: settings.data.chembl_parquet,
            DataSource.FAERS.value: settings.data.faers_parquet,
        }

        for source, path in source_paths.items():
            if Path(path).exists():
                status[source] = "ready"
            else:
                status[source] = "missing"

        return status

    @property
    def is_initialized(self) -> bool:
        """Return True if the pipeline successfully loaded models and features."""
        return self._initialized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_features(self, trial_id: str) -> pd.DataFrame:
        """Retrieve or build features for a single trial.

        For the MVP, features come from the pre-computed cache.  If the trial
        is not in the cache, we attempt to load trial data and return an
        error-producing placeholder that downstream callers can detect.
        """
        if self._feature_cache is not None:
            # Look up by trial_id column (or nctId / nct_id)
            for id_col in ("trial_id", "nctId", "nct_id"):
                if id_col in self._feature_cache.columns:
                    match = self._feature_cache.filter(pl.col(id_col) == trial_id)
                    if len(match) > 0:
                        # Drop the ID column(s) -- only return feature columns
                        feature_cols = [
                            c
                            for c in match.columns
                            if c not in ("trial_id", "nctId", "nct_id", "label", "y")
                        ]
                        # Convert to pandas at the ML boundary
                        return match.select(feature_cols).to_pandas()

        # Cache miss -- in the future this would trigger LLM agent feature
        # extraction.  For now, raise a clear error.
        raise LookupError(
            f"Trial {trial_id} not found in the feature cache.  "
            "Run the MCTS feature pipeline first, or add the trial to "
            f"{_FEATURE_CACHE_FILENAME}."
        )

    def _resolve_wrapper(self, model: str | None, phase: int | None = None) -> tuple[Any, str]:
        """Select the classifier wrapper to use.

        If ``phase`` is specified and a phase-specific model is loaded,
        use it.  Otherwise fall back to the legacy (unnamespaced) wrappers.

        If ``model`` is specified, returns that wrapper.  Otherwise returns
        the best available wrapper (preferring the one with highest training
        ROC-AUC, falling back to whatever is loaded).
        """
        # Select the wrapper pool: phase-specific or legacy
        wrappers = self._wrappers
        if phase is not None and phase in self._phase_wrappers:
            wrappers = self._phase_wrappers[phase]

        if not wrappers:
            raise RuntimeError(
                f"No models are loaded for phase={phase}.  Place trained "
                f"model artifacts in the output directory and restart the API."
            )

        if model is not None:
            if model not in wrappers:
                available = list(wrappers.keys())
                raise ValueError(
                    f"Requested model '{model}' is not loaded for phase={phase}.  "
                    f"Available models: {available}"
                )
            return wrappers[model], model

        # Auto-select: pick the wrapper with the best training ROC-AUC
        best_name: str | None = None
        best_score: float = -1.0
        for name in wrappers:
            res = self._results.get(name)
            if res is not None and hasattr(res, "metrics"):
                score = res.metrics.get("roc_auc", 0.0)
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            # Fallback: just pick the first loaded wrapper
            best_name = next(iter(wrappers))

        return wrappers[best_name], best_name

    async def _load_model_artifacts_from_dir(
        self,
        phase_dir: Path,
        target: dict[str, Any],
    ) -> None:
        """Load a ``best_model.pkl`` (dill) from a phase-specific output dir.

        This is the format produced by ``train_mcts.py`` via dill.dump.
        """
        model_path = phase_dir / "best_model.pkl"
        if not model_path.exists():
            return

        try:
            import dill

            wrapper = await asyncio.to_thread(lambda: dill.loads(model_path.read_bytes()))
            # Determine classifier name from type
            clf_name = "xgboost"
            try:
                from ctra.models.xgboost_classifier import XGBoostWrapper

                if not isinstance(wrapper, XGBoostWrapper):
                    clf_name = "tabpfn"
            except ImportError:
                pass

            target[clf_name] = wrapper
            logger.info("Loaded %s from %s", clf_name, model_path)
        except Exception:
            logger.exception("Failed to load model from %s", model_path)

    async def _load_model_artifacts(self, output_dir: Path) -> None:
        """Attempt to load persisted XGBoost and TabPFN model files.

        Searches two locations in order:
            1. ``ModelStore`` versioned directory (``models/v_*/``), loading
               the latest production or most recent version via ``joblib``.
            2. Legacy flat paths (``output/models/xgboost.pkl``,
               ``output/models/tabpfn.pkl``) using the wrapper ``load()``
               method (pickle-based).
        """
        # ------ Strategy 1: Load from ModelStore (joblib, versioned) ------
        try:
            from ctra.mlops.model_store import ModelStore

            settings = get_settings()
            store_dir = str(settings.mlops.model_store_dir)
            store = ModelStore(store_dir=store_dir)
            wrapper, feature_plans = await asyncio.to_thread(store.load_model)

            # Determine the model name from the wrapper type
            clf_name = "xgboost"
            from ctra.models.xgboost_classifier import XGBoostWrapper

            if not isinstance(wrapper, XGBoostWrapper):
                clf_name = "tabpfn"

            self._wrappers[clf_name] = wrapper
            logger.info(
                "Loaded %s from ModelStore (%s) with %d feature plans",
                clf_name,
                store_dir,
                len(feature_plans),
            )

            # Update feature plans if they were loaded from the store
            if feature_plans:
                self._feature_plans = feature_plans
                self._feature_set_id = (
                    feature_plans[0].get("set_id", "model_store")
                    if feature_plans
                    else "model_store"
                )
        except FileNotFoundError:
            logger.info("No versioned models in ModelStore; trying legacy paths")
        except Exception:
            logger.debug("ModelStore load failed; trying legacy paths", exc_info=True)

        # ------ Strategy 2: Legacy flat paths (output/models/*.pkl) ------
        # XGBoost
        xgb_path = output_dir / "models" / "xgboost.pkl"
        if xgb_path.exists() and "xgboost" not in self._wrappers:
            try:
                from ctra.models.xgboost_classifier import XGBoostWrapper as XGBWrap

                wrapper = XGBWrap()
                await asyncio.to_thread(wrapper.load, xgb_path)
                self._wrappers["xgboost"] = wrapper
                logger.info("Loaded XGBoost model from %s", xgb_path)
            except Exception:
                logger.exception("Failed to load XGBoost model from %s", xgb_path)

        # TabPFN
        tabpfn_path = output_dir / "models" / "tabpfn.pkl"
        if tabpfn_path.exists() and "tabpfn" not in self._wrappers:
            try:
                from ctra.models.tabpfn_classifier import TabPFNWrapper, is_tabpfn_available

                if is_tabpfn_available():
                    wrapper = TabPFNWrapper()
                    await asyncio.to_thread(wrapper.load, tabpfn_path)
                    self._wrappers["tabpfn"] = wrapper
                    logger.info("Loaded TabPFN model from %s", tabpfn_path)
                else:
                    logger.warning("TabPFN not installed; skipping model load")
            except Exception:
                logger.exception("Failed to load TabPFN model from %s", tabpfn_path)

    @staticmethod
    def _build_explanations(
        shap_values: NDArray[Any],
        feature_names: list[str],
        feature_values: pd.Series,
    ) -> list[FeatureExplanation]:
        """Build sorted FeatureExplanation list from SHAP values."""
        explanations: list[FeatureExplanation] = []

        for i, fname in enumerate(feature_names):
            if i >= len(shap_values):
                break

            raw_val = feature_values.get(fname)
            # Coerce numpy/pandas scalars to native Python types
            if hasattr(raw_val, "item"):
                raw_val = raw_val.item()  # type: ignore[union-attr]
            elif pd.isna(raw_val):
                raw_val = None

            explanations.append(
                FeatureExplanation(
                    name=fname,
                    value=raw_val,
                    shap_value=float(shap_values[i]),
                )
            )

        # Sort by absolute SHAP value descending (most important first)
        explanations.sort(key=lambda e: abs(e.shap_value), reverse=True)
        return explanations

    def _check_initialized(self) -> None:
        """Verify that the pipeline is initialized.

        Raises:
            RuntimeError: If the pipeline has not been successfully initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "PredictionPipeline is not initialized.  Call await pipeline.initialize() first."
            )
