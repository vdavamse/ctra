"""API endpoint definitions for the CTRA prediction service.

All endpoints are prefixed with ``/api/v1/``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ctra import __version__
from ctra.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


def _get_pipeline(request: Request) -> Any:
    """Retrieve the PredictionPipeline from application state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction pipeline is not initialized.  The server is still starting up.",
        )
    return pipeline


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    responses={503: {"model": ErrorResponse}},
)
async def health_check(request: Request) -> HealthResponse:
    """Return service health status including loaded models and data-source readiness."""
    pipeline = getattr(request.app.state, "pipeline", None)

    if pipeline is None or not pipeline.is_initialized:
        return HealthResponse(
            status="degraded",
            version=__version__,
            models_loaded=[],
            index_status={},
        )

    models = pipeline.get_loaded_model_names()
    index_status = pipeline.get_index_status()

    status = "healthy" if models else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        models_loaded=models,
        index_status=index_status,
    )


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict trial outcome",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict_trial(request: Request, body: PredictRequest) -> PredictResponse:
    """Predict the outcome of a single clinical trial.

    The prediction pipeline:
    1. Look up the trial record (from feature cache, CTG parquet, or API)
    2. Extract features using the best MCTS feature set
    3. Run the selected or best-available classifier (XGBoost or TabPFN)
    4. Generate SHAP explanations (if requested)
    5. Return prediction with probabilities and explanations
    """
    pipeline = _get_pipeline(request)

    try:
        return await pipeline.predict(  # type: ignore[no-any-return]
            trial_id=body.trial_id,
            model=body.model,
            include_shap=body.include_shap,
            phase=body.phase,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error predicting %s", body.trial_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error predicting {body.trial_id}: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch predict trial outcomes",
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict_batch(request: Request, body: BatchPredictRequest) -> BatchPredictResponse:
    """Predict outcomes for a batch of clinical trials.

    Runs predictions concurrently with bounded parallelism.  Individual
    failures do not abort the entire batch -- they are counted in
    ``batch_metadata.failed_predictions``.
    """
    pipeline = _get_pipeline(request)

    try:
        return await pipeline.predict_batch(  # type: ignore[no-any-return]
            trial_ids=body.trial_ids,
            model=body.model,
            include_shap=body.include_shap,
            phase=body.phase,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in batch prediction")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error in batch prediction: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


@router.get(
    "/models",
    response_model=list[ModelInfo],
    summary="List loaded models",
    responses={503: {"model": ErrorResponse}},
)
async def list_models(request: Request) -> list[ModelInfo]:
    """Return information about all currently loaded classifiers."""
    pipeline = _get_pipeline(request)
    return pipeline.get_model_info()  # type: ignore[no-any-return]
