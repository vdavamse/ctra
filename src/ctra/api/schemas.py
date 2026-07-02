"""Pydantic request/response models for the CTRA API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Metadata models
# ---------------------------------------------------------------------------


class PredictionMetadata(BaseModel):
    """Metadata attached to each individual prediction."""

    model_version: str = Field(description="Semantic version of the model artifact")
    feature_set_id: str = Field(
        description="Identifier of the MCTS node whose feature plan was used"
    )
    n_features: int = Field(description="Number of features used for prediction")
    inference_time_ms: float = Field(description="Wall-clock inference time in milliseconds")


class BatchMetadata(BaseModel):
    """Summary statistics for a batch prediction run."""

    total_trials: int
    successful_predictions: int
    failed_predictions: int
    total_time_ms: float


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for a single trial outcome prediction."""

    trial_id: str = Field(
        ...,
        description="ClinicalTrials.gov NCT ID (e.g. NCT04280705) or internal trial ID",
    )
    phase: int | None = Field(
        None,
        ge=1,
        le=3,
        description="Clinical trial phase (1, 2, or 3). Required for phase-specific models.",
    )
    model: str | None = Field(
        None,
        description='Classifier to use: "xgboost", "tabpfn", or null for best available',
    )
    include_shap: bool = Field(
        True,
        description="Whether to include SHAP feature explanations in the response",
    )


class BatchPredictRequest(BaseModel):
    """Request body for batch predictions across multiple trials."""

    trial_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of trial identifiers (NCT IDs or internal IDs)",
    )
    phase: int | None = Field(
        None,
        ge=1,
        le=3,
        description="Clinical trial phase (1, 2, or 3). Required for phase-specific models.",
    )
    model: str | None = Field(
        None,
        description='Classifier to use: "xgboost", "tabpfn", or null for best available',
    )
    include_shap: bool = Field(
        True,
        description="Whether to include SHAP feature explanations in the response",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FeatureExplanation(BaseModel):
    """SHAP-based explanation for a single feature."""

    name: str = Field(description="Machine-readable feature name")
    value: float | str | bool | None = Field(description="Raw feature value used for prediction")
    shap_value: float = Field(description="SHAP contribution to the positive-class prediction")


class PredictResponse(BaseModel):
    """Response for a single trial prediction."""

    trial_id: str
    phase: int | None = Field(None, description="Phase of the model used for prediction")
    prediction: str = Field(description='"success" or "failure"')
    probability_success: float = Field(ge=0.0, le=1.0)
    probability_failure: float = Field(ge=0.0, le=1.0)
    model_used: str = Field(description="Which classifier produced this prediction")
    features: list[FeatureExplanation] = Field(
        default_factory=list,
        description="Top SHAP explanations sorted by |shap_value| descending",
    )
    metadata: PredictionMetadata


class BatchPredictResponse(BaseModel):
    """Response for batch predictions."""

    predictions: list[PredictResponse]
    batch_metadata: BatchMetadata


class ModelInfo(BaseModel):
    """Description of a loaded model."""

    name: str
    type: str = Field(description='"xgboost" or "tabpfn"')
    version: str
    feature_set_id: str
    n_features: int
    training_metrics: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description='"healthy" or "degraded"')
    version: str
    models_loaded: list[str] = Field(default_factory=list)
    index_status: dict[str, str] = Field(
        default_factory=dict,
        description='Data source readiness: source -> "ready" | "stale" | "missing"',
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str | None = None
    trial_id: str | None = None
