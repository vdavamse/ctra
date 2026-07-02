"""Tests for ctra.api.schemas — Pydantic validation and construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctra.api.schemas import (
    BatchPredictRequest,
    ErrorResponse,
    FeatureExplanation,
    HealthResponse,
    PredictionMetadata,
    PredictRequest,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# PredictRequest
# ---------------------------------------------------------------------------


class TestPredictRequest:
    def test_valid_request(self) -> None:
        req = PredictRequest(trial_id="NCT04280705")
        assert req.trial_id == "NCT04280705"
        assert req.model is None
        assert req.include_shap is True
        assert req.phase is None

    def test_missing_trial_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest()  # type: ignore[call-arg]

    def test_with_model_override(self) -> None:
        req = PredictRequest(trial_id="NCT001", model="xgboost", include_shap=False)
        assert req.model == "xgboost"
        assert req.include_shap is False

    def test_phase_valid_values(self) -> None:
        for p in (1, 2, 3):
            req = PredictRequest(trial_id="NCT001", phase=p)
            assert req.phase == p

    def test_phase_none_default(self) -> None:
        req = PredictRequest(trial_id="NCT001")
        assert req.phase is None

    def test_phase_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            PredictRequest(trial_id="NCT001", phase=0)

    def test_phase_four_rejected(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 3"):
            PredictRequest(trial_id="NCT001", phase=4)

    def test_phase_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(trial_id="NCT001", phase=-1)


# ---------------------------------------------------------------------------
# PredictResponse
# ---------------------------------------------------------------------------


class TestPredictResponse:
    def test_valid_response(self) -> None:
        resp = PredictResponse(
            trial_id="NCT001",
            prediction="success",
            probability_success=0.85,
            probability_failure=0.15,
            model_used="xgboost",
            metadata=PredictionMetadata(
                model_version="0.1.0",
                feature_set_id="default",
                n_features=10,
                inference_time_ms=12.5,
            ),
        )
        assert resp.probability_success == 0.85

    def test_probability_out_of_bounds_high(self) -> None:
        with pytest.raises(ValidationError):
            PredictResponse(
                trial_id="NCT001",
                prediction="success",
                probability_success=1.5,
                probability_failure=0.15,
                model_used="xgboost",
                metadata=PredictionMetadata(
                    model_version="0.1.0",
                    feature_set_id="default",
                    n_features=10,
                    inference_time_ms=12.5,
                ),
            )

    def test_probability_out_of_bounds_low(self) -> None:
        with pytest.raises(ValidationError):
            PredictResponse(
                trial_id="NCT001",
                prediction="failure",
                probability_success=-0.1,
                probability_failure=1.1,
                model_used="tabpfn",
                metadata=PredictionMetadata(
                    model_version="0.1.0",
                    feature_set_id="default",
                    n_features=5,
                    inference_time_ms=1.0,
                ),
            )


# ---------------------------------------------------------------------------
# BatchPredictRequest
# ---------------------------------------------------------------------------


class TestBatchPredictRequest:
    def test_valid_batch(self) -> None:
        req = BatchPredictRequest(trial_ids=["NCT001", "NCT002"])
        assert len(req.trial_ids) == 2

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictRequest(trial_ids=[])

    def test_max_length_101_raises(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictRequest(trial_ids=[f"NCT{i:04d}" for i in range(101)])

    def test_exactly_100_allowed(self) -> None:
        req = BatchPredictRequest(trial_ids=[f"NCT{i:04d}" for i in range(100)])
        assert len(req.trial_ids) == 100

    def test_phase_valid(self) -> None:
        req = BatchPredictRequest(trial_ids=["NCT001"], phase=2)
        assert req.phase == 2

    def test_phase_none_default(self) -> None:
        req = BatchPredictRequest(trial_ids=["NCT001"])
        assert req.phase is None

    def test_phase_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictRequest(trial_ids=["NCT001"], phase=0)

    def test_phase_four_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictRequest(trial_ids=["NCT001"], phase=4)


# ---------------------------------------------------------------------------
# PredictResponse phase field
# ---------------------------------------------------------------------------


class TestPredictResponsePhase:
    def test_phase_included_in_response(self) -> None:
        resp = PredictResponse(
            trial_id="NCT001",
            phase=2,
            prediction="success",
            probability_success=0.85,
            probability_failure=0.15,
            model_used="xgboost",
            metadata=PredictionMetadata(
                model_version="0.1.0",
                feature_set_id="test",
                n_features=5,
                inference_time_ms=10.0,
            ),
        )
        assert resp.phase == 2

    def test_phase_none_default(self) -> None:
        resp = PredictResponse(
            trial_id="NCT001",
            prediction="failure",
            probability_success=0.3,
            probability_failure=0.7,
            model_used="tabpfn",
            metadata=PredictionMetadata(
                model_version="0.1.0",
                feature_set_id="test",
                n_features=3,
                inference_time_ms=5.0,
            ),
        )
        assert resp.phase is None


# ---------------------------------------------------------------------------
# HealthResponse, FeatureExplanation, ErrorResponse
# ---------------------------------------------------------------------------


class TestHealthResponse:
    def test_construction(self) -> None:
        hr = HealthResponse(status="healthy", version="0.1.0", models_loaded=["xgboost"])
        assert hr.status == "healthy"
        assert hr.models_loaded == ["xgboost"]

    def test_default_empty_lists(self) -> None:
        hr = HealthResponse(status="degraded", version="0.1.0")
        assert hr.models_loaded == []
        assert hr.index_status == {}


class TestFeatureExplanation:
    def test_construction(self) -> None:
        fe = FeatureExplanation(name="drug_target_count", value=3, shap_value=0.42)
        assert fe.name == "drug_target_count"
        assert fe.value == 3
        assert fe.shap_value == 0.42

    def test_none_value(self) -> None:
        fe = FeatureExplanation(name="missing_feat", value=None, shap_value=-0.1)
        assert fe.value is None


class TestErrorResponse:
    def test_construction(self) -> None:
        er = ErrorResponse(error="not_found", detail="Trial NCT999 not found")
        assert er.error == "not_found"
        assert er.detail == "Trial NCT999 not found"
        assert er.trial_id is None

    def test_with_trial_id(self) -> None:
        er = ErrorResponse(error="bad_request", trial_id="NCT999")
        assert er.trial_id == "NCT999"
