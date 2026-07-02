"""Tests for ctra.api.routes — FastAPI endpoint integration tests with mocked pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ctra import __version__
from ctra.api.routes import router
from ctra.api.schemas import (
    BatchMetadata,
    BatchPredictResponse,
    FeatureExplanation,
    ModelInfo,
    PredictionMetadata,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_predict_response(trial_id: str = "NCT001") -> PredictResponse:
    """Build a valid PredictResponse for mocking."""
    return PredictResponse(
        trial_id=trial_id,
        prediction="success",
        probability_success=0.82,
        probability_failure=0.18,
        model_used="xgboost",
        features=[
            FeatureExplanation(name="drug_targets", value=3, shap_value=0.15),
        ],
        metadata=PredictionMetadata(
            model_version=__version__,
            feature_set_id="test-set",
            n_features=5,
            inference_time_ms=10.0,
        ),
    )


@pytest.fixture()
def mock_pipeline() -> MagicMock:
    """Create a mock PredictionPipeline with async predict methods."""
    pipeline = MagicMock()
    pipeline.is_initialized = True
    pipeline.get_loaded_model_names.return_value = ["xgboost"]
    pipeline.get_index_status.return_value = {"ctg": "ready", "pubmed": "missing"}

    # predict returns an awaitable (AsyncMock)
    pipeline.predict = AsyncMock(return_value=_make_predict_response())

    # predict_batch
    pipeline.predict_batch = AsyncMock(
        return_value=BatchPredictResponse(
            predictions=[_make_predict_response("NCT001"), _make_predict_response("NCT002")],
            batch_metadata=BatchMetadata(
                total_trials=2,
                successful_predictions=2,
                failed_predictions=0,
                total_time_ms=25.0,
            ),
        )
    )

    # get_model_info
    pipeline.get_model_info.return_value = [
        ModelInfo(
            name="xgboost",
            type="xgboost",
            version=__version__,
            feature_set_id="test-set",
            n_features=5,
        ),
    ]
    return pipeline


@pytest.fixture()
def client(mock_pipeline: MagicMock) -> TestClient:
    """TestClient with the mock pipeline injected into app.state."""
    app = FastAPI()
    app.include_router(router)
    app.state.pipeline = mock_pipeline
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_200(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "xgboost" in data["models_loaded"]
        assert data["version"] == __version__

    def test_health_degraded_no_pipeline(self) -> None:
        """When pipeline is None, health should report degraded."""
        app = FastAPI()
        app.include_router(router)
        app.state.pipeline = None
        c = TestClient(app)
        resp = c.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"


class TestPredictEndpoint:
    def test_predict_200(self, client: TestClient) -> None:
        resp = client.post("/api/v1/predict", json={"trial_id": "NCT001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["trial_id"] == "NCT001"
        assert data["prediction"] == "success"
        assert 0.0 <= data["probability_success"] <= 1.0

    def test_predict_404_trial_not_found(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.predict = AsyncMock(side_effect=LookupError("Trial not found"))
        resp = client.post("/api/v1/predict", json={"trial_id": "NCT_MISSING"})
        assert resp.status_code == 404

    def test_predict_400_bad_request(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.predict = AsyncMock(side_effect=ValueError("Invalid model"))
        resp = client.post("/api/v1/predict", json={"trial_id": "NCT001", "model": "bad"})
        assert resp.status_code == 400

    def test_predict_503_pipeline_unavailable(self) -> None:
        """When pipeline is not set on app.state, predict should return 503."""
        app = FastAPI()
        app.include_router(router)
        # No pipeline set at all
        c = TestClient(app)
        resp = c.post("/api/v1/predict", json={"trial_id": "NCT001"})
        assert resp.status_code == 503


class TestBatchPredictEndpoint:
    def test_batch_predict_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict/batch",
            json={"trial_ids": ["NCT001", "NCT002"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2
        assert data["batch_metadata"]["total_trials"] == 2


class TestModelsEndpoint:
    def test_list_models_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "xgboost"
        assert data[0]["type"] == "xgboost"

    def test_models_503_no_pipeline(self) -> None:
        app = FastAPI()
        app.include_router(router)
        c = TestClient(app)
        resp = c.get("/api/v1/models")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Phase routing
# ---------------------------------------------------------------------------


class TestPredictWithPhase:
    """Tests that the phase parameter is passed through to the pipeline."""

    def test_predict_passes_phase_to_pipeline(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        resp = client.post(
            "/api/v1/predict",
            json={"trial_id": "NCT001", "phase": 2},
        )
        assert resp.status_code == 200
        mock_pipeline.predict.assert_called_once()
        call_kwargs = mock_pipeline.predict.call_args.kwargs
        assert call_kwargs["phase"] == 2

    def test_predict_phase_none_by_default(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        resp = client.post(
            "/api/v1/predict",
            json={"trial_id": "NCT001"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_pipeline.predict.call_args.kwargs
        assert call_kwargs["phase"] is None

    def test_predict_phase_1(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        resp = client.post(
            "/api/v1/predict",
            json={"trial_id": "NCT001", "phase": 1},
        )
        assert resp.status_code == 200
        call_kwargs = mock_pipeline.predict.call_args.kwargs
        assert call_kwargs["phase"] == 1

    def test_predict_phase_3(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        resp = client.post(
            "/api/v1/predict",
            json={"trial_id": "NCT001", "phase": 3},
        )
        assert resp.status_code == 200
        call_kwargs = mock_pipeline.predict.call_args.kwargs
        assert call_kwargs["phase"] == 3

    def test_predict_invalid_phase_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict",
            json={"trial_id": "NCT001", "phase": 4},
        )
        assert resp.status_code == 422

    def test_batch_passes_phase_to_pipeline(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        resp = client.post(
            "/api/v1/predict/batch",
            json={"trial_ids": ["NCT001"], "phase": 3},
        )
        assert resp.status_code == 200
        call_kwargs = mock_pipeline.predict_batch.call_args.kwargs
        assert call_kwargs["phase"] == 3
