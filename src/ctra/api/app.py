"""FastAPI application factory and entry point."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ctra import __version__
from ctra.api.pipeline import PredictionPipeline
from ctra.api.routes import router
from ctra.api.schemas import ErrorResponse
from ctra.config.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def create_app() -> FastAPI:
    """Create and configure the CTRA FastAPI application.

    Sets up the prediction pipeline lifecycle, middleware (CORS, timing, auth),
    routes, and exception handlers. The prediction pipeline loads pre-trained
    models and cached features on startup.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage startup and shutdown of the prediction pipeline.

        On startup:
            - Initializes logging (debug or info level)
            - Loads the PredictionPipeline (models + feature cache)
            - Stores pipeline in app.state for request handlers

        On shutdown:
            - Cleans up pipeline resources
            - Logs shutdown message
        """
        logging.basicConfig(
            level=logging.DEBUG if settings.debug else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _logger = logging.getLogger("ctra.api")
        _logger.info("CTRA API starting (version %s)", __version__)

        # Initialize the prediction pipeline (load models + feature cache)
        pipeline = PredictionPipeline()
        try:
            await pipeline.initialize()
        except Exception:
            _logger.exception(
                "Failed to initialize prediction pipeline.  The API will start in degraded mode."
            )
        app.state.pipeline = pipeline

        yield  # Application runs here

        _logger = logging.getLogger("ctra.api")
        pipeline = getattr(app.state, "pipeline", None)  # type: ignore[assignment]
        if pipeline is not None:
            await pipeline.shutdown()
        _logger.info("CTRA API shut down")

    app = FastAPI(
        title="CTRA -- Clinical Trial Risk Assessment",
        description=(
            "Predict clinical trial outcomes (success/failure) using "
            "LLM-engineered features with XGBoost/TabPFN classifiers "
            "and SHAP interpretability."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing -- adds X-Process-Time header to every response
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Add X-Process-Time-Ms response header with request duration.

        Measures wall-clock time from request entry to response completion
        and includes it as a millisecond float in the response headers.
        """
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # Optional API key authentication
    if settings.api.api_key:
        expected_key = settings.api.api_key

        @app.middleware("http")
        async def verify_api_key(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
            # Allow unauthenticated access to docs and health
            if request.url.path in ("/docs", "/redoc", "/openapi.json", "/api/v1/health"):
                return await call_next(request)  # type: ignore[no-any-return]

            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != expected_key:
                return JSONResponse(
                    status_code=401,
                    content=ErrorResponse(
                        error="unauthorized",
                        detail="Invalid or missing API key.  Use 'Authorization: Bearer <key>'.",
                    ).model_dump(),
                )
            return await call_next(request)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    app.include_router(router)

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError by returning 400 Bad Request.

        Converts application ValueError exceptions to HTTP 400 responses
        with standardized error format.
        """
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="bad_request", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(LookupError)
    async def lookup_error_handler(request: Request, exc: LookupError) -> JSONResponse:
        """Handle LookupError by returning 404 Not Found.

        Converts LookupError (e.g., KeyError, IndexError) to HTTP 404 responses.
        """
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(error="not_found", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        """Handle RuntimeError by returning 503 Service Unavailable.

        Converts RuntimeError (e.g., pipeline not initialized) to HTTP 503
        responses indicating temporary service degradation.
        """
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(error="service_unavailable", detail=str(exc)).model_dump(),
        )

    return app


def main() -> None:
    """Entry point for the ``ctra`` CLI command.

    Launches the CTRA API server using Uvicorn with settings from config.
    The server listens on the configured host and port, with the number of
    worker processes and reload behavior controlled by environment settings.
    """
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "ctra.api.app:create_app",
        factory=True,
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.reload,
        log_level=settings.api.log_level,
    )


if __name__ == "__main__":
    main()
