"""FastAPI REST API for clinical trial risk prediction."""

from ctra.api.app import create_app, main
from ctra.api.pipeline import PredictionPipeline

__all__ = ["PredictionPipeline", "create_app", "main"]
