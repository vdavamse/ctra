"""Downstream classifier wrappers (XGBoost, TabPFN) with SHAP support."""

from ctra.models.model_registry import ModelRegistry
from ctra.models.tabpfn_classifier import (
    PredictionResult,
    TabPFNWrapper,
    compute_metrics,
    is_tabpfn_available,
)
from ctra.models.xgboost_classifier import XGBoostWrapper

__all__ = [
    "ModelRegistry",
    "PredictionResult",
    "TabPFNWrapper",
    "XGBoostWrapper",
    "compute_metrics",
    "is_tabpfn_available",
]
