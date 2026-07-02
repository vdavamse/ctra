"""Downstream classifier wrappers (XGBoost, TabPFN) with SHAP support."""

from ctra.models.tabpfn_classifier import (
    PredictionResult,
    TabPFNWrapper,
    compute_metrics,
    is_tabpfn_available,
)
from ctra.models.model_registry import ModelRegistry
from ctra.models.xgboost_classifier import XGBoostWrapper

__all__ = [
    "PredictionResult",
    "compute_metrics",
    "is_tabpfn_available",
    "TabPFNWrapper",
    "XGBoostWrapper",
    "ModelRegistry",
]
