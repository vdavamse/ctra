"""Multi-objective definitions for model training and retraining (MLOps).

Two objectives (per-phase isolation means no cross-phase stability):
    1. **Accuracy**  -- Validation ROC-AUC (maximize)
    2. **Parsimony** -- Feature efficiency: 1 - n/max (maximize, fewer = higher)

All objectives are oriented so that *higher is better* and normalized to
roughly [0, 1] for fair comparison in Pareto ranking.

Note: MultiObjectiveEvaluator is used by the retraining pipeline (mlops/retraining.py)
for classical ML model evaluation. The MCTS feature search loop uses the lightweight
extract_objectives() function from agents/runner.py instead.
"""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ctra.config.settings import MCTSConfig, get_settings

if TYPE_CHECKING:
    from ctra.search.objectives import ObjectiveResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Objective(abc.ABC):
    """Base class for a single objective function."""

    name: str = ""
    direction: str = "maximize"  # all objectives are "maximize" after normalization

    @abc.abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> float:
        """Return a scalar in [0, 1].  Higher is better."""


# ---------------------------------------------------------------------------
# Concrete objectives
# ---------------------------------------------------------------------------


class PredictiveAccuracy(Objective):
    """Objective 1: Validation ROC-AUC (primary metric).

    With per-phase isolation, each MCTS tree optimizes a single phase.
    This objective simply receives the pre-computed ROC-AUC and clamps
    it to [0, 1].
    """

    name = "accuracy"
    direction = "maximize"

    def evaluate(
        self,
        feature_set: list[str] | None = None,
        roc_auc: float | None = None,
        **_kwargs: Any,
    ) -> float:
        """Return ROC-AUC in [0, 1].

        Args:
            feature_set: Not used directly; kept for interface uniformity.
            roc_auc: Pre-computed single ROC-AUC.
        """
        if roc_auc is not None:
            return float(np.clip(roc_auc, 0.0, 1.0))

        # Fallback -- no data means random-chance performance
        return 0.5


class Parsimony(Objective):
    """Objective 2: Feature efficiency (fewer features = better, lower LLM cost).

    Computed as ``1 - (n_features / max_features)``, clamped to [0, 1].
    A feature set with 0 features scores 1.0 (maximum parsimony) and one
    with ``max_features`` features scores 0.0.
    """

    name = "parsimony"
    direction = "maximize"

    def __init__(self, max_features: int | None = None) -> None:
        self._max_features = max_features or get_settings().mcts.max_features

    def evaluate(
        self,
        feature_set: list[str],
        **_kwargs: Any,
    ) -> float:
        """Return ``1 - (n_features / max_features)``, range [0, 1]."""
        n = len(feature_set)
        if n <= 0:
            return 1.0
        ratio = n / max(self._max_features, 1)
        return float(np.clip(1.0 - ratio, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Multi-objective evaluator
# ---------------------------------------------------------------------------

# Registry mapping objective name -> class
_OBJECTIVE_REGISTRY: dict[str, type[Objective]] = {
    "accuracy": PredictiveAccuracy,
    "parsimony": Parsimony,
}


class MultiObjectiveEvaluator:
    """Evaluates a feature set against all active objectives.

    Args:
        objectives: List of objective names to activate. Defaults to the value in
            ``MCTSConfig.objectives``.
        config: Optional explicit MCTS configuration override.
    """

    def __init__(
        self,
        objectives: list[str] | None = None,
        config: MCTSConfig | None = None,
    ) -> None:
        cfg = config or get_settings().mcts
        names = objectives or list(cfg.objectives)

        self._objectives: list[Objective] = []
        self._names: list[str] = []

        for name in names:
            cls = _OBJECTIVE_REGISTRY.get(name)
            if cls is None:
                raise ValueError(
                    f"Unknown objective {name!r}. Available: {list(_OBJECTIVE_REGISTRY)}"
                )
            self._objectives.append(cls())
            self._names.append(name)

    @property
    def n_objectives(self) -> int:
        return len(self._objectives)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        feature_set: list[str],
        roc_auc: float | None = None,
        **extra: Any,
    ) -> ObjectiveResult:
        """Run all objectives and return combined result.

        Args:
            feature_set: The list of feature names being evaluated.
            roc_auc: Pre-computed single validation ROC-AUC (for accuracy objective).
            extra: Forwarded to individual objective ``evaluate()`` calls.
        """
        # Import here to avoid circular dependency
        from ctra.search.objectives import ObjectiveResult

        values = np.zeros(self.n_objectives, dtype=np.float64)
        details: dict[str, Any] = {}

        for i, (name, obj) in enumerate(zip(self._names, self._objectives, strict=True)):
            if name == "accuracy":
                val = obj.evaluate(
                    feature_set=feature_set,
                    roc_auc=roc_auc,
                    **extra,
                )
            elif name == "parsimony":
                val = obj.evaluate(feature_set=feature_set, **extra)
            else:
                val = obj.evaluate(feature_set=feature_set, **extra)

            values[i] = val
            details[name] = val

        return ObjectiveResult(values=values, names=list(self._names), details=details)
