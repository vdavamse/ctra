"""Multi-objective definitions for MCTS feature set optimization.

Two objectives (per-phase isolation means no cross-phase stability):
    1. **Accuracy**  -- Validation ROC-AUC (maximize)
    2. **Parsimony** -- Feature efficiency: 1 - n/max (maximize, fewer = higher)

All objectives are oriented so that *higher is better* and normalized to
roughly [0, 1] for fair comparison in Pareto ranking.

Note: MultiObjectiveEvaluator and objective classes have been moved to
mlops/objectives.py since they are only used by the retraining pipeline,
not by the MCTS search loop which uses extract_objectives() instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Feature set type alias (list of feature names)
# ---------------------------------------------------------------------------

FeatureSet = list[str]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveResult:
    """Result of evaluating a feature set against all objectives."""

    values: NDArray[Any]  # shape (n_objectives,) -- higher is better for all
    names: list[str]  # e.g. ["accuracy", "parsimony"]
    details: dict[str, Any] = field(default_factory=dict)  # per-objective breakdown

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> float:
        """Lookup a single objective value by name."""
        try:
            idx = self.names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return float(self.values[idx])

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.values)
