"""NER evaluation pipeline for clinical trial benchmarks.

Provides two evaluation modes:
- Mode 1 (standard): Reproduce published GLiNER-BioMed paper scores
- Mode 2 (ctra): Evaluate CTRA's 16-label set with benchmark mapping

Also includes threshold sweep, label phrasing ablation, and baseline persistence.
"""

from __future__ import annotations

from ctra.rag.eval.data_models import (
    Benchmark,
    EvalMode,
    EvalResult,
    GoldAnnotation,
    MatchStrategy,
    PerLabelMetrics,
    PredictedEntity,
)
from ctra.rag.eval.evaluator import NERExperiment, gap_analysis
from ctra.rag.eval.loaders import DatasetNotAvailableError, load_benchmark
from ctra.rag.eval.metrics import compute_ner_metrics, match_entities
from ctra.rag.eval.reporting import (
    generate_report,
    plot_confusion_matrix,
    plot_pr_curves,
    save_baseline,
    save_label_ablation_csv,
    save_per_label_csv,
    save_threshold_sweep_csv,
)

__all__ = [
    "Benchmark",
    "DatasetNotAvailableError",
    "EvalMode",
    "EvalResult",
    "GoldAnnotation",
    "MatchStrategy",
    "NERExperiment",
    "PerLabelMetrics",
    "PredictedEntity",
    "compute_ner_metrics",
    "gap_analysis",
    "generate_report",
    "load_benchmark",
    "match_entities",
    "plot_confusion_matrix",
    "plot_pr_curves",
    "save_baseline",
    "save_label_ablation_csv",
    "save_per_label_csv",
    "save_threshold_sweep_csv",
]
