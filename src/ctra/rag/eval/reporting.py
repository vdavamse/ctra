"""Evaluation report generation, baseline persistence, and plotting.

Generates:
- Markdown evaluation report with all metrics and recommendations
- JSON baseline files for regression tracking
- CSV files for detailed per-label metrics
- PNG confusion matrices and PR curves (optional, requires matplotlib)

All CSV/JSON outputs are always generated. PNG plots require matplotlib
(installed via the [eval] optional dependency group).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path  # noqa: TC003
from typing import Any

from ctra.rag.eval.data_models import EvalResult  # noqa: TC001

logger = logging.getLogger(__name__)


def save_baseline(
    results: list[EvalResult],
    output_path: Path,
    filename: str,
) -> Path:
    """Save evaluation results as a JSON baseline for regression tracking.

    Args:
        results: List of EvalResult from Mode 1 or Mode 2.
        output_path: Directory to write to.
        filename: e.g., "mode1_standard_baseline.json"

    Returns:
        Path to the written file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename

    baseline_data = []
    for result in results:
        per_label_data = [
            {
                "label": m.label,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "support": m.support,
                "predicted_count": m.predicted_count,
            }
            for m in result.per_label
        ]

        baseline_data.append(
            {
                "benchmark": result.benchmark,
                "mode": result.mode,
                "match_strategy": result.match_strategy,
                "micro_f1": result.micro_f1,
                "macro_f1": result.macro_f1,
                "weighted_f1": result.weighted_f1,
                "micro_precision": result.micro_precision,
                "micro_recall": result.micro_recall,
                "total_gold": result.total_gold,
                "total_predicted": result.total_predicted,
                "total_matched": result.total_matched,
                "per_label": per_label_data,
                "metadata": result.metadata,
            }
        )

    with open(output_file, "w") as f:
        json.dump(baseline_data, f, indent=2)

    logger.info(f"Saved baseline to {output_file}")
    return output_file


def save_per_label_csv(
    results: list[EvalResult],
    output_path: Path,
    filename: str = "per_label_metrics.csv",
) -> Path:
    """Save per-label P/R/F1 metrics as CSV.

    Columns: benchmark, label, precision, recall, f1, support, predicted_count
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename

    rows = []
    for result in results:
        for metric in result.per_label:
            rows.append(
                {
                    "benchmark": result.benchmark,
                    "label": metric.label,
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "f1": metric.f1,
                    "support": metric.support,
                    "predicted_count": metric.predicted_count,
                }
            )

    fieldnames = ["benchmark", "label", "precision", "recall", "f1", "support", "predicted_count"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved per-label metrics to {output_file}")
    return output_file


def save_threshold_sweep_csv(
    results: list[EvalResult],
    output_path: Path,
) -> Path:
    """Save threshold sweep results to threshold_sweep.csv.

    Columns: threshold, label, precision, recall, f1, support
    Plus aggregate rows for micro/macro/weighted F1.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "threshold_sweep.csv"

    rows = []
    for result in results:
        threshold = result.metadata.get("threshold", "")

        for metric in result.per_label:
            rows.append(
                {
                    "threshold": threshold,
                    "label": metric.label,
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "f1": metric.f1,
                    "support": metric.support,
                }
            )

        # Add aggregate rows
        rows.append(
            {
                "threshold": threshold,
                "label": "micro_f1",
                "precision": result.micro_precision,
                "recall": result.micro_recall,
                "f1": result.micro_f1,
                "support": result.total_gold,
            }
        )
        rows.append(
            {
                "threshold": threshold,
                "label": "macro_f1",
                "precision": 0.0,
                "recall": 0.0,
                "f1": result.macro_f1,
                "support": result.total_gold,
            }
        )
        rows.append(
            {
                "threshold": threshold,
                "label": "weighted_f1",
                "precision": 0.0,
                "recall": 0.0,
                "f1": result.weighted_f1,
                "support": result.total_gold,
            }
        )

    fieldnames = ["threshold", "label", "precision", "recall", "f1", "support"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved threshold sweep to {output_file}")
    return output_file


def save_label_ablation_csv(
    ablation_results: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Save label ablation results to label_phrasing_ablation.csv.

    Columns: original_label, variant, f1_original, f1_variant, f1_delta
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "label_phrasing_ablation.csv"

    fieldnames = ["original_label", "variant", "f1_original", "f1_variant", "f1_delta"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation_results)

    logger.info(f"Saved label ablation to {output_file}")
    return output_file


def generate_report(
    mode1_results: list[EvalResult] | None,
    mode2_results: list[EvalResult] | None,
    sweep_results: list[EvalResult] | None,
    ablation_results: list[dict[str, Any]] | None,
    gap_analysis_results: list[dict[str, Any]] | None,
    output_path: Path,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Generate comprehensive Markdown evaluation report.

    Report sections:
    1. Environment & Configuration (model, device, versions)
    2. Mode 1 Results (if available) -- per-benchmark, vs paper scores
    3. Mode 2 Results (if available) -- per-label, per-benchmark, combined
    4. Threshold Sweep (if available) -- optimal thresholds
    5. Label Ablation (if available) -- best phrasings
    6. Gap Analysis -- labels with no benchmark coverage
    7. Recommendations -- actionable next steps

    Returns:
        Path to the written report.md file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "report.md"

    lines = []
    lines.append("# NER Evaluation Report\n")

    # 1. Environment & Configuration
    lines.append("## Environment & Configuration\n")
    if metadata:
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")
    lines.append("")

    # 2. Mode 1 Results
    if mode1_results:
        lines.append("## Mode 1: Standard Benchmark Evaluation\n")
        lines.append("Results with benchmark native entity labels:\n")
        for result in mode1_results:
            lines.append(f"### {result.benchmark.upper()}\n")
            lines.append(f"- **Micro F1**: {result.micro_f1:.4f}")
            lines.append(f"- **Macro F1**: {result.macro_f1:.4f}")
            lines.append(f"- **Weighted F1**: {result.weighted_f1:.4f}")
            lines.append(f"- **Total Gold Entities**: {result.total_gold}")
            lines.append(f"- **Total Matched**: {result.total_matched}\n")
            lines.append("| Label | Precision | Recall | F1 | Support |")
            lines.append("|-------|-----------|--------|-----|---------|")
            for metric in result.per_label:
                lines.append(
                    f"| {metric.label} | {metric.precision:.3f} | "
                    f"{metric.recall:.3f} | {metric.f1:.3f} | {metric.support} |"
                )
            lines.append("")

    # 3. Mode 2 Results
    if mode2_results:
        lines.append("## Mode 2: CTRA-Mapped Evaluation\n")
        lines.append("Results with CTRA's 16-label set:\n")
        for result in mode2_results:
            lines.append(f"### {result.benchmark.upper()}\n")
            lines.append(f"- **Micro F1**: {result.micro_f1:.4f}")
            lines.append(f"- **Macro F1**: {result.macro_f1:.4f}")
            lines.append(f"- **Weighted F1**: {result.weighted_f1:.4f}")
            lines.append(f"- **Total Gold Entities**: {result.total_gold}")
            lines.append(f"- **Total Mapped**: {result.total_matched}\n")
            lines.append("| Label | Precision | Recall | F1 | Support |")
            lines.append("|-------|-----------|--------|-----|---------|")
            for metric in result.per_label:
                lines.append(
                    f"| {metric.label} | {metric.precision:.3f} | "
                    f"{metric.recall:.3f} | {metric.f1:.3f} | {metric.support} |"
                )
            lines.append("")

    # 4. Threshold Sweep
    if sweep_results:
        lines.append("## Threshold Sweep Analysis\n")
        lines.append("F1 scores across confidence thresholds:\n")
        lines.append("| Threshold | Micro F1 | Macro F1 | Weighted F1 |")
        lines.append("|-----------|----------|----------|-------------|")
        for result in sweep_results:
            threshold = result.metadata.get("threshold", "")
            lines.append(
                f"| {threshold} | {result.micro_f1:.4f} | "
                f"{result.macro_f1:.4f} | {result.weighted_f1:.4f} |"
            )
        lines.append("")

        # Find optimal threshold
        if sweep_results:
            best_result = max(sweep_results, key=lambda r: r.micro_f1)
            best_threshold = best_result.metadata.get("threshold", "")
            lines.append(
                f"**Optimal threshold**: {best_threshold} (micro F1: {best_result.micro_f1:.4f})\n"
            )

    # 5. Label Ablation
    if ablation_results:
        lines.append("## Label Phrasing Ablation\n")
        lines.append("F1 delta for alternative label phrasings:\n")
        lines.append("| Original Label | Variant | F1 Original | F1 Variant | Delta |")
        lines.append("|---|---|---|---|---|")
        for ablation in ablation_results:
            lines.append(
                f"| {ablation['original_label']} | {ablation['variant']} | "
                f"{ablation['f1_original']:.3f} | {ablation['f1_variant']:.3f} | "
                f"{ablation['f1_delta']:+.3f} |"
            )
        lines.append("")

        # Highlight best variants
        if ablation_results:
            best_deltas: dict[str, tuple[str, float]] = {}
            for ablation in ablation_results:
                label = ablation["original_label"]
                if label not in best_deltas or ablation["f1_delta"] > best_deltas[label][1]:
                    best_deltas[label] = (ablation["variant"], ablation["f1_delta"])

            lines.append("**Best variant improvements**:\n")
            for label, (variant, delta) in best_deltas.items():
                if delta > 0:
                    lines.append(f"- {label}: {variant} (+{delta:.3f})")

    # 6. Gap Analysis
    if gap_analysis_results:
        lines.append("## Gap Analysis\n")
        lines.append("Benchmark coverage by entity label:\n")
        lines.append("| Label | Total Gold | Status | Recommendation |")
        lines.append("|-------|----------|--------|---|")
        for gap in gap_analysis_results:
            lines.append(
                f"| {gap['label']} | {gap['total_gold']} | "
                f"{gap['status']} | {gap['recommendation']} |"
            )
        lines.append("")

    # 7. Recommendations
    lines.append("## Recommendations\n")
    lines.append("- Review low-coverage labels and consider targeted benchmark expansion")
    lines.append("- Consider threshold tuning based on sweep analysis")
    lines.append("- Test promising label variants in production")
    lines.append("- Monitor F1 scores across benchmark updates")
    lines.append("")

    # Write report
    with open(report_file, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved report to {report_file}")
    return report_file


def plot_confusion_matrix(
    confusion_matrix: list[list[int]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: Path,
) -> Path | None:
    """Plot and save a confusion matrix as PNG.

    Returns None if matplotlib is not installed (graceful degradation).
    Import matplotlib only inside this function to keep it optional.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed; skipping confusion matrix plot. "
            "Install with: pip install ctra[eval]"
        )
        return None

    output_path.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    import numpy as np

    matrix = np.array(confusion_matrix)
    im = ax.imshow(matrix, cmap="YlOrRd")

    # Set ticks and labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color="black",
            )

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Gold Label")

    plt.tight_layout()

    output_file = output_path / "confusion_matrix.png"
    fig.savefig(output_file, dpi=100)
    plt.close(fig)

    logger.info(f"Saved confusion matrix to {output_file}")
    return output_file


def plot_pr_curves(
    sweep_results: list[EvalResult],
    output_dir: Path,
) -> list[Path]:
    """Plot per-label precision-recall curves from threshold sweep data.

    Creates one PNG per CTRA label in output_dir/pr_curves/.
    Returns list of created file paths (empty if matplotlib unavailable).
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
    except ImportError:
        logger.warning(
            "matplotlib not installed; skipping PR curves. Install with: pip install ctra[eval]"
        )
        return []

    pr_dir = output_dir / "pr_curves"
    pr_dir.mkdir(parents=True, exist_ok=True)

    # Extract data by label
    label_data: dict[str, list[tuple[float, float, float]]] = {}
    for result in sweep_results:
        threshold = float(result.metadata.get("threshold", "0.4"))
        for metric in result.per_label:
            if metric.label not in label_data:
                label_data[metric.label] = []
            label_data[metric.label].append((threshold, metric.precision, metric.recall))

    # Sort by threshold
    for label in label_data:
        label_data[label].sort(key=lambda x: x[0])

    # Plot PR curves
    output_files = []
    for label, data in label_data.items():
        if not data:
            continue

        precisions = [x[1] for x in data]
        recalls = [x[2] for x in data]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recalls, precisions, marker="o", label=label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve: {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_file = pr_dir / f"{label.replace(' ', '_')}_pr_curve.png"
        fig.savefig(output_file, dpi=100)
        plt.close(fig)

        output_files.append(output_file)
        logger.info(f"Saved PR curve for {label}")

    return output_files
