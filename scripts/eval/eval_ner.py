#!/usr/bin/env python
"""Evaluate GLiNER-BioMed NER on clinical trial benchmarks.

Two evaluation modes:
- Mode 1 (standard): Run with each benchmark's native entity labels,
  reproducing the GLiNER-BioMed paper's reported scores.
- Mode 2 (ctra): Run with CTRA's 16 entity labels, mapping gold
  annotations to CTRA label space.

Usage::

    # Full evaluation -- both modes on freely available benchmarks
    python scripts/eval/eval_ner.py --benchmarks chia,tac --mode both

    # Mode 1 only -- reproduce paper scores
    python scripts/eval/eval_ner.py --benchmarks chia,tac --mode standard

    # Mode 2 with threshold sweep
    python scripts/eval/eval_ner.py --benchmarks chia --mode ctra --sweep-threshold

    # Label phrasing ablation
    python scripts/eval/eval_ner.py --benchmarks chia --mode ctra --ablate-labels

    # Custom model and output directory
    python scripts/eval/eval_ner.py --gliner-model Ihor/gliner-biomed-bi-large-v1.0 \\
        --output-dir output/ner_eval_bi/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ctra.rag.eval import (
    Benchmark,
    EvalMode,
    MatchStrategy,
    NERExperiment,
    gap_analysis,
    generate_report,
    save_baseline,
    save_label_ablation_csv,
    save_per_label_csv,
    save_threshold_sweep_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ctra.eval_ner")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for NER evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate GLiNER-BioMed NER on clinical trial benchmarks.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="chia,tac",
        help="Comma-separated benchmarks to evaluate (chia,n2c2,tac). Default: chia,tac",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "ctra", "both"],
        default="both",
        help="Evaluation mode: standard (Mode 1), ctra (Mode 2), or both. Default: both",
    )
    parser.add_argument(
        "--sweep-threshold",
        action="store_true",
        help="Run threshold sweep on specified benchmark (default: CHIA). Requires mode=ctra or both.",
    )
    parser.add_argument(
        "--ablate-labels",
        action="store_true",
        help="Run label phrasing ablation on specified benchmark (default: CHIA). Requires mode=ctra or both.",
    )
    parser.add_argument(
        "--gliner-model",
        type=str,
        default="Ihor/gliner-biomed-large-v1.0",
        help="GLiNER-BioMed model identifier. Default: Ihor/gliner-biomed-large-v1.0",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/ner_eval",
        help="Output directory for results. Default: output/ner_eval",
    )
    parser.add_argument(
        "--match-strategy",
        type=str,
        choices=["strict", "relaxed", "text_only"],
        default="text_only",
        help="Entity matching strategy. Default: text_only",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Parse arguments
    benchmark_names = [b.strip() for b in args.benchmarks.split(",")]
    benchmarks = []
    for name in benchmark_names:
        try:
            benchmarks.append(Benchmark(name.lower()))
        except ValueError:
            logger.error(f"Unknown benchmark: {name}")
            return 1

    mode = EvalMode(args.mode)
    match_strategy = MatchStrategy(args.match_strategy)
    output_path = Path(args.output_dir)

    logger.info("Starting NER evaluation")
    logger.info(f"  Benchmarks: {[b.value for b in benchmarks]}")
    logger.info(f"  Mode: {mode.value}")
    logger.info(f"  Match strategy: {match_strategy.value}")
    logger.info(f"  GLiNER model: {args.gliner_model}")
    logger.info(f"  Output dir: {output_path}")

    # Create experiment
    experiment = NERExperiment(
        gliner_model=args.gliner_model,
        match_strategy=match_strategy,
    )

    mode1_results = None
    mode2_results = None
    sweep_results = None
    ablation_results = None

    # Run evaluations
    if mode in (EvalMode.STANDARD, EvalMode.BOTH):
        logger.info("Running Mode 1 (standard benchmark evaluation)...")
        mode1_results = experiment.run_mode1(benchmarks)
        if mode1_results:
            save_baseline(mode1_results, output_path, "mode1_standard_baseline.json")
            save_per_label_csv(mode1_results, output_path, "mode1_per_label_metrics.csv")

    if mode in (EvalMode.CTRA, EvalMode.BOTH):
        logger.info("Running Mode 2 (CTRA-mapped evaluation)...")
        mode2_results = experiment.run_mode2(benchmarks)
        if mode2_results:
            save_baseline(mode2_results, output_path, "mode2_ctra_baseline.json")
            save_per_label_csv(mode2_results, output_path, "mode2_per_label_metrics.csv")

        if args.sweep_threshold:
            logger.info("Running threshold sweep...")
            sweep_results = experiment.run_threshold_sweep(benchmarks[0])
            if sweep_results:
                save_threshold_sweep_csv(sweep_results, output_path)

        if args.ablate_labels:
            logger.info("Running label phrasing ablation...")
            ablation_results = experiment.run_label_ablation(benchmarks[0])
            if ablation_results:
                save_label_ablation_csv(ablation_results, output_path)

    # Gap analysis
    gap_results = None
    if mode2_results:
        logger.info("Running gap analysis...")
        gap_results = gap_analysis(mode2_results)

    # Generate report
    logger.info("Generating report...")
    metadata = {
        "gliner_model": args.gliner_model,
        "match_strategy": match_strategy.value,
        "benchmarks": ", ".join(b.value for b in benchmarks),
    }
    generate_report(
        mode1_results,
        mode2_results,
        sweep_results,
        ablation_results,
        gap_results,
        output_path,
        metadata,
    )

    logger.info(f"Evaluation complete. Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
