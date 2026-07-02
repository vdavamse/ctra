#!/usr/bin/env python
"""Compare GLiNER-BioMed vs scispaCy BioBERT NER on CHIA.

Runs both models over all CHIA documents and compares:
1. Total entities found
2. Label-agnostic recall (span overlap with gold — what LinearRAG cares about)
3. Label-agnostic recall (text match)
4. Entity type distribution
5. Per-CHIA-type recall comparison

scispaCy models available:
- en_core_sci_scibert: general biomedical NER, single ENTITY label
- en_ner_bc5cdr_md: CHEMICAL + DISEASE labels (trained on BC5CDR)
- en_ner_bionlp13cg_md: 16 biomedical types (CANCER, CELL, GENE_OR_GENE_PRODUCT, etc.)

Usage:
    python scripts/diagnose_ner_biobert.py
    python scripts/diagnose_ner_biobert.py --scispacy-model en_ner_bionlp13cg_md
    python scripts/diagnose_ner_biobert.py --skip-gliner  # only run scispaCy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose_biobert")


def compute_label_agnostic_recall(
    gold_by_doc: dict[str, list],
    pred_by_doc: dict[str, list[tuple[str, str, int, int]]],
    doc_ids: list[str],
) -> dict[str, float | int]:
    """Compute label-agnostic recall metrics.

    Returns dict with text_match, span_overlap counts and percentages,
    plus per-CHIA-type span recall breakdown.
    """
    total_mappable = 0
    text_matched = 0
    span_matched = 0
    per_chia_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "span_matched": 0})

    for doc_id in doc_ids:
        doc_gold = gold_by_doc.get(doc_id, [])
        doc_preds = pred_by_doc.get(doc_id, [])

        for ann in doc_gold:
            if ann.mapped_ctra_label is None:
                continue
            total_mappable += 1
            per_chia_type[ann.gold_label]["total"] += 1

            ann_lower = ann.text.lower()

            # Text match under ANY label
            for pred_text, _, _, _ in doc_preds:
                if pred_text.lower() == ann_lower:
                    text_matched += 1
                    break

            # Span overlap under ANY label
            for _, _, pred_start, pred_end in doc_preds:
                if max(pred_start, ann.start_char) < min(pred_end, ann.end_char):
                    span_matched += 1
                    per_chia_type[ann.gold_label]["span_matched"] += 1
                    break

    return {
        "total_mappable": total_mappable,
        "text_matched": text_matched,
        "text_recall_pct": 100 * text_matched / max(total_mappable, 1),
        "span_matched": span_matched,
        "span_recall_pct": 100 * span_matched / max(total_mappable, 1),
        "per_chia_type": dict(per_chia_type),
    }


def run_gliner(doc_texts: dict[str, str], doc_ids: list[str], threshold: float) -> dict[str, list]:
    """Run GLiNER-BioMed on all documents."""
    from tqdm import tqdm

    from ctra.config.settings import RAGConfig
    from ctra.rag.ner_config import build_ner_pipeline

    logger.info("Building GLiNER-BioMed pipeline...")
    config = RAGConfig()
    config.ner_threshold = threshold
    nlp = build_ner_pipeline(config)

    pred_by_doc: dict[str, list[tuple[str, str, int, int]]] = {}
    for doc_id in tqdm(doc_ids, desc="GLiNER inference"):
        doc = nlp(doc_texts[doc_id])
        pred_by_doc[doc_id] = [
            (ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents
        ]

    return pred_by_doc


def run_scispacy(
    doc_texts: dict[str, str],
    doc_ids: list[str],
    model_name: str,
) -> dict[str, list]:
    """Run scispaCy BioBERT model on all documents."""
    import spacy
    from tqdm import tqdm

    logger.info(f"Loading scispaCy model: {model_name}...")
    nlp = spacy.load(model_name)

    # Log pipeline components
    logger.info(f"Pipeline components: {nlp.pipe_names}")
    logger.info(f"Labels: {nlp.get_pipe('ner').labels if 'ner' in nlp.pipe_names else 'N/A'}")

    pred_by_doc: dict[str, list[tuple[str, str, int, int]]] = {}
    for doc_id in tqdm(doc_ids, desc=f"scispaCy ({model_name}) inference"):
        doc = nlp(doc_texts[doc_id])
        pred_by_doc[doc_id] = [
            (ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents
        ]

    return pred_by_doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare GLiNER vs scispaCy BioBERT on CHIA")
    parser.add_argument(
        "--scispacy-model",
        type=str,
        default="en_core_sci_scibert",
        help="scispaCy model name (default: en_core_sci_scibert)",
    )
    parser.add_argument("--threshold", type=float, default=0.4, help="GLiNER threshold")
    parser.add_argument("--skip-gliner", action="store_true", help="Skip GLiNER, only run scispaCy")
    parser.add_argument(
        "--skip-scispacy", action="store_true", help="Skip scispaCy, only run GLiNER"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/ner_comparison", help="Output directory"
    )
    args = parser.parse_args()

    from ctra.rag.eval.data_models import Benchmark
    from ctra.rag.eval.loaders import get_document_texts, load_benchmark

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load CHIA
    logger.info("Loading CHIA benchmark...")
    gold_annotations = load_benchmark(Benchmark.CHIA)
    doc_texts = get_document_texts(Benchmark.CHIA)

    gold_by_doc: dict[str, list] = defaultdict(list)
    for ann in gold_annotations:
        gold_by_doc[ann.doc_id].append(ann)

    doc_ids = list(doc_texts.keys())
    logger.info(f"Loaded {len(doc_ids)} documents, {len(gold_annotations)} gold entities")

    results: dict[str, dict] = {}

    # Run GLiNER
    if not args.skip_gliner:
        logger.info("=" * 80)
        logger.info("Running GLiNER-BioMed...")
        gliner_preds = run_gliner(doc_texts, doc_ids, args.threshold)
        total_gliner_ents = sum(len(p) for p in gliner_preds.values())
        logger.info(f"GLiNER total predictions: {total_gliner_ents}")

        gliner_recall = compute_label_agnostic_recall(gold_by_doc, gliner_preds, doc_ids)
        results["gliner"] = {
            "model": "Ihor/gliner-biomed-large-v1.0",
            "threshold": args.threshold,
            "total_predictions": total_gliner_ents,
            **gliner_recall,
        }

        # Count label distribution
        gliner_label_dist: Counter = Counter()
        for preds in gliner_preds.values():
            for _, label, _, _ in preds:
                gliner_label_dist[label] += 1
        results["gliner"]["label_distribution"] = dict(gliner_label_dist.most_common())

    # Run scispaCy
    if not args.skip_scispacy:
        logger.info("=" * 80)
        logger.info(f"Running scispaCy ({args.scispacy_model})...")
        scispacy_preds = run_scispacy(doc_texts, doc_ids, args.scispacy_model)
        total_scispacy_ents = sum(len(p) for p in scispacy_preds.values())
        logger.info(f"scispaCy total predictions: {total_scispacy_ents}")

        scispacy_recall = compute_label_agnostic_recall(gold_by_doc, scispacy_preds, doc_ids)
        results["scispacy"] = {
            "model": args.scispacy_model,
            "total_predictions": total_scispacy_ents,
            **scispacy_recall,
        }

        # Count label distribution
        scispacy_label_dist: Counter = Counter()
        for preds in scispacy_preds.values():
            for _, label, _, _ in preds:
                scispacy_label_dist[label] += 1
        results["scispacy"]["label_distribution"] = dict(scispacy_label_dist.most_common())

    # Print comparison
    print("\n" + "=" * 100)
    print("NER MODEL COMPARISON ON CHIA")
    print(f"Documents: {len(doc_ids)}, Gold entities: {len(gold_annotations)}")
    mappable = results.get("gliner", results.get("scispacy", {})).get("total_mappable", 0)
    print(f"Mappable gold entities: {mappable}")
    print("=" * 100)

    print("\n### SUMMARY")
    print(f"  {'Metric':<35} ", end="")
    for name in results:
        print(
            f"{'GLiNER-BioMed':>18}" if name == "gliner" else f"{args.scispacy_model:>18}", end=""
        )
    print()
    print("  " + "-" * (35 + 18 * len(results)))

    metrics = [
        ("Total predictions", "total_predictions"),
        ("Label-agnostic text recall", "text_recall_pct"),
        ("Label-agnostic span recall", "span_recall_pct"),
    ]
    for label, key in metrics:
        print(f"  {label:<35} ", end="")
        for name in results:
            val = results[name][key]
            if isinstance(val, float):
                print(f"{val:>17.1f}%", end="")
            else:
                print(f"{val:>18}", end="")
        print()

    # Per-CHIA-type comparison
    print("\n### PER CHIA TYPE — LABEL-AGNOSTIC SPAN RECALL")
    all_chia_types = set()
    for name in results:
        all_chia_types.update(results[name].get("per_chia_type", {}).keys())

    print(f"  {'CHIA Type':<20} {'Gold':>6}", end="")
    for name in results:
        model_label = "GLiNER" if name == "gliner" else "scispaCy"
        print(f"  {model_label + ' found':>14} {model_label + ' %':>10}", end="")
    print()
    print("  " + "-" * (26 + 26 * len(results)))

    for chia_type in sorted(all_chia_types):
        # Get total from any result
        total = 0
        for name in results:
            ct = results[name].get("per_chia_type", {}).get(chia_type, {})
            if ct.get("total", 0) > total:
                total = ct.get("total", 0)

        print(f"  {chia_type:<20} {total:>6}", end="")
        for name in results:
            ct = results[name].get("per_chia_type", {}).get(chia_type, {})
            found = ct.get("span_matched", 0)
            t = ct.get("total", 0)
            pct = 100 * found / max(t, 1)
            print(f"  {found:>14} {pct:>9.1f}%", end="")
        print()

    # Label distributions
    for name in results:
        model_label = "GLiNER-BioMed" if name == "gliner" else args.scispacy_model
        dist = results[name].get("label_distribution", {})
        print(f"\n### {model_label} — LABEL DISTRIBUTION (top 20)")
        for label, count in sorted(dist.items(), key=lambda x: -x[1])[:20]:
            print(f"  {label:<30} {count:>6}")

    # Save results
    summary_path = output_path / "comparison_summary.json"
    # Convert per_chia_type for JSON serialization
    for name in results:
        if "per_chia_type" in results[name]:
            results[name]["per_chia_type"] = {
                k: dict(v) for k, v in results[name]["per_chia_type"].items()
            }
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved comparison to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
