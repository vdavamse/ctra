#!/usr/bin/env python
"""Full-scale NER diagnostic: run GLiNER over ALL CHIA documents and classify
every gold entity into one of these categories:

1. MATCHED_EXACT     — correct label + exact text match
2. MATCHED_RELAXED   — correct label + overlapping span
3. WRONG_LABEL       — GLiNER found the text but assigned a different CTRA label
4. PARTIAL_SPAN      — GLiNER found overlapping text but wrong label
5. TEXT_FOUND_NO_LABEL — entity text appears in predictions under ANY label
6. NOT_FOUND         — GLiNER did not extract this entity at all

Categories 3-5 are "evaluation artifacts" — GLiNER found the entity but the
label mapping or matching strategy rejected it. Category 6 is the true recall
gap. This distinction tells us whether low F1 is a model problem or a mapping
problem.

Also computes label-agnostic recall: what fraction of gold entity TEXTS does
GLiNER extract under any label? This is what LinearRAG actually cares about.

Usage:
    python scripts/diagnose_ner_full.py
    python scripts/diagnose_ner_full.py --threshold 0.3
    python scripts/diagnose_ner_full.py --output-dir output/ner_diagnosis/
"""

from __future__ import annotations

import argparse
import csv
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
logger = logging.getLogger("diagnose_ner_full")


def classify_gold_entity(
    ann_text: str,
    ann_start: int,
    ann_end: int,
    ann_mapped_label: str,
    predictions: list[tuple[str, str, int, int]],
) -> str:
    """Classify a gold entity into one of 6 categories based on predictions.

    predictions: list of (text, label, start_char, end_char)
    """
    ann_text_lower = ann_text.lower()

    for pred_text, pred_label, pred_start, pred_end in predictions:
        # Check exact text + correct label
        if pred_label == ann_mapped_label and pred_text.lower() == ann_text_lower:
            return "MATCHED_EXACT"

    for pred_text, pred_label, pred_start, pred_end in predictions:
        # Check overlapping span + correct label
        if pred_label == ann_mapped_label:
            if max(pred_start, ann_start) < min(pred_end, ann_end):
                return "MATCHED_RELAXED"

    for pred_text, pred_label, pred_start, pred_end in predictions:
        # Check exact text but wrong label
        if pred_text.lower() == ann_text_lower and pred_label != ann_mapped_label:
            return "WRONG_LABEL"

    for pred_text, pred_label, pred_start, pred_end in predictions:
        # Check overlapping span but wrong label
        if pred_label != ann_mapped_label:
            if max(pred_start, ann_start) < min(pred_end, ann_end):
                return "PARTIAL_SPAN"

    for pred_text, pred_label, pred_start, pred_end in predictions:
        # Check if gold text appears as substring of any prediction or vice versa
        if ann_text_lower in pred_text.lower() or pred_text.lower() in ann_text_lower:
            return "TEXT_FOUND_NO_LABEL"

    return "NOT_FOUND"


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-scale NER diagnostic on CHIA")
    parser.add_argument("--threshold", type=float, default=0.4, help="GLiNER threshold")
    parser.add_argument(
        "--output-dir", type=str, default="output/ner_diagnosis", help="Output directory"
    )
    args = parser.parse_args()

    from tqdm import tqdm

    from ctra.config.settings import RAGConfig
    from ctra.rag.eval.data_models import Benchmark
    from ctra.rag.eval.loaders import get_document_texts, load_benchmark
    from ctra.rag.ner_config import build_ner_pipeline

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load CHIA
    logger.info("Loading CHIA benchmark...")
    gold_annotations = load_benchmark(Benchmark.CHIA)
    doc_texts = get_document_texts(Benchmark.CHIA)

    # Group gold by doc_id
    gold_by_doc: dict[str, list] = defaultdict(list)
    for ann in gold_annotations:
        gold_by_doc[ann.doc_id].append(ann)

    doc_ids = list(doc_texts.keys())
    logger.info(f"Loaded {len(doc_ids)} documents, {len(gold_annotations)} gold entities")

    # Build GLiNER pipeline with CTRA labels
    logger.info("Building GLiNER pipeline...")
    config = RAGConfig()
    config.ner_threshold = args.threshold
    nlp = build_ner_pipeline(config)

    # Run inference on ALL documents
    logger.info("Running inference on all documents...")
    pred_by_doc: dict[str, list[tuple[str, str, int, int]]] = {}

    for doc_id in tqdm(doc_ids, desc="Inference"):
        text = doc_texts[doc_id]
        doc = nlp(text)
        pred_by_doc[doc_id] = [
            (ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents
        ]

    # Classify every gold entity
    logger.info("Classifying gold entities...")
    categories = Counter()
    per_label_categories: dict[str, Counter] = defaultdict(Counter)
    per_chia_type_categories: dict[str, Counter] = defaultdict(Counter)
    wrong_label_pairs: Counter = Counter()  # (expected, actual) pairs
    detailed_rows: list[dict] = []

    total_mappable = 0
    total_unmappable = 0

    for doc_id in tqdm(doc_ids, desc="Classifying"):
        doc_gold = gold_by_doc.get(doc_id, [])
        doc_preds = pred_by_doc.get(doc_id, [])

        for ann in doc_gold:
            if ann.mapped_ctra_label is None:
                total_unmappable += 1
                continue

            total_mappable += 1
            category = classify_gold_entity(
                ann.text,
                ann.start_char,
                ann.end_char,
                ann.mapped_ctra_label,
                doc_preds,
            )

            categories[category] += 1
            per_label_categories[ann.mapped_ctra_label][category] += 1
            per_chia_type_categories[ann.gold_label][category] += 1

            # Track wrong label details
            if category == "WRONG_LABEL":
                for pred_text, pred_label, _, _ in doc_preds:
                    if pred_text.lower() == ann.text.lower():
                        wrong_label_pairs[(ann.mapped_ctra_label, pred_label)] += 1
                        break

            detailed_rows.append(
                {
                    "doc_id": doc_id,
                    "gold_text": ann.text,
                    "gold_chia_type": ann.gold_label,
                    "gold_ctra_label": ann.mapped_ctra_label,
                    "start": ann.start_char,
                    "end": ann.end_char,
                    "category": category,
                }
            )

    # Also compute label-agnostic recall
    label_agnostic_found = 0
    label_agnostic_found_relaxed = 0
    for doc_id in doc_ids:
        doc_gold = gold_by_doc.get(doc_id, [])
        doc_preds = pred_by_doc.get(doc_id, [])

        for ann in doc_gold:
            if ann.mapped_ctra_label is None:
                continue

            # Text match under ANY label
            ann_lower = ann.text.lower()
            for pred_text, _, _, _ in doc_preds:
                if pred_text.lower() == ann_lower:
                    label_agnostic_found += 1
                    break

            # Span overlap under ANY label
            for _, _, pred_start, pred_end in doc_preds:
                if max(pred_start, ann.start_char) < min(pred_end, ann.end_char):
                    label_agnostic_found_relaxed += 1
                    break

    # Print results
    print("\n" + "=" * 100)
    print("FULL-SCALE NER DIAGNOSTIC — ALL CHIA DOCUMENTS")
    print(f"Documents: {len(doc_ids)}, Gold entities: {len(gold_annotations)}")
    print(f"Mappable (have CTRA label): {total_mappable}, Unmappable: {total_unmappable}")
    print(f"Total predictions: {sum(len(p) for p in pred_by_doc.values())}")
    print(f"Threshold: {args.threshold}")
    print("=" * 100)

    print("\n### CATEGORY BREAKDOWN (mappable gold entities only)")
    print(f"{'Category':<25} {'Count':>8} {'%':>8}  Meaning")
    print("-" * 90)
    desc = {
        "MATCHED_EXACT": "Correct label + exact text (counted in text_only eval)",
        "MATCHED_RELAXED": "Correct label + overlapping span (counted in relaxed eval)",
        "WRONG_LABEL": "GLiNER found text, assigned DIFFERENT label (mapping problem)",
        "PARTIAL_SPAN": "Overlapping span, wrong label (mapping + span problem)",
        "TEXT_FOUND_NO_LABEL": "Gold text is substring of a prediction (partial extraction)",
        "NOT_FOUND": "GLiNER did not extract this entity at all (true recall gap)",
    }
    for cat in [
        "MATCHED_EXACT",
        "MATCHED_RELAXED",
        "WRONG_LABEL",
        "PARTIAL_SPAN",
        "TEXT_FOUND_NO_LABEL",
        "NOT_FOUND",
    ]:
        count = categories[cat]
        pct = 100 * count / max(total_mappable, 1)
        print(f"  {cat:<23} {count:>8} {pct:>7.1f}%  {desc[cat]}")

    eval_artifact = (
        categories["WRONG_LABEL"] + categories["PARTIAL_SPAN"] + categories["TEXT_FOUND_NO_LABEL"]
    )
    true_found = categories["MATCHED_EXACT"] + categories["MATCHED_RELAXED"] + eval_artifact
    print(
        f"\n  Evaluation artifacts (found but wrong label): {eval_artifact} ({100 * eval_artifact / max(total_mappable, 1):.1f}%)"
    )
    print(
        f"  True model recall (found under any criteria):  {true_found} ({100 * true_found / max(total_mappable, 1):.1f}%)"
    )
    print(
        f"  True NOT found by GLiNER at all:               {categories['NOT_FOUND']} ({100 * categories['NOT_FOUND'] / max(total_mappable, 1):.1f}%)"
    )

    print("\n### LABEL-AGNOSTIC RECALL (what LinearRAG cares about)")
    print(
        f"  Text match (any label):  {label_agnostic_found}/{total_mappable} ({100 * label_agnostic_found / max(total_mappable, 1):.1f}%)"
    )
    print(
        f"  Span overlap (any label): {label_agnostic_found_relaxed}/{total_mappable} ({100 * label_agnostic_found_relaxed / max(total_mappable, 1):.1f}%)"
    )

    print("\n### TOP WRONG LABEL PAIRS (expected → actual)")
    print(f"  {'Expected CTRA label':<25} {'GLiNER assigned':<25} {'Count':>6}")
    print("  " + "-" * 60)
    for (expected, actual), count in wrong_label_pairs.most_common(15):
        print(f"  {expected:<25} {actual:<25} {count:>6}")

    print("\n### PER CHIA TYPE BREAKDOWN")
    print(
        f"  {'CHIA Type':<20} {'Total':>6} {'Matched':>8} {'WrongLbl':>8} {'NotFound':>8} {'Recall%':>8}"
    )
    print("  " + "-" * 70)
    for chia_type in sorted(per_chia_type_categories.keys()):
        cats = per_chia_type_categories[chia_type]
        total = sum(cats.values())
        matched = cats["MATCHED_EXACT"] + cats["MATCHED_RELAXED"]
        wrong = cats["WRONG_LABEL"] + cats["PARTIAL_SPAN"]
        not_found = cats["NOT_FOUND"]
        recall = 100 * matched / max(total, 1)
        print(
            f"  {chia_type:<20} {total:>6} {matched:>8} {wrong:>8} {not_found:>8} {recall:>7.1f}%"
        )

    print("\n### PER CTRA LABEL BREAKDOWN")
    print(
        f"  {'CTRA Label':<25} {'Total':>6} {'Matched':>8} {'WrongLbl':>8} {'NotFound':>8} {'Recall%':>8}"
    )
    print("  " + "-" * 75)
    for ctra_label in sorted(per_label_categories.keys()):
        cats = per_label_categories[ctra_label]
        total = sum(cats.values())
        matched = cats["MATCHED_EXACT"] + cats["MATCHED_RELAXED"]
        wrong = cats["WRONG_LABEL"] + cats["PARTIAL_SPAN"]
        not_found = cats["NOT_FOUND"]
        recall = 100 * matched / max(total, 1)
        print(
            f"  {ctra_label:<25} {total:>6} {matched:>8} {wrong:>8} {not_found:>8} {recall:>7.1f}%"
        )

    # Save detailed CSV
    csv_path = output_path / "entity_classification.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "gold_text",
                "gold_chia_type",
                "gold_ctra_label",
                "start",
                "end",
                "category",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)
    logger.info(f"Saved detailed entity classification to {csv_path}")

    # Save summary JSON
    summary = {
        "total_documents": len(doc_ids),
        "total_gold": len(gold_annotations),
        "total_mappable": total_mappable,
        "total_unmappable": total_unmappable,
        "total_predictions": sum(len(p) for p in pred_by_doc.values()),
        "threshold": args.threshold,
        "categories": dict(categories),
        "label_agnostic_text_recall": label_agnostic_found / max(total_mappable, 1),
        "label_agnostic_span_recall": label_agnostic_found_relaxed / max(total_mappable, 1),
        "wrong_label_pairs": {f"{e}->{a}": c for (e, a), c in wrong_label_pairs.most_common(30)},
    }
    summary_path = output_path / "diagnosis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
