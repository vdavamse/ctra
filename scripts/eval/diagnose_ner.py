#!/usr/bin/env python
"""Diagnostic script: dump GLiNER predictions vs gold annotations for a sample CHIA document.

Shows exactly what GLiNER finds, what the gold annotations expect, and why
matches fail. Helps diagnose whether low recall is a model problem or an
evaluation artifact.

Usage:
    python scripts/diagnose_ner.py --num-docs 3
    python scripts/diagnose_ner.py --doc-index 42
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose_ner")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose GLiNER NER on CHIA samples")
    parser.add_argument("--num-docs", type=int, default=3, help="Number of documents to inspect")
    parser.add_argument(
        "--doc-index", type=int, default=None, help="Specific document index to inspect"
    )
    parser.add_argument("--threshold", type=float, default=0.4, help="GLiNER threshold")
    parser.add_argument("--mode", choices=["native", "ctra"], default="both", help="Label mode")
    args = parser.parse_args()

    from ctra.config.settings import RAGConfig
    from ctra.rag.eval.data_models import BENCHMARK_NATIVE_LABELS, Benchmark
    from ctra.rag.eval.loaders import get_document_texts, load_benchmark
    from ctra.rag.ner_config import build_ner_pipeline

    # Load CHIA
    logger.info("Loading CHIA benchmark...")
    gold_annotations = load_benchmark(Benchmark.CHIA)
    doc_texts = get_document_texts(Benchmark.CHIA)

    # Group gold by doc_id
    gold_by_doc: dict[str, list] = {}
    for ann in gold_annotations:
        if ann.doc_id not in gold_by_doc:
            gold_by_doc[ann.doc_id] = []
        gold_by_doc[ann.doc_id].append(ann)

    doc_ids = list(doc_texts.keys())
    logger.info(f"Loaded {len(doc_ids)} documents, {len(gold_annotations)} gold entities")

    # Select documents
    if args.doc_index is not None:
        selected = [doc_ids[args.doc_index]]
    else:
        # Pick docs with moderate entity count for readability
        docs_with_counts = [(did, len(gold_by_doc.get(did, []))) for did in doc_ids]
        docs_with_counts.sort(key=lambda x: x[1])
        # Pick from 25th, 50th, 75th percentile by entity count
        n = len(docs_with_counts)
        indices = [n // 4, n // 2, 3 * n // 4]
        selected = [docs_with_counts[i][0] for i in indices[: args.num_docs]]

    # Build pipelines
    modes_to_run = []
    if args.mode in ("native", "both"):
        modes_to_run.append("native")
    if args.mode in ("ctra", "both"):
        modes_to_run.append("ctra")

    native_labels = BENCHMARK_NATIVE_LABELS[Benchmark.CHIA]
    ctra_labels = RAGConfig().ner_labels

    for doc_id in selected:
        text = doc_texts[doc_id]
        doc_gold = gold_by_doc.get(doc_id, [])

        print("\n" + "=" * 100)
        print(f"DOCUMENT: {doc_id}")
        print(f"Text length: {len(text)} chars")
        print(f"Gold entities: {len(doc_gold)}")
        print("=" * 100)

        # Show first 500 chars of text
        print("\n--- TEXT (first 500 chars) ---")
        print(text[:500])
        if len(text) > 500:
            print(f"... [{len(text) - 500} more chars]")

        # Show gold annotations
        print(f"\n--- GOLD ANNOTATIONS ({len(doc_gold)} entities) ---")
        gold_by_type: dict[str, list] = {}
        for ann in doc_gold:
            if ann.gold_label not in gold_by_type:
                gold_by_type[ann.gold_label] = []
            gold_by_type[ann.gold_label].append(ann)

        for label in sorted(gold_by_type.keys()):
            anns = gold_by_type[label]
            texts = [f'"{a.text}" [{a.start_char}:{a.end_char}]' for a in anns[:5]]
            suffix = f" ... +{len(anns) - 5} more" if len(anns) > 5 else ""
            print(f"  {label} ({len(anns)}): {', '.join(texts)}{suffix}")

        # Run GLiNER with native labels
        for mode in modes_to_run:
            if mode == "native":
                labels = native_labels
                label_name = "CHIA native"
            else:
                labels = ctra_labels
                label_name = "CTRA 16"

            print(f"\n--- GLiNER PREDICTIONS ({label_name} labels, threshold={args.threshold}) ---")

            config = RAGConfig()
            config.ner_labels = labels
            config.ner_threshold = args.threshold
            nlp = build_ner_pipeline(config)

            doc = nlp(text)
            ents = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

            if not ents:
                print("  NO ENTITIES FOUND")
                continue

            # Group predictions by label
            pred_by_type: dict[str, list] = {}
            for ent_text, ent_label, start, end in ents:
                if ent_label not in pred_by_type:
                    pred_by_type[ent_label] = []
                pred_by_type[ent_label].append((ent_text, start, end))

            for label in sorted(pred_by_type.keys()):
                preds = pred_by_type[label]
                texts = [f'"{p[0]}" [{p[1]}:{p[2]}]' for p in preds[:5]]
                suffix = f" ... +{len(preds) - 5} more" if len(preds) > 5 else ""
                print(f"  {label} ({len(preds)}): {', '.join(texts)}{suffix}")

            print(f"\n  Total predicted: {len(ents)}")

            # Match analysis (for CTRA mode, use mapped labels)
            if mode == "ctra":
                print("\n--- MATCH ANALYSIS (text_only) ---")
                matched = 0
                missed_examples: list[str] = []
                for ann in doc_gold:
                    if ann.mapped_ctra_label is None:
                        continue
                    # Check if any prediction matches
                    found = False
                    for ent_text, ent_label, start, end in ents:
                        if (
                            ent_label == ann.mapped_ctra_label
                            and ent_text.lower() == ann.text.lower()
                        ):
                            found = True
                            break
                    if found:
                        matched += 1
                    elif len(missed_examples) < 10:
                        # Check if there's a partial text match (substring)
                        partial = ""
                        for ent_text, ent_label, start, end in ents:
                            if ent_label == ann.mapped_ctra_label:
                                if (
                                    ann.text.lower() in ent_text.lower()
                                    or ent_text.lower() in ann.text.lower()
                                ):
                                    partial = f' (PARTIAL: pred="{ent_text}")'
                                    break
                        missed_examples.append(
                            f'  gold="{ann.text}" [{ann.gold_label}->{ann.mapped_ctra_label}]{partial}'
                        )

                mappable_gold = [a for a in doc_gold if a.mapped_ctra_label is not None]
                print(
                    f"  Matched: {matched}/{len(mappable_gold)} ({100 * matched / max(len(mappable_gold), 1):.1f}%)"
                )

                if missed_examples:
                    print(f"\n  First {len(missed_examples)} missed entities:")
                    for ex in missed_examples:
                        print(f"    {ex}")

                # Also check relaxed matching
                print("\n--- MATCH ANALYSIS (relaxed) ---")
                matched_relaxed = 0
                for ann in doc_gold:
                    if ann.mapped_ctra_label is None:
                        continue
                    for ent_text, ent_label, start, end in ents:
                        if ent_label == ann.mapped_ctra_label:
                            if max(start, ann.start_char) < min(end, ann.end_char):
                                matched_relaxed += 1
                                break

                print(
                    f"  Matched: {matched_relaxed}/{len(mappable_gold)} ({100 * matched_relaxed / max(len(mappable_gold), 1):.1f}%)"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
