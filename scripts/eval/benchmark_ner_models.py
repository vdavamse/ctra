#!/usr/bin/env python
"""Benchmark all available NER models on CHIA clinical trial text.

Compares label-agnostic span recall (what LinearRAG cares about) across:
1. GLiNER-BioMed (zero-shot, 16 CTRA labels)
2. scispaCy en_core_sci_scibert (supervised, generic ENTITY)
3. HunFlair v2 (BiLSTM-CRF, 5 biomedical types)
4. d4data/biomedical-ner-all (DeBERTa, 107 entity types)
5. Stanza biomedical (bc5cdr: CHEMICAL+DISEASE, i2b2: clinical)

Usage:
    python scripts/benchmark_ner_models.py
    python scripts/benchmark_ner_models.py --models scibert,hunflair,d4data
    python scripts/benchmark_ner_models.py --models all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_ner")

AVAILABLE_MODELS = [
    "gliner",
    "scibert",
    "hunflair",
    "d4data",
    "stanza_bc5cdr",
    "stanza_i2b2",
]


def compute_recall(
    gold_by_doc: dict[str, list],
    pred_by_doc: dict[str, list[tuple[str, str, int, int]]],
    doc_ids: list[str],
) -> dict:
    """Compute label-agnostic recall metrics."""
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
            for pred_text, _, _, _ in doc_preds:
                if pred_text.lower() == ann_lower:
                    text_matched += 1
                    break

            for _, _, ps, pe in doc_preds:
                if max(ps, ann.start_char) < min(pe, ann.end_char):
                    span_matched += 1
                    per_chia_type[ann.gold_label]["span_matched"] += 1
                    break

    return {
        "total_mappable": total_mappable,
        "text_matched": text_matched,
        "text_recall_pct": round(100 * text_matched / max(total_mappable, 1), 1),
        "span_matched": span_matched,
        "span_recall_pct": round(100 * span_matched / max(total_mappable, 1), 1),
        "per_chia_type": {k: dict(v) for k, v in per_chia_type.items()},
    }


def _run_spacy_batched(
    nlp,
    doc_texts: dict[str, str],
    doc_ids: list[str],
    desc: str,
    batch_size: int = 32,
) -> dict[str, list]:
    """Run a spaCy-compatible NER pipeline with batched inference."""
    from tqdm import tqdm

    texts = [doc_texts[did] for did in doc_ids]
    preds: dict[str, list] = {}
    for i, doc in enumerate(
        tqdm(
            nlp.pipe(texts, batch_size=batch_size),
            total=len(texts),
            desc=f"{desc} (batch={batch_size})",
        )
    ):
        preds[doc_ids[i]] = [(e.text, e.label_, e.start_char, e.end_char) for e in doc.ents]
    return preds


def run_gliner(doc_texts: dict[str, str], doc_ids: list[str]) -> dict[str, list]:
    from ctra.config.settings import RAGConfig
    from ctra.rag.ner_config import build_ner_pipeline

    logger.info("Loading GLiNER-BioMed...")
    config = RAGConfig()
    nlp = build_ner_pipeline(config)
    return _run_spacy_batched(nlp, doc_texts, doc_ids, desc="GLiNER")


def run_scibert(doc_texts: dict[str, str], doc_ids: list[str]) -> dict[str, list]:
    import spacy

    logger.info("Loading scispaCy en_core_sci_scibert...")
    nlp = spacy.load("en_core_sci_scibert")
    return _run_spacy_batched(nlp, doc_texts, doc_ids, desc="scibert")


def run_hunflair(doc_texts: dict[str, str], doc_ids: list[str]) -> dict[str, list]:
    from flair.data import Sentence
    from flair.nn import Classifier
    from tqdm import tqdm

    logger.info("Loading HunFlair v2...")
    tagger = Classifier.load("hunflair2")

    preds: dict[str, list] = {}
    for doc_id in tqdm(doc_ids, desc="HunFlair"):
        text = doc_texts[doc_id]
        # HunFlair has a max sentence length; split on newlines
        sentences = [Sentence(s) for s in text.split("\n") if s.strip()]
        if not sentences:
            preds[doc_id] = []
            continue

        tagger.predict(sentences)

        doc_ents = []
        offset = 0
        for sent_text, sent in zip(text.split("\n"), sentences, strict=False):
            for entity in sent.get_spans("ner"):
                start = offset + entity.start_position
                end = offset + entity.end_position
                doc_ents.append((entity.text, entity.tag, start, end))
            offset += len(sent_text) + 1  # +1 for newline

        preds[doc_id] = doc_ents
    return preds


def run_d4data(doc_texts: dict[str, str], doc_ids: list[str]) -> dict[str, list]:
    from tqdm import tqdm
    from transformers import pipeline

    logger.info("Loading d4data/biomedical-ner-all...")
    pipe = pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
        device=0,  # GPU
    )

    preds: dict[str, list] = {}
    for doc_id in tqdm(doc_ids, desc="d4data"):
        text = doc_texts[doc_id]
        # Transformers pipeline has max token limit; chunk the text
        try:
            results = pipe(text[:10000])  # Truncate very long docs
        except Exception as e:
            logger.warning(f"d4data error on {doc_id}: {e}")
            preds[doc_id] = []
            continue

        doc_ents = []
        for r in results:
            doc_ents.append(
                (
                    r["word"],
                    r["entity_group"],
                    r["start"],
                    r["end"],
                )
            )
        preds[doc_id] = doc_ents
    return preds


def run_stanza(
    doc_texts: dict[str, str],
    doc_ids: list[str],
    ner_package: str,
) -> dict[str, list]:
    import stanza
    from tqdm import tqdm

    logger.info(f"Loading Stanza biomedical (ner={ner_package})...")
    nlp = stanza.Pipeline(
        "en",
        package="mimic",
        processors={"ner": ner_package},
        use_gpu=True,
    )

    preds: dict[str, list] = {}
    for doc_id in tqdm(doc_ids, desc=f"Stanza-{ner_package}"):
        text = doc_texts[doc_id]
        try:
            doc = nlp(text)
        except Exception as e:
            logger.warning(f"Stanza error on {doc_id}: {e}")
            preds[doc_id] = []
            continue

        doc_ents = []
        for ent in doc.ents:
            doc_ents.append((ent.text, ent.type, ent.start_char, ent.end_char))
        preds[doc_id] = doc_ents
    return preds


MODEL_RUNNERS = {
    "gliner": run_gliner,
    "scibert": run_scibert,
    "hunflair": run_hunflair,
    "d4data": run_d4data,
    "stanza_bc5cdr": lambda dt, di: run_stanza(dt, di, "bc5cdr"),
    "stanza_i2b2": lambda dt, di: run_stanza(dt, di, "i2b2"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NER models on CHIA")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=f"Comma-separated model names or 'all'. Available: {','.join(AVAILABLE_MODELS)}",
    )
    parser.add_argument("--output-dir", type=str, default="output/ner_benchmark")
    args = parser.parse_args()

    from ctra.rag.eval.data_models import Benchmark
    from ctra.rag.eval.loaders import get_document_texts, load_benchmark

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse model list
    if args.models == "all":
        models_to_run = AVAILABLE_MODELS
    else:
        models_to_run = [m.strip() for m in args.models.split(",")]
        for m in models_to_run:
            if m not in MODEL_RUNNERS:
                logger.error(f"Unknown model: {m}. Available: {AVAILABLE_MODELS}")
                return 1

    # Load CHIA
    logger.info("Loading CHIA benchmark...")
    gold_annotations = load_benchmark(Benchmark.CHIA)
    doc_texts = get_document_texts(Benchmark.CHIA)

    gold_by_doc: dict[str, list] = defaultdict(list)
    for ann in gold_annotations:
        gold_by_doc[ann.doc_id].append(ann)

    doc_ids = list(doc_texts.keys())
    logger.info(f"Loaded {len(doc_ids)} documents, {len(gold_annotations)} gold entities")

    # Run each model
    results: dict[str, dict] = {}
    for model_name in models_to_run:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running {model_name}...")

        runner = MODEL_RUNNERS[model_name]
        start_time = time.time()

        try:
            pred_by_doc = runner(doc_texts, doc_ids)
        except Exception as e:
            logger.error(f"Failed to run {model_name}: {e}")
            results[model_name] = {"error": str(e)}
            continue

        elapsed = time.time() - start_time
        total_preds = sum(len(p) for p in pred_by_doc.values())

        recall = compute_recall(gold_by_doc, pred_by_doc, doc_ids)

        # Label distribution
        label_dist: Counter = Counter()
        for preds in pred_by_doc.values():
            for _, label, _, _ in preds:
                label_dist[label] += 1

        results[model_name] = {
            "total_predictions": total_preds,
            "inference_time_sec": round(elapsed, 1),
            "docs_per_sec": round(len(doc_ids) / elapsed, 1),
            **recall,
            "label_distribution": dict(label_dist.most_common(20)),
        }

        logger.info(
            f"{model_name}: {total_preds} predictions, "
            f"span_recall={recall['span_recall_pct']}%, "
            f"text_recall={recall['text_recall_pct']}%, "
            f"time={elapsed:.1f}s"
        )

    # Print comparison table
    print("\n" + "=" * 120)
    print("NER MODEL BENCHMARK ON CHIA (1,000 documents, label-agnostic recall)")
    print("=" * 120)

    # Header
    print(
        f"\n{'Model':<20} {'Predictions':>12} {'Span Recall':>12} {'Text Recall':>12} {'Time (s)':>10} {'Docs/sec':>10}"
    )
    print("-" * 80)

    # Sort by span recall descending
    sorted_models = sorted(
        [(m, r) for m, r in results.items() if "error" not in r],
        key=lambda x: x[1]["span_recall_pct"],
        reverse=True,
    )

    for model_name, r in sorted_models:
        print(
            f"{model_name:<20} {r['total_predictions']:>12} "
            f"{r['span_recall_pct']:>11.1f}% "
            f"{r['text_recall_pct']:>11.1f}% "
            f"{r['inference_time_sec']:>10.1f} "
            f"{r['docs_per_sec']:>10.1f}"
        )

    # Print errors
    for model_name, r in results.items():
        if "error" in r:
            print(f"{model_name:<20} ERROR: {r['error']}")

    # Per CHIA type comparison
    print(f"\n{'=' * 120}")
    print("PER CHIA TYPE — LABEL-AGNOSTIC SPAN RECALL (%)")
    print(f"{'=' * 120}")

    chia_types = [
        "Condition",
        "Drug",
        "Measurement",
        "Procedure",
        "Temporal",
        "Person",
        "Reference_point",
        "Visit",
    ]

    header = f"{'CHIA Type':<20} {'Gold':>6}"
    for model_name, _ in sorted_models:
        header += f" {model_name:>14}"
    print(header)
    print("-" * (26 + 15 * len(sorted_models)))

    for ct in chia_types:
        row = f"{ct:<20}"
        total = 0
        for _, r in sorted_models:
            t = r.get("per_chia_type", {}).get(ct, {}).get("total", 0)
            if t > total:
                total = t
        row += f" {total:>6}"

        for _, r in sorted_models:
            ct_data = r.get("per_chia_type", {}).get(ct, {})
            t = ct_data.get("total", 0)
            matched = ct_data.get("span_matched", 0)
            pct = 100 * matched / max(t, 1) if t > 0 else 0
            row += f" {pct:>13.1f}%"
        print(row)

    # Label distributions
    for model_name, r in sorted_models:
        if "error" in r:
            continue
        dist = r.get("label_distribution", {})
        print(f"\n{model_name} — top labels: ", end="")
        top = list(dist.items())[:5]
        print(", ".join(f"{l}({c})" for l, c in top))

    # Save
    summary_path = output_path / "benchmark_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
