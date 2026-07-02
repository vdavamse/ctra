#!/usr/bin/env python
"""Benchmark LinearRAG retrieval quality with different NER backends
using TREC Clinical Trials 2021 as ground truth.

Pipeline:
1. Download TREC-CT 2021 corpus (ClinicalTrials.gov snapshot) + queries + qrels
2. Build a test index from a subset of trial documents
3. Index the same documents with GLiNER vs scibert NER backends
4. Run TREC queries against both indices
5. Compute MRR, NDCG@10, Recall@10 against physician-labeled relevance

Usage:
    # Step 1: Download and prepare TREC data
    python scripts/benchmark_retrieval.py prepare --output-dir output/trec_retrieval/

    # Step 2: Build indices with different NER backends
    python scripts/benchmark_retrieval.py index --backend gliner --output-dir output/trec_retrieval/
    python scripts/benchmark_retrieval.py index --backend scibert --output-dir output/trec_retrieval/

    # Step 3: Run retrieval benchmark
    python scripts/benchmark_retrieval.py evaluate --output-dir output/trec_retrieval/

    # All steps at once
    python scripts/benchmark_retrieval.py run --output-dir output/trec_retrieval/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_retrieval")


# ---------------------------------------------------------------------------
# Step 1: Prepare TREC Clinical Trials data
# ---------------------------------------------------------------------------


def prepare_trec_data(output_dir: Path, max_docs: int = 5000) -> dict:
    """Download TREC-CT 2021 and prepare passages + queries + qrels.

    Args:
        output_dir: Where to write prepared data.
        max_docs: Max documents to include (for faster iteration).
            Set to 0 for full corpus.

    Returns:
        Dict with paths and stats.
    """
    import ir_datasets

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading TREC Clinical Trials 2021...")
    dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

    # Load queries
    queries = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text
    logger.info(f"Loaded {len(queries)} queries")

    # Load qrels (relevance judgments)
    # TREC-CT uses: 0=not relevant, 1=excluded, 2=eligible
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    relevant_doc_ids: set[str] = set()
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        if qrel.relevance >= 1:  # excluded or eligible
            relevant_doc_ids.add(qrel.doc_id)
    logger.info(
        f"Loaded qrels: {sum(len(v) for v in qrels.values())} judgments across {len(qrels)} queries"
    )
    logger.info(f"Relevant documents: {len(relevant_doc_ids)}")

    # Load documents — prioritize judged documents + fill with others
    logger.info("Loading trial documents...")
    passages = []
    doc_count = 0
    judged_count = 0

    for doc in dataset.docs_iter():
        # Build passage text from trial fields
        parts = []
        if hasattr(doc, "title") and doc.title:
            parts.append(f"Title: {doc.title}")
        if hasattr(doc, "condition") and doc.condition:
            parts.append(f"Condition: {doc.condition}")
        if hasattr(doc, "summary") and doc.summary:
            parts.append(f"Summary: {doc.summary}")
        if hasattr(doc, "detailed_description") and doc.detailed_description:
            parts.append(f"Description: {doc.detailed_description}")
        if hasattr(doc, "eligibility") and doc.eligibility:
            parts.append(f"Eligibility: {doc.eligibility}")

        text = "\n".join(parts)
        if not text.strip():
            continue

        is_judged = doc.doc_id in relevant_doc_ids
        if is_judged:
            judged_count += 1

        # Prioritize judged documents, then fill up to max_docs
        if is_judged or (max_docs == 0 or doc_count < max_docs):
            passages.append(
                {
                    "doc_id": doc.doc_id,
                    "text": text,
                    "source": "ctg",
                    "date": None,
                }
            )
            doc_count += 1

        if max_docs > 0 and doc_count >= max_docs and judged_count >= len(relevant_doc_ids):
            break

    logger.info(f"Prepared {len(passages)} passages ({judged_count} judged)")

    # Save as parquet for LinearRAG indexing
    import polars as pl

    passages_path = output_dir / "trec_passages.parquet"
    df = pl.DataFrame(passages)
    df.write_parquet(passages_path)
    logger.info(f"Saved passages to {passages_path}")

    # Save queries
    queries_path = output_dir / "trec_queries.json"
    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)

    # Save qrels
    qrels_path = output_dir / "trec_qrels.json"
    with open(qrels_path, "w") as f:
        json.dump(dict(qrels), f, indent=2)

    stats = {
        "num_queries": len(queries),
        "num_documents": len(passages),
        "num_judged_documents": judged_count,
        "num_qrels": sum(len(v) for v in qrels.values()),
    }
    stats_path = output_dir / "trec_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"TREC data prepared: {stats}")
    return stats


# ---------------------------------------------------------------------------
# Step 2: Build index with specific NER backend
# ---------------------------------------------------------------------------


def build_index(output_dir: Path, backend: str) -> Path:
    """Build a LinearRAG index from TREC passages with specified NER backend.

    Args:
        output_dir: Directory containing trec_passages.parquet.
        backend: NER backend — "gliner" or "scibert".

    Returns:
        Path to the built index directory.
    """
    import polars as pl

    from ctra.config.settings import RAGConfig
    from ctra.rag.ner_config import build_ner_pipeline

    passages_path = output_dir / "trec_passages.parquet"
    index_dir = output_dir / f"index_{backend}"
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building index with {backend} NER backend...")

    # Load passages
    df = pl.read_parquet(passages_path)
    texts = df["text"].to_list()
    doc_ids = df["doc_id"].to_list()
    logger.info(f"Loaded {len(texts)} passages")

    # Build NER pipeline
    config = RAGConfig()
    if backend == "scibert":
        config.ner_model = "en_core_sci_scibert"
    elif backend == "gliner":
        config.ner_model = "gliner_bio"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    nlp = build_ner_pipeline(config)

    # Run NER on all passages using nlp.pipe() for GPU batching
    logger.info("Running NER on passages (batched)...")
    from tqdm import tqdm

    batch_size = 32  # Reasonable batch size for T4/A10 GPU with transformer NER

    passage_entities: dict[str, list[str]] = {}
    total_entities = 0
    start_time = time.time()

    for i, doc in enumerate(
        tqdm(
            nlp.pipe(texts, batch_size=batch_size),
            total=len(texts),
            desc=f"NER ({backend}, batch={batch_size})",
        )
    ):
        ents = list(
            {ent.text.lower() for ent in doc.ents if ent.label_ not in ("ORDINAL", "CARDINAL")}
        )
        passage_entities[doc_ids[i]] = ents
        total_entities += len(ents)

    elapsed = time.time() - start_time
    throughput = len(texts) / elapsed
    logger.info(
        f"NER complete: {total_entities} total entities, "
        f"avg {total_entities / len(texts):.1f} per passage, "
        f"throughput={throughput:.1f} docs/sec (elapsed={elapsed:.1f}s)"
    )

    # Save NER results and passages for retrieval
    ner_path = index_dir / "passage_entities.json"
    with open(ner_path, "w") as f:
        json.dump(passage_entities, f)

    # Copy passages parquet to index dir
    import shutil

    shutil.copy(passages_path, index_dir / "passages.parquet")

    # Save index metadata
    meta = {
        "backend": backend,
        "num_passages": len(texts),
        "total_entities": total_entities,
        "avg_entities_per_passage": round(total_entities / len(texts), 1),
        "unique_entities": len({e for ents in passage_entities.values() for e in ents}),
    }
    with open(index_dir / "index_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Index built at {index_dir}: {meta}")
    return index_dir


# ---------------------------------------------------------------------------
# Step 3: Evaluate retrieval quality
# ---------------------------------------------------------------------------


def evaluate_retrieval(output_dir: Path, top_k: int = 10) -> dict:
    """Run TREC queries against both indices and compute retrieval metrics.

    Uses entity-overlap scoring: for each query, extract query entities via
    NER, then rank passages by the number of shared entities. This simulates
    LinearRAG's entity-based retrieval without needing full graph construction
    (which requires embeddings + GPU).

    Args:
        output_dir: Directory containing indices and TREC data.
        top_k: Number of results to evaluate.

    Returns:
        Dict with per-backend and per-query metrics.
    """
    # Load queries and qrels
    with open(output_dir / "trec_queries.json") as f:
        queries = json.load(f)
    with open(output_dir / "trec_qrels.json") as f:
        qrels = json.load(f)

    # Find available backends
    backends = []
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("index_"):
            backends.append(d.name.removeprefix("index_"))

    if not backends:
        logger.error("No indices found. Run 'index' command first.")
        return {}

    logger.info(f"Evaluating backends: {backends}")

    from ctra.config.settings import RAGConfig
    from ctra.rag.ner_config import build_ner_pipeline

    results: dict[str, dict] = {}

    for backend in backends:
        index_dir = output_dir / f"index_{backend}"
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating {backend}...")

        # Load passage entities
        with open(index_dir / "passage_entities.json") as f:
            passage_entities = json.load(f)

        # Build NER pipeline for query processing
        config = RAGConfig()
        if backend == "scibert":
            config.ner_model = "en_core_sci_scibert"
        else:
            config.ner_model = "gliner_bio"
        nlp = build_ner_pipeline(config)

        # Evaluate each query
        per_query: list[dict] = []
        query_entity_counts: list[int] = []
        seed_match_counts: list[int] = []

        for qid, query_text in queries.items():
            if qid not in qrels:
                continue

            # Extract query entities
            doc = nlp(query_text)
            query_ents = list(
                {ent.text.lower() for ent in doc.ents if ent.label_ not in ("ORDINAL", "CARDINAL")}
            )
            query_entity_counts.append(len(query_ents))

            # Score passages by entity overlap
            scored: list[tuple[str, float]] = []
            for doc_id, passage_ents in passage_entities.items():
                if not query_ents:
                    score = 0.0
                else:
                    overlap = len(set(query_ents) & set(passage_ents))
                    score = overlap / len(query_ents)  # Jaccard-like
                scored.append((doc_id, score))

            # Sort by score descending
            scored.sort(key=lambda x: -x[1])
            top_results = scored[:top_k]

            # Count how many query entities found seed matches in the index
            all_index_ents = {e for ents in passage_entities.values() for e in ents}
            seed_matches = len(set(query_ents) & all_index_ents)
            seed_match_counts.append(seed_matches)

            # Compute metrics against qrels
            query_qrels = qrels[qid]

            # MRR: reciprocal rank of first relevant result
            mrr = 0.0
            for rank, (doc_id, _score) in enumerate(top_results, 1):
                if doc_id in query_qrels and query_qrels[doc_id] >= 1:
                    mrr = 1.0 / rank
                    break

            # Recall@K: fraction of relevant docs in top-K
            relevant_in_qrels = {did for did, rel in query_qrels.items() if rel >= 1}
            retrieved_relevant = {did for did, _ in top_results if did in relevant_in_qrels}
            recall_at_k = len(retrieved_relevant) / max(len(relevant_in_qrels), 1)

            # NDCG@K
            ndcg = compute_ndcg(top_results, query_qrels, top_k)

            # Eligible@K: fraction of top-K that are "eligible" (rel=2)
            eligible_docs = {did for did, rel in query_qrels.items() if rel == 2}
            retrieved_eligible = {did for did, _ in top_results if did in eligible_docs}
            eligible_at_k = (
                len(retrieved_eligible) / max(len(eligible_docs), 1) if eligible_docs else 0.0
            )

            per_query.append(
                {
                    "query_id": qid,
                    "query_entities": len(query_ents),
                    "seed_matches": seed_matches,
                    "mrr": mrr,
                    "recall_at_k": recall_at_k,
                    "ndcg_at_k": ndcg,
                    "eligible_at_k": eligible_at_k,
                    "top_result_relevant": top_results[0][0] in relevant_in_qrels
                    if top_results
                    else False,
                }
            )

        # Aggregate metrics
        n = len(per_query)
        avg_mrr = sum(q["mrr"] for q in per_query) / max(n, 1)
        avg_recall = sum(q["recall_at_k"] for q in per_query) / max(n, 1)
        avg_ndcg = sum(q["ndcg_at_k"] for q in per_query) / max(n, 1)
        avg_eligible = sum(q["eligible_at_k"] for q in per_query) / max(n, 1)
        avg_query_ents = sum(query_entity_counts) / max(n, 1)
        avg_seed_matches = sum(seed_match_counts) / max(n, 1)
        queries_with_entities = sum(1 for c in query_entity_counts if c > 0)
        queries_with_seeds = sum(1 for c in seed_match_counts if c > 0)

        with open(index_dir / "index_meta.json") as f:
            meta = json.load(f)

        results[backend] = {
            "backend": backend,
            "num_queries": n,
            "avg_mrr": round(avg_mrr, 4),
            "avg_recall_at_k": round(avg_recall, 4),
            "avg_ndcg_at_k": round(avg_ndcg, 4),
            "avg_eligible_at_k": round(avg_eligible, 4),
            "avg_query_entities": round(avg_query_ents, 1),
            "avg_seed_matches": round(avg_seed_matches, 1),
            "queries_with_entities": queries_with_entities,
            "queries_with_seed_matches": queries_with_seeds,
            "index_total_entities": meta["total_entities"],
            "index_unique_entities": meta["unique_entities"],
            "index_avg_entities_per_passage": meta["avg_entities_per_passage"],
            "per_query": per_query,
        }

    # Print comparison
    print("\n" + "=" * 100)
    print("LINEARRAG RETRIEVAL BENCHMARK — TREC Clinical Trials 2021")
    print(f"Queries: {len(queries)}, Top-K: {top_k}")
    print("=" * 100)

    print(f"\n{'Metric':<40}", end="")
    for b in sorted(results.keys()):
        print(f"{b:>15}", end="")
    print()
    print("-" * (40 + 15 * len(results)))

    metrics_to_show = [
        ("MRR (Mean Reciprocal Rank)", "avg_mrr"),
        ("Recall@10", "avg_recall_at_k"),
        ("NDCG@10", "avg_ndcg_at_k"),
        ("Eligible@10", "avg_eligible_at_k"),
        ("Avg query entities", "avg_query_entities"),
        ("Avg seed matches in index", "avg_seed_matches"),
        ("Queries with entities", "queries_with_entities"),
        ("Queries with seed matches", "queries_with_seed_matches"),
        ("Index: total entities", "index_total_entities"),
        ("Index: unique entities", "index_unique_entities"),
        ("Index: avg entities/passage", "index_avg_entities_per_passage"),
    ]

    for label, key in metrics_to_show:
        print(f"{label:<40}", end="")
        for b in sorted(results.keys()):
            val = results[b][key]
            if isinstance(val, float):
                print(f"{val:>15.4f}", end="")
            else:
                print(f"{val:>15}", end="")
        print()

    # Save results
    results_path = output_dir / "retrieval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved results to {results_path}")

    return results


def compute_ndcg(
    ranked_results: list[tuple[str, float]],
    qrels: dict[str, int],
    k: int,
) -> float:
    """Compute NDCG@K."""
    dcg = 0.0
    for i, (doc_id, _score) in enumerate(ranked_results[:k]):
        rel = qrels.get(doc_id, 0)
        dcg += rel / math.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark LinearRAG retrieval with TREC-CT 2021")
    parser.add_argument(
        "command",
        choices=["prepare", "index", "evaluate", "run"],
        help="Step to execute (run = all steps)",
    )
    parser.add_argument("--output-dir", type=str, default="output/trec_retrieval")
    parser.add_argument("--backend", type=str, default="gliner", help="NER backend for index step")
    parser.add_argument(
        "--max-docs", type=int, default=5000, help="Max documents for prepare step (0=all)"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for evaluation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.command == "prepare":
        prepare_trec_data(output_dir, max_docs=args.max_docs)

    elif args.command == "index":
        build_index(output_dir, args.backend)

    elif args.command == "evaluate":
        evaluate_retrieval(output_dir, top_k=args.top_k)

    elif args.command == "run":
        logger.info("Running full benchmark pipeline...")

        # Step 1: Prepare
        prepare_trec_data(output_dir, max_docs=args.max_docs)

        # Step 2: Build both indices
        for backend in ["gliner", "scibert"]:
            build_index(output_dir, backend)

        # Step 3: Evaluate
        evaluate_retrieval(output_dir, top_k=args.top_k)

    return 0


if __name__ == "__main__":
    sys.exit(main())
