#!/usr/bin/env python
"""Benchmark NER throughput with different batching strategies.

Measures actual docs/sec for GLiNER and scibert on a small sample with:
1. No batching (one-at-a-time)
2. nlp.pipe(batch_size=1)
3. nlp.pipe(batch_size=16)
4. nlp.pipe(batch_size=32)
5. nlp.pipe(batch_size=64)
6. nlp.pipe(batch_size=128)

This tells us whether:
- Batching provides real speedup (GPU parallelization works)
- What batch size is optimal
- Whether gliner-spacy supports batching at all

Usage:
    python scripts/benchmark_ner_throughput.py --num-docs 200 --backend gliner
    python scripts/benchmark_ner_throughput.py --num-docs 200 --backend scibert
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("throughput")


def verify_gpu():
    """Verify CUDA is available and log GPU details."""
    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA NOT AVAILABLE — running on CPU!")
        return False

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"CUDA available: {device_name} ({total_mem:.1f} GB)")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    return True


def check_model_on_gpu(model):
    """Verify the GLiNER model is actually on GPU."""
    import torch

    try:
        # Find any parameter and check its device
        for p in model.parameters() if hasattr(model, "parameters") else []:
            device = p.device
            logger.info(f"Model parameter device: {device}")
            if device.type != "cuda":
                logger.warning(f"MODEL IS ON {device.type.upper()}, not CUDA!")
                return False
            break

        # Check GPU memory used
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        return True
    except Exception as e:
        logger.warning(f"Could not verify GPU usage: {e}")
        return False


def build_nlp(backend: str):
    from ctra.config.settings import RAGConfig
    from ctra.rag.ner_config import build_ner_pipeline

    if backend == "gliner":
        config = RAGConfig()
        config.ner_model = "gliner_bio"
        return build_ner_pipeline(config)
    elif backend == "gliner_raw":
        # Direct GLiNER model without spaCy wrapper
        from gliner import GLiNER

        model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0", map_location="cuda")
        return model
    elif backend == "gliner_fp16":
        from gliner import GLiNER

        model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0", map_location="cuda")
        # gliner 0.2.26 doesn't have quantize() — use PyTorch's native .half()
        if hasattr(model, "quantize"):
            model.quantize("fp16")
        else:
            model.model.half()  # Manual fp16 conversion
            logger.info("Applied manual .half() fp16 conversion")
        return model
    elif backend == "gliner_compile":
        import torch
        from gliner import GLiNER

        model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0", map_location="cuda")
        if hasattr(model, "quantize"):
            model.quantize("fp16")
        else:
            model.model.half()
        # torch.compile on the underlying model
        if hasattr(model, "compile"):
            model.compile()
        else:
            model.model = torch.compile(model.model, dynamic=True)
            logger.info("Applied torch.compile(dynamic=True) to model.model")
        return model
    elif backend == "scibert":
        import spacy

        return spacy.load("en_core_sci_scibert")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def measure_no_batching(nlp, texts: list[str]) -> float:
    """Measure throughput with no batching (one-at-a-time)."""
    start = time.time()
    for text in texts:
        _ = nlp(text)
    return time.time() - start


def measure_pipe(nlp, texts: list[str], batch_size: int) -> float:
    """Measure throughput with nlp.pipe(batch_size=N)."""
    start = time.time()
    for _ in nlp.pipe(texts, batch_size=batch_size):
        pass
    return time.time() - start


def measure_gliner_raw(model, texts: list[str], labels: list, batch_size: int) -> float:
    """Measure throughput with raw GLiNER.inference() — no spaCy overhead."""
    start = time.time()
    _ = model.inference(texts, labels, threshold=0.4, batch_size=batch_size)
    return time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NER throughput vs batch size")
    parser.add_argument("--num-docs", type=int, default=200, help="Sample size")
    parser.add_argument(
        "--backend",
        type=str,
        default="gliner",
        choices=["gliner", "gliner_raw", "gliner_fp16", "gliner_compile", "scibert"],
    )
    parser.add_argument("--output-dir", type=str, default="output/ner_throughput")
    args = parser.parse_args()

    from ctra.rag.eval.data_models import Benchmark
    from ctra.rag.eval.loaders import get_document_texts

    # Verify GPU is available FIRST
    logger.info("=" * 60)
    logger.info("GPU VERIFICATION")
    logger.info("=" * 60)
    verify_gpu()
    logger.info("=" * 60)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load sample documents
    logger.info(f"Loading {args.num_docs} CHIA documents for throughput test...")
    doc_texts = get_document_texts(Benchmark.CHIA)
    doc_ids = list(doc_texts.keys())[: args.num_docs]
    texts = [doc_texts[d] for d in doc_ids]

    total_chars = sum(len(t) for t in texts)
    logger.info(
        f"Loaded {len(texts)} docs, total {total_chars} chars, avg {total_chars / len(texts):.0f} chars/doc"
    )

    # Build NER pipeline
    logger.info(f"Building {args.backend} pipeline...")
    nlp = build_nlp(args.backend)

    # Verify the model is actually on GPU
    logger.info("Checking model device...")
    is_raw = args.backend.startswith("gliner_") and args.backend != "gliner_spacy"
    if is_raw:
        # nlp IS the GLiNER model
        check_model_on_gpu(nlp.model if hasattr(nlp, "model") else nlp)
    else:
        # nlp is a spaCy pipeline — find the GLiNER component inside
        try:
            component = nlp.get_pipe("gliner_batched")
            check_model_on_gpu(component.model)
        except Exception as e:
            logger.warning(f"Could not find gliner_batched component: {e}")

    results = {}
    is_raw_gliner = args.backend.startswith("gliner_") and args.backend != "gliner_spacy"

    if is_raw_gliner:
        # Raw GLiNER model — use inference() directly
        from ctra.config.settings import RAGConfig

        labels = RAGConfig().ner_labels

        # Warm up
        logger.info("Warming up...")
        _ = nlp.inference([texts[0]], labels, threshold=0.4, batch_size=1)

        # Test safe batch sizes for GLiNER-large on T4 (16GB).
        # batch_size=32 OOMs due to span enumeration (B * L * max_width * D).
        for batch_size in [1, 2, 4, 8]:
            logger.info(f"\n[gliner_raw] inference(batch_size={batch_size})...")
            try:
                elapsed = measure_gliner_raw(nlp, texts, labels, batch_size)
                throughput = len(texts) / elapsed
                results[f"inference_batch_{batch_size}"] = {
                    "batch_size": batch_size,
                    "elapsed_sec": round(elapsed, 2),
                    "throughput_docs_per_sec": round(throughput, 2),
                }
                logger.info(f"  → {throughput:.1f} docs/sec ({elapsed:.1f}s)")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"  → OOM at batch_size={batch_size}, skipping")
                    import torch

                    torch.cuda.empty_cache()
                    results[f"inference_batch_{batch_size}"] = {"error": "OOM"}
                else:
                    raise
    else:
        # spaCy pipeline
        # Warm up
        logger.info("Warming up...")
        _ = nlp(texts[0])

        # Test 1: No batching
        logger.info("\n[1/6] No batching (one-at-a-time)...")
        elapsed = measure_no_batching(nlp, texts)
        throughput = len(texts) / elapsed
        results["no_batching"] = {
            "elapsed_sec": round(elapsed, 2),
            "throughput_docs_per_sec": round(throughput, 2),
        }
        logger.info(f"  → {throughput:.1f} docs/sec ({elapsed:.1f}s)")

        # Test 2-6: nlp.pipe with different batch sizes
        for batch_size in [1, 16, 32, 64, 128]:
            logger.info(f"\n[{batch_size}] nlp.pipe(batch_size={batch_size})...")
            elapsed = measure_pipe(nlp, texts, batch_size)
            throughput = len(texts) / elapsed
            results[f"pipe_batch_{batch_size}"] = {
                "batch_size": batch_size,
                "elapsed_sec": round(elapsed, 2),
                "throughput_docs_per_sec": round(throughput, 2),
            }
            logger.info(f"  → {throughput:.1f} docs/sec ({elapsed:.1f}s)")

    # Print comparison
    print("\n" + "=" * 80)
    print(f"NER THROUGHPUT BENCHMARK — {args.backend}")
    print(f"Sample: {len(texts)} CHIA documents, avg {total_chars / len(texts):.0f} chars/doc")
    print("=" * 80)

    print(f"\n{'Strategy':<35} {'Elapsed (s)':>12} {'Docs/sec':>12}")
    print("-" * 60)

    # Baseline: first valid result (smallest batch size)
    valid_for_print = {k: v for k, v in results.items() if "throughput_docs_per_sec" in v}
    if not valid_for_print:
        logger.error("No successful runs to print.")
        return 1
    baseline = next(iter(valid_for_print.values()))["throughput_docs_per_sec"]

    for key, data in results.items():
        if "error" in data:
            print(f"{key:<35} {'ERROR':>12} {data['error']:>12}")
            continue
        thr = data["throughput_docs_per_sec"]
        speedup = thr / max(baseline, 0.001)
        print(f"{key:<35} {data['elapsed_sec']:>12.1f} {thr:>12.1f}  ({speedup:.1f}x)")

    # Extrapolation — filter out errored runs
    valid_results = {k: v for k, v in results.items() if "throughput_docs_per_sec" in v}
    if not valid_results:
        logger.error("No valid results — all runs failed.")
        return 1
    best_throughput = max(r["throughput_docs_per_sec"] for r in valid_results.values())
    best_key = next(
        k for k, v in valid_results.items() if v["throughput_docs_per_sec"] == best_throughput
    )

    print(f"\nBest: {best_key} at {best_throughput:.1f} docs/sec")
    print("\nExtrapolation to full CTRA dataset (3.9M passages):")
    print(
        f"  At {best_throughput:.1f} docs/sec: {3_900_000 / best_throughput / 3600:.1f} hours = {3_900_000 / best_throughput / 86400:.1f} days"
    )

    # Save
    with open(output_path / f"throughput_{args.backend}.json", "w") as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
