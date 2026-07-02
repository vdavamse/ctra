"""Benchmark data loaders using bigbio HuggingFace collection.

All three benchmarks are loaded via the `datasets` library using bigbio
dataset identifiers. N2C2 requires a Harvard DBMI DUA and is gracefully
skipped if data is unavailable.

bigbio dataset identifiers:
- CHIA: "chia_bigbio_kb"
- N2C2 2018: "n2c2_2018_track2_bigbio_kb"
- TAC 2017 (SPL ADR 200db): "spl_adr_200db_bigbio_kb"
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from ctra.rag.eval.data_models import (
    BENCHMARK_MAPPINGS,
    Benchmark,
    GoldAnnotation,
)

logger = logging.getLogger(__name__)

# bigbio dataset configuration: (repo_id, config_name)
_BIGBIO_CONFIGS: dict[Benchmark, tuple[str, str]] = {
    Benchmark.CHIA: ("bigbio/chia", "chia_bigbio_kb"),
    Benchmark.N2C2: ("bigbio/n2c2_2018", "n2c2_2018_track2_bigbio_kb"),
    Benchmark.TAC: ("bigbio/spl_adr_200db", "spl_adr_200db_train_bigbio_kb"),
}


class DatasetNotAvailableError(Exception):
    """Raised when a benchmark dataset cannot be loaded."""

    pass


@functools.lru_cache(maxsize=3)
def _load_raw_dataset(benchmark: Benchmark) -> Any:
    """Load and cache the raw bigbio dataset for a benchmark.

    Cached so that ``load_benchmark()`` and ``get_document_texts()``
    share a single ``load_dataset()`` call per benchmark.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        msg = (
            "datasets library required for benchmark loading. Install with: pip install ctra[eval]"
        )
        raise DatasetNotAvailableError(msg) from e

    repo, config = _BIGBIO_CONFIGS[benchmark]
    logger.info(f"Loading {benchmark.value} from bigbio ({repo})...")

    try:
        dataset = load_dataset(repo, name=config, trust_remote_code=True)
        if isinstance(dataset, dict):
            dataset = dataset.get("train", next(iter(dataset.values())))
    except Exception as e:
        if benchmark == Benchmark.N2C2:
            msg = (
                "Failed to load N2C2 from bigbio. This benchmark requires a "
                "Harvard DBMI Data Use Agreement (DUA). If you have accepted the DUA, "
                "ensure the datasets library can access it via HuggingFace authentication. "
                f"Error: {e}"
            )
        else:
            msg = f"Failed to load {benchmark.value} from bigbio: {e}"
        raise DatasetNotAvailableError(msg) from e

    return dataset


def load_benchmark(benchmark: Benchmark) -> list[GoldAnnotation]:
    """Load gold annotations for a benchmark.

    Args:
        benchmark: Which benchmark to load.

    Returns:
        List of gold annotations normalized to the common schema.

    Raises:
        DatasetNotAvailableError: If the benchmark data cannot be accessed
            (e.g., N2C2 without DUA).
    """
    dataset = _load_raw_dataset(benchmark)
    mapping = BENCHMARK_MAPPINGS[benchmark]
    annotations = _bigbio_kb_to_gold_annotations(dataset, benchmark, mapping)
    logger.info(f"Loaded {len(annotations)} entities from {benchmark.value}")
    return annotations


def _bigbio_kb_to_gold_annotations(
    dataset: Any,
    benchmark: Benchmark,
    mapping: dict[str, str | None],
) -> list[GoldAnnotation]:
    """Convert bigbio KB-format dataset to GoldAnnotation list.

    The bigbio KB schema normalizes all NER datasets to a common format:
    - Each example has 'id', 'document_id', 'passages', 'entities'
    - entities[i] has 'id', 'type', 'text' (list[str]), 'offsets' (list[[start, end]])
    - entities[i]['normalized'] has optional KB links

    This function iterates over entities, extracts spans, and applies
    the benchmark-to-CTRA mapping for mapped_ctra_label.
    """
    annotations: list[GoldAnnotation] = []

    for example in dataset:
        doc_id = example.get("document_id", example.get("id", ""))

        # Collect entities
        for entity in example.get("entities", []):
            entity_type = entity.get("type", "")
            # Handle text which can be list or string
            text_list = entity.get("text", [])
            if isinstance(text_list, str):
                text_list = [text_list]
            offsets = entity.get("offsets", [])

            # Each entity can have multiple spans; match text to spans
            for idx, offset in enumerate(offsets):
                if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                    start_char = offset[0]
                    end_char = offset[1]
                else:
                    continue

                # Get surface text; use from list if available
                surface_text = text_list[idx] if idx < len(text_list) else ""

                # Map entity type
                mapped_label = mapping.get(entity_type)

                annotation = GoldAnnotation(
                    doc_id=doc_id,
                    start_char=start_char,
                    end_char=end_char,
                    text=surface_text,
                    gold_label=entity_type,
                    mapped_ctra_label=mapped_label,
                )
                annotations.append(annotation)

    return annotations


def get_document_texts(benchmark: Benchmark) -> dict[str, str]:
    """Return {doc_id: full_text} for a loaded benchmark.

    Needed for running GLiNER inference over benchmark documents.
    Extracted from bigbio passages[].text, concatenated per document.

    Uses the same cached ``_load_raw_dataset()`` as ``load_benchmark()``
    so the dataset is only downloaded/decoded once per benchmark.
    """
    dataset = _load_raw_dataset(benchmark)

    doc_texts: dict[str, str] = {}
    for example in dataset:
        doc_id = example.get("document_id", example.get("id", ""))
        passages = example.get("passages", [])
        # Concatenate all passages — handle text as str or list[str]
        parts: list[str] = []
        for p in passages:
            text_val = p.get("text", "")
            if isinstance(text_val, list):
                parts.extend(text_val)
            elif text_val:
                parts.append(text_val)
        doc_texts[doc_id] = "\n".join(parts)

    return doc_texts
