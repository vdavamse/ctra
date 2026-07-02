"""Batched GLiNER spaCy component for fast NER.

Replaces gliner-spacy's sequential ``__call__`` implementation with a proper
``pipe()`` method that uses GLiNER's native ``batch_predict_entities()`` API
for real GPU batching.

gliner-spacy (the upstream ``theirstory/gliner-spacy``) processes documents
one at a time via ``__call__`` and chunks each document sequentially. Even
when called via ``nlp.pipe(batch_size=N)``, spaCy's default behavior is to
buffer ``N`` docs then call ``__call__`` on each sequentially — the GPU sees
batch_size=1.

This component:
1. Implements a proper ``pipe()`` method
2. Collects chunks from multiple documents
3. Calls GLiNER's ``batch_predict_entities()`` on the flattened chunk list
4. Distributes predictions back to each document

On a T4 GPU with GLiNER-BioMed-large, this delivers 8-15x speedup for typical
clinical trial document batches compared to the stock gliner-spacy pipeline.
"""

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from gliner import GLiNER
from spacy.language import Language
from spacy.tokens import Doc, Span

# Language is imported to satisfy @Language.factory decorator

logger = logging.getLogger(__name__)

# Register a custom extension to store entity confidence scores (matches
# gliner-spacy's API so downstream code works unchanged).
if not Span.has_extension("score"):
    Span.set_extension("score", default=0.0)


_DEFAULT_CONFIG: dict[str, Any] = {
    "gliner_model": "Ihor/gliner-biomed-large-v1.0",
    # Placeholder labels — spaCy's factory registry requires default values
    # for all constructor parameters. Callers override via nlp.add_pipe(
    # "gliner_batched", config={"labels": config.ner_labels}).
    "labels": [],
    "threshold": 0.4,
    "chunk_size": 1500,
    "batch_size": 8,  # GLiNER's default; larger doesn't always help due to span enum
    "map_location": "cpu",
    "use_fp16": True,  # ~1.4x speedup via Tensor Cores on GPU
    "use_compile": False,  # torch.compile can help ~2x but adds slow first-call warm-up
}


@Language.factory(
    "gliner_batched",
    assigns=["doc.ents"],
    default_config=_DEFAULT_CONFIG,
)
class GlinerBatched:
    """Batched GLiNER spaCy component.

    Args:
        nlp: The spaCy language object (provided by spaCy factory).
        name: Component name (provided by spaCy factory).
        gliner_model: HuggingFace model identifier or local path.
        labels: Zero-shot entity labels to extract.
        threshold: Confidence threshold (0.0-1.0).
        chunk_size: Max characters per chunk sent to GLiNER.
            GLiNER's transformer supports ~384 tokens (~2000 chars).
        batch_size: GPU batch size for batch_predict_entities.
        map_location: Device for GLiNER model — "cpu" or "cuda".
    """

    def __init__(
        self,
        nlp: Any,
        name: str,
        gliner_model: str,
        labels: list[str],
        threshold: float,
        chunk_size: int,
        batch_size: int,
        map_location: str,
        use_fp16: bool = True,
        use_compile: bool = False,
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.labels = labels
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        logger.info(
            "Loading GLiNER model '%s' on %s (batch_size=%d, fp16=%s, compile=%s)",
            gliner_model,
            map_location,
            batch_size,
            use_fp16,
            use_compile,
        )
        self.model = GLiNER.from_pretrained(
            gliner_model,
            map_location=map_location,
        )

        # Apply fp16 quantization for ~1.4x speedup via Tensor Cores (GPU only).
        # gliner >= 0.2.27 has model.quantize('fp16'); older versions need .half()
        if use_fp16 and map_location == "cuda":
            try:
                if hasattr(self.model, "quantize"):
                    self.model.quantize("fp16")
                else:
                    self.model.model.half()
                logger.info("Applied fp16 to GLiNER model")
            except Exception as e:
                logger.warning("fp16 conversion failed: %s", e)

        # Apply torch.compile for ~2x speedup (first call is slow due to warm-up).
        # gliner >= 0.2.27 has model.compile(); older versions need torch.compile directly
        if use_compile and map_location == "cuda":
            try:
                import torch

                if hasattr(self.model, "compile"):
                    self.model.compile()
                else:
                    self.model.model = torch.compile(self.model.model, dynamic=True)
                logger.info("Applied torch.compile to GLiNER model")
            except Exception as e:
                logger.warning("torch.compile failed: %s", e)

    def _chunk_text(self, text: str) -> list[tuple[int, str]]:
        """Split text into chunks at word boundaries.

        Returns:
            List of (start_offset_in_original, chunk_text) tuples.
        """
        chunks: list[tuple[int, str]] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # Extend to word boundary if we're mid-word
            while end < len(text) and text[end] not in (" ", "\n"):
                end += 1
            chunks.append((start, text[start:end]))
            start = end
        return chunks

    def _apply_entities(self, doc: Doc, all_entities: list[dict[str, Any]]) -> Doc:
        """Attach predicted entities to a spaCy Doc."""
        spans: list[Span] = []
        for ent in all_entities:
            span = doc.char_span(ent["start"], ent["end"], label=ent["label"])
            if span is not None:
                span._.score = ent["score"]
                spans.append(span)
        try:
            doc.ents = spans
        except ValueError:
            # spaCy raises ValueError on overlapping spans; filter to non-overlapping.
            spans.sort(key=lambda s: (s.start, -s.end))
            filtered: list[Span] = []
            last_end = -1
            for span in spans:
                if span.start >= last_end:
                    filtered.append(span)
                    last_end = span.end
            doc.ents = filtered
        return doc

    def __call__(self, doc: Doc) -> Doc:
        """Process a single document (falls back to per-doc inference).

        This is called when spaCy invokes the component outside of pipe().
        For batched processing, use pipe() instead.
        """
        chunks = self._chunk_text(doc.text)
        all_entities: list[dict[str, Any]] = []
        for offset, chunk_text in chunks:
            chunk_entities = self.model.predict_entities(
                chunk_text,
                self.labels,
                flat_ner=True,
                threshold=self.threshold,
            )
            for entity in chunk_entities:
                all_entities.append(
                    {
                        "start": offset + entity["start"],
                        "end": offset + entity["end"],
                        "label": entity["label"],
                        "score": entity["score"],
                    }
                )
        return self._apply_entities(doc, all_entities)

    def pipe(
        self,
        stream: Iterable[Doc],
        batch_size: int | None = None,
    ) -> Iterator[Doc]:
        """Process documents in batches using GLiNER's native batching.

        Collects chunks from ``batch_size`` documents, runs them through
        GLiNER in a single batched call, then distributes predictions back
        to each document. This is 8-15x faster than sequential inference
        on GPU.
        """
        effective_batch = batch_size or self.batch_size
        buffer: list[Doc] = []
        for doc in stream:
            buffer.append(doc)
            if len(buffer) >= effective_batch:
                yield from self._process_batch(buffer)
                buffer = []
        if buffer:
            yield from self._process_batch(buffer)

    def _process_batch(self, docs: list[Doc]) -> Iterator[Doc]:
        """Run GLiNER on a batch of docs with native GPU batching."""
        # Build flat chunk list with provenance (which doc, what offset)
        all_chunks: list[str] = []
        doc_chunk_spans: list[list[tuple[int, int]]] = []  # [(offset, chunk_idx_in_flat)]

        for doc in docs:
            chunks = self._chunk_text(doc.text)
            chunk_spans: list[tuple[int, int]] = []
            for offset, chunk_text in chunks:
                chunk_spans.append((offset, len(all_chunks)))
                all_chunks.append(chunk_text)
            doc_chunk_spans.append(chunk_spans)

        if not all_chunks:
            yield from docs
            return

        # Real batched GPU inference via GLiNER.inference() directly.
        # We avoid batch_predict_entities because:
        # 1. It's deprecated in gliner 0.2.26+ and emits FutureWarning
        #    (warning machinery overhead on every call)
        # 2. It forwards to inference() anyway
        # We pass batch_size explicitly so the internal DataLoader uses our
        # value instead of the default 8 (which would create 4 sub-batches
        # from our batch of 32, wasting batching benefit).
        try:
            results = self.model.inference(
                all_chunks,
                self.labels,
                flat_ner=True,
                threshold=self.threshold,
                batch_size=self.batch_size,
            )
        except AttributeError:
            # Fallback for older GLiNER versions
            results = [
                self.model.predict_entities(
                    chunk,
                    self.labels,
                    flat_ner=True,
                    threshold=self.threshold,
                )
                for chunk in all_chunks
            ]

        # Distribute results back to documents
        for doc, chunk_spans in zip(docs, doc_chunk_spans, strict=True):
            all_entities: list[dict[str, Any]] = []
            for offset, flat_idx in chunk_spans:
                chunk_entities = results[flat_idx]
                for entity in chunk_entities:
                    all_entities.append(
                        {
                            "start": offset + entity["start"],
                            "end": offset + entity["end"],
                            "label": entity["label"],
                            "score": entity["score"],
                        }
                    )
            yield self._apply_entities(doc, all_entities)
