"""NER pipeline builder for LinearRAG graph construction.

Supports two backends:

1. **GLiNER-BioMed** (default) — Zero-shot biomedical NER via
   ``urchade/gliner_large_bio-v0.1``.  Extracts typed entities (drug,
   disease, gene, protein, organization, etc.) specified at runtime.
   Native spaCy integration via ``gliner-spacy``.

2. **scispaCy** (fallback) — ``en_core_sci_scibert`` or other scispaCy
   models.  Uses a single generic ``ENTITY`` label for all biomedical
   spans.

LinearRAG only uses ``ent.text`` (entity surface form) for graph
construction — it never inspects ``ent.label_``.  Both backends produce
``doc.ents`` which is all LinearRAG needs.  The advantage of GLiNER is
higher recall on brand drug names (Keytruda), short gene symbols (F2),
disease abbreviations (NSCLC), and organization names (Pfizer).

Note: LinearRAG's entity filter (``ent.label_ in ("ORDINAL", "CARDINAL")``)
will never trigger for GLiNER entities since GLiNER uses labels like
``"drug"``, ``"disease"``, etc.  This is correct — all GLiNER entities
should pass through to the graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import spacy

if TYPE_CHECKING:
    from spacy.language import Language

    from ctra.config.settings import RAGConfig

logger = logging.getLogger(__name__)

# Common biomedical abbreviations that should NOT trigger sentence breaks.
# The sentencizer splits on periods — these exceptions prevent splits like
# "5 mg/kg i.v. on day 1" → ["5 mg/kg i.v.", "on day 1"].
_BIOMEDICAL_ABBREVIATIONS = {
    "i.v.",
    "i.m.",
    "i.p.",
    "s.c.",
    "p.o.",
    "b.i.d.",
    "t.i.d.",
    "q.d.",
    "q.i.d.",
    "p.r.n.",
    "vs.",
    "et al.",
    "e.g.",
    "i.e.",
    "approx.",
    "ca.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Fig.",
    "Eq.",
    "No.",
    "Vol.",
    "max.",
    "min.",
    "avg.",
    "inc.",
    "corp.",
    "ltd.",
}


def build_ner_pipeline(config: RAGConfig) -> Language:
    """Build a spaCy ``Language`` pipeline with NER for LinearRAG.

    Parameters:
        config: RAG configuration.  Uses ``ner_model``, ``ner_labels``,
            and ``ner_threshold`` fields.

    Returns:
        A spaCy ``Language`` instance with sentence segmentation and NER.

    Raises:
        ImportError: If neither GLiNER nor scispaCy is installed.
    """
    if config.ner_model == "gliner_bio":
        return _build_gliner_pipeline(config)

    # Fallback: load any spaCy model by name (e.g., en_core_sci_scibert)
    logger.info("Loading spaCy NER model: %s", config.ner_model)
    return spacy.load(config.ner_model)


def _build_gliner_pipeline(config: RAGConfig) -> Language:
    """Build a GLiNER-BioMed NER pipeline.

    Uses ``gliner-spacy`` to wrap the GLiNER model as a spaCy component.
    A custom sentencizer with biomedical abbreviation handling provides
    sentence segmentation.

    Falls back to scispaCy if GLiNER is not installed, with a clear
    error if neither is available.
    """
    try:
        # Import the CTRA batched component (registers gliner_batched factory)
        import ctra.rag.gliner_batched  # noqa: F401 — registers the component
    except ImportError:
        logger.warning("gliner not installed; attempting scispaCy fallback.")
        try:
            return spacy.load("en_core_sci_scibert")
        except OSError:
            raise ImportError(
                "Neither gliner nor en_core_sci_scibert is installed.  "
                "Install one of:\n"
                "  pip install gliner\n"
                "  pip install scispacy && pip install "
                "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz"
            ) from None

    logger.info(
        "Building GLiNER-BioMed NER pipeline (labels=%s, threshold=%.2f)",
        config.ner_labels,
        config.ner_threshold,
    )

    nlp = spacy.blank("en")

    # Custom sentencizer with biomedical abbreviation handling.
    # The default sentencizer splits on ALL periods, which breaks on
    # "i.v.", "vs.", "et al.", etc. in biomedical text.
    sentencizer = nlp.add_pipe("sentencizer")
    sentencizer.punct_chars = [".", "!", "?", "\n"]  # type: ignore[attr-defined,unused-ignore]

    # Determine GLiNER device — use CUDA if available for ~10x faster NER.
    _map_location = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            _map_location = "cuda"
    except ImportError:
        pass

    nlp.add_pipe(
        "gliner_batched",
        config={
            # GLiNER-BioMed v1.0 from ds4dh — trained on biomedical corpora
            # with synthetic annotations from OpenBioLLM.
            # https://github.com/ds4dh/GLiNER-biomed
            "gliner_model": "Ihor/gliner-biomed-large-v1.0",
            # Labels are zero-shot natural language prompts — the model
            # computes similarity between label text and candidate spans.
            # Title Case and descriptive phrasing work best (matching the
            # model's training convention).
            "labels": config.ner_labels,
            "threshold": config.ner_threshold,
            # chunk_size is in CHARACTERS (not tokens). GLiNER's transformer
            # supports ~384 tokens (~2000 chars). Setting to 1500 chars
            # avoids splitting entities at chunk boundaries while staying
            # within the model's context window.
            "chunk_size": 1500,
            # GPU batch size for GLiNER's internal DataLoader. GLiNER's
            # default is 8; larger values don't always help due to span
            # enumeration overhead (O(B * T * max_width)).
            "batch_size": 8,
            "map_location": _map_location,
            # fp16 quantization: ~1.4x speedup via Tensor Cores on GPU
            "use_fp16": True,
            # torch.compile: ~2x speedup but slow first-call warm-up
            # Enable for production indexing, disable for interactive use
            "use_compile": False,
        },
    )

    # Validate the pipeline produces entities on a test sentence.
    _validate_pipeline(nlp)

    return nlp


def _validate_pipeline(nlp: Language) -> None:
    """Quick sanity check that the NER pipeline extracts entities."""
    test_text = "Pembrolizumab is used to treat melanoma at Merck."
    doc = nlp(test_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        logger.info("NER pipeline validation passed: %s", entities)
    else:
        logger.warning(
            "NER pipeline validation: no entities extracted from test "
            "sentence '%s'. Check ner_labels and ner_threshold settings.",
            test_text,
        )
