"""LinearRAG wrapper with date filtering for label-leakage prevention.

This module wraps the upstream LinearRAG library (DEEP-PolyU/LinearRAG) to add:
    - Temporal filtering: only return passages published before a cutoff date
    - Source filtering: restrict retrieval to specific data sources
    - Structured output: return ``RetrievedChunk`` dataclass instances

The wrapper delegates scoring to LinearRAG's Personalized PageRank over an
entity co-occurrence graph built by ``ctra.rag.indexer``.  When the graph is
unavailable (e.g., during early development), it falls back to dense
passage retrieval via sentence-transformers cosine similarity.

The date-filtering strategy is **Layer 4** (post-retrieval filtering):
    1. Retrieve ``top_k * oversample_factor`` passages from LinearRAG
    2. Post-filter by ``before_date`` to remove temporally leaked passages
    3. Return the top ``top_k`` remaining passages

This is the simplest and safest approach: it guarantees no future information
leaks into the feature builder regardless of the retrieval model's internal
behavior.
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ctra.config.settings import DataSource, RAGConfig, get_settings

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Oversample factor for post-retrieval date filtering.  We retrieve this many
# multiples of ``top_k`` from LinearRAG before applying the date filter.
_OVERSAMPLE_FACTOR = 3


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single passage returned by the RAG layer.

    Attributes:
        text: The passage text (sentence or short paragraph).
        source: Origin data source identifier.
        doc_id: Unique document identifier (NCT ID, PMID, ChEMBL ID, ...).
        date: Publication or posting date for temporal filtering.
        score: Relevance score from Personalized PageRank (higher is better).
        metadata: Arbitrary extra metadata from the index.
    """

    text: str
    source: str
    doc_id: str
    date: datetime.date | None = None
    score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


class LinearRAGWrapper:
    """Date-filtered retrieval over a LinearRAG index.

    Wraps the upstream ``LinearRAG`` class (from ``DEEP-PolyU/LinearRAG``) and
    adds temporal and source filtering.  The upstream library uses spaCy NER to
    extract entities, builds a co-occurrence graph, and scores passages via
    Personalized PageRank seeded from query entities.

    Args:
        config: RAG configuration.  If ``None``, uses the global settings.
        index_dir: Override path to the LinearRAG index directory.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        index_dir: Path | None = None,
        entity_resolver: Any | None = None,
    ) -> None:
        self._config = config or get_settings().rag
        self._index_dir = Path(index_dir or self._config.index_dir)
        self._linearrag: Any | None = None  # lazy-loaded LinearRAG instance
        self._passages_df: pl.DataFrame | None = None
        self._resolver = entity_resolver

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazily initialize the LinearRAG index from disk artifacts.

        On first call, loads the passages.parquet metadata file and the
        LinearRAG graph artifacts (embeddings, graph, NER results), then
        initializes the ``LinearRAG`` instance. Subsequent calls are no-ops
        since ``_linearrag`` is cached.

        If GLiNER-BioMed NER is configured, it is initialized here with
        the entity type labels and threshold settings. Drug synonym expansion
        is also patched at this point if an entity resolver is available.

        Raises:
            FileNotFoundError: If index artifacts are missing.
            RuntimeError: If CUDA is not available (required for embeddings).
        """
        if self._linearrag is not None:
            return

        index_path = self._index_dir
        if not index_path.exists():
            raise FileNotFoundError(
                f"LinearRAG index not found at {index_path}.  "
                "Run `python -m ctra.rag.indexer build` first."
            )

        # -- Load passage metadata (text, source, doc_id, date) --
        passages_path = index_path / "passages.parquet"
        if not passages_path.exists():
            raise FileNotFoundError(f"Passage store not found: {passages_path}")

        self._passages_df = pl.read_parquet(passages_path)
        logger.info("Loaded %d passages from %s", len(self._passages_df), passages_path)

        # -- Initialize LinearRAG --
        # LinearRAG expects embedding_model to be a SentenceTransformer object,
        # not a string path. The config dataclass declares it as str but run.py
        # always passes a pre-loaded SentenceTransformer instance.
        # CUDA is required — LinearRAG's NER + embedding pipeline needs GPU.
        import torch

        from ctra.rag.linearrag import LinearRAG, LinearRAGConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device required for LinearRAG. No CUDA-capable GPU detected. "
                "Install CUDA toolkit and a CUDA-enabled PyTorch build."
            )

        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(self._config.embedding_model, device="cuda")

        # For GLiNER-BioMed: the vendored SpacyNER.__init__ accepts either
        # a model name (str) or a pre-built spaCy Language pipeline, so we
        # pass the GLiNER pipeline directly — no monkey-patching needed.
        use_gliner = self._config.ner_model == "gliner_bio"
        ner_model_for_config: Any = self._config.ner_model

        if use_gliner:
            from ctra.rag.ner_config import build_ner_pipeline

            ner_model_for_config = build_ner_pipeline(self._config)

        config = LinearRAGConfig(
            dataset_name=index_path.name,
            embedding_model=st_model,
            spacy_model=ner_model_for_config,
            working_dir=str(index_path.parent),
            damping=0.5,  # LinearRAG default; in igraph PPR this is link-follow probability
            max_iterations=3,  # BFS entity propagation depth, NOT PPR iterations
            top_k_sentence=3,  # sentences per entity per BFS iteration (run.py default)
            passage_ratio=2.0,  # DPR score weight in passage scoring (run.py default)
            iteration_threshold=0.4,  # BFS pruning threshold (run.py default)
            retrieval_top_k=self._config.top_k * _OVERSAMPLE_FACTOR,
        )

        rag = LinearRAG(global_config=config)
        if use_gliner:
            logger.info("LinearRAG initialized with GLiNER-BioMed NER pipeline")

        # Build the passage list in "idx:text" format that LinearRAG expects.
        passages_for_index = [
            f"{i}:{row['text']}" for i, row in enumerate(self._passages_df.iter_rows(named=True))
        ]

        # LinearRAG writes artifacts to {working_dir}/{dataset_name}/
        # which equals index_path since working_dir=parent, dataset_name=name.
        graph_path = index_path / "LinearRAG.graphml"
        ner_path = index_path / "ner_results.json"

        if graph_path.exists() and ner_path.exists():
            logger.info("Loading pre-built LinearRAG graph from %s", graph_path)
            # Insert passages into the embedding store without re-computing NER/graph
            rag.passage_embedding_store.insert_text(passages_for_index)

            import json

            import igraph as ig

            ner_data = json.loads(ner_path.read_text())
            existing_phe = ner_data["passage_hash_id_to_entities"]
            existing_ste = ner_data["sentence_to_entities"]

            entity_nodes, sentence_nodes, _phte, rag.entity_to_sentence, rag.sentence_to_entity = (
                rag.extract_nodes_and_edges(existing_phe, existing_ste)
            )
            rag.sentence_embedding_store.insert_text(list(sentence_nodes))
            rag.entity_embedding_store.insert_text(list(entity_nodes))

            # Rebuild hash-id mappings
            rag.entity_hash_id_to_sentence_hash_ids = {}
            for entity, sentences in rag.entity_to_sentence.items():
                ehid = rag.entity_embedding_store.text_to_hash_id[entity]
                rag.entity_hash_id_to_sentence_hash_ids[ehid] = [
                    rag.sentence_embedding_store.text_to_hash_id[s] for s in sentences
                ]
            rag.sentence_hash_id_to_entity_hash_ids = {}
            for sentence, entities in rag.sentence_to_entity.items():
                shid = rag.sentence_embedding_store.text_to_hash_id[sentence]
                rag.sentence_hash_id_to_entity_hash_ids[shid] = [
                    rag.entity_embedding_store.text_to_hash_id[e] for e in entities
                ]

            rag.graph = ig.Graph.Read_GraphML(str(graph_path))
        else:
            # Build the index from scratch (NER, embeddings, graph).
            logger.info("Building LinearRAG index (NER + graph) -- this may take a while")
            rag.index(passages_for_index)

        # Patch query NER with drug synonym expansion if resolver is available.
        # This ensures that when an agent queries "mechanism of pembrolizumab",
        # LinearRAG seeds PPR from BOTH "Pembrolizumab" AND "Keytruda" graph
        # nodes, bridging brand/generic/research-code name fragmentation.
        if self._resolver is not None:
            self._patch_query_ner_with_synonyms(rag)

        self._linearrag = rag
        logger.info("LinearRAG wrapper initialized (index_dir=%s)", index_path)

    # ------------------------------------------------------------------
    # Drug synonym expansion
    # ------------------------------------------------------------------

    def _patch_query_ner_with_synonyms(self, rag: Any) -> None:
        """Wrap LinearRAG's query NER to expand drug entities with synonyms.

        LinearRAG's ``get_seed_entities()`` calls::

            question_entities = list(self.spacy_ner.question_ner(question))

        which returns a ``set[str]`` of lowercased entity texts.  We wrap
        ``question_ner()`` to append drug synonym variants after NER
        extraction.  This gives ``get_seed_entities()`` more candidate
        strings to match against graph entity nodes via embedding
        similarity, solving the brand-vs-generic name disconnect.

        Only drug entities are expanded (entities found in the ChEMBL
        synonym table).  Non-drug entities (diseases, genes, etc.) pass
        through unchanged.

        Args:
            rag: The ``LinearRAG`` instance whose ``spacy_ner`` to patch.
        """
        original_question_ner = rag.spacy_ner.question_ner
        resolver = self._resolver
        assert resolver is not None

        def _expanded_question_ner(question: str) -> set[str]:
            # Step 1: Run original NER (returns lowercased entity set)
            entities = original_question_ner(question)

            # Step 2: Expand drug entities with synonyms from ChEMBL.
            # get_synonyms() returns [entity] for non-drugs (no expansion).
            # For known drugs, it returns [canonical, syn1, syn2, ...].
            expanded: set[str] = set(entities)
            for entity in entities:
                synonyms = resolver.get_synonyms(entity)
                if len(synonyms) > 1:  # has synonyms → known drug
                    for syn in synonyms[:5]:  # cap to avoid query explosion
                        expanded.add(syn.lower())

            if len(expanded) > len(entities):
                logger.debug(
                    "Query NER expanded %d → %d entities (drug synonyms)",
                    len(entities),
                    len(expanded),
                )

            return expanded

        rag.spacy_ner.question_ner = _expanded_question_ner
        logger.info("Patched LinearRAG query NER with drug synonym expansion")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        before_date: datetime.date | None = None,
        sources: Sequence[DataSource | str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve passages relevant to *query* with optional filters.

        Implements Layer 4 (post-retrieval filtering):
            1. Retrieve ``top_k * oversample_factor`` from LinearRAG
            2. Post-filter by source and date
            3. Return top ``top_k``

        Args:
            query: Natural-language search query.
            before_date: If set, exclude passages with a date on or after this
                value.  Primary label-leakage prevention mechanism.
            sources: Restrict to these data sources (e.g., ``["ctg", "pubmed"]``).
                Defaults to all configured sources.
            top_k: Number of passages to return.  Defaults to ``config.top_k``.

        Returns:
            List of :class:`RetrievedChunk` sorted by descending score.
        """
        self._ensure_loaded()
        assert self._passages_df is not None

        top_k = top_k or self._config.top_k
        source_filter = {
            s.value if isinstance(s, DataSource) else str(s)
            for s in (sources or self._config.sources)
        }

        # -- Step 1: Retrieve from LinearRAG (oversampled) --
        retrieve_k = top_k * _OVERSAMPLE_FACTOR
        scored_passages = self._retrieve_raw(query, retrieve_k)

        # -- Step 2: Post-filter by source and date --
        results: list[RetrievedChunk] = []
        for passage_text, score in scored_passages:
            row_idx = self._resolve_passage_row(passage_text)
            if row_idx is None:
                continue

            row = self._passages_df.row(row_idx, named=True)
            row_source = str(row.get("source", ""))

            # Source filter
            if row_source not in source_filter:
                continue

            # Date filter (Layer 4: post-retrieval)
            if before_date is not None and self._config.date_filter_enabled:
                row_date = self._parse_row_date(row.get("date"))
                if row_date is None:
                    # Conservatively exclude undated documents to prevent leakage
                    continue
                if row_date >= before_date:
                    continue

            date_val = self._parse_row_date(row.get("date"))
            results.append(
                RetrievedChunk(
                    text=str(row["text"]),
                    source=row_source,
                    doc_id=str(row.get("doc_id", "")),
                    date=date_val,
                    score=float(score),
                    metadata={
                        k: str(v)
                        for k, v in row.items()
                        if k not in ("text", "source", "doc_id", "date") and v is not None
                    },
                )
            )

            if len(results) >= top_k:
                break

        if not results:
            logger.warning(
                "No passages match filters (sources=%s, before=%s, query=%.60s...)",
                source_filter,
                before_date,
                query,
            )

        return results

    # ------------------------------------------------------------------
    # Raw retrieval — uses LinearRAG's public retrieve() API
    # ------------------------------------------------------------------

    def _retrieve_raw(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Retrieve scored passages from LinearRAG using the public API.

        Calls ``LinearRAG.retrieve()`` with the query and returns the top-k
        passages along with their Personalized PageRank scores. This is the
        raw retrieval stage before source/date filtering is applied in
        ``search()``.

        Args:
            query: Natural-language search query.
            top_k: Number of passages to retrieve (oversampled for later filtering).

        Returns:
            List of ``(passage_text, score)`` tuples sorted by descending score.
        """
        rag = self._linearrag
        assert rag is not None
        old_top_k = rag.config.retrieval_top_k
        rag.config.retrieval_top_k = top_k

        try:
            # Use the public retrieve() API. It expects a list of question dicts.
            # The "answer" field is only used for passthrough — empty string is fine.
            questions = [{"question": query, "answer": ""}]
            results = rag.retrieve(questions)

            if not results:
                return []

            result = results[0]
            passages = result.get("sorted_passage", [])
            scores = result.get("sorted_passage_scores", [])

            return list(zip(passages, scores, strict=True))
        finally:
            rag.config.retrieval_top_k = old_top_k

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_passage_row(self, passage_text: str) -> int | None:
        """Map a retrieved passage text back to its DataFrame row index.

        LinearRAG formats passages as ``"idx:text"`` where idx is the
        sequential row index from the passages DataFrame. This method:
        1. Extracts the row index from the prefix if present
        2. Falls back to text-based matching if the prefix is invalid
        3. Returns ``None`` if no match is found

        This is used during ``search()`` to retrieve metadata (source, date, doc_id).

        Args:
            passage_text: The passage text returned by LinearRAG.

        Returns:
            The row index in ``_passages_df``, or ``None`` if not found.
        """
        assert self._passages_df is not None

        # Strip the "idx:" prefix that LinearRAG prepends.
        match = re.match(r"^\d+:", passage_text)
        if match:
            # The prefix IS the DataFrame row index (sequential from 0).
            try:
                idx = int(passage_text[: match.end() - 1])
                if 0 <= idx < len(self._passages_df):
                    return idx
            except ValueError:
                pass

        # Fallback: search by text content (slower but robust).
        clean_text = re.sub(r"^\d+:", "", passage_text).strip()
        mask = self._passages_df.get_column("text").cast(pl.Utf8) == clean_text
        indices = [i for i, m in enumerate(mask.to_list()) if m]
        if indices:
            return indices[0]

        return None

    @staticmethod
    def _parse_row_date(value: Any) -> datetime.date | None:
        """Parse a date value from a passage metadata row.

        Handles multiple input types (datetime.datetime, datetime.date,
        ISO string, float NaN) and returns a normalized ``datetime.date``
        or ``None`` if parsing fails or the value is null.

        Used during ``search()`` to enforce temporal filtering (layer 4).

        Args:
            value: A potential date value from the passages DataFrame.

        Returns:
            A ``datetime.date`` object, or ``None`` if null/invalid.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            if isinstance(value, datetime.datetime):
                return value.date()
            if isinstance(value, datetime.date):
                return value
            return datetime.date.fromisoformat(str(value)[:10])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Index building (static)
    # ------------------------------------------------------------------

    @staticmethod
    def build_index(
        passages: list[dict[str, Any]],
        output_dir: str | Path,
        rag_config: RAGConfig | None = None,
    ) -> Path:
        """Build a LinearRAG index from passage dicts with metadata.

        Each passage dict must contain:
            - ``text`` (str): The passage text.
            - ``source`` (str): Data source identifier (ctg, pubmed, ...).
            - ``doc_id`` (str): Unique document identifier.
            - ``date`` (str | None): ISO date string (YYYY-MM-DD).

        Additional keys are stored as metadata.

        The method writes:
            1. ``passages.parquet`` -- metadata for date/source filtering
            2. LinearRAG index artifacts (embeddings, graph, NER results)

        Args:
            passages: List of passage dicts.
            output_dir: Directory to write the index.
            rag_config: Optional RAG configuration override.

        Returns:
            Path to the output directory.
        """
        config = rag_config or get_settings().rag
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # -- Write passages.parquet --
        df = pl.DataFrame(passages)
        required_cols = {"text", "source", "doc_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Passages missing required columns: {missing}")

        if "date" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("date"))

        passages_path = out / "passages.parquet"
        df.write_parquet(passages_path)
        logger.info("Wrote %d passages to %s", len(df), passages_path)

        # -- Build LinearRAG index --
        # Load SentenceTransformer object — LinearRAG requires the model instance,
        # not a string path. CUDA is required for embedding computation.
        import torch

        from ctra.rag.linearrag import LinearRAG, LinearRAGConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device required for LinearRAG indexing. No CUDA-capable GPU detected. "
                "Install CUDA toolkit and a CUDA-enabled PyTorch build."
            )

        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(config.embedding_model, device="cuda")

        # For GLiNER-BioMed: the vendored SpacyNER accepts either a model
        # name (str) or a pre-built spaCy pipeline — pass it directly.
        use_gliner = config.ner_model == "gliner_bio"
        ner_model_for_config: Any = config.ner_model

        if use_gliner:
            from ctra.rag.ner_config import build_ner_pipeline

            ner_model_for_config = build_ner_pipeline(config)

        lr_config = LinearRAGConfig(
            dataset_name=out.name,
            embedding_model=st_model,
            spacy_model=ner_model_for_config,
            working_dir=str(out.parent),
            damping=0.5,  # LinearRAG default; link-follow probability in igraph PPR
            max_iterations=3,  # BFS entity propagation depth (NOT PPR iterations)
            top_k_sentence=3,  # sentences per entity per BFS iteration
            passage_ratio=2.0,  # DPR score weight in passage scoring
            iteration_threshold=0.4,  # BFS pruning threshold
            retrieval_top_k=config.top_k,
        )

        rag = LinearRAG(global_config=lr_config)
        if use_gliner:
            logger.info("LinearRAG indexing with GLiNER-BioMed NER pipeline")

        # Format passages as "idx:text" for LinearRAG (sequential from 0)
        formatted = [f"{i}:{row['text']}" for i, row in enumerate(df.iter_rows(named=True))]
        logger.info(
            "Indexing %d passages with LinearRAG (NER + graph construction)", len(formatted)
        )
        rag.index(formatted)

        logger.info("LinearRAG index built at %s", out)
        return out
