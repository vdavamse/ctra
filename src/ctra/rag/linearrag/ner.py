"""Named Entity Recognition — vendored from LinearRAG (GPL-3).

Modified: ``SpacyNER.__init__`` now accepts either a model name (``str``)
or a pre-built spaCy ``Language`` pipeline object.  This eliminates the
need to monkey-patch when injecting GLiNER-BioMed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import spacy


class SpacyNER:
    """Named Entity Recognition wrapper using spaCy pipelines.

    Extracts named entities from passages and sentences for LinearRAG indexing
    and question processing. Supports both standard spaCy models and custom
    pipelines (e.g., GLiNER-BioMed for biomedical NER). Entities are extracted
    at both passage and sentence levels, building the entity-to-sentence and
    sentence-to-entity mappings that form the core of LinearRAG's knowledge graph.

    Attributes:
        spacy_model: The spaCy Language pipeline (loaded model or injected instance).
    """

    def __init__(self, spacy_model: str | Any) -> None:
        """Initialize SpacyNER with a spaCy model.

        Args:
            spacy_model: Either a model name string (e.g., 'en_core_web_sm')
                         or a pre-built spaCy Language pipeline object.
                         If a string, the model is loaded via spacy.load().
                         If an object, it is used as-is. This allows for
                         injection of custom pipelines (e.g., GLiNER-BioMed).
        """
        if isinstance(spacy_model, str):
            self.spacy_model = spacy.load(spacy_model)
        else:
            # Accept a pre-built spaCy Language pipeline (e.g. GLiNER-BioMed).
            self.spacy_model = spacy_model

    def batch_ner(
        self, hash_id_to_passage: dict[str, str], max_workers: int
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Extract entities from multiple passages in batch.

        Processes a collection of passages through the spaCy NER pipeline,
        extracting entities at both passage and sentence levels. Supports
        parallel batch processing via max_workers parameter. This is the
        main entry point for LinearRAG indexing; it builds the mappings
        required for entity-based graph retrieval.

        Args:
            hash_id_to_passage: Mapping from unique passage identifiers
                                (hash IDs) to passage text strings.
            max_workers: Number of workers for batch processing. Controls
                         the batch size (total passages / max_workers).

        Returns:
            A tuple of two dicts:
            - passage_hash_id_to_entities: Maps each passage hash ID to its
                                           unique entity list.
            - sentence_to_entities: Maps each sentence to the entities
                                     extracted from it (deduplicated).
        """
        passage_list = list(hash_id_to_passage.values())
        # Fixed batch size for GPU-friendly transformer inference.
        # Original computed batch_size=len(passages)//max_workers which produced
        # absurdly large batches (935+ for 15K passages) that OOM'd or silently
        # fell back to sequential processing. A fixed 32-64 works well for T4/A10.
        batch_size = 32
        docs_list = self.spacy_model.pipe(passage_list, batch_size=batch_size)
        passage_hash_id_to_entities: dict[str, list[str]] = {}
        sentence_to_entities: dict[str, list[str]] = defaultdict(list)
        for idx, doc in enumerate(docs_list):
            passage_hash_id = list(hash_id_to_passage.keys())[idx]
            single_passage_hash_id_to_entities, single_sentence_to_entities = (
                self.extract_entities_sentences(doc, passage_hash_id)
            )
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
            for sent, ents in single_sentence_to_entities.items():
                for e in ents:
                    if e not in sentence_to_entities[sent]:
                        sentence_to_entities[sent].append(e)
        return passage_hash_id_to_entities, sentence_to_entities

    def extract_entities_sentences(
        self, doc: Any, passage_hash_id: str
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Extract entities from a parsed spaCy document at passage and sentence levels.

        Processes a spaCy Doc object to extract named entities, filtering out
        ORDINAL and CARDINAL entity types (which are typically not useful for
        biomedical retrieval). Builds two mappings: passage-level unique entities
        and sentence-level entity lists. Used internally by batch_ner().

        Args:
            doc: A parsed spaCy Doc object (output of spacy_model(text)).
            passage_hash_id: The unique identifier (hash) for the passage
                             that was parsed into this doc.

        Returns:
            A tuple of two dicts:
            - passage_hash_id_to_entities: Maps this passage's hash to its
                                           unique entity list.
            - sentence_to_entities: Maps each sentence text to its extracted
                                     entities (without duplicates across sentences).
        """
        sentence_to_entities: dict[str, list[str]] = defaultdict(list)
        unique_entities: set[str] = set()
        passage_hash_id_to_entities: dict[str, list[str]] = {}
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            sent_text = ent.sent.text
            ent_text = ent.text
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            unique_entities.add(ent_text)
        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)
        return passage_hash_id_to_entities, sentence_to_entities

    def question_ner(self, question: str) -> set[str]:
        """Extract entities from a query question.

        Parses a question string to extract named entities, filtering out
        ORDINAL and CARDINAL types. Returns entities in lowercase for
        consistent matching against passage entities during graph search.
        This is used in LinearRAG retrieval to find "seed entities" that
        anchor the graph traversal.

        Args:
            question: The query text to extract entities from.

        Returns:
            A set of entity strings (lowercase) found in the question.
                      Empty set if no entities are detected.
        """
        doc = self.spacy_model(question)
        question_entities: set[str] = set()
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            question_entities.add(ent.text.lower())
        return question_entities
