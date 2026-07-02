"""Embedding storage — vendored from LinearRAG (GPL-3)."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .utils import compute_mdhash_id

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EmbeddingStore:
    """Persistent embedding storage for LinearRAG passages, sentences, and entities.

    Manages embeddings for text nodes in the LinearRAG knowledge graph using
    parquet files for persistence. Maintains multiple index mappings for
    efficient lookups (hash_id <-> text, hash_id <-> position). Supports
    incremental insertion with automatic deduplication — new texts are embedded
    and appended; existing texts are skipped. Used during indexing to store
    embeddings for passages, sentences, and entities, and during retrieval
    to access pre-computed embeddings for similarity computations.

    Attributes:
        embedding_model: The sentence embedding model (e.g., SentenceTransformer).
        db_filename: Path to the parquet file storing embeddings.
        batch_size: Batch size for embedding encoding.
        namespace: Prefix for hash IDs (e.g., 'passage', 'entity', 'sentence').
        hash_ids: Ordered list of unique hash IDs.
        texts: Ordered list of text strings (parallel to hash_ids).
        embeddings: Ordered list of embedding vectors (parallel to hash_ids).
        hash_id_to_text: Mapping from hash ID to text string.
        hash_id_to_idx: Mapping from hash ID to position in the lists.
        text_to_hash_id: Mapping from text string to hash ID.
    """

    def __init__(
        self, embedding_model: Any, db_filename: str, batch_size: int, namespace: str
    ) -> None:
        """Initialize an embedding store for a given namespace.

        Loads existing embeddings from disk if available. If the parquet file
        doesn't exist, creates an empty store that will be populated by
        subsequent insert_text() calls.

        Args:
            embedding_model: A sentence embedding model with an encode() method
                             (e.g., SentenceTransformer).
            db_filename: Path to the parquet file for persistence (will be
                         created if it doesn't exist).
            batch_size: Batch size for encoding texts into embeddings.
            namespace: A string identifier (e.g., 'passage', 'entity') used as
                       a prefix for generated hash IDs.
        """
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace

        self.hash_ids: list[str] = []
        self.texts: list[str] = []
        self.embeddings: list[Any] = []
        self.hash_id_to_text: dict[str, str] = {}
        self.hash_id_to_idx: dict[str, int] = {}
        self.text_to_hash_id: dict[str, str] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Load embeddings from parquet file if it exists.

        Reads the parquet database and reconstructs all index mappings
        (hash_id_to_idx, hash_id_to_text, text_to_hash_id). Called during
        initialization. If the file doesn't exist, leaves the store empty
        to be populated by insert_text() calls.
        """
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["text"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()

            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts, strict=True)}
            self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids, strict=True)}
            print(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")

    def insert_text(self, text_list: list[str]) -> None:
        """Insert texts and their embeddings into the store.

        Deduplicates texts by computing their hash IDs. New texts are encoded
        via the embedding model and appended to the store. Existing texts
        (same hash ID) are skipped. The store is persisted to disk after
        insertion. This is the main entry point for adding passages, sentences,
        or entities during LinearRAG indexing.

        Args:
            text_list: A list of text strings to insert (may contain duplicates).
        """
        nodes_dict = {}
        for text in text_list:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {"content": text}

        all_hash_ids = list(nodes_dict.keys())

        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        all_embeddings = self.embedding_model.encode(
            texts_to_encode,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )

        self._upsert(missing_ids, texts_to_encode, all_embeddings)

    def _upsert(self, hash_ids: list[str], texts: list[str], embeddings: Any) -> None:
        """Add new texts and embeddings, then rebuild index mappings and save.

        Appends the provided hash IDs, texts, and embeddings to the store,
        then reconstructs all index mappings. Called internally by insert_text().
        Persists the updated store to disk via _save_data().

        Args:
            hash_ids: List of hash IDs for the new texts.
            texts: List of text strings (parallel to hash_ids).
            embeddings: List or array of embedding vectors (parallel to hash_ids).
        """
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)

        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts, strict=True)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids, strict=True)}

        self._save_data()

    def _save_data(self) -> None:
        """Persist the current store to a parquet file.

        Writes all hash IDs, texts, and embeddings to the configured parquet file,
        creating parent directories if needed. Called after each _upsert() to
        ensure data persistence across runs.
        """
        data_to_save = pd.DataFrame(
            {
                "hash_id": self.hash_ids,
                "text": self.texts,
                "embedding": self.embeddings,
            }
        )
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        data_to_save.to_parquet(self.db_filename, index=False)

    def get_hash_id_to_text(self) -> dict[str, str]:
        """Return a deep copy of the hash_id_to_text mapping.

        Returns a copy rather than a reference to prevent external modifications
        to the store's internal mapping.

        Returns:
            A dict mapping hash IDs to their text strings.
        """
        return deepcopy(self.hash_id_to_text)

    def encode_texts(self, texts: list[str]) -> Any:
        """Encode a list of texts into embeddings using the embedding model.

        Wrapper around the embedding model's encode() method. Used for
        embedding questions during retrieval without storing them in the store.

        Args:
            texts: List of text strings to encode.

        Returns:
            Array or list of embedding vectors (one per text).
        """
        return self.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size
        )

    def get_embeddings(self, hash_ids: list[str]) -> NDArray[Any]:
        """Retrieve embeddings for a list of hash IDs.

        Looks up pre-stored embeddings by their hash IDs. Efficient for
        bulk retrieval during graph search. Returns an empty array if
        hash_ids is empty.

        Args:
            hash_ids: List of hash IDs to look up.

        Returns:
            A numpy array of shape (len(hash_ids), embedding_dim) containing
            the requested embeddings in the same order as hash_ids.
        """
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings)[indices]
        return embeddings
