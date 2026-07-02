"""Core LinearRAG class — vendored from LinearRAG (GPL-3).

Modified: converted ``from src.*`` imports to relative imports.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import igraph as ig
import numpy as np
import torch
from tqdm import tqdm

from .embedding_store import EmbeddingStore
from .ner import SpacyNER
from .utils import min_max_normalize

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LinearRAG:
    """Graph-based retrieval system with Personalized PageRank and entity-aware scoring.

    LinearRAG builds a knowledge graph from passages via entity extraction and
    implements a two-stage retrieval pipeline: (1) entity-aware graph search
    using BFS/vectorized matrix operations, and (2) Personalized PageRank ranking.
    The system is designed for efficient multi-hop retrieval over large document
    collections. It supports both BFS iteration and vectorized (PyTorch sparse
    tensor) retrieval modes for flexibility in speed vs. memory trade-offs.

    Attributes:
        config: Configuration object with hyperparameters (embedding model,
                retrieval_top_k, max_iterations, iteration_threshold, etc.).
        dataset_name: Name of the dataset for organizing embeddings.
        passage_embedding_store: EmbeddingStore for passages.
        entity_embedding_store: EmbeddingStore for entities.
        sentence_embedding_store: EmbeddingStore for sentences.
        spacy_ner: SpacyNER pipeline for entity extraction during indexing.
        graph: igraph Graph object representing the knowledge structure.
        device: PyTorch device (cuda or cpu) for vectorized retrieval.
    """

    def __init__(self, global_config: Any) -> None:
        """Initialize LinearRAG with configuration and load embeddings.

        Sets up the three embedding stores (passage, entity, sentence) from
        disk if they exist, or creates empty stores to be populated during
        indexing. Initializes the NER pipeline and knowledge graph. Detects
        GPU availability for vectorized retrieval acceleration.

        Args:
            global_config: A configuration object with required attributes:
                          - dataset_name: Name for organizing embeddings
                          - embedding_model: SentenceTransformer or similar
                          - spacy_model: Spacy model name or Language object
                          - working_dir: Root directory for embeddings
                          - batch_size: Batch size for encoding
                          - use_vectorized_retrieval: Whether to use PyTorch
                          - damping: Damping factor for PersonalizedPageRank
                          - max_iterations: Max BFS iterations
                          - iteration_threshold: Minimum score to continue
                          - retrieval_top_k: Number of top passages to return
                          - top_k_sentence: Top sentences per entity in BFS
                          - passage_ratio, passage_node_weight: Scoring parameters
        """
        self.config = global_config
        logger.info(f"Initializing LinearRAG with config: {self.config}")
        retrieval_method = (
            "Vectorized Matrix-based" if self.config.use_vectorized_retrieval else "BFS Iteration"
        )
        logger.info(f"Using retrieval method: {retrieval_method}")

        # Setup device for GPU acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.config.use_vectorized_retrieval:
            logger.info(f"Using device: {self.device} for vectorized retrieval")

        self.dataset_name = global_config.dataset_name
        self.load_embedding_store()
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.graph = ig.Graph(directed=False)

    def load_embedding_store(self) -> None:
        """Initialize three separate embedding stores for passages, entities, and sentences.

        Each store uses the same embedding model but maintains separate parquet files
        with different namespaces (for hash ID prefixes). Embeddings are loaded from
        disk if the files exist, or created empty to be populated during indexing.
        Called during __init__().
        """
        self.passage_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(
                self.config.working_dir, self.dataset_name, "passage_embedding.parquet"
            ),
            batch_size=self.config.batch_size,
            namespace="passage",
        )
        self.entity_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(
                self.config.working_dir, self.dataset_name, "entity_embedding.parquet"
            ),
            batch_size=self.config.batch_size,
            namespace="entity",
        )
        self.sentence_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(
                self.config.working_dir, self.dataset_name, "sentence_embedding.parquet"
            ),
            batch_size=self.config.batch_size,
            namespace="sentence",
        )

    def load_existing_data(
        self, passage_hash_ids: Any
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        """Load previously extracted NER results and identify new passages to process.

        Checks if a NER results cache exists from a prior indexing run. If it does,
        loads the cached entity-to-passage and entity-to-sentence mappings and
        identifies which passages are new (not in the cache). This enables incremental
        indexing without re-processing existing passages. Called at the beginning
        of the index() method.

        Args:
            passage_hash_ids: The set of hash IDs for all passages being indexed.

        Returns:
            A tuple of three items:
            - existing_passage_hash_id_to_entities: Cached mapping (empty dict if no cache)
            - existing_sentence_to_entities: Cached mapping (empty dict if no cache)
            - new_passage_hash_ids: Set of passage hash IDs not yet processed
                                    (equals passage_hash_ids if no cache exists)
        """
        self.ner_results_path = os.path.join(
            self.config.working_dir, self.dataset_name, "ner_results.json"
        )
        if os.path.exists(self.ner_results_path):
            with open(self.ner_results_path) as f:
                existing_ner_results = json.load(f)
            existing_passage_hash_id_to_entities = existing_ner_results[
                "passage_hash_id_to_entities"
            ]
            existing_sentence_to_entities = existing_ner_results["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return (
                existing_passage_hash_id_to_entities,
                existing_sentence_to_entities,
                new_passage_hash_ids,
            )
        else:
            return {}, {}, passage_hash_ids

    def retrieve(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Retrieve top-k passages for a batch of questions using graph-based retrieval.

        Main retrieval entry point. For each question:
        1. Embed the question and extract seed entities via NER
        2. If seed entities found: perform entity-aware graph search (BFS + PPR)
        3. Else: fallback to dense passage retrieval (pure embedding similarity)
        4. Return top-k passages with scores

        The graph search combines two signals: (a) entity relevance via BFS traversal
        propagating seed entity scores through entity-sentence-entity paths, and
        (b) passage relevance via embedding similarity, boosted by entity matches.
        Personalized PageRank over the combined scores determines final ranking.

        Args:
            questions: List of dicts, each with keys:
                      - "question": the query string
                      - "answer": gold answer (for logging, not used in retrieval)

        Returns:
            List of dicts, one per question, with keys:
            - "question": the original query
            - "sorted_passage": list of top-k passage texts
            - "sorted_passage_scores": list of scores (float)
            - "gold_answer": the provided answer
        """
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        self.vertex_idx_to_node_name = {
            v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()
        }

        if self.config.use_vectorized_retrieval:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
            logger.info(
                f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})"
            )
            logger.info(
                f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})"
            )
            logger.info(f"Device: {self.device}")

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            question_embedding = self.config.embedding_model.encode(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = (
                self.get_seed_entities(question)
            )
            if len(seed_entities) != 0:
                sorted_passage_hash_ids, sorted_passage_scores = (
                    self.graph_search_with_seed_entities(
                        question,
                        question_embedding,
                        seed_entity_indices,
                        seed_entities,
                        seed_entity_hash_ids,
                        seed_entity_scores,
                    )
                )
                final_passage_hash_ids = sorted_passage_hash_ids[: self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[: self.config.retrieval_top_k]
                final_passages = [
                    self.passage_embedding_store.hash_id_to_text[pid]
                    for pid in final_passage_hash_ids
                ]
            else:
                sorted_passage_indices, sorted_passage_scores = self.dense_passage_retrieval(
                    question_embedding
                )
                final_passage_indices = sorted_passage_indices[: self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[: self.config.retrieval_top_k]
                final_passages = [
                    self.passage_embedding_store.texts[idx] for idx in final_passage_indices
                ]
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"],
            }
            retrieval_results.append(result)
        return retrieval_results

    def _precompute_sparse_matrices(self) -> None:
        """Precompute sparse adjacency matrices for vectorized entity-sentence traversal.

        Builds PyTorch sparse COO tensors representing entity-to-sentence and
        sentence-to-entity bipartite relationships. These matrices enable fast
        matrix-vector products in vectorized BFS (instead of iterating manually).
        Called during retrieve() if use_vectorized_retrieval is True.

        Sets instance attributes:
        - entity_to_sentence_sparse: Sparse tensor of shape (num_entities, num_sentences)
        - sentence_to_entity_sparse: Sparse tensor of shape (num_sentences, num_entities)
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)

        entity_to_sentence_indices = []
        entity_to_sentence_values = []

        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)

        sentence_to_entity_indices = []
        sentence_to_entity_values = []

        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)

        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences),
                device=self.device,
            )

        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities),
                device=self.device,
            )

    def graph_search_with_seed_entities(
        self,
        question: str,
        question_embedding: Any,
        seed_entity_indices: list[int],
        seed_entities: list[str],
        seed_entity_hash_ids: list[str],
        seed_entity_scores: list[float],
    ) -> tuple[list[str], list[float]]:
        """Execute entity-aware graph search and return ranked passages.

        Orchestrates the core LinearRAG retrieval algorithm:
        1. Compute entity scores via multi-hop BFS (spreading seed entity signal)
        2. Compute passage scores combining embedding + entity bonus
        3. Run Personalized PageRank with node weights = entity + passage scores
        4. Return ranked passage list

        Args:
            question: The original query string.
            question_embedding: The embedded question (1-D or 2-D array).
            seed_entity_indices: Indices of seed entities in entity_embedding_store.
            seed_entities: Text strings of seed entities.
            seed_entity_hash_ids: Hash IDs of seed entities.
            seed_entity_scores: Similarity scores of seed entities (entity-question).

        Returns:
            A tuple of (passage_hash_ids, passage_scores):
            - passage_hash_ids: Ranked list of passage hash IDs
            - passage_scores: Corresponding Personalized PageRank scores
        """
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(
                question_embedding,
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            )
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(
                question_embedding,
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            )
        passage_weights = self.calculate_passage_scores(
            question, question_embedding, actived_entities
        )
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices, ppr_sorted_passage_scores

    def run_ppr(self, node_weights: NDArray[Any]) -> tuple[list[str], list[float]]:
        """Run Personalized PageRank over the knowledge graph with given node weights.

        Executes igraph's personalized_pagerank algorithm using entity and passage
        scores as reset probabilities. PPR amplifies paths through high-weight nodes,
        combining local relevance (seed entities) with graph structure (entity-sentence
        relationships). Returns only passage nodes, ranked by PPR score.

        Args:
            node_weights: Array of shape (num_nodes,) where index i corresponds to
                         graph vertex i. Contains entity_weights + passage_weights.
                         NaN and negative values are converted to 0.

        Returns:
            A tuple of (passage_hash_ids, passage_scores):
            - passage_hash_ids: List of passage hash IDs sorted by PPR score (descending)
            - passage_scores: Corresponding PPR scores (float list)
        """
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]

        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]]
            for i in sorted_indices_in_doc_scores
        ]

        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(
        self,
        question_embedding: Any,
        seed_entity_indices: list[int],
        seed_entities: list[str],
        seed_entity_hash_ids: list[str],
        seed_entity_scores: list[float],
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        """Compute entity scores using iterative BFS over entity-sentence-entity paths.

        Implements the core LinearRAG entity relevance computation: breadth-first
        traversal of the entity-sentence bipartite graph, propagating seed entity
        scores through sentence nodes based on question similarity. Each iteration:
        1. For each active entity, retrieve its associated sentences
        2. Score sentences by embedding similarity to the question
        3. For each sentence, retrieve associated entities
        4. Propagate entity score = current_score * sentence_similarity
        5. Add newly activated entities to the next iteration (threshold-filtered)

        This is the iterative (non-vectorized) implementation. See
        calculate_entity_scores_vectorized() for the matrix-based version.

        Args:
            question_embedding: Embedded question (1-D or 2-D array).
            seed_entity_indices: Indices of seed entities.
            seed_entities: Text strings of seed entities.
            seed_entity_hash_ids: Hash IDs of seed entities.
            seed_entity_scores: Seed entity similarity scores.

        Returns:
            A tuple of (entity_weights, actived_entities):
            - entity_weights: Array of shape (num_nodes,) with cumulative scores
            - actived_entities: Dict mapping entity_hash_id to (idx, score, tier)
        """
        actived_entities: dict[str, Any] = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx, _seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
            strict=True,
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        used_sentence_hash_ids: set[str] = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities: dict[str, Any] = {}
            for entity_hash_id, (_entity_id, entity_score, _tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [
                    sid
                    for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id])
                    if sid not in used_sentence_hash_ids
                ]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [
                    self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids
                ]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = (
                    question_embedding.reshape(-1, 1)
                    if len(question_embedding.shape) == 1
                    else question_embedding
                )
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][
                    : self.config.top_k_sentence
                ]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[
                        top_sentence_hash_id
                    ]
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        next_entity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_entity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (
                            next_entity_node_idx,
                            next_entity_score,
                            iteration + 1,
                        )
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def calculate_entity_scores_vectorized(
        self,
        question_embedding: Any,
        seed_entity_indices: list[int],
        seed_entities: list[str],
        seed_entity_hash_ids: list[str],
        seed_entity_scores: list[float],
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        """Compute entity scores using vectorized sparse matrix operations (GPU-accelerated).

        Functionally equivalent to calculate_entity_scores() but uses PyTorch sparse
        tensors for efficient matrix-vector products instead of explicit iteration.
        Uses GPU (if available) for massive speedup on large graphs. BFS iterations
        are computed as: entity_scores @ entity_to_sentence_sparse @ sentence_weights.

        Args:
            question_embedding: Embedded question (1-D or 2-D array).
            seed_entity_indices: Indices of seed entities.
            seed_entities: Text strings of seed entities.
            seed_entity_hash_ids: Hash IDs of seed entities.
            seed_entity_scores: Seed entity similarity scores.

        Returns:
            A tuple of (entity_weights, actived_entities):
            - entity_weights: Array of shape (num_nodes,) with cumulative scores
            - actived_entities: Dict mapping entity_hash_id to (idx, score, tier)
        """
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)

        question_emb = (
            question_embedding.reshape(-1, 1)
            if len(question_embedding.shape) == 1
            else question_embedding
        )
        sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()
        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.device)

        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)

        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()

        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
        entity_scores_dense.scatter_(
            0,
            torch.tensor(seed_entity_indices, device=self.device),
            torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device),
        )

        actived_entities: dict[str, Any] = {}
        for seed_entity_idx, _seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
            strict=True,
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score

        current_entity_scores_sparse = entity_scores_sparse

        for iteration in range(1, self.config.max_iterations):
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()
            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold,
                current_entity_scores_dense,
                torch.zeros_like(current_entity_scores_dense),
            )

            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)

            if len(nonzero_indices) == 0:
                break

            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0), nonzero_values, (num_entities,), device=self.device
            ).coalesce()

            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device,
            ).coalesce()

            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(), current_scores_2d
            )
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()

            sentence_activation = torch.where(
                used_sentence_mask, torch.zeros_like(sentence_activation), sentence_activation
            )

            selected_sentence_indices_list = []

            if len(nonzero_indices) > 0 and self.config.top_k_sentence > 0:
                for _i, entity_idx in enumerate(nonzero_indices):
                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]

                    if len(entity_sentence_indices) == 0:
                        continue

                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]

                    if len(available_sentence_indices) == 0:
                        continue

                    sentence_sims = sentence_similarities[available_sentence_indices]
                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        _top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)

                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)
                    used_sentence_mask[unique_selected_sentences] = True

                    weighted_sentence_scores = sentence_activation * sentence_similarities
                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask, weighted_sentence_scores, torch.zeros_like(weighted_sentence_scores)
                    )
                else:
                    weighted_sentence_scores = torch.zeros(
                        num_sentences, dtype=torch.float32, device=self.device
                    )
            else:
                weighted_sentence_scores = torch.zeros(
                    num_sentences, dtype=torch.float32, device=self.device
                )

            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(
                -1
            )

            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack(
                        [weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]
                    ),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device,
                ).coalesce()

                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(), weighted_scores_2d
                )
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(
                    num_entities, dtype=torch.float32, device=self.device
                )

            entity_scores_dense += next_entity_scores_dense

            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.config.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)

            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0),
                    next_nonzero_values,
                    (num_entities,),
                    device=self.device,
                ).coalesce()
            else:
                break

        entity_scores_final = entity_scores_dense.cpu().numpy()
        nonzero_indices_np = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices_np:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)

        return entity_weights, actived_entities

    def calculate_passage_scores(
        self, question: str, question_embedding: Any, actived_entities: dict[str, Any]
    ) -> NDArray[Any]:
        """Compute passage scores combining embedding similarity and entity match bonuses.

        For each passage: score = passage_ratio * dpr_score + log(1 + entity_bonus).
        The DPR score comes from embedding similarity to the question. The entity bonus
        accumulates for each activated entity found in the passage, scaled by the entity's
        BFS tier (inversely — earlier tiers get higher weight). This two-signal combination
        ensures both semantic relevance and entity-topical alignment.

        Args:
            question: The original query string (not used directly, for logging).
            question_embedding: Embedded question vector.
            actived_entities: Dict from calculate_entity_scores mapping entity_hash_id
                             to (entity_idx, entity_score, bfs_tier).

        Returns:
            Array of shape (num_nodes,) with passage scores (0 for non-passages).
        """
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores_list = self.dense_passage_retrieval(
            question_embedding
        )
        dpr_passage_scores = min_max_normalize(np.array(dpr_passage_scores_list))

        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[
                passage_hash_id
            ].lower()
            for entity_hash_id, (_entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus

            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(
                1 + total_entity_bonus
            )

            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question_embedding: Any) -> tuple[NDArray[Any], list[float]]:
        """Retrieve passages by pure embedding similarity (fallback for no seed entities).

        Computes cosine similarity between the question embedding and all passage
        embeddings via dot product (embeddings are L2-normalized). Returns the
        sorted list of passage indices and scores. Used as a fallback when NER
        does not extract seed entities.

        Args:
            question_embedding: Embedded question vector.

        Returns:
            A tuple of (passage_indices, passage_scores):
            - passage_indices: Sorted indices in passage_embedding_store (descending score)
            - passage_scores: Corresponding similarity scores (list)
        """
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores

    def get_seed_entities(
        self, question: str
    ) -> tuple[list[int], list[str], list[str], list[float]]:
        """Extract and link entities from the question to the knowledge graph.

        Executes NER on the question, then for each extracted entity, finds the
        most similar entity in the knowledge graph via embedding similarity. Returns
        the matched entities and their similarity scores, which serve as starting
        points (seeds) for graph-based traversal. If no entities are extracted,
        returns empty lists (triggering fallback to dense retrieval).

        Args:
            question: The query string to extract entities from.

        Returns:
            A tuple of four lists (all parallel):
            - seed_entity_indices: Indices in entity_embedding_store
            - seed_entity_texts: Text strings of matched entities
            - seed_entity_hash_ids: Hash IDs of matched entities
            - seed_entity_scores: Similarity scores (cosine, entity-question)
            Empty lists if no question entities extracted.
        """
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return [], [], [], []
        question_entity_embeddings = self.config.embedding_model.encode(
            question_entities,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
        )
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(int(best_entity_idx))
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(float(best_entity_score))
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages: list[str]) -> None:
        """Build the LinearRAG knowledge graph from a collection of passages.

        Orchestrates the complete indexing pipeline:
        1. Insert passages and compute embeddings
        2. Load cached NER results (if available) for incremental indexing
        3. Run NER on new passages to extract entities and sentences
        4. Build entity-sentence-passage graph
        5. Compute embeddings for all entities and sentences
        6. Build entity-to-passage edges with entity match statistics
        7. Add adjacent-passage edges (sequential ordering)
        8. Construct the knowledge graph in igraph
        9. Save the graph to GraphML format

        Supports incremental indexing: if NER results cache exists, new passages
        are NER-processed while old results are reused.

        Args:
            passages: List of passage (document) strings to index.
        """
        self.node_to_node_stats: dict[str, dict[str, float]] = defaultdict(dict)
        self.entity_to_sentence_stats: dict[str, dict[str, float]] = defaultdict(dict)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        (
            existing_passage_hash_id_to_entities,
            existing_sentence_to_entities,
            new_passage_hash_ids,
        ) = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k: hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities, new_sentence_to_entities = self.spacy_ner.batch_ner(
                new_hash_id_to_passage, self.config.max_workers
            )
            self.merge_ner_results(
                existing_passage_hash_id_to_entities,
                existing_sentence_to_entities,
                new_passage_hash_id_to_entities,
                new_sentence_to_entities,
            )
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        (
            entity_nodes,
            sentence_nodes,
            passage_hash_id_to_entities,
            self.entity_to_sentence,
            self.sentence_to_entity,
        ) = self.extract_nodes_and_edges(
            existing_passage_hash_id_to_entities, existing_sentence_to_entities
        )
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [
                self.sentence_embedding_store.text_to_hash_id[s] for s in sentence
            ]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sent, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sent]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [
                self.entity_embedding_store.text_to_hash_id[e] for e in entities
            ]
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
        output_graphml_path = os.path.join(
            self.config.working_dir, self.dataset_name, "LinearRAG.graphml"
        )
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)
        self.graph.write_graphml(output_graphml_path)

    def add_adjacent_passage_edges(self) -> None:
        """Create edges between sequentially indexed passages.

        Parses passage texts for index prefixes (e.g., "123: passage text") and
        creates weighted edges from passage i to passage i+1. This encodes document
        structure (e.g., chunks from the same paper) into the graph, enabling
        retrieval of contextually adjacent passages. Only applies to passages with
        detected index prefixes; others are skipped.
        """
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r"^(\d+):")
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self) -> None:
        """Add all graph nodes and edges to the igraph Graph object.

        Orchestrates graph construction by calling add_nodes() and add_edges().
        After this, self.graph contains the complete knowledge graph topology.
        """
        self.add_nodes()
        self.add_edges()

    def add_nodes(self) -> None:
        """Add entity and passage nodes to the igraph Graph.

        Creates a vertex for each entity and passage with attributes:
        - name: the hash ID (unique identifier)
        - content: the text string
        Skips nodes already in the graph (idempotent). Also builds:
        - node_name_to_vertex_idx: mapping from hash ID to igraph vertex index
        - passage_node_indices: list of vertex indices for only passage nodes
            (used for final PPR ranking)
        """
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}

        passage_hash_ids = set(passage_hash_id_to_text.keys())

        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)

        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id]
            for passage_id in passage_hash_ids
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self) -> None:
        """Add weighted edges to the igraph Graph.

        Converts the node_to_node_stats dictionary (built by add_entity_to_passage_edges
        and add_adjacent_passage_edges) into igraph edges with weights. Skips self-loops.
        Called as part of augment_graph().
        """
        edges = []
        weights = []
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities: dict[str, Any]) -> None:
        """Create weighted edges from passages to their extracted entities.

        Computes passage-entity edge weights as the normalized entity occurrence count:
        weight = entity_count_in_passage / total_entity_count_in_passage.
        This encodes passage-entity affinity for graph search. Higher weight means
        the entity is a dominant concept in that passage.

        Args:
            passage_hash_id_to_entities: Dict mapping passage hash IDs to lists of
                                         extracted entity strings.
        """
        passage_to_entity_count: dict[tuple[str, str], int] = {}
        passage_to_all_score: dict[str, int] = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(
        self,
        existing_passage_hash_id_to_entities: dict[str, Any],
        existing_sentence_to_entities: dict[str, Any],
    ) -> tuple[set[str], set[str], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
        """Extract unique entities, sentences, and bipartite relationships from NER results.

        Converts the NER output (passage->entities, sentence->entities mappings)
        into graph node and edge sets: unique entity texts, unique sentence texts,
        and entity<->sentence relationships. These are used to build the bipartite
        entity-sentence subgraph for BFS traversal during retrieval.

        Args:
            existing_passage_hash_id_to_entities: Dict mapping passage hash IDs
                                                 to entity lists.
            existing_sentence_to_entities: Dict mapping sentence texts to entity lists.

        Returns:
            A tuple of five items:
            - entity_nodes: Set of unique entity text strings
            - sentence_nodes: Set of unique sentence text strings
            - passage_hash_id_to_entities: Re-keyed version (same as input)
            - entity_to_sentence: Dict mapping entity -> set of sentence texts
            - sentence_to_entity: Dict mapping sentence -> set of entity texts
        """
        entity_nodes: set[str] = set()
        sentence_nodes: set[str] = set()
        passage_hash_id_to_entities: dict[str, set[str]] = defaultdict(set)
        entity_to_sentence: dict[str, set[str]] = defaultdict(set)
        sentence_to_entity: dict[str, set[str]] = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence, entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return (
            entity_nodes,
            sentence_nodes,
            passage_hash_id_to_entities,
            entity_to_sentence,
            sentence_to_entity,
        )

    def merge_ner_results(
        self,
        existing_passage_hash_id_to_entities: dict[str, Any],
        existing_sentence_to_entities: dict[str, Any],
        new_passage_hash_id_to_entities: dict[str, Any],
        new_sentence_to_entities: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Merge cached NER results with newly extracted NER results.

        Combines old (from cache) and new (freshly extracted) NER mappings.
        Used for incremental indexing: new passages are NER-processed while
        existing results are preserved. Modifies the existing dicts in-place.

        Args:
            existing_passage_hash_id_to_entities: Cached mapping (modified in-place).
            existing_sentence_to_entities: Cached mapping (modified in-place).
            new_passage_hash_id_to_entities: Newly extracted passage->entities.
            new_sentence_to_entities: Newly extracted sentence->entities.

        Returns:
            The merged dicts (same objects that were passed in, now updated).
        """
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(
        self,
        existing_passage_hash_id_to_entities: dict[str, Any],
        existing_sentence_to_entities: dict[str, Any],
    ) -> None:
        """Persist NER results to JSON for incremental indexing.

        Saves the merged NER mappings to disk (ner_results.json) so that future
        indexing runs can reuse these results without re-processing passages.
        Called at the end of the index() method.

        Args:
            existing_passage_hash_id_to_entities: The merged passage->entities mapping.
            existing_sentence_to_entities: The merged sentence->entities mapping.
        """
        with open(self.ner_results_path, "w") as f:
            json.dump(
                {
                    "passage_hash_id_to_entities": existing_passage_hash_id_to_entities,
                    "sentence_to_entities": existing_sentence_to_entities,
                },
                f,
            )
