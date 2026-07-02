"""Configuration — vendored from LinearRAG (GPL-3).

Modified: replaced ``LLM_Model`` type with ``Any`` since CTRA does not
use LinearRAG's built-in QA pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LinearRAGConfig:
    dataset_name: str
    embedding_model: Any = "all-mpnet-base-v2"
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: Any = "en_core_web_trf"  # str or pre-built spaCy pipeline
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False
