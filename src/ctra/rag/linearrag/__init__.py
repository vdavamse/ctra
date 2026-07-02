"""Vendored LinearRAG library (DEEP-PolyU/LinearRAG, GPL-3).

This is a modified copy of the upstream LinearRAG source code.  Changes:

- Converted ``from src.*`` imports to relative imports.
- ``SpacyNER.__init__`` accepts either a model name (``str``) or a
  pre-built spaCy ``Language`` pipeline, removing the need for
  monkey-patching when using GLiNER-BioMed.
- Removed ``LLM_Model`` dependency from config (replaced with ``Any``).
- Removed ``pdb`` import from ``ner.py``.

Original repository: https://github.com/DEEP-PolyU/LinearRAG
License: GPL-3.0 (see LICENSE.txt in the original repository)
"""

from ctra.rag.linearrag.config import LinearRAGConfig
from ctra.rag.linearrag.core import LinearRAG
from ctra.rag.linearrag.embedding_store import EmbeddingStore
from ctra.rag.linearrag.ner import SpacyNER

__all__ = ["EmbeddingStore", "LinearRAG", "LinearRAGConfig", "SpacyNER"]
