"""Utility functions vendored from LinearRAG (GPL-3)."""

from __future__ import annotations

from hashlib import md5
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute a unique MD5-based hash ID for content.

    Generates a stable, deterministic hash identifier for a text string.
    Used in LinearRAG to create unique keys for passages, sentences, and
    entities in the knowledge graph, ensuring consistent indexing across
    runs and allowing deduplication.

    Args:
        content: The text to hash.
        prefix: Optional prefix to prepend to the hash (e.g., "passage-", "entity-").
                Default: empty string.

    Returns:
        A string combining the prefix with the MD5 hex digest of the content.
    """
    return prefix + md5(content.encode()).hexdigest()


def min_max_normalize(x: NDArray[Any]) -> NDArray[Any]:
    """Normalize an array to [0, 1] range using min-max scaling.

    Rescales array values to the [0, 1] interval by subtracting the minimum
    and dividing by the range (max - min). Handles edge case where all values
    are identical (range = 0) by returning a ones array. Used in LinearRAG
    to normalize passage retrieval scores before combining with entity-based
    scoring in the graph search pipeline.

    Args:
        x: Input array to normalize.

    Returns:
        Normalized array with values in [0, 1]. If all values in x are equal,
        returns an array of ones (uniform score).
    """
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val  # type: ignore[no-any-return]
