"""Tests for ctra.search.objectives — ObjectiveResult data container.

Pure math tests: no mocks, no external services.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctra.search.objectives import ObjectiveResult

# ======================================================================
# ObjectiveResult
# ======================================================================


class TestObjectiveResult:
    """Tests for the ObjectiveResult data container."""

    def test_getitem_by_name(self) -> None:
        r = ObjectiveResult(
            values=np.array([0.8, 0.6]),
            names=["accuracy", "parsimony"],
        )
        np.testing.assert_allclose(r["accuracy"], 0.8, atol=1e-6)
        np.testing.assert_allclose(r["parsimony"], 0.6, atol=1e-6)

    def test_getitem_missing_name_raises(self) -> None:
        r = ObjectiveResult(
            values=np.array([0.5]),
            names=["accuracy"],
        )
        with pytest.raises(KeyError, match="unknown"):
            _ = r["unknown"]

    def test_as_tuple(self) -> None:
        r = ObjectiveResult(
            values=np.array([0.8, 0.6]),
            names=["a", "b"],
        )
        t = r.as_tuple()
        assert isinstance(t, tuple)
        np.testing.assert_allclose(t, (0.8, 0.6), atol=1e-6)
