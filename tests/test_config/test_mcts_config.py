"""Tests for ``MCTSConfig`` (``ctra.config.settings``): the hypervolume reference point."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctra.config.settings import MCTSConfig


class TestReferencePoint:
    def test_default_sits_at_the_roc_auc_chance_baseline(self):
        """Accuracy is a raw ROC-AUC, so its worst acceptable value is 0.5, not 0 (issue #18)."""
        config = MCTSConfig()

        assert config.objectives == ["accuracy", "parsimony"]
        assert config.reference_point == [0.5, 0.0]

    def test_one_coordinate_per_objective_is_accepted(self):
        config = MCTSConfig(objectives=["accuracy"], reference_point=[0.5])

        assert config.reference_point == [0.5]

    def test_fewer_coordinates_than_objectives_is_rejected(self):
        with pytest.raises(ValidationError, match="one reference value per objective"):
            MCTSConfig(objectives=["accuracy", "parsimony"], reference_point=[0.5])

    def test_more_coordinates_than_objectives_is_rejected(self):
        with pytest.raises(ValidationError, match="reference_point has 2 coordinates"):
            MCTSConfig(objectives=["accuracy"], reference_point=[0.5, 0.0])
