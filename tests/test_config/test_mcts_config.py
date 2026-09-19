"""Tests for ``MCTSConfig`` (``ctra.config.settings``): the hypervolume reference point."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctra.config.settings import MCTSConfig, Settings

_ENV_KEYS = (
    "CTRA_MCTS_REFERENCE_POINT",
    "CTRA_MCTS_OBJECTIVES",
    "CTRA_MCTS__REFERENCE_POINT",
    "CTRA_MCTS__OBJECTIVES",
)


@pytest.fixture(autouse=True)
def _isolate_mcts_env(monkeypatch):
    """Keep the live environment (flat and nested spellings) out of these tests."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


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


class TestDerivedReferencePoint:
    """A reference that was not supplied is derived from the objectives' floors.

    Narrowing ``objectives`` alone must not make ``get_settings()`` raise for
    every caller (the API, loaders, RAG and agents all build ``Settings``);
    only an explicit mismatch is an error.
    """

    def test_explicit_single_objective_without_reference_derives_the_floor(self):
        config = MCTSConfig(objectives=["accuracy"])

        assert config.reference_point == [0.5]

    def test_reordered_objectives_derive_reordered_floors(self):
        config = MCTSConfig(objectives=["parsimony", "accuracy"])

        assert config.reference_point == [0.0, 0.5]

    def test_env_single_objective_derives_the_floor(self, monkeypatch):
        monkeypatch.setenv("CTRA_MCTS_OBJECTIVES", '["accuracy"]')

        config = MCTSConfig()

        assert config.objectives == ["accuracy"]
        assert config.reference_point == [0.5]

    def test_nested_env_single_objective_derives_the_floor_through_settings(self, monkeypatch):
        monkeypatch.setenv("CTRA_MCTS__OBJECTIVES", '["accuracy"]')

        mcts = Settings().mcts

        assert mcts.objectives == ["accuracy"]
        assert mcts.reference_point == [0.5]

    def test_env_reference_mismatching_default_objectives_is_rejected(self, monkeypatch):
        """An env-sourced reference counts as supplied: its length is checked."""
        monkeypatch.setenv("CTRA_MCTS_REFERENCE_POINT", "[0.5]")

        with pytest.raises(ValidationError, match="one reference value per objective"):
            MCTSConfig()

    def test_env_reference_matching_env_objectives_is_kept(self, monkeypatch):
        monkeypatch.setenv("CTRA_MCTS_OBJECTIVES", '["accuracy"]')
        monkeypatch.setenv("CTRA_MCTS_REFERENCE_POINT", "[0.6]")

        assert MCTSConfig().reference_point == [0.6]
