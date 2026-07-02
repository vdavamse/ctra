"""Tests for ctra.search.pareto — Pareto front, hypervolume, selection.

Pure math tests: no mocks, no external services.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctra.search.pareto import (
    _compute_2d_hypervolume,
    hypervolume_contribution,
    pareto_front,
    pareto_front_indices,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def two_non_dominated() -> np.ndarray:
    """Two points that do not dominate each other (trade-off)."""
    return np.array([[1.0, 0.0], [0.0, 1.0]])


@pytest.fixture
def with_dominated() -> np.ndarray:
    """Three points: two on the front, one dominated."""
    return np.array(
        [
            [3.0, 1.0],  # front
            [1.0, 3.0],  # front
            [1.0, 1.0],  # dominated by both
        ]
    )


@pytest.fixture
def three_objectives() -> np.ndarray:
    """Points in 3-objective space for Pareto tests."""
    return np.array(
        [
            [0.9, 0.1, 0.5],
            [0.1, 0.9, 0.5],
            [0.5, 0.5, 0.5],
            [0.3, 0.3, 0.3],  # dominated by [0.5, 0.5, 0.5]
        ]
    )


# ======================================================================
# pareto_front
# ======================================================================


class TestParetoFront:
    """Tests for the boolean mask identifying non-dominated points."""

    def test_two_non_dominated(self, two_non_dominated: np.ndarray) -> None:
        mask = pareto_front(two_non_dominated)
        np.testing.assert_array_equal(mask, [True, True])

    def test_with_dominated_point(self, with_dominated: np.ndarray) -> None:
        mask = pareto_front(with_dominated)
        np.testing.assert_array_equal(mask, [True, True, False])

    def test_single_point(self) -> None:
        pts = np.array([[0.5, 0.5]])
        mask = pareto_front(pts)
        np.testing.assert_array_equal(mask, [True])

    def test_empty(self) -> None:
        pts = np.array([]).reshape(0, 2)
        mask = pareto_front(pts)
        assert len(mask) == 0
        assert mask.dtype == bool

    def test_all_identical(self) -> None:
        pts = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        mask = pareto_front(pts)
        # None dominates another when equal, so all are on the front
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_chain_dominance(self) -> None:
        """A > B > C — only A should be on the front."""
        pts = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
        mask = pareto_front(pts)
        np.testing.assert_array_equal(mask, [True, False, False])

    def test_three_objectives(self, three_objectives: np.ndarray) -> None:
        mask = pareto_front(three_objectives)
        # Point [0.3, 0.3, 0.3] is dominated by [0.5, 0.5, 0.5]
        np.testing.assert_array_equal(mask, [True, True, True, False])


# ======================================================================
# pareto_front_indices
# ======================================================================


class TestParetoFrontIndices:
    """Tests for integer-index variant of pareto_front."""

    def test_matches_mask(self, with_dominated: np.ndarray) -> None:
        mask = pareto_front(with_dominated)
        indices = pareto_front_indices(with_dominated)
        np.testing.assert_array_equal(indices, np.where(mask)[0])

    def test_single_point(self) -> None:
        pts = np.array([[1.0, 2.0]])
        indices = pareto_front_indices(pts)
        np.testing.assert_array_equal(indices, [0])

    def test_empty(self) -> None:
        pts = np.array([]).reshape(0, 2)
        indices = pareto_front_indices(pts)
        assert len(indices) == 0


# ======================================================================
# _compute_2d_hypervolume
# ======================================================================


class TestCompute2dHypervolume:
    """Tests for the sweep-line 2D hypervolume computation."""

    def test_known_area_three_points(self) -> None:
        """Classic example: [[3,1],[1,3],[2,2]] with ref=[0,0] -> 6.0.

        Staircase: (0,0)->(3,0)->(3,1)->(2,1)->(2,2)->(1,2)->(1,3)->(0,3)
        Area = 3*1 + 2*(2-1) + 1*(3-2) = 3 + 2 + 1 = 6.
        """
        pts = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        ref = np.array([0.0, 0.0])
        hv = _compute_2d_hypervolume(pts, ref)
        np.testing.assert_allclose(hv, 6.0, atol=1e-6)

    def test_single_point(self) -> None:
        pts = np.array([[2.0, 3.0]])
        ref = np.array([0.0, 0.0])
        hv = _compute_2d_hypervolume(pts, ref)
        np.testing.assert_allclose(hv, 6.0, atol=1e-6)

    def test_empty(self) -> None:
        pts = np.array([]).reshape(0, 2)
        ref = np.array([0.0, 0.0])
        hv = _compute_2d_hypervolume(pts, ref)
        np.testing.assert_allclose(hv, 0.0, atol=1e-6)

    def test_points_at_reference(self) -> None:
        """Points exactly at the reference contribute zero volume."""
        pts = np.array([[0.0, 0.0]])
        ref = np.array([0.0, 0.0])
        hv = _compute_2d_hypervolume(pts, ref)
        np.testing.assert_allclose(hv, 0.0, atol=1e-6)

    def test_non_zero_reference(self) -> None:
        pts = np.array([[4.0, 5.0]])
        ref = np.array([1.0, 2.0])
        hv = _compute_2d_hypervolume(pts, ref)
        np.testing.assert_allclose(hv, 3.0 * 3.0, atol=1e-6)

    def test_dominated_point_ignored(self) -> None:
        """A point behind the reference should not contribute."""
        pts = np.array([[0.5, 0.5], [3.0, 3.0]])
        ref = np.array([1.0, 1.0])
        hv = _compute_2d_hypervolume(pts, ref)
        # Only (3,3) counts: (3-1)*(3-1) = 4
        np.testing.assert_allclose(hv, 4.0, atol=1e-6)


# ======================================================================
# hypervolume_contribution
# ======================================================================


class TestHypervolumeContribution:
    """Tests for per-point hypervolume contribution."""

    def test_single_point(self) -> None:
        pts = np.array([[2.0, 3.0]])
        ref = np.array([0.0, 0.0])
        hvc = hypervolume_contribution(pts, ref)
        np.testing.assert_allclose(hvc, [6.0], atol=1e-6)

    def test_two_points_contributions(self) -> None:
        """Each point's exclusive hypervolume contribution."""
        pts = np.array([[3.0, 1.0], [1.0, 3.0]])
        ref = np.array([0.0, 0.0])
        hvc = hypervolume_contribution(pts, ref)
        # Total HV = 3*1 + 1*(3-1) = 3+2 = 5
        # Without point 0: HV of [[1,3]] = 1*3 = 3 -> contribution 0 = 5-3 = 2
        # Without point 1: HV of [[3,1]] = 3*1 = 3 -> contribution 1 = 5-3 = 2
        np.testing.assert_allclose(hvc[0], 2.0, atol=1e-6)
        np.testing.assert_allclose(hvc[1], 2.0, atol=1e-6)

    def test_at_reference_zero_contribution(self) -> None:
        """A point at the reference contributes nothing."""
        pts = np.array([[0.0, 0.0], [2.0, 2.0]])
        ref = np.array([0.0, 0.0])
        hvc = hypervolume_contribution(pts, ref)
        np.testing.assert_allclose(hvc[0], 0.0, atol=1e-6)

    def test_empty(self) -> None:
        pts = np.array([]).reshape(0, 2)
        ref = np.array([0.0, 0.0])
        hvc = hypervolume_contribution(pts, ref)
        assert len(hvc) == 0

    def test_default_reference_is_origin(self) -> None:
        """When reference is None, it defaults to zeros."""
        pts = np.array([[2.0, 3.0]])
        hvc = hypervolume_contribution(pts, reference=None)
        np.testing.assert_allclose(hvc, [6.0], atol=1e-6)

    def test_contributions_sum_le_total(self) -> None:
        """Sum of contributions should not exceed total hypervolume."""
        pts = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        ref = np.array([0.0, 0.0])
        hvc = hypervolume_contribution(pts, ref)
        total = _compute_2d_hypervolume(pts, ref)
        assert np.sum(hvc) <= total + 1e-6
