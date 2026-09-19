"""Pareto front, hypervolume ranking, and Pareto-based selection for MCTS.

Implements the multi-objective selection logic for MCTS nodes.  Each node
has a reward vector of shape ``(n_objectives,)`` where objectives are:

    0. Accuracy   (ROC-AUC, maximize)
    1. Parsimony  (1 - n/max, maximize)

All objectives are normalized to [0, 1] and oriented so that higher = better.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ctra.search.mcts import MCTSNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------


def pareto_front(points: NDArray[Any]) -> NDArray[Any]:
    """Identify the first (best) Pareto front — the set of non-dominated solutions.

    A point is *non-dominated* if no other point in the set is better on all
    objectives simultaneously.  Visually, in 2-D these form the "staircase"
    boundary along the upper-right edge of the point cloud.

    The algorithm is O(n^2) pairwise comparison: for each point, check
    whether any other (non-already-dominated) point dominates it.  This is
    efficient enough for the typical 10-50 nodes in an MCTS tree.

    Parameters:
        points: Array of shape ``(n_points, n_objectives)``.

    Returns:
        1-D boolean array of length *n_points* — ``True`` for points on
        the Pareto front, ``False`` for dominated points.
    """
    n = len(points)
    if n == 0:
        return np.array([], dtype=bool)

    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # b dominates a iff b >= a on all objectives and b > a on at least one
            if bool(np.all(points[j] >= points[i]) and np.any(points[j] > points[i])):
                is_dominated[i] = True
                break

    return ~is_dominated


def pareto_front_indices(points: NDArray[Any]) -> NDArray[Any]:
    """Return integer indices of non-dominated points (convenience wrapper).

    Identical to ``pareto_front`` but returns an integer array of positions
    (e.g. ``[0, 3, 7]``) instead of a boolean mask.  This is more convenient
    when you need to index into a parallel list of objects (e.g. the list of
    ``MCTSNode`` in ``_select_best``).
    """
    mask = pareto_front(points)
    return np.where(mask)[0]


# ---------------------------------------------------------------------------
# Hypervolume contribution
# ---------------------------------------------------------------------------


def hypervolume_contribution(
    points: NDArray[Any],
    reference: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Compute each point's *exclusive* hypervolume contribution to the set.

    The hypervolume contribution of point *i* is the volume of objective
    space that is dominated by *i* but **not** by any other point.  In other
    words: "how much objective coverage would we lose if we removed this
    point?"  Points with high HV contribution occupy unique, valuable
    regions of the trade-off surface.

    This is the primary ranking mechanism used by ``MCTSSearch._select_best``
    and ``pareto_select`` to break ties among Pareto-front nodes.

    **Dispatch by dimensionality:**
    - **2 objectives** → exact sweep-line (``_hypervolume_2d``), O(n^2 log n)
    - **3 objectives** → Monte Carlo with 20k samples (``_hypervolume_3d``)
    - **4+ objectives** → Monte Carlo with 10k samples (``_hypervolume_mc``)

    Points that do not strictly dominate the reference on all objectives
    are assigned zero contribution (they add no volume above the baseline).

    Parameters:
        points: shape ``(n_points, n_objectives)`` — objective vectors.
        reference: shape ``(n_objectives,)`` — the lower corner of the
            hypervolume box (worst acceptable performance).  Defaults to
            the origin ``[0, 0, ...]``.

    Returns:
        Array of shape ``(n_points,)`` — each entry is the exclusive
        hypervolume contribution of that point.
    """
    n_points, n_obj = points.shape
    if reference is None:
        reference = np.zeros(n_obj)

    # Edge cases
    if n_points == 0:
        return np.array([], dtype=np.float64)

    if n_points == 1:
        vol = np.prod(np.maximum(points[0] - reference, 0.0))
        return np.array([vol])

    # Filter out points that are not strictly better than the reference in
    # at least one objective (they contribute zero volume).
    valid = np.all(points > reference, axis=1)
    contributions = np.zeros(n_points, dtype=np.float64)

    if not np.any(valid):
        return contributions

    valid_points = points[valid]

    if n_obj == 2:
        valid_contributions = _hypervolume_2d(valid_points, reference)
    elif n_obj == 3:
        valid_contributions = _hypervolume_3d(valid_points, reference)
    else:
        valid_contributions = _hypervolume_mc(valid_points, reference)

    # Map back to original indices
    valid_indices = np.where(valid)[0]
    for i, idx in enumerate(valid_indices):
        contributions[idx] = valid_contributions[i]

    return contributions


# ---------------------------------------------------------------------------
# 2-D exact hypervolume contribution (sweep-line)
# ---------------------------------------------------------------------------


def _hypervolume_2d(points: NDArray[Any], reference: NDArray[Any]) -> NDArray[Any]:
    """Exact hypervolume contribution for 2 objectives using leave-one-out.

    For each point *i*, computes::

        contribution[i] = HV(all_points) - HV(all_points without i)

    where HV is the total 2-D hypervolume (area) computed by
    ``_compute_2d_hypervolume``.  This isolates the *exclusive* area that
    only point *i* covers — the "slice" of the staircase that disappears
    when *i* is removed.

    Complexity: O(n^2 log n) — calls the O(n log n) sweep-line *n* times.
    """
    n = len(points)
    contributions = np.zeros(n, dtype=np.float64)

    total_hv = _compute_2d_hypervolume(points, reference)

    for i in range(n):
        # HV without point i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        hv_without = _compute_2d_hypervolume(points[mask], reference)
        contributions[i] = total_hv - hv_without

    return contributions


def _compute_2d_hypervolume(points: NDArray[Any], reference: NDArray[Any]) -> float:
    """Compute the total 2-D hypervolume (area) dominated by a set of points.

    Uses a sweep-line algorithm: sort points by descending first-objective
    value, then sweep left-to-right accumulating rectangles.  Each point
    contributes a rectangle whose width is ``(x - ref_x)`` and whose height
    is the *new* portion of objective-1 it covers above the running maximum.

    Visually, this traces the "staircase" boundary from upper-left to
    lower-right and sums the area of each step above the reference point.

    Only points that strictly dominate the reference on both objectives
    contribute area.  Complexity: O(n log n) due to the sort.
    """
    if len(points) == 0:
        return 0.0
    # Filter to points dominating the reference
    valid = points[(points[:, 0] > reference[0]) & (points[:, 1] > reference[1])]
    if len(valid) == 0:
        return 0.0
    # Sort by first objective descending
    order = np.argsort(-valid[:, 0])
    sorted_pts = valid[order]
    hv = 0.0
    max_y = reference[1]
    for pt in sorted_pts:
        if pt[1] > max_y:
            hv += (pt[0] - reference[0]) * (pt[1] - max_y)
            max_y = pt[1]
    return hv


# ---------------------------------------------------------------------------
# 3-D exact hypervolume contribution (leave-one-out via MC)
# ---------------------------------------------------------------------------


def _hypervolume_3d(
    points: NDArray[Any],
    reference: NDArray[Any],
    n_samples: int = 20_000,
) -> NDArray[Any]:
    """Hypervolume contribution for 3-objective problems via Monte Carlo.

    Delegates to ``_hypervolume_mc`` with a higher sample count (20k vs 10k)
    to compensate for the increased variance that comes with an extra
    objective dimension.  Exact 3-D algorithms exist but the MC approach is
    simpler and accurate enough for the 10-50 point sets typical in MCTS.
    """
    return _hypervolume_mc(points, reference, n_samples=n_samples)


# ---------------------------------------------------------------------------
# General Monte Carlo hypervolume contribution
# ---------------------------------------------------------------------------


def _hypervolume_mc(
    points: NDArray[Any],
    reference: NDArray[Any],
    n_samples: int = 10_000,
) -> NDArray[Any]:
    """Estimate hypervolume contributions via Monte Carlo sampling.

    This is the general-purpose fallback for 3+ objectives where exact
    algorithms are complex.  The algorithm:

    1. Define a bounding box from ``reference`` (lower corner) to
       ``max(points)`` (upper corner) — the smallest box that contains
       all dominated space.
    2. Draw ``n_samples`` uniform random points within this box.
    3. For each sample, determine which solution points dominate it
       (a sample at ``s`` is dominated by point ``p`` if ``p[j] >= s[j]``
       for all objectives ``j``).
    4. A sample is *exclusively* dominated by point ``i`` if ``i`` dominates
       it but no other point does.
    5. The contribution of point ``i`` is::

           (exclusive_count[i] / n_samples) * box_volume

    The estimate is unbiased with variance decreasing as O(1/n_samples).
    A fixed seed (42) ensures reproducible results across runs.
    """
    n_points, n_obj = points.shape
    contributions = np.zeros(n_points, dtype=np.float64)

    upper = np.max(points, axis=0)
    lower = reference.copy()

    # If the bounding box has zero volume on any axis, contribution is 0
    box_dims = upper - lower
    if np.any(box_dims <= 0):
        return contributions

    rng = np.random.default_rng(42)
    samples = rng.uniform(lower, upper, size=(n_samples, n_obj))

    # Pre-compute which samples are dominated by each point.
    # dominated_by[i, s] is True if sample s is dominated by point i.
    dominated_by = np.ones((n_points, n_samples), dtype=bool)
    for obj in range(n_obj):
        dominated_by &= samples[:, obj][np.newaxis, :] <= points[:, obj][:, np.newaxis]

    for i in range(n_points):
        # Samples dominated by i but not by any other point
        others_mask = np.zeros(n_samples, dtype=bool)
        for j in range(n_points):
            if j == i:
                continue
            others_mask |= dominated_by[j]

        exclusive = dominated_by[i] & ~others_mask
        contributions[i] = exclusive.sum() / n_samples

    box_volume = float(np.prod(box_dims))
    contributions *= box_volume
    return contributions


# ---------------------------------------------------------------------------
# Pareto-based MCTS node selection
# ---------------------------------------------------------------------------


def pareto_select(
    nodes: Sequence[MCTSNode],
    exploration_constant: float = 1.414,
    rng: np.random.Generator | None = None,
) -> MCTSNode:
    """Select the best child node during MCTS tree traversal using Pareto-UCB.

    This is the multi-objective replacement for scalar UCB argmax.  Called by
    ``MCTSSearch._select`` at each level of the tree to decide which child
    to descend into.  The algorithm:

    1. **Unvisited priority** — If any child has ``visit_count == 0``, pick
       one at random.  This guarantees every child is tried at least once
       before revisiting (same as standard UCT).

    2. **UCB vectors** — For each visited child, compute a per-objective
       UCB1 score via ``MCTSNode.ucb_scores``.  This yields an array of
       shape ``(n_children, n_objectives)`` where each row balances
       exploitation (mean reward) and exploration (visit-count bonus).

    3. **Pareto front** — Find the non-dominated set among UCB vectors.
       These are the children where no other child is better on *all*
       objectives simultaneously.

    4. **Hypervolume tiebreak** — Among front nodes, select the one with
       the highest hypervolume contribution (most unique coverage of
       objective space).

    5. **Random fallback** — If all HV contributions are zero (e.g., all
       front nodes have identical UCB vectors), pick randomly.

    Parameters:
        nodes: Sibling child nodes to choose from.
        exploration_constant: The C parameter in the UCB1 formula, controlling
            the exploration-vs-exploitation balance (default √2 ≈ 1.414).
        rng: Generator for the random picks in steps 1 and 5.  ``None`` (the
            default) keeps the historical behaviour and draws from numpy's
            global RNG; in practice this default is unreachable from
            ``MCTSSearch.search``, which always passes a generator seeded per
            rollout (PR #28). Per-rollout seeding lets resumed runs replay
            the pre-crash selection path and find cached evaluations (issue #12).

    Returns:
        The single selected child node to descend into.
    """
    if len(nodes) == 1:
        return nodes[0]

    def _randint(n: int) -> int:
        return int(np.random.randint(n)) if rng is None else int(rng.integers(n))

    # Prioritize unvisited nodes
    unvisited = [n for n in nodes if n.visit_count == 0]
    if unvisited:
        return unvisited[_randint(len(unvisited))]

    # Compute UCB vectors for every child
    ucb_vectors = np.array([n.ucb_scores(exploration_constant) for n in nodes])

    # Find Pareto front among UCB vectors
    front_mask = pareto_front(ucb_vectors)
    front_indices = np.where(front_mask)[0]

    if len(front_indices) == 1:
        return nodes[front_indices[0]]  # type: ignore[no-any-return]

    # Among front nodes, rank by hypervolume contribution
    front_ucb = ucb_vectors[front_indices]
    contributions = hypervolume_contribution(front_ucb)

    # If all contributions are zero (e.g., identical points), pick randomly
    if np.all(contributions == 0):
        idx = _randint(len(front_indices))
        return nodes[front_indices[idx]]  # type: ignore[no-any-return]

    best_front_idx = int(np.argmax(contributions))
    return nodes[front_indices[best_front_idx]]  # type: ignore[no-any-return]
