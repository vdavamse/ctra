"""Multi-objective MCTS for feature set optimization.

Adapted from AutoCT's ``treesearch.py`` with the following modifications
for CTRA:

    - ``MCTSNode.total_reward``: ``float`` -> ``np.ndarray`` of shape ``(n_objectives,)``
      to support multi-objective evaluation via Pareto ranking.
    - Pareto UCT replaces scalar UCT argmax during tree policy.
    - AB-MCTS adaptive branching controls the expansion factor per node.
    - Feature value cache (cross-branch reuse) reduces redundant LLM calls.
    - ``_select_best()`` ranks each node's *own* evaluation by Pareto front
      then hypervolume contribution, instead of scalar max over subtree means.

Architecture:
    ``MCTSSearch`` accepts a **runner** callable
    ``(node_id, task, previous_output) -> AgentOutput`` that executes each
    evaluation as an isolated subprocess via ``run_agent_as_subprocess``.
    ``AgentOutput`` is stored on ``MCTSNode.eval_output`` and parent state
    flows through the tree structure (``node.parent.eval_output``).

The MCTS loop:
    1. **Select** -- traverse the tree using vector UCB + Pareto selection
    2. **Expand** -- generate child nodes from evaluator suggestions
    3. **Simulate** -- call runner to evaluate the feature set in subprocess
    4. **Backpropagate** -- update ancestor nodes with objective vectors
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ctra.agents.data_models import FeaturePlan
from ctra.agents.feature_store import plan_content_hash
from ctra.agents.runner import extract_objectives
from ctra.config.settings import MCTSConfig, get_settings
from ctra.search.objectives import ObjectiveResult
from ctra.search.pareto import (
    hypervolume_contribution,
    pareto_front_indices,
    pareto_select,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Absolute tolerance under which two scores count as a tie in final selection
# (issue #15): well above float noise, well below any ROC-AUC resolution.
_TIE_ATOL = 1e-9


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    """A single node in the MCTS feature-set search tree.

    Each node represents a specific feature configuration (set of feature
    names).  Rewards are stored as vectors of shape ``(n_objectives,)``
    where every objective is oriented so that higher = better.
    """

    features: list[str]
    operation: Literal["root", "add", "remove", "refine", "suggestion"] = "root"
    operation_detail: str = ""  # e.g., "add:prior_drug_approvals"

    # Which suggestion from the parent's eval_output this child follows
    suggestion_index: int = 0

    # Statistics ----------------------------------------------------------
    visit_count: int = 0
    total_reward: NDArray[Any] = field(
        default_factory=lambda: np.zeros(2)
    )  # shape: (n_objectives,)

    # History of all evaluations at this node (for diagnostics / re-ranking)
    objective_history: list[ObjectiveResult] = field(default_factory=list)

    # Tree structure ------------------------------------------------------
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)

    # Full evaluation output (AgentOutput from runner, set after simulation)
    eval_output: Any | None = None

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        """Check whether this node has no children (i.e. is at the frontier of the tree).

        Leaf nodes are candidates for expansion during the MCTS loop. A node
        starts as a leaf when created and stops being one once ``_expand``
        attaches children to it.
        """
        return len(self.children) == 0

    @property
    def mean_reward(self) -> NDArray[Any]:
        """Compute the average reward vector across all visits to this node.

        Returns an array of shape ``(n_objectives,)`` where each element is
        ``total_reward[j] / visit_count``.  For an unvisited node (visit_count
        == 0) returns zeros to avoid division-by-zero.  The mean reward is what
        UCB scoring consumes during the search.  It is deliberately *not* what
        ``_select_best`` ranks by — the final pick uses each node's own
        evaluations (``MCTSSearch._best_own_objectives``, issue #15).
        """
        if self.visit_count == 0:
            return np.zeros_like(self.total_reward)
        return self.total_reward / self.visit_count

    def ucb_scores(self, exploration_constant: float = 1.414) -> NDArray[Any]:
        """Compute the Upper Confidence Bound (UCB1) score for each objective.

        UCB1 balances exploitation (high mean reward) with exploration (rarely
        visited nodes).  For each objective j the score is::

            mean_reward[j]  +  C * sqrt( ln(parent_visits) / visit_count )

        The exploration term is the same scalar added to every objective
        dimension, so the resulting vector has shape ``(n_objectives,)``.

        When ``visit_count == 0`` every element is set to ``inf`` so that
        unvisited nodes are always selected first during the tree-policy
        traversal in ``_select``.

        This vector is consumed by ``pareto_select`` which performs
        non-dominated sorting over the UCB vectors of all siblings to decide
        which child to descend into.
        """
        if self.visit_count == 0:
            return np.full_like(self.total_reward, fill_value=float("inf"))

        parent_visits = self.parent.visit_count if self.parent else self.visit_count
        exploit = self.mean_reward
        explore = exploration_constant * np.sqrt(math.log(max(parent_visits, 1)) / self.visit_count)
        return exploit + explore  # type: ignore[no-any-return]

    def value(self, reference: NDArray[Any] | None = None) -> float:
        """Collapse the multi-objective mean reward into a single scalar for ranking.

        The scalar is the *hypervolume* of the hyper-rectangle whose corners
        are the reference point and the node's mean reward vector, i.e.::

            product( max(mean_reward[j] - reference[j], 0)  for j in objectives )

        A higher value means the node dominates more of the objective space.
        When ``n_objectives == 1`` this reduces to ``mean_reward[0] - ref[0]``
        (simple scalar difference).

        Args:
            reference: Lower-bound corner of the hypervolume rectangle. Defaults to the
                origin ``[0, 0, ...]``, **not** to the configured
                ``MCTSConfig.reference_point`` (``[0.5, 0.0]``, issue #18):
                production code never calls this method, so nothing passes the
                search's reference in.  A caller that wants the search's
                geometry must pass it explicitly.

        Reporting helper for external callers.  ``_select_best`` does not use
        it: it ranks nodes by their own evaluations, not by the subtree mean
        this collapses (issue #15).
        """
        mean = self.mean_reward
        if reference is None:
            reference = np.zeros_like(mean)

        # Hypervolume of a single point = product of (coord - ref) for
        # each objective (a hyper-rectangle).
        deltas = np.maximum(mean - reference, 0.0)
        return float(np.prod(deltas))


# ---------------------------------------------------------------------------
# MCTS Search
# ---------------------------------------------------------------------------


class MCTSSearch:
    """Multi-objective Monte Carlo Tree Search for feature set optimization.

    Args:
        runner: Callable ``(node_id: str, task: str|Task, previous_output: AgentOutput|None) -> AgentOutput``
            that evaluates a feature set. Default: ``run_agent_as_subprocess``
            which spawns each evaluation as an isolated subprocess.
        task: Task identifier forwarded to the runner (e.g. ``Task.TRIAL_OUTCOME_PHASE_2``
            or ``"phase2"``).
        expand_fn: Optional override for child generation. Default reads suggestions
            from ``node.eval_output`` to generate candidates.  Candidates beyond the
            parent's suggestion count are dropped when the parent output is an
            ``AgentOutput`` with suggestions (issue #7): each child's
            ``suggestion_index`` must index that list.
        config: MCTS configuration override.
    """

    def __init__(
        self,
        runner: Callable[..., Any],
        task: Any,
        *,
        expand_fn: Callable[..., list[tuple[list[str], str, str]]] | None = None,
        config: MCTSConfig | None = None,
    ) -> None:
        """Initialize the MCTS search engine.

        Stores the runner callable, task identifier, expansion function, and
        config.  Derives ``_n_objectives`` from the config's objective list
        and builds the ``_reference`` point array (padded/truncated to match
        the number of objectives) used for all hypervolume calculations.

        No tree is created here — the root node is built lazily in
        ``search()`` when ``start_rollout == 0``.
        """
        self._runner = runner
        self._task = task
        self._config = config or get_settings().mcts
        self._expand_fn = expand_fn or self._suggestion_expand
        self._root: MCTSNode | None = None
        self._all_nodes: list[MCTSNode] = []
        # Issue #7: nodes whose exhaustion skip has already been logged at INFO.
        # Read via ``getattr(self, "_skip_logged", None)`` in ``_log_skip`` so
        # checkpoints pickled before this attribute existed still resume.
        self._skip_logged: set[int] = set()
        # Issue #7: once-per-search INFO markers, also ``getattr``-guarded.
        self._frontier_logged = False
        self._truncation_logged = False

        # Resolve the number of objectives from config
        self._n_objectives = len(self._config.objectives)

        # Reference point for hypervolume
        ref = self._config.reference_point
        self._reference = np.array(ref[: self._n_objectives], dtype=np.float64)
        if len(self._reference) < self._n_objectives:
            self._reference = np.pad(
                self._reference,
                (0, self._n_objectives - len(self._reference)),
            )

    @property
    def root(self) -> MCTSNode | None:
        return self._root

    @property
    def all_nodes(self) -> list[MCTSNode]:
        return list(self._all_nodes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_runner(self, runner: Callable[..., Any]) -> None:
        """Replace the evaluation runner, e.g. after loading a checkpoint.

        A checkpoint pickles the runner the search was built with; a resumed
        process re-binds it here so evaluations hit *this* run's agent cache
        (``scripts/train_mcts.py``).  The tree is untouched.
        """
        self._runner = runner

    def search(
        self,
        initial_features: list[str],
        on_rollout: Any | None = None,
        start_rollout: int = 0,
    ) -> MCTSNode:
        """Run the full MCTS search loop.

        Args:
            initial_features: Starting feature set (from initial proposal).
            on_rollout: Optional callback ``(rollout_idx, node, objective_vector) -> None``
                called after each rollout's backpropagation. Useful for progress
                bars, logging, and checkpointing.  It is **not** invoked for a
                rollout that evaluated nothing (the selected node and its children
                were exhausted, issue #7) — the tree state is unchanged, so a
                periodic checkpoint keyed on the rollout index may land one
                interval later and a progress bar may end short of ``total``.
            start_rollout: Resume from this rollout index. When > 0, root evaluation is
                skipped (assumed restored from checkpoint).

        Returns:
            The best node found: among the nodes that were evaluated at least
            once, the one whose *own* best objective vector wins on Pareto rank,
            then hypervolume contribution (``_select_best``).  Its score is
            ``best_own_objectives(node)``; ``node.mean_reward`` is the subtree
            mean that UCB used during the search and is generally lower.
        """
        if start_rollout == 0:
            # Fresh start: create root and evaluate it
            self._root = MCTSNode(
                features=list(initial_features),
                total_reward=np.zeros(self._n_objectives),
            )
            self._all_nodes = [self._root]

            # Evaluate root
            root_obj = self._call_evaluate(self._root, rollout=0)
            # The root has no parent output, so it can never be exhausted.
            if root_obj is None:
                raise RuntimeError("root evaluation returned no objective")
            self._root.total_reward = root_obj.copy()
            self._root.visit_count = 1
            self._root.objective_history.append(self._wrap_result(root_obj, self._root.features))
        else:
            # Resuming: root and previous nodes already restored
            assert self._root is not None, "Cannot resume: no root node. Load checkpoint first."
            # Issue #7: the pickled dedupe set holds ``id()`` values from the
            # previous process.  A fresh node can collide with one of them and
            # have its first skip logged at DEBUG (a missing line), so start
            # clean — one INFO line per node per process is the contract.
            self._skip_logged = set()

        logger.info(
            "MCTS starting: %d initial features, %d rollouts planned (from %d)",
            len(initial_features),
            self._config.num_rollouts,
            start_rollout,
        )

        for rollout in range(start_rollout, self._config.num_rollouts):
            logger.info(
                "--- Rollout %d/%d ---",
                rollout + 1,
                self._config.num_rollouts,
            )

            # 1. SELECT — Pareto UCT traversal to a leaf.  Seeded per rollout
            # (like ``_simulate_deep``) so a resumed run replays the pre-crash
            # path and finds its cached evaluations (issue #12); an unseeded
            # pick of an unvisited child would hit only by chance.
            # SELECT gets its own stream: ``_simulate_deep`` seeds
            # ``default_rng(rollout)``, and sharing that state would rank-link
            # the two picks made in the same rollout.
            node = self._select(self._root, rng=np.random.default_rng([rollout, 1]))

            # 2. EXPAND — generate children from feature operations
            all_children_exhausted = False
            if node.visit_count > 0 and len(node.features) > 0:
                children = self._expand(node)
                if children:
                    # Issue #7 (R6): pick the first child that can still make
                    # progress.  With the expansion cap this is ``children[0]``
                    # in every current scenario — children are born capped in
                    # this same rollout — so the empty branch is unreachable
                    # today; it covers a predicate or expander change that
                    # could exhaust fresh children.  Re-evaluating the visited
                    # leaf instead would be a pointless subprocess, so the
                    # rollout ends with no evaluation.
                    viable = [c for c in children if not self._is_exhausted(c)]
                    if viable:
                        node = viable[0]
                    else:
                        logger.info(
                            "Rollout %d: every child of the selected leaf is exhausted (issue #7)",
                            rollout + 1,
                        )
                        all_children_exhausted = True

            # 3. SIMULATE + 4. BACKPROPAGATE
            try:
                if all_children_exhausted:
                    rollout_reward = None
                elif self._config.deep_simulation:
                    best_obj, _best_node = self._simulate_deep(node, rollout)
                    # Backpropagation is handled inside _simulate_deep
                    # for each node on the deep path.  ``None`` = the whole
                    # path was exhausted (issue #7): nothing was evaluated.
                    rollout_reward = best_obj
                else:
                    # Shallow mode: evaluate only this single node
                    obj_vector = self._call_evaluate(node, rollout=rollout)
                    if obj_vector is None:
                        # Issue #7 (R7): a skipped evaluation has no reward to
                        # record — no history entry, no backprop, no visit.
                        rollout_reward = None
                    else:
                        node.objective_history.append(self._wrap_result(obj_vector, node.features))
                        self._backpropagate(node, obj_vector)
                        rollout_reward = obj_vector
            except Exception:
                logger.warning(
                    "Rollout %d failed, skipping",
                    rollout + 1,
                    exc_info=True,
                )
                continue

            logger.info(
                "Rollout %d: node depth %d, %d features, total nodes %d",
                rollout + 1,
                self._node_depth(node),
                len(node.features),
                len(self._all_nodes),
            )

            # Per-rollout callback (for checkpointing, progress bars, etc.)
            if rollout_reward is None:
                # Issue #7 (R9): scripts/train_mcts.py indexes
                # objective_vector[0]/[1]; a skipped rollout has no vector to
                # hand it, so the callback (and its checkpoint) is skipped too.
                logger.debug(
                    "Rollout %d ended without an evaluation: the selected node and its "
                    "children are exhausted (issue #7)",
                    rollout + 1,
                )
                if not node.children and not getattr(self, "_frontier_logged", False):
                    # Nothing left to expand anywhere on this path.  Say so once;
                    # the remaining rollouts still run (``num_rollouts`` is a
                    # contract) but are no-ops until the frontier changes.
                    self._frontier_logged = True
                    logger.info(
                        "Rollout %d: search frontier is exhausted — the remaining %d "
                        "rollouts will be no-ops unless selection reaches an "
                        "unexhausted node (issue #7)",
                        rollout + 1,
                        self._config.num_rollouts - rollout - 1,
                    )
            elif on_rollout is not None:
                on_rollout(rollout, node, rollout_reward)

        # Return best node
        best = self._select_best()
        logger.info(
            "MCTS complete: best node has %d features, own objectives=%s (subtree mean=%s)",
            len(best.features),
            self.best_own_objectives(best).round(4),
            best.mean_reward.round(4),
        )
        return best

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode, rng: np.random.Generator | None = None) -> MCTSNode:
        """Phase 1 — SELECTION: walk down the tree from root to a leaf node.

        Starting at ``node`` (usually the root), repeatedly pick the best
        child via ``pareto_select`` — which computes UCB1 vectors for every
        sibling, performs non-dominated sorting, and returns a winner — until
        either a leaf is reached or ``max_depth`` is hit.

        The returned leaf is the node where the next expansion + simulation
        will happen.  Because UCB scores give ``inf`` to unvisited children,
        any child that has never been evaluated will be selected before
        revisiting an already-explored child.  ``rng`` drives the random
        picks among unvisited children (and the zero-contribution fallback);
        ``search()`` seeds it with the rollout index so the traversal is
        reproducible across a resume.  ``None`` uses numpy's global RNG.
        """
        current = node
        depth = 0

        while not current.is_leaf and depth < self._config.max_depth:
            current = pareto_select(
                current.children,
                exploration_constant=self._config.exploration_constant,
                rng=rng,
            )
            depth += 1

        return current

    def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        """Phase 2 — EXPANSION: create child nodes representing new feature-set variants.

        Given a leaf node that has already been evaluated, this method
        generates a set of candidate child configurations by calling
        ``_expand_fn`` (by default ``_suggestion_expand``, which reads the
        evaluator's suggestions from the node's ``eval_output``).

        **AB-MCTS adaptive branching** — When ``adaptive_branching`` is
        enabled in config, the number of children (branch factor) scales
        logarithmically with the node's visit count::

            branch_factor = clamp( floor(log2(visit_count)) + 2,
                                   min_branch_factor, max_branch_factor )

        This means frequently visited (promising) nodes get more children
        to explore, while rarely visited nodes stay narrow, saving budget.

        Each child ``MCTSNode`` is wired into the tree (``parent`` pointer)
        and registered in ``_all_nodes`` so it can later be found by
        ``_select_best``.

        Returns the list of newly created children (empty if the expand
        function produced no candidates).
        """
        if self._config.adaptive_branching:
            branch_factor = min(
                self._config.max_branch_factor,
                max(
                    self._config.min_branch_factor,
                    int(math.log2(max(node.visit_count, 1)) + 2),
                ),
            )
        else:
            branch_factor = self._config.max_branch_factor

        candidates = self._expand_fn(node, max_children=branch_factor)

        # Issue #7 (R2): ``suggestion_index=i`` below indexes the *parent's*
        # suggestion list, but a custom ``expand_fn`` knows nothing about that
        # list and returns up to ``branch_factor`` candidates
        # (``mlops/retraining.py:156-171`` and ten test modules do exactly
        # this).  Cap the candidate count so no child is born past the end.
        # The default ``_suggestion_expand`` already slices
        # ``suggestions[:max_children]``, so this is a no-op for it.
        n_suggestions = self._suggestion_count(node)
        if n_suggestions is not None and len(candidates) > n_suggestions:
            msg = "Expansion truncated to the suggestion list: %d candidates -> %d (issue #7)"
            if getattr(self, "_truncation_logged", False):
                logger.debug(msg, len(candidates), n_suggestions)
            else:
                self._truncation_logged = True
                logger.warning(msg, len(candidates), n_suggestions)
            candidates = candidates[:n_suggestions]

        children: list[MCTSNode] = []
        for i, (features, operation, detail) in enumerate(candidates):
            child = MCTSNode(
                features=features,
                operation=operation,  # type: ignore[arg-type]
                operation_detail=detail,
                suggestion_index=i,
                parent=node,
                total_reward=np.zeros(self._n_objectives),
            )
            children.append(child)
            self._all_nodes.append(child)

        node.children.extend(children)

        logger.debug(
            "Expanded node (%d features) -> %d children (branch_factor=%d)",
            len(node.features),
            len(children),
            branch_factor,
        )
        return children

    def _simulate_deep(
        self, start_node: MCTSNode, rollout: int
    ) -> tuple[NDArray[Any] | None, MCTSNode]:
        """Phase 3 (deep mode) — SIMULATION: run a full rollout from start_node down to max_depth.

        This is the AutoCT-style "deep simulation" used when
        ``config.deep_simulation`` is True.  Instead of evaluating only the
        single selected leaf (shallow mode), it chains expand-evaluate steps
        to walk deeper into the tree:

        1. Evaluate ``start_node`` and backpropagate its reward.
        2. Expand ``start_node`` to create children.
        3. Pick a random *unvisited* child (or any random child if all have
           been visited).
        4. Evaluate that child, backpropagate, track the best-so-far.
        5. Repeat from step 2 with the chosen child until ``max_depth`` is
           reached or expansion yields no candidates.

        Every node along this path is fully integrated into the tree (stored
        in ``_all_nodes``, back-propagated, available for future selection).
        This is more expensive per rollout but explores deeper feature
        combinations faster.

        The "best node" on the path is determined by hypervolume contribution
        — the node whose objective vector covers the largest hyper-rectangle
        above the reference point.

        Returns:
            Tuple of (best_obj, best_node) where:
                - best_obj: Objective vector of the best node found on the path,
                  or ``None`` when nothing on the path could be evaluated
                  because every candidate was exhausted (issue #7).
                - best_node: The corresponding tree node (``start_node`` when
                  ``best_obj`` is ``None``).
        """
        rng = np.random.default_rng(rollout)
        current = start_node

        # NOTE: on rollout 0 ``start_node`` is the root, which ``search()`` has
        # already evaluated.  In the normal path it is not evaluated twice:
        # ``_select`` descends to a leaf and ``search()`` moves to a freshly
        # expanded child before calling here.  When expansion yields no children
        # (or ``features`` is empty) the root does reach this line and is
        # evaluated a second time.  See test_root_visit_count_not_inflated.

        # Evaluate start node and backpropagate
        start_obj = self._call_evaluate(current, rollout=rollout)
        best_node = current
        if start_obj is None:
            # Issue #7: nothing was evaluated — no history entry, no backprop,
            # no visit increment (R7).  A later child on this path may still win.
            best_obj: NDArray[Any] | None = None
            best_hv = -np.inf
            nodes_evaluated = 0
        else:
            current.objective_history.append(self._wrap_result(start_obj, current.features))
            self._backpropagate(current, start_obj)
            best_obj = start_obj
            best_hv = self._point_hypervolume(start_obj)
            nodes_evaluated = 1

        if start_obj is None and current.eval_output is None:
            # Never evaluated: nothing to expand from.  Expanding here would
            # hand every child ``previous_output=None`` — a root-style fresh
            # initialization mid-tree (a full initializer subprocess on the
            # real pipeline).  A skipped node that *was* evaluated earlier
            # still carries its own output and may expand below.
            return None, current

        depth = self._node_depth(current)

        while depth < self._config.max_depth:
            # Expand current node if not already expanded
            if current.is_leaf and len(current.features) > 0:
                children = self._expand(current)
                if not children:
                    break
            elif not current.children:
                break

            # Issue #7 (decision 2): a child whose suggestion_index is past the
            # parent's suggestions can only replay a rejected suggestion.
            # Filter before picking, so the rng is only ever asked about
            # children that can make progress; if nothing is left, the path
            # stops here rather than burning a subprocess.
            viable = [c for c in current.children if not self._is_exhausted(c)]
            if not viable:
                logger.debug(
                    "Deep rollout stopped at depth %d: all children exhausted (issue #7)",
                    depth,
                )
                break

            # Pick a random unvisited child (AutoCT's simulation policy)
            unvisited = [c for c in viable if c.visit_count == 0]
            if unvisited:
                child = unvisited[rng.integers(len(unvisited))]
            else:
                # All children visited — pick random (exploration)
                child = viable[rng.integers(len(viable))]

            # Evaluate the child and backpropagate
            child_obj = self._call_evaluate(child, rollout=rollout)
            if child_obj is None:
                # Defensive only.  ``_is_exhausted`` reads the same parent output
                # ``_call_evaluate`` builds, so a child filtered as viable is
                # not exhausted at this point; this branch exists so a future
                # change cannot turn a skip into a phantom reward.  Because it
                # masks the pre-pick filter, that filter has its own test
                # (``test_deep_simulation_filters_exhausted_children_before_picking``).
                break
            child.objective_history.append(self._wrap_result(child_obj, child.features))
            self._backpropagate(child, child_obj)
            nodes_evaluated += 1

            # Track best on path by hypervolume
            child_hv = self._point_hypervolume(child_obj)
            if child_hv > best_hv:
                best_hv = child_hv
                best_obj = child_obj
                best_node = child

            current = child
            depth += 1

        logger.debug(
            "Deep rollout: evaluated %d nodes to depth %d, best HV=%.4f at depth %d",
            nodes_evaluated,
            depth,
            best_hv,
            self._node_depth(best_node),
        )
        return best_obj, best_node

    def _point_hypervolume(self, objectives: NDArray[Any]) -> float:
        """Compute the hypervolume dominated by a single objective vector.

        Measures how much of the objective space this point "covers" above
        the reference point by computing the volume of the axis-aligned
        hyper-rectangle between ``self._reference`` and ``objectives``::

            product( max(objectives[j] - reference[j], 0)  for j in range(n) )

        Negative deltas are clamped to 0 (a point below the reference on
        any axis contributes zero volume).

        Used inside ``_simulate_deep`` to compare nodes along a deep rollout
        path and pick the best intermediate result.
        """
        deltas = np.maximum(objectives - self._reference, 0.0)
        return float(np.prod(deltas))

    @staticmethod
    def _node_depth(node: MCTSNode) -> int:
        """Count the number of edges from this node back to the root (root = depth 0).

        Walks the ``parent`` chain upward.  Used by ``_select`` to enforce
        ``max_depth``, by ``_simulate_deep`` to know when to stop descending,
        and in log messages for diagnostics.
        """
        d = 0
        current = node
        while current.parent is not None:
            d += 1
            current = current.parent
        return d

    def _backpropagate(self, node: MCTSNode, objectives: NDArray[Any]) -> None:
        """Phase 4 — BACKPROPAGATION: propagate a simulation result up to the root.

        Starting at the just-evaluated ``node``, walk the ``parent`` chain
        all the way to the root, and at every ancestor:

        1. Increment ``visit_count`` by 1.
        2. Add the objective vector element-wise to ``total_reward``.

        This ensures that ``mean_reward`` (total_reward / visit_count)
        reflects the average performance of *all* descendants, which is what
        the UCB formula relies on during future selection phases.
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visit_count += 1
            current.total_reward = current.total_reward + objectives
            current = current.parent

    def _best_own_objectives(self, node: MCTSNode) -> NDArray[Any] | None:
        """The best objective vector this node scored *itself*, or ``None``.

        ``objective_history`` holds one entry per evaluation of *this* feature
        set (appended in ``search()`` and ``_simulate_deep`` right after
        ``_call_evaluate`` returns a vector).  It is empty for a node that was
        never evaluated — one expanded but never simulated, or one whose
        suggestion was exhausted (issue #7) — and such a node is not a
        selectable result.

        A node can hold several entries: a deep rollout may re-pick an already
        visited child, and an unexpandable leaf can be re-evaluated in shallow
        mode.  Each entry snapshots the feature set it scored
        (``details["features"]``, written by ``_wrap_result``).  A
        re-evaluation that changed the plans rewrites ``node.features``
        (``_call_evaluate`` step 5), and an entry scored for an earlier set is
        skipped here, so the node is ranked by the set it now holds — the one
        its ``eval_output`` (and so ``feature_plans.json`` and
        ``best_model.pkl``) describes.  An entry without a snapshot (a
        checkpoint written before it was recorded) counts unconditionally, as
        every entry did before.  The entries that remain usually share a
        parsimony (it is a function of ``len(node.features)``) and differ in
        accuracy, so "best" is decided by
        ``(self._point_hypervolume(v), v[0])`` — the file's one notion of
        vector quality, with accuracy as the tie-break.  The tie-break is
        load-bearing, not a corner case: the reference point sits at the
        ROC-AUC chance baseline (``[0.5, 0.0]``, issue #18), so every entry
        of a below-chance feature set has hypervolume 0 and only its
        accuracy ranks it against the node's other entries.

        Returns a *copy*, so a caller cannot mutate the stored history.

        Entries with a non-finite component (a NaN or infinite score) are
        skipped: NaN compares false against everything, so such an entry
        would otherwise stick as the "best" one and poison the ranking.  A
        node whose every entry is non-finite has no usable score and is
        treated like one with no history.

        All entries are assumed to have ``n_objectives`` components — that is
        what ``_call_evaluate`` pads/truncates to before ``_wrap_result``.
        """
        best: NDArray[Any] | None = None
        best_key: tuple[float, float] | None = None
        current = list(node.features)
        for result in node.objective_history:
            snapshot = result.details.get("features")
            if snapshot is not None and list(snapshot) != current:
                continue
            values = np.asarray(result.values, dtype=float)
            if not np.all(np.isfinite(values)):
                continue
            key = (self._point_hypervolume(values), float(values[0]))
            if best_key is None or key > best_key:
                best_key = key
                best = values
        return None if best is None else best.copy()

    def best_own_objectives(self, node: MCTSNode) -> NDArray[Any]:
        """The objective vector ``_select_best`` ranked ``node`` by.

        This is the node's own best evaluation, as opposed to
        ``node.mean_reward``, which is the average over the node's whole
        subtree and is what UCB consumes during the search (issue #15).
        Callers that report "the best feature set scored X" want this one;
        ``scripts/train_mcts.py`` writes it to ``results.json["best_objectives"]``
        and keeps the subtree mean beside it as ``best_mean_objectives``.

        It is the score of ``node.features``: every history entry records the
        feature set it scored, and ``_best_own_objectives`` ranks only the
        entries matching the node's current set, so a re-evaluation that
        changed the plans (``_call_evaluate`` rewrites ``node.features`` from
        the new ``feature_plans``) cannot leave the reported score belonging
        to an earlier set than the ``eval_output`` the node now carries.  The
        one exception is a checkpoint written before the snapshot existed:
        its entries carry no set and all count.

        Never returns ``None``: a node with no evaluation of its own (never
        simulated, or restored from a checkpoint written before histories were
        kept) falls back to ``node.mean_reward``, which is zeros for an
        unvisited node.  Such a node is never returned by ``_select_best``
        unless it is the root fallback.
        """
        own = self._best_own_objectives(node)
        if own is not None:
            return own
        return node.mean_reward

    @staticmethod
    def _break_ties(candidates: list[MCTSNode], indices: NDArray[Any]) -> MCTSNode:
        """Pick one node out of an equally-ranked set, deterministically.

        Prefers the smaller feature set (AutoCT's parsimony bias, and the
        tie-break that matters when every candidate sits below the reference
        point and hypervolume contributions are all zero), then the node that
        entered ``_all_nodes`` first — which is the shallower, earlier-created
        one, and makes the result stable across runs.
        """
        winner = min(indices, key=lambda i: (len(candidates[int(i)].features), int(i)))
        return candidates[int(winner)]

    def _select_best(self) -> MCTSNode:
        """After all rollouts, pick the single best node from the entire tree.

        This is called once at the end of ``search()`` to produce the final
        result.  Nodes are ranked by their **own** evaluations
        (``_best_own_objectives``), not by ``mean_reward``: the mean is the
        average over a node's subtree, so expanding the best feature set and
        evaluating mediocre children *lowers* its rank and the winning set
        could be discarded in favour of a worse, unexpanded one (issue #15).
        AutoCT selects on the node's own reward and so does CTRA's replica of
        it in ``tests/test_search/test_autoct_mcts_comparison.py``.
        ``mean_reward`` is untouched and still drives UCB during the search.

        Candidates are the nodes with at least one entry in
        ``objective_history`` (the root included).  A node that was expanded
        but never simulated, or whose evaluation was skipped as exhausted
        (issue #7), has no score of its own and cannot be the answer, even
        though backpropagation may have given it a ``visit_count``.

        **Single-objective** (fast path): the highest own accuracy.

        **Multi-objective** (2+ objectives):
        1. Build the matrix of per-node best own vectors.
        2. Compute the Pareto front — the set of nodes that are not
           dominated by any other node on all objectives simultaneously.
        3. Among Pareto-front nodes, rank by *hypervolume contribution*
           (how much each point uniquely adds to the front's total
           hypervolume).  Return the one with the largest contribution,
           as it represents the most valuable trade-off point.

        Ties at either step are resolved by ``_break_ties`` (fewer features,
        then earliest in ``_all_nodes``); a front whose every contribution is
        0 — no candidate above the reference point — is logged at WARNING
        first.  A tie is decided with
        ``np.isclose`` at an absolute tolerance of ``_TIE_ATOL`` (1e-9): two
        contributions (or accuracies) that are equal on paper can differ by a
        few ULPs in floating point, and exact equality would hand the pick to
        rounding noise instead of the tie-break.  The tolerance is far below
        any ROC-AUC resolution (one ordered pair on a 500/500 split is 4e-6),
        so a measurably better set is never tied away.  Falls
        back to the root when no node has been evaluated at all (edge case).
        """
        candidates: list[MCTSNode] = []
        own_points: list[NDArray[Any]] = []
        for node in self._all_nodes:
            own = self._best_own_objectives(node)
            if own is not None:
                candidates.append(node)
                own_points.append(own)

        if not candidates:
            assert self._root is not None
            return self._root

        points = np.array(own_points)
        # Every candidate vector is finite (``_best_own_objectives`` drops the
        # rest), so the maximum exists and the maximiser set is non-empty.
        accuracy = points[:, 0]
        scalar_max = np.flatnonzero(np.isclose(accuracy, accuracy.max(), rtol=0.0, atol=_TIE_ATOL))

        # Single-objective fast path: just pick the max
        if self._n_objectives == 1:
            return self._break_ties(candidates, scalar_max)

        # Multi-objective: Pareto front -> hypervolume contribution ranking
        front_idx = pareto_front_indices(points)

        if len(front_idx) == 0:
            # Shouldn't happen, but fall back to scalar objective 0
            return self._break_ties(candidates, scalar_max)

        if len(front_idx) == 1:
            return candidates[int(front_idx[0])]

        # 2. Among front nodes, rank by hypervolume contribution.  All-zero
        # contributions (every point below the reference point) are a tie, so
        # the answer stays deterministic instead of falling to ``argmax``'s
        # first index.  Say so: under the production reference (accuracy at
        # the ROC-AUC chance baseline, issue #18) it means no feature set
        # scored above chance, which the operator should know.
        contributions = self._front_contributions(points[front_idx])
        if not np.any(contributions > 0.0):
            logger.warning(
                "No Pareto-front candidate scores above the reference point %s: "
                "all %d hypervolume contributions are 0, so the smallest feature "
                "set is selected",
                self._reference.tolist(),
                len(front_idx),
            )
        winners = front_idx[
            np.isclose(contributions, contributions.max(), rtol=0.0, atol=_TIE_ATOL)
        ]
        return self._break_ties(candidates, winners)

    def _front_contributions(self, front_points: NDArray[Any]) -> NDArray[Any]:
        """Hypervolume contribution of every front point, immune to duplicates.

        ``hypervolume_contribution`` is leave-one-out: two points at the same
        coordinates each cover nothing the other does not, so both score 0.
        Ranking by *own* evaluations makes coincident points ordinary — the
        same feature set is reached by several orderings of the same ADDs, a
        re-picked node repeats its vector — and a feature set the search
        rediscovered would be ranked *below* one it found once.  (This is why
        feeding every entry of ``objective_history`` in as its own candidate
        was rejected: near-duplicates from one node cancel each other.)

        So the contribution is computed over the *distinct* points and
        broadcast back to every front position holding that point; the nodes
        that share the winning point are then separated by ``_break_ties``.

        "Distinct" is decided by tolerance, not by exact equality: two
        evaluations of one feature set can differ at 1e-13 (float noise in a
        metric), and ``np.unique`` on the raw values would keep both, each
        with a near-zero exclusive contribution — the cancellation this method
        exists to prevent.  Rounding to a decimal grid is not enough either
        (twins straddling a grid boundary stay distinct), so the front is
        sorted lexicographically and consecutive rows within ``_TIE_ATOL`` of
        each other are merged into one representative; the representatives
        are what gets scored.  The tolerance is far below the resolution of
        any ROC-AUC or parsimony score, so genuinely distinct points rank
        exactly as before.
        """
        order = np.lexsort(front_points.T[::-1])
        group_of_row = np.empty(len(front_points), dtype=int)
        representatives: list[NDArray[Any]] = []
        for row in order:
            point = front_points[row]
            if representatives and np.allclose(
                representatives[-1], point, rtol=0.0, atol=_TIE_ATOL
            ):
                group_of_row[row] = len(representatives) - 1
            else:
                representatives.append(point)
                group_of_row[row] = len(representatives) - 1
        contributions = hypervolume_contribution(np.array(representatives), self._reference)
        return np.asarray(contributions[group_of_row])

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _suggestion_count(node: MCTSNode) -> int | None:
        """How many suggestions this node's own ``eval_output`` offers.

        Returns ``None`` when the question does not apply: no output yet, a
        runner result that is not an ``AgentOutput`` (``mlops/retraining.py``
        returns an ``ObjectiveResult``), or an empty suggestion list.  Empty is
        deliberately *not* zero — ``FeatureProposer`` degrades to an empty
        suggestion on purpose and ``AgentOutput.suggestions_exhausted`` treats
        empty as "not exhausted" (``data_models.py``); returning ``None`` keeps
        the two predicates agreeing.
        """
        output = node.eval_output
        if output is None or not hasattr(output, "get_best_eval_output"):
            return None
        try:
            best_eval, _ = output.get_best_eval_output()
        except (ValueError, KeyError, AttributeError):
            return None
        suggestions = getattr(best_eval, "suggestions", None)
        return len(suggestions) if suggestions else None

    @staticmethod
    def _output_exhausted(parent_output: Any) -> bool:
        """Issue #7 predicate, evaluated on an output that already carries the
        child's ``suggestion_index``.  Non-``AgentOutput`` runner results have
        no such property and are never exhausted."""
        # ``is True`` rather than ``bool(...)``: a Mock (or any truthy stand-in)
        # standing in for the parent output must not be silently skipped.
        return getattr(parent_output, "suggestions_exhausted", False) is True

    def _is_exhausted(self, node: MCTSNode) -> bool:
        """Would evaluating ``node`` replay a suggestion past the end of the list?

        Derived only (R8): builds the same probe ``_call_evaluate`` would build
        and asks it.  The root (no parent, or a parent with no output) is never
        exhausted.  ``_replace`` here is ``dataclasses.replace`` — it aliases
        the mutable fields rather than copying them, so the probe is cheap and
        must not be mutated.
        """
        parent = node.parent
        if parent is None or parent.eval_output is None:
            return False
        parent_output = parent.eval_output
        if not hasattr(parent_output, "_replace"):
            return False
        try:
            probe = parent_output._replace(suggestion_index=node.suggestion_index)
        except (TypeError, ValueError):
            return False
        return self._output_exhausted(probe)

    def _call_evaluate(self, node: MCTSNode, rollout: int) -> NDArray[Any] | None:
        """Invoke the external runner to evaluate a node's feature set and extract objective scores.

        This is the bridge between the MCTS tree and the agent pipeline.  The
        steps are:

        1. **Build parent context** — If this node has a parent with a stored
           ``eval_output``, pass it to the runner so the agent pipeline knows
           what was tried before.  The ``suggestion_index`` on the parent
           output tells the proposer agent *which* suggestion to follow for
           this child.  For the root node, ``parent_output`` is None and the
           runner starts a fresh initialization.

        2. **Skip exhausted nodes** (issue #7) — If that parent output, now
           carrying this child's ``suggestion_index``, reports
           ``suggestions_exhausted``, the runner could only replay a
           suggestion the proposer already rejected.  Return ``None`` without
           calling it.  The root is never exhausted (no parent output).

        3. **Call the runner** — ``self._runner(node_id, task, parent_output)``
           spawns a subprocess (by default) that runs the full agent pipeline
           (initializer -> proposer -> planner -> builder -> grouper ->
           evaluator) and returns an ``AgentOutput`` containing feature plans,
           built features, and evaluation metrics.

        4. **Store output** — The ``AgentOutput`` is saved on
           ``node.eval_output`` so future children can use it as context and
           so ``_suggestion_expand`` can read the evaluator's suggestions.

           The output's own ``suggestion_index`` is written back to the node
           **only when it advances** past the one sent (issue #14).  A skipped
           iteration returns ``sent + 1`` and the node must take it, or the
           next rollout replays the dead suggestion; a *successful* iteration
           returns a hard-coded ``0`` that says nothing about which suggestion
           this node followed, and adopting it wiped the index assigned at
           expansion — every re-evaluation then asked the proposer for
           suggestion 0.  The counter is therefore monotonic: what a skip
           burned stays burned, which is the semantics
           ``suggestions_exhausted`` — and the guard in step 2 — is defined
           against.

        5. **Sync features** — If the subprocess produced different features
           than expected (the proposer may modify the set), update
           ``node.features`` to reflect reality.

        6. **Extract objectives** — Pull the objective scores (e.g. ROC-AUC,
           parsimony) from the output, pad or truncate to ``n_objectives``,
           and return as an ndarray.

        Returns:
            An ndarray of shape ``(n_objectives,)`` containing the objective scores for
            this node, ready for backpropagation — or ``None`` when the evaluation
            was skipped because the node's suggestion is exhausted.  Callers must
            treat ``None`` as "no evaluation happened": nothing to append to
            ``objective_history``, backpropagate, or hand to ``on_rollout``.
        """
        # Build parent output with correct suggestion_index.
        # For root (no parent), pass None — the runner/subprocess handles
        # iteration 0 (initializer) when previous_output is None.
        parent_output = None
        if node.parent is not None and node.parent.eval_output is not None:
            parent_output = node.parent.eval_output
            # Set suggestion_index so the subprocess proposer follows
            # the correct suggestion for this child
            if hasattr(parent_output, "_replace"):
                parent_output = parent_output._replace(
                    suggestion_index=node.suggestion_index,
                )

        # Issue #7: the parent's output, carrying *this child's*
        # suggestion_index, is past the end of the evaluator's suggestions.
        # ``get_next_suggestion`` would clamp and hand the proposer a suggestion
        # already tried and rejected (``data_models.py``), the orchestrator would
        # early-skip it (``orchestrator.py``) and return the input unchanged — a
        # full subprocess round-trip (dill-serialised AgentOutput incl. pipelines
        # and DataFrames) for a guaranteed no-op, whose duplicate objective
        # vector would then be backpropagated.  Skip it: ``None`` means "no
        # evaluation happened".
        if parent_output is not None and self._output_exhausted(parent_output):
            self._log_skip(node)
            return None

        # What is sent is captured before the runner call: it is the value
        # ``parent_output`` carries and the baseline the write-back below compares against.
        sent = self._as_index(node.suggestion_index)
        node_id = self._make_node_id(node, rollout, parent_output)
        output = self._runner(node_id, self._task, parent_output)

        # Store full AgentOutput on the node
        node.eval_output = output

        # Honour a suggestion_index *advance* from the orchestrator (issue #14).
        # ``Agent.forward`` returns three shapes, and only the first one carries
        # a counter the node should adopt:
        #   - skipped iteration (proposer failure ``orchestrator.py:396-399``,
        #     unhandled operation ``:522-525``) -> ``sent + 1``: the suggestion was
        #     consumed and must not be replayed, so take it.
        #   - successful iteration (``:619-633``) -> a hard-coded ``0`` that says
        #     nothing about which suggestion this node followed.  Taking it reset
        #     the child's expansion-assigned index and made every re-evaluation
        #     replay suggestion 0 (issue #14).
        #   - exhausted early-skip (``:370``) -> the input index unchanged (no
        #     suggestion was consumed), so there is nothing to adopt.
        # ``returned > sent`` separates them without a schema change: indices are
        # non-negative, so a success (0) never advances and the early-skip is equal.
        # Defensive about stand-in runners (R5: this runs outside ``search()``'s
        # rollout try/except for the root): a Mock, a missing attribute or a bool
        # leaves the node's int alone rather than raising or poisoning the counter.
        # ``_as_index`` rejects anything that is not a builtin or numpy integer
        # (``Mock``, ``Mock(spec=int)``, ``bool``, ``str``, a missing attribute).
        returned = self._as_index(getattr(output, "suggestion_index", None))
        if sent is not None and returned is not None and returned > sent:
            if returned > sent + 1:
                logger.warning(
                    "Runner advanced suggestion_index by %d (from %d to %d) for "
                    "node %r; Agent.forward only ever advances by one",
                    returned - sent,
                    sent,
                    returned,
                    node.operation_detail,
                )
            node.suggestion_index = returned

        # Update node features to match actual output (subprocess may
        # produce different features via the proposer). Skip if the output
        # mirrors the parent (e.g., in stub runners for testing).
        if hasattr(output, "feature_plans") and output.feature_plans:
            actual_features = list(output.feature_plans.keys())
            parent_features = (
                list(node.parent.eval_output.feature_plans.keys())
                if node.parent
                and hasattr(node.parent.eval_output, "feature_plans")
                and node.parent.eval_output is not None
                else None
            )
            if actual_features != parent_features:
                node.features = actual_features

        # Extract objectives, truncate/pad to n_objectives
        obj = extract_objectives(
            output,
            n_features=len(node.features),
            max_features=self._config.max_features,
        )
        if len(obj) < self._n_objectives:
            obj = np.pad(obj, (0, self._n_objectives - len(obj)))
        return obj[: self._n_objectives]

    def _log_skip(self, node: MCTSNode) -> None:
        """Log an exhaustion skip: INFO the first time for a given node, DEBUG after.

        The dedupe set lives on the search object, not the node: no new
        ``MCTSNode`` state (R8) means dill checkpoints stay round-trippable
        (issue #14 likewise added no node state).  ``getattr`` with a default
        is what lets a checkpoint pickled *before* this change resume —
        it unpickles without ``_skip_logged``.  ``id(node)`` is stable for the
        lifetime of a process and meaningless across a resume — a pickled set
        would carry stale ids that a fresh node can collide with, silencing its
        first INFO line — so ``search()`` resets the set on resume.
        """
        logged = getattr(self, "_skip_logged", None)
        if logged is None:
            logged = self._skip_logged = set()
        msg = (
            "Skipping evaluation of node %r whose suggestion_index=%d is past its "
            "parent's suggestions — the runner would replay a rejected suggestion "
            "(issue #7)"
        )
        detail = getattr(node, "operation_detail", "") or "<no detail>"
        if id(node) in logged:
            logger.debug(msg, detail, node.suggestion_index)
        else:
            logged.add(id(node))
            logger.info(msg, detail, node.suggestion_index)

    def _make_node_id(self, node: MCTSNode, rollout: int, parent_output: Any) -> str:
        """Name this evaluation for the runner's cache, e.g. ``"r3-9f2c...e1"``.

        ``run_agent_as_subprocess`` stores each output under
        ``{task}--{node_id}.output.pkl`` and returns the pickle on a repeated
        id without running the agent.  That cache is **crash recovery**, not
        memoisation: a resumed run re-executes the rollouts after its last
        checkpoint and must find their pre-crash outputs, while anything that
        is not literally the same evaluation must miss.  The id is therefore
        a digest of what defines the evaluation (issue #12):

        - ``rollout``: re-evaluating a node in a later rollout is intentional
          exploration of a stochastic pipeline (deep-mode revisits), never a
          replay.
        - ``lineage``: the child-position path from the root (``_lineage``).
          Its length is the depth, which separates a child from a parent
          whose plans it repeats verbatim (the orchestrator's proposer-failure
          skip and the exhausted early-skip both return the input plans);
          its last element is the sibling index *assigned at expansion*, so
          it is immune to the ``suggestion_index`` write-back at the call
          site (issue #14) and to numpy integer indices; and cousins whose
          parents both returned content-identical plans differ structurally
          (``[0, 0]`` vs ``[1, 0]``) instead of colliding on
          ``(depth, index)``.
        - parent plan digest: the plan *content* the child is built from.  A
          REFINE keeps the feature name and changes only ``feature_idea``, so
          the feature-name set alone cannot tell a refined child from its
          parent.
        - ``features``: sorted, so insertion order is irrelevant.

        SHA-256 over the canonical JSON keeps the id byte-identical across
        processes (issue #13: builtin ``hash()`` is salted per process, and its
        negative values formatted to ``r3--...`` filenames).  Because the key is
        stable across *runs* as well, the cache directory must be per run (as
        ``scripts/train_mcts.py`` does with ``<output_dir>/agent_cache``): a
        shared directory would replay results across unrelated runs, starting
        with the root evaluation at rollout 0.  Runner stand-ins
        (``Mock``, plain objects, outputs without plans, detached nodes)
        never raise: in later rollouts a raise here would land in
        ``search()``'s rollout ``try``/``except`` and skip the rollout
        silently (R9); the rollout-0 root evaluation sits outside that
        ``try`` and would abort the search.
        """
        payload = {
            "rollout": int(rollout),
            "lineage": self._lineage(node),
            "parent": self._parent_plan_digest(parent_output),
            "features": sorted(node.features),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"r{rollout}-{hashlib.sha256(raw).hexdigest()[:16]}"

    @staticmethod
    def _as_index(value: Any) -> int | None:
        """``int`` for a builtin or numpy integer, ``None`` otherwise.

        ``type`` rather than ``isinstance``: a ``Mock(spec=int)`` spoofs
        ``__class__`` but not ``type()``, so ``int()`` can never raise here,
        and ``bool`` is excluded because ``True == 1`` would read as an advance.
        """
        kind = type(value)
        if kind is bool or not issubclass(kind, (int, np.integer)):
            return None
        return int(value)

    @staticmethod
    def _lineage(node: MCTSNode) -> list[int | str]:
        """Child-position path from the root to ``node`` (see ``_make_node_id``).

        ``[]`` for the root, ``[1, 0]`` for the first child of the root's
        second child.  Each entry is the node's position in
        ``node.parent.children`` — fixed by ``_expand`` when the child is
        created and preserved by dill across a checkpoint — found by
        identity, because the dataclass ``__eq__`` compares fields (numpy
        arrays included) and ``list.index`` would misfire on it.  O(depth).
        A node whose parent does not list it among its children (a detached
        test stand-in) contributes the ``"opaque-position"`` sentinel for
        that level rather than raising.
        """
        path: list[int | str] = []
        current = node
        while current.parent is not None:
            siblings = getattr(current.parent, "children", None) or []
            position: int | str = "opaque-position"
            for index, sibling in enumerate(siblings):
                if sibling is current:
                    position = index
                    break
            path.append(position)
            current = current.parent
        path.reverse()
        return path

    @staticmethod
    def _parent_plan_digest(parent_output: Any) -> str | list[str]:
        """Digest of the parent plans this evaluation is built from (see ``_make_node_id``).

        ``"no-parent"`` for the root, ``"opaque-parent"`` for an output whose
        ``feature_plans`` are missing, empty or not hashable (test stand-ins);
        the two are distinct so a root and an opaque-parent child never share
        an id.  Otherwise one ``plan_content_hash`` per plan, in name order.
        A hashing failure on real ``FeaturePlan`` values is logged at WARNING
        with the traceback (it silently weakens the key); a stand-in that is
        not a plan mapping only at DEBUG.
        Only the plans are hashed: the rest of an ``AgentOutput`` (DataFrames,
        fitted pipelines) has no stable representation.
        """
        if parent_output is None:
            return "no-parent"
        plans = getattr(parent_output, "feature_plans", None)
        if not plans:
            return "opaque-parent"
        try:
            return [plan_content_hash(plans[name]) for name in sorted(plans)]
        except Exception:
            values = list(plans.values()) if isinstance(plans, dict) else []
            if values and all(isinstance(plan, FeaturePlan) for plan in values):
                logger.warning(
                    "Parent output FeaturePlans failed to hash; using the opaque sentinel",
                    exc_info=True,
                )
            else:
                logger.debug("Parent output plans are not hashable; using the opaque sentinel")
            return "opaque-parent"

    # ------------------------------------------------------------------
    # Default expansion
    # ------------------------------------------------------------------

    def _suggestion_expand(
        self,
        node: MCTSNode,
        *,
        max_children: int = 5,
    ) -> list[tuple[list[str], str, str]]:
        """Default expansion strategy: turn evaluator suggestions into child candidates.

        After a node is evaluated, its ``eval_output`` contains an evaluator
        result with a list of natural-language suggestions like "add a feature
        for prior drug approvals" or "remove the sparse endpoint feature".
        This method converts up to ``max_children`` of those suggestions into
        candidate tuples that ``_expand`` will turn into ``MCTSNode`` objects.

        Each candidate is a tuple of:
        - ``features``: a *copy* of the parent's current feature list (the
          actual feature modification happens later when the child is
          evaluated, because the proposer agent interprets the suggestion
          and decides the concrete add/remove/refine operation).
        - ``operation``: always ``"suggestion"`` (distinguishes these from
          manually injected candidates).
        - ``detail``: a human-readable label like
          ``"suggestion_0: add prior drug approval count"`` (truncated to
          80 chars) for logging and diagnostics.

        Returns an empty list if the node has no eval_output or no
        suggestions, which causes ``_expand`` to produce zero children and
        the rollout to stop expanding at this node.
        """
        output = node.eval_output
        if output is None:
            return []

        # AgentOutput has eval_outputs dict
        if not hasattr(output, "eval_outputs") or not output.eval_outputs:
            return []

        best_eval, _ = output.get_best_eval_output()
        suggestions = best_eval.suggestions
        current_features = list(output.feature_plans.keys())

        candidates = []
        for i, suggestion in enumerate(suggestions[:max_children]):
            # Each child starts with the parent's features.
            # Actual features are updated after evaluation by _call_evaluate.
            candidates.append(
                (
                    list(current_features),
                    "suggestion",
                    f"suggestion_{i}: {suggestion[:80]}",
                )
            )

        return candidates

    @staticmethod
    def _wrap_result(
        obj_vector: NDArray[Any], features: Sequence[str] | None = None
    ) -> ObjectiveResult:
        """Wrap a raw objective vector into an ``ObjectiveResult`` for the node's history.

        Assigns auto-generated names (``obj_0``, ``obj_1``, ...) and stores
        a copy so that mutations to the original array don't affect the
        history.  The ``objective_history`` list on each node accumulates
        these records for diagnostics, re-ranking, and checkpoint export.

        ``features`` is the feature set the vector scored, snapshotted as
        ``details["features"]`` (a list, so it survives a JSON dump) so that
        ``_best_own_objectives`` can ignore an entry that belongs to a set
        the node no longer holds.  Every append site in this file passes
        ``node.features``; ``None`` writes no snapshot and the entry counts
        unconditionally, like one restored from an older checkpoint.
        ``mlops/experiment_tracker.py::log_rollout`` and
        ``mlops/model_store.py`` only read the numeric entries of
        ``details`` by name, so the list is passed through untouched.
        """
        names = [f"obj_{i}" for i in range(len(obj_vector))]
        details: dict[str, Any] = {n: float(v) for n, v in zip(names, obj_vector, strict=True)}
        if features is not None:
            details["features"] = list(features)
        return ObjectiveResult(values=obj_vector.copy(), names=names, details=details)
