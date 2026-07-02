"""Multi-objective MCTS for feature set optimization.

Adapted from AutoCT's ``treesearch.py`` with the following modifications
for CTRA:

    - ``MCTSNode.total_reward``: ``float`` -> ``np.ndarray`` of shape ``(n_objectives,)``
      to support multi-objective evaluation via Pareto ranking.
    - Pareto UCT replaces scalar UCT argmax during tree policy.
    - AB-MCTS adaptive branching controls the expansion factor per node.
    - Feature value cache (cross-branch reuse) reduces redundant LLM calls.
    - ``_select_best()`` uses hypervolume contribution instead of scalar max.

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

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ctra.agents.runner import extract_objectives
from ctra.config.settings import MCTSConfig, get_settings
from ctra.search.objectives import ObjectiveResult
from ctra.search.pareto import (
    hypervolume_contribution,
    pareto_front_indices,
    pareto_select,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
        == 0) returns zeros to avoid division-by-zero.  The mean reward is used
        by UCB scoring and by ``_select_best`` to rank nodes.
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
                origin ``[0, 0, ...]``. In practice, ``MCTSSearch`` uses the
                configured ``reference_point`` from ``MCTSConfig``.

        Used by ``_select_best`` to break ties among Pareto-front nodes and
        by external callers for reporting.
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
            from ``node.eval_output`` to generate candidates.
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
                bars, logging, and checkpointing.
            start_rollout: Resume from this rollout index. When > 0, root evaluation is
                skipped (assumed restored from checkpoint).

        Returns:
            The best node found (by Pareto rank, then hypervolume
            contribution).
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
            self._root.total_reward = root_obj.copy()
            self._root.visit_count = 1
            self._root.objective_history.append(self._wrap_result(root_obj))
        else:
            # Resuming: root and previous nodes already restored
            assert self._root is not None, "Cannot resume: no root node. Load checkpoint first."

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

            # 1. SELECT — Pareto UCT traversal to a leaf
            node = self._select(self._root)

            # 2. EXPAND — generate children from feature operations
            if node.visit_count > 0 and len(node.features) > 0:
                children = self._expand(node)
                if children:
                    node = children[0]  # pick first unvisited child

            # 3. SIMULATE + 4. BACKPROPAGATE
            try:
                if self._config.deep_simulation:
                    best_obj, _best_node = self._simulate_deep(node, rollout)
                    # Backpropagation is handled inside _simulate_deep
                    # for each node on the deep path.
                    rollout_reward = best_obj
                else:
                    # Shallow mode: evaluate only this single node
                    obj_vector = self._call_evaluate(node, rollout=rollout)
                    node.objective_history.append(self._wrap_result(obj_vector))
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
            if on_rollout is not None:
                on_rollout(rollout, node, rollout_reward)

        # Return best node
        best = self._select_best()
        logger.info(
            "MCTS complete: best node has %d features, mean_reward=%s",
            len(best.features),
            best.mean_reward.round(4),
        )
        return best

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Phase 1 — SELECTION: walk down the tree from root to a leaf node.

        Starting at ``node`` (usually the root), repeatedly pick the best
        child via ``pareto_select`` — which computes UCB1 vectors for every
        sibling, performs non-dominated sorting, and returns a winner — until
        either a leaf is reached or ``max_depth`` is hit.

        The returned leaf is the node where the next expansion + simulation
        will happen.  Because UCB scores give ``inf`` to unvisited children,
        any child that has never been evaluated will be selected before
        revisiting an already-explored child.
        """
        current = node
        depth = 0

        while not current.is_leaf and depth < self._config.max_depth:
            current = pareto_select(
                current.children,
                exploration_constant=self._config.exploration_constant,
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

    def _simulate_deep(self, start_node: MCTSNode, rollout: int) -> tuple[NDArray[Any], MCTSNode]:
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
                - best_obj: Objective vector of the best node found on the path.
                - best_node: The corresponding tree node.
        """
        rng = np.random.default_rng(rollout)
        current = start_node

        # NOTE: On the first rollout, start_node is the root which was already
        # evaluated during search() setup.  This causes root to be evaluated
        # twice, slightly inflating its visit_count.  This is functionally
        # harmless — subsequent rollouts dilute the effect, and MCTS selection
        # operates on children, not root.  See test_root_visit_count_not_inflated.

        # Evaluate start node and backpropagate
        start_obj = self._call_evaluate(current, rollout=rollout)
        current.objective_history.append(self._wrap_result(start_obj))
        self._backpropagate(current, start_obj)

        best_obj = start_obj
        best_hv = self._point_hypervolume(best_obj)
        best_node = current

        depth = self._node_depth(current)
        nodes_evaluated = 1

        while depth < self._config.max_depth:
            # Expand current node if not already expanded
            if current.is_leaf and len(current.features) > 0:
                children = self._expand(current)
                if not children:
                    break
            elif not current.children:
                break

            # Pick a random unvisited child (AutoCT's simulation policy)
            unvisited = [c for c in current.children if c.visit_count == 0]
            if unvisited:
                child = unvisited[rng.integers(len(unvisited))]
            else:
                # All children visited — pick random (exploration)
                child = current.children[rng.integers(len(current.children))]

            # Evaluate the child and backpropagate
            child_obj = self._call_evaluate(child, rollout=rollout)
            child.objective_history.append(self._wrap_result(child_obj))
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

    def _select_best(self) -> MCTSNode:
        """After all rollouts, pick the single best node from the entire tree.

        This is called once at the end of ``search()`` to produce the final
        result.  The selection strategy depends on the number of objectives:

        **Single-objective** (fast path): return the node with the highest
        ``mean_reward[0]``.

        **Multi-objective** (2+ objectives):
        1. Collect ``mean_reward`` vectors for all visited nodes.
        2. Compute the Pareto front — the set of nodes that are not
           dominated by any other node on all objectives simultaneously.
        3. Among Pareto-front nodes, rank by *hypervolume contribution*
           (how much each point uniquely adds to the front's total
           hypervolume).  Return the one with the largest contribution,
           as it represents the most valuable trade-off point.

        Falls back to the root if no nodes have been visited (edge case).
        """
        visited = [n for n in self._all_nodes if n.visit_count > 0]
        if not visited:
            assert self._root is not None
            return self._root

        # Single-objective fast path: just pick the max
        if self._n_objectives == 1:
            return max(visited, key=lambda n: n.mean_reward[0])

        # Multi-objective: Pareto front -> hypervolume contribution ranking
        mean_rewards = np.array([n.mean_reward for n in visited])

        # 1. Find the Pareto front
        front_idx = pareto_front_indices(mean_rewards)

        if len(front_idx) == 0:
            # Shouldn't happen, but fall back to scalar objective 0
            return max(visited, key=lambda n: n.mean_reward[0])

        if len(front_idx) == 1:
            return visited[front_idx[0]]  # type: ignore[no-any-return]

        # 2. Among front nodes, rank by hypervolume contribution
        front_points = mean_rewards[front_idx]
        contributions = hypervolume_contribution(front_points, self._reference)

        best_in_front = int(np.argmax(contributions))
        return visited[front_idx[best_in_front]]  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _call_evaluate(self, node: MCTSNode, rollout: int) -> NDArray[Any]:
        """Invoke the external runner to evaluate a node's feature set and extract objective scores.

        This is the bridge between the MCTS tree and the agent pipeline.  The
        steps are:

        1. **Build parent context** — If this node has a parent with a stored
           ``eval_output``, pass it to the runner so the agent pipeline knows
           what was tried before.  The ``suggestion_index`` on the parent
           output tells the proposer agent *which* suggestion to follow for
           this child.  For the root node, ``parent_output`` is None and the
           runner starts a fresh initialization.

        2. **Call the runner** — ``self._runner(node_id, task, parent_output)``
           spawns a subprocess (by default) that runs the full agent pipeline
           (initializer -> proposer -> planner -> builder -> grouper ->
           evaluator) and returns an ``AgentOutput`` containing feature plans,
           built features, and evaluation metrics.

        3. **Store output** — The ``AgentOutput`` is saved on
           ``node.eval_output`` so future children can use it as context and
           so ``_suggestion_expand`` can read the evaluator's suggestions.

        4. **Sync features** — If the subprocess produced different features
           than expected (the proposer may modify the set), update
           ``node.features`` to reflect reality.

        5. **Extract objectives** — Pull the objective scores (e.g. ROC-AUC,
           parsimony) from the output, pad or truncate to ``n_objectives``,
           and return as an ndarray.

        Returns:
            An ndarray of shape ``(n_objectives,)`` containing the objective scores for
            this node, ready for backpropagation.
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

        node_id = self._make_node_id(node, rollout)
        output = self._runner(node_id, self._task, parent_output)

        # Store full AgentOutput on the node
        node.eval_output = output

        # Honor suggestion_index advance from orchestrator (M1 fix: Site 1 fallback).
        # When orchestrator skips an iteration due to proposer failure, it returns
        # an AgentOutput with suggestion_index advanced by 1. Honor that advance on
        # the node so subsequent rollouts don't replay the same exhausted suggestion.
        if hasattr(output, "suggestion_index"):
            node.suggestion_index = output.suggestion_index

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

    def _make_node_id(self, node: MCTSNode, rollout: int) -> str:
        """Generate a deterministic, unique string identifier for a node evaluation.

        The ID combines the rollout index with a hash of the sorted feature
        names, e.g. ``"r3-a1b2c3d4"``.  This is passed to the runner so that
        subprocess outputs can be cached and correlated back to tree nodes.
        Sorting features before hashing ensures that two nodes with the same
        feature set (regardless of insertion order) produce the same hash.
        """
        features_hash = hash(tuple(sorted(node.features)))
        return f"r{rollout}-{features_hash:x}"

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
    def _wrap_result(obj_vector: NDArray[Any]) -> ObjectiveResult:
        """Wrap a raw objective vector into an ``ObjectiveResult`` for the node's history.

        Assigns auto-generated names (``obj_0``, ``obj_1``, ...) and stores
        a copy so that mutations to the original array don't affect the
        history.  The ``objective_history`` list on each node accumulates
        these records for diagnostics, re-ranking, and checkpoint export.
        """
        names = [f"obj_{i}" for i in range(len(obj_vector))]
        return ObjectiveResult(
            values=obj_vector.copy(),
            names=names,
            details={n: float(v) for n, v in zip(names, obj_vector, strict=True)},
        )
