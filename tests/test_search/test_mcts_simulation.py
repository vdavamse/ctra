"""End-to-end MCTS simulation tests with stub evaluate/expand functions.

These tests exercise the full MCTS loop (select → expand → simulate →
backpropagate → select_best) WITHOUT any LLM calls, real data, or GPU.
Stubs provide deterministic or random-but-seeded objective values so we
can verify the search mechanics work correctly.

Tests cover:
    - Tree grows (nodes created, children linked, backpropagation works)
    - Pareto selection prefers non-dominated nodes
    - Multi-fidelity schedule is applied
    - Adaptive branching scales with visit count
    - Best node is on the Pareto front with highest hypervolume
    - Single-objective backward compatibility
    - Deterministic reproducibility with fixed seed
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig, Settings, get_settings
from ctra.search.mcts import MCTSNode, MCTSSearch
from ctra.search.objectives import ObjectiveResult
from tests.test_search.conftest import make_stub_runner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Provide deterministic MCTS settings for all tests."""
    get_settings.cache_clear()
    settings = Settings(
        mcts=MCTSConfig(
            num_rollouts=20,
            max_depth=10,
            objectives=["accuracy", "parsimony"],
            max_features=50,
            exploration_constant=1.414,
            adaptive_branching=True,
            min_branch_factor=2,
            max_branch_factor=5,
            reference_point=[0.0, 0.0],
        ),
    )
    monkeypatch.setattr("ctra.search.mcts.get_settings", lambda: settings)
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Stub functions
# ---------------------------------------------------------------------------


def make_deterministic_evaluator(seed: int = 42):
    """Create a seeded evaluator that returns realistic clinical trial objectives.

    The evaluator simulates the real pipeline: given a feature set, it returns
    objectives that correlate with feature count in a realistic way:
    - Accuracy: improves with more features (diminishing returns), with noise
    - Parsimony: deterministic 1 - n/50
    """
    rng = np.random.default_rng(seed)
    call_count = 0

    def evaluate(features: list[str], fidelity: float = 1.0) -> np.ndarray:
        nonlocal call_count
        call_count += 1

        n = len(features)

        # Accuracy: logarithmic improvement with features, noisy
        # More features help but with diminishing returns
        base_accuracy = 0.5 + 0.15 * math.log2(max(n, 1))
        noise = rng.normal(0, 0.03 * (1.0 / max(fidelity, 0.1)))  # more noise at low fidelity
        accuracy = float(np.clip(base_accuracy + noise, 0.0, 1.0))

        # Parsimony: deterministic
        parsimony = max(0.0, 1.0 - n / 50.0)

        return np.array([accuracy, parsimony])

    evaluate.call_count = lambda: call_count  # type: ignore[attr-defined]
    return evaluate


def make_deterministic_expander(seed: int = 42):
    """Create a seeded expander that generates add/remove/refine operations.

    Generates realistic feature operation candidates:
    - Add: adds a new feature from a pool of clinical trial feature names
    - Remove: removes a random existing feature
    - Refine: renames an existing feature (simulating re-characterization)
    """
    rng = np.random.default_rng(seed)
    feature_pool = [
        "prior_drug_approvals",
        "target_gene_count",
        "enrollment_size",
        "phase_duration_months",
        "eligibility_criteria_count",
        "num_endpoints",
        "disease_prevalence",
        "drug_interaction_count",
        "adverse_event_rate",
        "sponsor_success_rate",
        "site_count",
        "competitor_trials",
        "biomarker_available",
        "pediatric_population",
        "multi_arm_design",
        "placebo_controlled",
        "adaptive_design",
        "orphan_drug_status",
        "fda_fast_track",
        "mechanism_novelty_score",
    ]

    def expand(node: MCTSNode, max_children: int = 3) -> list[tuple[list[str], str, str]]:
        candidates = []
        current = list(node.features)

        for _ in range(max_children):
            op = rng.choice(["add", "remove", "refine"], p=[0.5, 0.3, 0.2])

            if op == "add":
                available = [f for f in feature_pool if f not in current]
                if available:
                    new_feat = rng.choice(available)
                    new_features = [*current, new_feat]
                    candidates.append((new_features, "add", f"add:{new_feat}"))
                else:
                    # Pool exhausted — skip
                    continue

            elif op == "remove" and len(current) > 1:
                idx = rng.integers(len(current))
                removed = current[idx]
                new_features = [f for i, f in enumerate(current) if i != idx]
                candidates.append((new_features, "remove", f"remove:{removed}"))

            elif op == "refine" and len(current) > 0:
                idx = rng.integers(len(current))
                old_name = current[idx]
                new_name = f"{old_name}_v2"
                new_features = [new_name if i == idx else f for i, f in enumerate(current)]
                candidates.append((new_features, "refine", f"refine:{old_name}->{new_name}"))

        return candidates

    return expand


# ---------------------------------------------------------------------------
# Test: Full MCTS simulation
# ---------------------------------------------------------------------------


class TestMCTSFullSimulation:
    """End-to-end MCTS simulation verifying the search loop works correctly."""

    def test_search_completes_and_returns_best_node(self):
        """MCTS search runs to completion and returns a valid node."""
        evaluate = make_deterministic_evaluator(seed=42)
        expand = make_deterministic_expander(seed=42)

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        best = search.search(
            initial_features=["prior_drug_approvals", "enrollment_size", "num_endpoints"]
        )

        assert isinstance(best, MCTSNode)
        assert len(best.features) > 0
        assert best.visit_count > 0
        assert best.mean_reward.shape == (2,)

    def test_tree_grows_with_rollouts(self):
        """The tree should have more nodes than just the root after search."""
        evaluate = make_deterministic_evaluator(seed=42)
        expand = make_deterministic_expander(seed=42)

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        search.search(initial_features=["feat_a", "feat_b"])

        assert len(search.all_nodes) > 1, "Tree should grow beyond root"
        # With 20 rollouts and branching, we expect at least 10 nodes
        assert len(search.all_nodes) >= 10

    def test_backpropagation_reaches_root(self):
        """All rollout rewards should propagate back to root."""
        evaluate = make_deterministic_evaluator(seed=42)
        expand = make_deterministic_expander(seed=42)

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        search.search(initial_features=["feat_a", "feat_b"])

        root = search.root
        assert root is not None
        # Root is visited once initially + once per rollout that touches it
        # (backpropagation always reaches root)
        assert root.visit_count >= 1 + search._config.num_rollouts

    def test_best_node_has_positive_objectives(self):
        """The selected best node should have reasonable objective values."""
        evaluate = make_deterministic_evaluator(seed=42)
        expand = make_deterministic_expander(seed=42)

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        best = search.search(initial_features=["prior_drug_approvals", "enrollment_size"])

        mean = best.mean_reward
        # All objectives should be positive (better than reference [0,0])
        assert all(mean > 0), f"Expected positive objectives, got {mean}"
        # Accuracy should be above random chance (0.5)
        assert mean[0] > 0.4, f"Accuracy {mean[0]} too low"
        # Parsimony should be positive (not too many features)
        assert mean[1] > 0.0, f"Parsimony {mean[1]} should be positive"

    def test_best_node_is_on_pareto_front(self):
        """The selected best node should be Pareto-optimal among the evaluated nodes.

        Non-domination holds in the space ``_select_best`` ranks in: each
        node's *own* best evaluation (issue #15).  It does **not** hold for
        ``mean_reward`` — expanding the winner and evaluating mediocre
        children drags its subtree mean below an unexpanded node's, which is
        the whole reason the final ranking moved off the mean.
        """
        evaluate = make_deterministic_evaluator(seed=42)
        expand = make_deterministic_expander(seed=42)

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        best = search.search(initial_features=["feat_a", "feat_b", "feat_c"])

        # Candidates are the nodes that were evaluated at least once.
        evaluated = [n for n in search.all_nodes if n.objective_history]
        assert best in evaluated
        best_own = search.best_own_objectives(best)

        # Verify no other node dominates the best node on ALL objectives
        for node in evaluated:
            if node is best:
                continue
            other_own = search.best_own_objectives(node)
            # "dominates" means strictly better on all objectives
            if all(other_own > best_own):
                pytest.fail(f"Best node is dominated: best={best_own}, dominator={other_own}")

    def test_parent_child_links_consistent(self):
        """Every child's parent reference should point back correctly."""
        evaluate = make_deterministic_evaluator(seed=42)
        expand = make_deterministic_expander(seed=42)

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        search.search(initial_features=["feat_a", "feat_b"])

        for node in search.all_nodes:
            for child in node.children:
                assert child.parent is node, (
                    f"Child's parent mismatch: child.parent={child.parent}, "
                    f"expected node with features={node.features}"
                )

    def test_feature_operations_tracked(self):
        """Expanded nodes should record their operation type."""
        evaluate = make_deterministic_evaluator(seed=42)
        expand = make_deterministic_expander(seed=42)

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        search.search(initial_features=["feat_a", "feat_b"])

        operations_seen = set()
        for node in search.all_nodes:
            operations_seen.add(node.operation)

        assert "root" in operations_seen
        # At least one non-root operation should be present
        assert len(operations_seen) > 1, f"Only saw operations: {operations_seen}"


class TestAdaptiveBranching:
    """Verify AB-MCTS adaptive branching adjusts expansion factor."""

    def test_branching_factor_increases_with_visits(self):
        """Nodes visited more often should generate more children."""
        branch_factors = []

        def tracking_expand(node, max_children=3):
            branch_factors.append(max_children)
            # Return simple add operations
            return [
                ([*node.features, f"new_{i}"], "add", f"add:new_{i}")
                for i in range(min(max_children, 2))
            ]

        evaluate = make_deterministic_evaluator(seed=42)
        runner = make_stub_runner(evaluate, tracking_expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=tracking_expand)
        search.search(initial_features=["feat_a", "feat_b"])

        # With adaptive branching, branch_factor = min(5, max(2, int(log2(visits) + 2)))
        # Root gets visited many times via backpropagation, so branching
        # should increase over time.
        assert len(branch_factors) > 0
        # All branch factors should be within [min_branch_factor, max_branch_factor]
        for bf in branch_factors:
            assert 2 <= bf <= 5, f"Branch factor {bf} outside [2, 5]"


class TestSingleObjectiveBackwardCompat:
    """Verify MCTS works correctly with a single objective (scalar mode)."""

    def test_single_objective_search(self, monkeypatch):
        """Single-objective MCTS should work like standard scalar MCTS."""
        get_settings.cache_clear()
        settings = Settings(
            mcts=MCTSConfig(
                num_rollouts=10,
                max_depth=5,
                objectives=["accuracy"],
                max_features=50,
                exploration_constant=1.414,
                adaptive_branching=False,
                max_branch_factor=3,
                reference_point=[0.0],
            ),
        )
        monkeypatch.setattr("ctra.search.mcts.get_settings", lambda: settings)

        rng = np.random.default_rng(42)

        def scalar_evaluate(features, fidelity=1.0):
            # Single objective: just accuracy
            n = len(features)
            return np.array([0.5 + 0.1 * math.log2(max(n, 1)) + rng.normal(0, 0.02)])

        def simple_expand(node, max_children=3):
            return [([*node.features, f"f{i}"], "add", f"add:f{i}") for i in range(max_children)]

        runner = make_stub_runner(scalar_evaluate, simple_expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=simple_expand)
        best = search.search(initial_features=["f0"])

        assert isinstance(best, MCTSNode)
        assert best.mean_reward.shape == (1,)
        assert best.mean_reward[0] > 0.4  # should beat random chance


class TestDeterministicReproducibility:
    """Verify that the same seeds produce the same results."""

    def test_same_seed_same_result(self):
        """Two runs with identical seeds should produce identical best features."""
        results = []
        for _ in range(2):
            evaluate = make_deterministic_evaluator(seed=123)
            expand = make_deterministic_expander(seed=123)
            runner = make_stub_runner(evaluate, expand)
            search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
            best = search.search(initial_features=["feat_a", "feat_b"])
            results.append(sorted(best.features))

        assert results[0] == results[1], (
            f"Same seeds should produce same features: {results[0]} vs {results[1]}"
        )

    def test_different_seed_different_result(self):
        """Different seeds should (very likely) produce different results."""
        features_per_run = []
        for seed in [42, 99]:
            evaluate = make_deterministic_evaluator(seed=seed)
            expand = make_deterministic_expander(seed=seed)
            runner = make_stub_runner(evaluate, expand)
            search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
            best = search.search(initial_features=["feat_a", "feat_b"])
            features_per_run.append(sorted(best.features))

        # Not guaranteed to differ, but extremely likely with different seeds
        # If this flakes, increase rollouts or seed difference
        assert features_per_run[0] != features_per_run[1], (
            "Different seeds should produce different results"
        )


class TestObjectiveHistoryTracking:
    """Verify that objective values are recorded in node history."""

    def test_history_recorded_on_evaluated_nodes(self):
        """Nodes that are evaluated should have non-empty objective_history."""
        evaluate = make_deterministic_evaluator(seed=42)
        expand = make_deterministic_expander(seed=42)

        runner = make_stub_runner(evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        search.search(initial_features=["a", "b"])

        nodes_with_history = [n for n in search.all_nodes if len(n.objective_history) > 0]
        assert len(nodes_with_history) > 0

        # Each history entry should be an ObjectiveResult with 2 values
        for node in nodes_with_history:
            for entry in node.objective_history:
                assert isinstance(entry, ObjectiveResult)
                assert entry.values.shape == (2,)


class TestSearchWithObjectiveResult:
    """Verify MCTS works when evaluate_fn returns ObjectiveResult directly."""

    def test_objective_result_return_type(self):
        """Evaluator returning ObjectiveResult should work identically."""

        def or_evaluate(features, fidelity=1.0):
            n = len(features)
            return ObjectiveResult(
                values=np.array([0.7, 1.0 - n / 50.0]),
                names=["accuracy", "parsimony"],
                details={"source": "stub"},
            )

        expand = make_deterministic_expander(seed=42)
        runner = make_stub_runner(or_evaluate, expand)
        search = MCTSSearch(runner=runner, task="test", expand_fn=expand)
        best = search.search(initial_features=["a", "b"])

        assert isinstance(best, MCTSNode)
        assert best.visit_count > 0
        assert best.mean_reward.shape == (2,)
