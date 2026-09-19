"""Tests for the ``MCTSConfig.backprop`` rules (issue #16).

Hand-built trees drive ``MCTSSearch._backpropagate`` directly, so each rule
is checked on the vectors it folds — the subtree mean, the elementwise
maximum, and the best realised vector by hypervolume — and on the one
invariant they share: ``total_reward == A(node) * visit_count``, which is
what keeps ``mean_reward``, ``ucb_scores`` and ``value`` unchanged code.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ctra.config.settings import MCTSConfig
from ctra.search.mcts import MCTSNode, MCTSSearch
from tests.test_search.conftest import make_stub_runner

REFERENCE = [0.5, 0.0]

# Realised vectors: above the reference (positive hypervolume) and below it.
ABOVE_A = np.array([0.7, 0.8])  # HV 0.16
ABOVE_B = np.array([0.9, 0.1])  # HV 0.04
ABOVE_C = np.array([0.8, 0.9])  # HV 0.27
BELOW_A = np.array([0.3, 0.5])  # HV 0
BELOW_B = np.array([0.45, 0.2])  # HV 0
BELOW_C = np.array([0.4, 0.9])  # HV 0


def _unused_runner(node_id, task, previous_output):  # pragma: no cover - never called
    raise AssertionError("hand-built trees do not evaluate")


def _search(backprop: str, **overrides) -> MCTSSearch:
    config = MCTSConfig(
        backprop=backprop,
        objectives=["accuracy", "parsimony"],
        reference_point=REFERENCE,
        **overrides,
    )
    return MCTSSearch(runner=_unused_runner, task="test", config=config)


def _chain() -> tuple[MCTSNode, MCTSNode, MCTSNode]:
    """root -> child -> leaf, all unvisited."""
    root = MCTSNode(features=["a"], total_reward=np.zeros(2))
    child = MCTSNode(features=["a", "b"], parent=root, total_reward=np.zeros(2))
    leaf = MCTSNode(features=["a", "b", "c"], parent=child, total_reward=np.zeros(2))
    root.children.append(child)
    child.children.append(leaf)
    return root, child, leaf


def _point_hv(vector: np.ndarray) -> float:
    return float(np.prod(np.maximum(vector - np.asarray(REFERENCE), 0.0)))


def _aggregate(mode: str, vectors: list[np.ndarray]) -> np.ndarray:
    """The rule's aggregate, computed independently of ``_backpropagate``."""
    stack = np.stack(vectors)
    if mode == "mean":
        return stack.mean(axis=0)
    if mode == "max":
        return stack.max(axis=0)
    return max(vectors, key=lambda v: (_point_hv(v), float(v[0])))


# ---------------------------------------------------------------------------
# mean (the default): frozen expectation
# ---------------------------------------------------------------------------


class TestMean:
    def test_sums_vectors_and_counts_visits_up_the_chain(self):
        search = _search("mean")
        root, child, leaf = _chain()

        search._backpropagate(leaf, np.array([0.6, 0.9]))
        search._backpropagate(leaf, np.array([0.8, 0.7]))
        search._backpropagate(child, np.array([0.7, 0.5]))

        assert leaf.visit_count == 2
        assert child.visit_count == 3
        assert root.visit_count == 3
        np.testing.assert_allclose(leaf.total_reward, [1.4, 1.6])
        np.testing.assert_allclose(child.total_reward, [2.1, 2.1])
        np.testing.assert_allclose(root.total_reward, [2.1, 2.1])
        np.testing.assert_allclose(root.mean_reward, [0.7, 0.7])

    def test_is_the_default_mode(self):
        search = MCTSSearch(runner=_unused_runner, task="test", config=MCTSConfig())

        assert search._backprop_mode == "mean"


# ---------------------------------------------------------------------------
# max: elementwise running maximum
# ---------------------------------------------------------------------------


class TestMax:
    def test_is_order_independent(self):
        vectors = [ABOVE_A, ABOVE_B, BELOW_A, ABOVE_C, BELOW_C]
        forward = _chain()
        backward = _chain()
        search = _search("max")

        for v in vectors:
            search._backpropagate(forward[2], v)
        for v in reversed(vectors):
            search._backpropagate(backward[2], v)

        for a, b in zip(forward, backward, strict=True):
            np.testing.assert_allclose(a.mean_reward, b.mean_reward)
            np.testing.assert_allclose(a.total_reward, b.total_reward)
            assert a.visit_count == b.visit_count

    def test_is_monotone_in_every_coordinate(self):
        search = _search("max")
        root, child, leaf = _chain()
        previous = {id(n): np.full(2, -np.inf) for n in (root, child, leaf)}

        for v in [ABOVE_B, BELOW_A, ABOVE_A, BELOW_B, ABOVE_C, BELOW_C]:
            search._backpropagate(leaf, v)
            for node in (root, child, leaf):
                assert np.all(node.mean_reward >= previous[id(node)] - 1e-12)
                previous[id(node)] = node.mean_reward.copy()

    def test_is_the_elementwise_maximum_of_everything_seen(self):
        search = _search("max")
        root, child, leaf = _chain()

        search._backpropagate(leaf, ABOVE_A)
        search._backpropagate(leaf, ABOVE_B)
        search._backpropagate(child, BELOW_C)

        # The root aggregate combines coordinates no single evaluation scored
        # (accuracy from ABOVE_B, parsimony from BELOW_C): the documented caveat.
        np.testing.assert_allclose(root.mean_reward, [0.9, 0.9])
        np.testing.assert_allclose(leaf.mean_reward, [0.9, 0.8])


# ---------------------------------------------------------------------------
# max_hv: best realised vector by (hypervolume, accuracy)
# ---------------------------------------------------------------------------


class TestMaxHV:
    def test_keeps_a_realised_vector_with_the_largest_hypervolume(self):
        search = _search("max_hv")
        root, child, leaf = _chain()

        search._backpropagate(leaf, ABOVE_A)
        search._backpropagate(leaf, ABOVE_B)  # higher accuracy, smaller HV
        np.testing.assert_allclose(root.mean_reward, ABOVE_A)

        search._backpropagate(leaf, ABOVE_C)  # larger HV replaces
        np.testing.assert_allclose(root.mean_reward, ABOVE_C)
        np.testing.assert_allclose(child.mean_reward, ABOVE_C)
        np.testing.assert_allclose(leaf.mean_reward, ABOVE_C)

    def test_aggregate_is_always_one_of_the_vectors_seen(self):
        rng = np.random.default_rng(16)
        vectors = [rng.uniform(0.0, 1.0, size=2) for _ in range(25)]
        search = _search("max_hv")
        root, _child, leaf = _chain()

        for v in vectors:
            search._backpropagate(leaf, v)
            assert any(np.allclose(root.mean_reward, seen) for seen in vectors)

    def test_zero_hypervolume_higher_accuracy_does_not_displace(self):
        search = _search("max_hv")
        root, _child, leaf = _chain()

        search._backpropagate(leaf, np.array([0.6, 0.5]))  # HV 0.05
        search._backpropagate(leaf, np.array([0.95, 0.0]))  # HV 0, best accuracy

        np.testing.assert_allclose(root.mean_reward, [0.6, 0.5])

    def test_accuracy_breaks_ties_below_the_reference(self):
        search = _search("max_hv")
        root, _child, leaf = _chain()

        search._backpropagate(leaf, BELOW_A)
        search._backpropagate(leaf, BELOW_B)  # HV 0 too, higher accuracy wins
        np.testing.assert_allclose(root.mean_reward, BELOW_B)

        search._backpropagate(leaf, BELOW_C)  # HV 0, lower accuracy: stays
        np.testing.assert_allclose(root.mean_reward, BELOW_B)


# ---------------------------------------------------------------------------
# Shared contract: the invariant, visit counts, the config guard
# ---------------------------------------------------------------------------

SEQUENCE = [ABOVE_B, BELOW_A, ABOVE_A, BELOW_B, ABOVE_C, BELOW_C, ABOVE_A]


@pytest.mark.parametrize("mode", ["mean", "max", "max_hv"])
class TestInvariant:
    def test_total_reward_is_the_aggregate_times_visit_count_after_every_call(self, mode):
        search = _search(mode)
        root, child, leaf = _chain()
        through: dict[int, list[np.ndarray]] = {id(n): [] for n in (root, child, leaf)}

        for i, v in enumerate(SEQUENCE):
            target = leaf if i % 3 else child  # some vectors stop at the child
            search._backpropagate(target, v)
            node: MCTSNode | None = target
            while node is not None:
                through[id(node)].append(v)
                node = node.parent
            for node in (root, child, leaf):
                seen = through[id(node)]
                assert node.visit_count == len(seen)
                if seen:
                    expected = _aggregate(mode, seen)
                    np.testing.assert_allclose(
                        node.total_reward, expected * node.visit_count, atol=1e-12
                    )
                    np.testing.assert_allclose(node.mean_reward, expected, atol=1e-12)
                    np.testing.assert_allclose(search.value_estimate(node), expected, atol=1e-12)

    def test_visit_counts_match_the_mean_rule(self, mode):
        reference = _search("mean")
        under_test = _search(mode)
        ref_nodes = _chain()
        nodes = _chain()

        for i, v in enumerate(SEQUENCE):
            reference._backpropagate(ref_nodes[i % 3], v)
            under_test._backpropagate(nodes[i % 3], v)

        assert [n.visit_count for n in nodes] == [n.visit_count for n in ref_nodes]

    def test_ucb_scores_read_the_aggregate(self, mode):
        """``ucb_scores`` is unchanged code: exploit term == aggregate."""
        search = _search(mode)
        root, child, leaf = _chain()
        for v in SEQUENCE:
            search._backpropagate(leaf, v)

        explore = 1.0 * np.sqrt(np.log(root.visit_count) / child.visit_count)
        np.testing.assert_allclose(
            child.ucb_scores(exploration_constant=1.0), child.mean_reward + explore
        )


class TestConfigGuard:
    def test_a_config_without_the_field_falls_back_to_mean(self):
        """A checkpoint pickled before ``backprop`` existed resumes under ``mean``."""
        search = _search("max")
        stripped = {k: v for k, v in search.config.model_dump().items() if k != "backprop"}
        search._config = SimpleNamespace(**stripped)  # type: ignore[assignment]
        root, _child, leaf = _chain()

        assert search._backprop_mode == "mean"
        search._backpropagate(leaf, ABOVE_A)
        search._backpropagate(leaf, ABOVE_B)

        np.testing.assert_allclose(root.total_reward, ABOVE_A + ABOVE_B)
        assert root.visit_count == 2

    def test_an_unknown_rule_is_an_error_not_a_silent_mean(self):
        search = _search("max")
        stand_in = SimpleNamespace(**{**search.config.model_dump(), "backprop": "sum"})
        search._config = stand_in  # type: ignore[assignment]
        _root, _child, leaf = _chain()

        with pytest.raises(ValueError, match="unknown backprop mode 'sum'"):
            search._backpropagate(leaf, ABOVE_A)

    def test_value_estimate_names_every_rule(self):
        doc = MCTSSearch.value_estimate.__doc__ or ""

        assert "mean" in doc and "max_hv" in doc and "max" in doc


# ---------------------------------------------------------------------------
# End to end: every rule completes a small search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["mean", "max", "max_hv"])
def test_a_small_search_completes_under_every_rule(mode):
    """No winner is asserted here — research/backprop-ablation.md measures that."""

    def evaluate(features):
        return np.array([0.5 + 0.05 * len(features), max(0.0, 1 - len(features) / 50)])

    def expand(node, max_children=2):
        return [
            ([*node.features, f"f{len(node.features)}_{i}"], "add", f"add f{i}")
            for i in range(max_children)
        ]

    runner = make_stub_runner(evaluate, expand_fn=expand, initial_features=["base"])
    config = MCTSConfig(
        backprop=mode, num_rollouts=4, max_depth=3, max_branch_factor=2, reference_point=REFERENCE
    )
    search = MCTSSearch(runner=runner, task="test", expand_fn=expand, config=config)
    best = search.search(initial_features=["base"])

    assert best in search.all_nodes
    assert best.visit_count >= 1
    assert search.root is not None and search.root.visit_count > 1
    own = search.best_own_objectives(best)
    assert own.shape == (2,)
    # The invariant survives a real search under every rule.
    for node in search.all_nodes:
        np.testing.assert_allclose(
            node.total_reward, node.mean_reward * max(node.visit_count, 1), atol=1e-12
        )


def test_stub_runner_search_completes_under_max_hv():
    """The conftest stub (the shape ``scripts/train_mcts.py`` drives) works with a non-default rule."""
    runner = make_stub_runner(lambda features: np.array([0.7, 0.9]), initial_features=["base"])
    config = MCTSConfig(backprop="max_hv", num_rollouts=3, max_depth=2, max_branch_factor=2)
    search = MCTSSearch(runner=runner, task="test", config=config)

    best = search.search(initial_features=["base"])

    np.testing.assert_allclose(search.best_own_objectives(best)[0], 0.7)
