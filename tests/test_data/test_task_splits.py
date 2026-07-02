"""Tests for TaskSplitGenerator.

All tests use synthetic in-memory data — no disk I/O, no external services.
"""

from __future__ import annotations

import pytest

from ctra.data.task_splits import TaskSplitGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def generator() -> TaskSplitGenerator:
    return TaskSplitGenerator(random_state=42)


@pytest.fixture()
def trial_data() -> tuple[list[str], list[int], list[str]]:
    """100 synthetic trials with roughly balanced labels and two phases."""
    n = 100
    ids = [f"NCT{i:08d}" for i in range(n)]
    labels = [0, 1] * (n // 2)
    phases = (["PHASE2"] * 50) + (["PHASE3"] * 50)
    return ids, labels, phases


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDefaultRatios:
    def test_default_split_sizes(
        self,
        generator: TaskSplitGenerator,
        trial_data: tuple[list[str], list[int], list[str]],
    ) -> None:
        ids, labels, phases = trial_data
        splits = generator.generate_splits(ids, labels, phases=phases)

        total = sum(len(splits[k]) for k in ("train", "val", "test"))
        assert total == len(ids)

    def test_sum_equals_total(
        self,
        generator: TaskSplitGenerator,
        trial_data: tuple[list[str], list[int], list[str]],
    ) -> None:
        ids, labels, phases = trial_data
        splits = generator.generate_splits(ids, labels, phases=phases)

        n_train = len(splits["train"])
        n_val = len(splits["val"])
        n_test = len(splits["test"])
        assert n_train + n_val + n_test == len(ids)

    def test_no_overlap_between_splits(
        self,
        generator: TaskSplitGenerator,
        trial_data: tuple[list[str], list[int], list[str]],
    ) -> None:
        ids, labels, phases = trial_data
        splits = generator.generate_splits(ids, labels, phases=phases)

        train_ids = set(splits["train"]["trial_id"].to_list())
        val_ids = set(splits["val"]["trial_id"].to_list())
        test_ids = set(splits["test"]["trial_id"].to_list())

        assert train_ids & val_ids == set()
        assert train_ids & test_ids == set()
        assert val_ids & test_ids == set()

    def test_approximate_ratios(
        self,
        generator: TaskSplitGenerator,
        trial_data: tuple[list[str], list[int], list[str]],
    ) -> None:
        ids, labels, phases = trial_data
        n = len(ids)
        splits = generator.generate_splits(ids, labels, phases=phases)

        # Allow +/- 10 % tolerance due to stratification
        assert abs(len(splits["train"]) / n - 0.6) < 0.10
        assert abs(len(splits["val"]) / n - 0.2) < 0.10
        assert abs(len(splits["test"]) / n - 0.2) < 0.10


class TestStratification:
    def test_label_ratio_preserved(
        self,
        generator: TaskSplitGenerator,
        trial_data: tuple[list[str], list[int], list[str]],
    ) -> None:
        ids, labels, phases = trial_data
        splits = generator.generate_splits(ids, labels, phases=phases)

        overall_ratio = sum(labels) / len(labels)
        for name in ("train", "val", "test"):
            df = splits[name]
            if len(df) == 0:
                continue
            split_ratio = df["label"].sum() / len(df)
            assert abs(split_ratio - overall_ratio) < 0.15, (
                f"{name} label ratio {split_ratio:.2f} deviates too much "
                f"from overall {overall_ratio:.2f}"
            )


class TestReproducibility:
    def test_same_seed_same_splits(
        self,
        trial_data: tuple[list[str], list[int], list[str]],
    ) -> None:
        ids, labels, phases = trial_data
        g1 = TaskSplitGenerator(random_state=123)
        g2 = TaskSplitGenerator(random_state=123)

        s1 = g1.generate_splits(ids, labels, phases=phases)
        s2 = g2.generate_splits(ids, labels, phases=phases)

        for name in ("train", "val", "test"):
            assert s1[name]["trial_id"].to_list() == s2[name]["trial_id"].to_list()

    def test_different_seed_different_splits(
        self,
        trial_data: tuple[list[str], list[int], list[str]],
    ) -> None:
        ids, labels, phases = trial_data
        g1 = TaskSplitGenerator(random_state=1)
        g2 = TaskSplitGenerator(random_state=9999)

        s1 = g1.generate_splits(ids, labels, phases=phases)
        s2 = g2.generate_splits(ids, labels, phases=phases)

        # Very unlikely for two different seeds to produce identical train sets
        assert s1["train"]["trial_id"].to_list() != s2["train"]["trial_id"].to_list()


class TestEdgeCases:
    def test_empty_input(self, generator: TaskSplitGenerator) -> None:
        splits = generator.generate_splits([], [])
        for name in ("train", "val", "test"):
            assert len(splits[name]) == 0

    def test_mismatched_lengths_raises(self, generator: TaskSplitGenerator) -> None:
        with pytest.raises(ValueError, match="same length"):
            generator.generate_splits(["A", "B"], [0])

    def test_mismatched_phases_raises(self, generator: TaskSplitGenerator) -> None:
        with pytest.raises(ValueError, match="phases"):
            generator.generate_splits(["A", "B"], [0, 1], phases=["PHASE2"])
