"""Tests for the feature-store counters (issue #17).

``FeatureStoreCounters`` arithmetic, the ``get_store_stats`` merge, and the
single-owner counting rule across ``compute_features``'s batch probe and
``WrappedFeatureBuilder``'s re-probe: a partially cached trial-group is
probed twice on disk but counted once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ctra.agents.data_models import (
    LLM_CALLS_PER_GROUP_BUILD,
    FeaturePlan,
    FeatureSource,
    FeatureStoreCounters,
    FeatureType,
)
from ctra.agents.feature_builder import WrappedFeatureBuilder, compute_features
from ctra.agents.feature_store import get_store_stats, put_cached_feature

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

NAMESPACE = "phase2"


def _make_plan(name: str) -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[],
        possible_values={},
        feature_instructions=f"Extract {name}.",
    )


def _prepopulate(store_dir: Path, nctid: str, plan: FeaturePlan) -> None:
    put_cached_feature(
        store_dir,
        NAMESPACE,
        nctid,
        plan.feature_name,
        plan,
        {plan.feature_name: {"value": "cached"}},
    )


@pytest.fixture()
def fake_builder() -> Iterator[list[tuple[str, list[str]]]]:
    """Stub ``FeatureBuilder`` that builds every requested plan; records calls."""
    calls: list[tuple[str, list[str]]] = []

    def fake_build(nctid: str, feature_plan_group: dict[str, FeaturePlan]) -> tuple:
        calls.append((nctid, sorted(feature_plan_group)))
        return (
            {fn: {"value": f"built_{nctid}"} for fn in feature_plan_group},
            {"builder_reasoning": "mock", "research_results": "mock"},
        )

    with (
        patch("ctra.agents.feature_builder.FeatureBuilder") as builder_cls,
        patch(
            "ctra.agents.feature_builder.ResettingRefine",
            side_effect=lambda module, **kw: module,
        ),
    ):
        builder_cls.return_value = MagicMock(side_effect=fake_build)
        yield calls


def _compute(
    store_dir: Path,
    nctids: list[str],
    plans: dict[str, FeaturePlan],
    counters: FeatureStoreCounters,
    *,
    plan_origin: str = "planner",
    grouper: MagicMock | None = None,
    enabled: bool = True,
) -> None:
    compute_features(
        grouper=grouper if grouper is not None else MagicMock(),
        nctids=nctids,
        task_description="test task",
        plans=plans,
        feature_store_dir=store_dir,
        task_namespace=NAMESPACE,
        feature_store_enabled=enabled,
        counters=counters,
        plan_origin=plan_origin,
    )


# ---------------------------------------------------------------------------
# Counter arithmetic
# ---------------------------------------------------------------------------


class TestFeatureStoreCounters:
    def test_defaults_are_zero_and_the_rate_is_safe(self) -> None:
        c = FeatureStoreCounters()
        assert c.as_dict() == {
            "feature_lookups": 0,
            "feature_hits": 0,
            "groups_dispatched": 0,
            "groups_skipped": 0,
            "hits_from_initializer_plans": 0,
            "hits_from_planner_plans": 0,
            "store_writes": 0,
            "hit_rate": 0.0,
            "llm_calls_avoided_estimate": 0,
        }

    def test_hit_rate_is_hits_over_lookups(self) -> None:
        c = FeatureStoreCounters(feature_lookups=8, feature_hits=2)
        assert c.hit_rate == pytest.approx(0.25)

    def test_record_hits_attributes_to_the_plan_origin(self) -> None:
        c = FeatureStoreCounters()
        c.record_hits(3, "initializer")
        c.record_hits(2, "planner")
        assert c.feature_hits == 5
        assert c.hits_from_initializer_plans == 3
        assert c.hits_from_planner_plans == 2
        with pytest.raises(ValueError, match="plan_origin"):
            c.record_hits(1, "somewhere")

    def test_avoided_calls_multiply_groups_skipped_only(self) -> None:
        """A hit inside a dispatched group avoids nothing: the builder still ran."""
        assert LLM_CALLS_PER_GROUP_BUILD == 6
        partial = FeatureStoreCounters(feature_lookups=2, feature_hits=1, groups_dispatched=1)
        assert partial.llm_calls_avoided_estimate == 0
        full = FeatureStoreCounters(feature_lookups=2, feature_hits=2, groups_skipped=1)
        assert full.llm_calls_avoided_estimate == 6
        many = FeatureStoreCounters(feature_hits=100, groups_skipped=3)
        assert many.llm_calls_avoided_estimate == 18

    def test_merge_sums_every_counter_in_place(self) -> None:
        a = FeatureStoreCounters(feature_lookups=1, feature_hits=1, store_writes=2)
        b = FeatureStoreCounters(feature_lookups=3, groups_skipped=1, hits_from_planner_plans=1)
        a.merge(b)
        assert a == FeatureStoreCounters(
            feature_lookups=4,
            feature_hits=1,
            groups_skipped=1,
            hits_from_planner_plans=1,
            store_writes=2,
        )
        assert b.feature_lookups == 3, "merge must not touch the source"

    def test_merge_skips_non_int_stand_ins(self) -> None:
        a = FeatureStoreCounters(feature_lookups=1)
        a.merge(MagicMock())
        assert a.feature_lookups == 1


# ---------------------------------------------------------------------------
# get_store_stats merge
# ---------------------------------------------------------------------------


class TestGetStoreStatsMerge:
    def test_without_counters_the_shape_is_unchanged(self, tmp_path: Path) -> None:
        assert get_store_stats(tmp_path / "missing") == {
            "features": 0,
            "trials": 0,
            "total_entries": 0,
        }

    def test_counters_are_merged_after_the_inventory(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        _prepopulate(store, "NCT001", _make_plan("feat_a"))
        counters = FeatureStoreCounters(feature_lookups=4, feature_hits=1, groups_skipped=1)
        stats = get_store_stats(store, NAMESPACE, counters)
        assert stats["features"] == 1
        assert stats["trials"] == 1
        assert stats["total_entries"] == 1
        assert stats["feature_lookups"] == 4
        assert stats["hit_rate"] == pytest.approx(0.25)
        assert stats["llm_calls_avoided_estimate"] == LLM_CALLS_PER_GROUP_BUILD
        assert set(stats) == {"features", "trials", "total_entries", *counters.as_dict()}


# ---------------------------------------------------------------------------
# Counting through compute_features (batch probe owns the counts)
# ---------------------------------------------------------------------------


class TestComputeFeaturesCounting:
    def test_cold_store_counts_lookups_misses_dispatches_and_writes(
        self, tmp_path: Path, fake_builder: list
    ) -> None:
        counters = FeatureStoreCounters()
        _compute(
            tmp_path / "store", ["NCT001", "NCT002"], {"feat_a": _make_plan("feat_a")}, counters
        )
        assert counters.feature_lookups == 2
        assert counters.feature_hits == 0
        assert counters.groups_dispatched == 2
        assert counters.groups_skipped == 0
        assert counters.store_writes == 2
        assert len(fake_builder) == 2

    def test_per_feature_hits_and_groups_skipped_vs_dispatched(
        self, tmp_path: Path, fake_builder: list
    ) -> None:
        store = tmp_path / "store"
        plan = _make_plan("feat_a")
        _prepopulate(store, "NCT001", plan)
        counters = FeatureStoreCounters()
        _compute(store, ["NCT001", "NCT002"], {"feat_a": plan}, counters)
        assert counters.feature_lookups == 2
        assert counters.feature_hits == 1
        assert counters.groups_skipped == 1
        assert counters.groups_dispatched == 1
        assert counters.store_writes == 1
        assert fake_builder == [("NCT002", ["feat_a"])]

        # A second branch sending the same plan is served entirely from the store.
        _compute(store, ["NCT001", "NCT002"], {"feat_a": plan}, counters)
        assert counters.feature_lookups == 4
        assert counters.feature_hits == 3
        assert counters.groups_skipped == 3
        assert counters.groups_dispatched == 1
        assert counters.store_writes == 1
        assert counters.llm_calls_avoided_estimate == 3 * LLM_CALLS_PER_GROUP_BUILD
        assert len(fake_builder) == 1

    def test_partial_group_is_counted_once(self, tmp_path: Path, fake_builder: list) -> None:
        """The fence for the double-probe trap.

        ``feat_a`` is in the store, ``feat_b`` is not, and the grouper puts
        both in one group.  The batch probe sees the (nctid, group) pair as
        partially cached and dispatches it; the wrapper then re-probes both
        plans on disk.  Counted once: two lookups, one hit, one dispatch.
        """
        store = tmp_path / "store"
        plan_a, plan_b = _make_plan("feat_a"), _make_plan("feat_b")
        _prepopulate(store, "NCT001", plan_a)
        grouper = MagicMock(return_value=[{"feat_a": plan_a, "feat_b": plan_b}])
        counters = FeatureStoreCounters()
        _compute(store, ["NCT001"], {"feat_a": plan_a, "feat_b": plan_b}, counters, grouper=grouper)
        assert counters.feature_lookups == 2
        assert counters.feature_hits == 1
        assert counters.groups_dispatched == 1
        assert counters.groups_skipped == 0
        assert counters.llm_calls_avoided_estimate == 0
        assert counters.store_writes == 1
        assert fake_builder == [("NCT001", ["feat_b"])]

    def test_hits_are_split_by_plan_origin(self, tmp_path: Path, fake_builder: list) -> None:
        store = tmp_path / "store"
        plan = _make_plan("feat_a")
        _prepopulate(store, "NCT001", plan)
        counters = FeatureStoreCounters()
        _compute(store, ["NCT001"], {"feat_a": plan}, counters, plan_origin="initializer")
        assert counters.hits_from_initializer_plans == 1
        assert counters.hits_from_planner_plans == 0
        _compute(store, ["NCT001"], {"feat_a": plan}, counters, plan_origin="planner")
        assert counters.hits_from_initializer_plans == 1
        assert counters.hits_from_planner_plans == 1
        assert counters.feature_hits == 2

    def test_store_disabled_counts_dispatches_but_no_lookups(
        self, tmp_path: Path, fake_builder: list
    ) -> None:
        counters = FeatureStoreCounters()
        _compute(
            tmp_path / "store",
            ["NCT001", "NCT002"],
            {"feat_a": _make_plan("feat_a")},
            counters,
            enabled=False,
        )
        assert counters.feature_lookups == 0
        assert counters.feature_hits == 0
        assert counters.groups_dispatched == 2
        assert counters.groups_skipped == 0
        assert counters.store_writes == 0

    def test_counters_default_to_a_throwaway(self, tmp_path: Path, fake_builder: list) -> None:
        """``counters=None`` keeps the old call signature working."""
        result, _, _ = compute_features(
            grouper=MagicMock(),
            nctids=["NCT001"],
            task_description="test task",
            plans={"feat_a": _make_plan("feat_a")},
            feature_store_dir=tmp_path / "store",
            task_namespace=NAMESPACE,
        )
        assert result["NCT001"]["feat_a"]["value"] == "built_NCT001"


# ---------------------------------------------------------------------------
# The wrapper on its own owns the counts
# ---------------------------------------------------------------------------


class TestWrapperCounting:
    def test_standalone_wrapper_counts_its_probe(self, tmp_path: Path, fake_builder: list) -> None:
        store = tmp_path / "store"
        plan_a, plan_b = _make_plan("feat_a"), _make_plan("feat_b")
        _prepopulate(store, "NCT001", plan_a)
        counters = FeatureStoreCounters()
        wrapper = WrappedFeatureBuilder(
            "test task", store, NAMESPACE, counters=counters, plan_origin="initializer"
        )
        wrapper(("NCT001", {"feat_a": plan_a, "feat_b": plan_b}))
        assert counters.feature_lookups == 2
        assert counters.feature_hits == 1
        assert counters.hits_from_initializer_plans == 1
        assert counters.groups_dispatched == 1
        assert counters.store_writes == 1

        wrapper(("NCT001", {"feat_a": plan_a, "feat_b": plan_b}))
        assert counters.feature_lookups == 4
        assert counters.feature_hits == 3
        assert counters.groups_skipped == 1
        assert counters.groups_dispatched == 1

    def test_batch_probed_wrapper_counts_only_writes(
        self, tmp_path: Path, fake_builder: list
    ) -> None:
        store = tmp_path / "store"
        plan_a, plan_b = _make_plan("feat_a"), _make_plan("feat_b")
        _prepopulate(store, "NCT001", plan_a)
        counters = FeatureStoreCounters()
        wrapper = WrappedFeatureBuilder(
            "test task", store, NAMESPACE, counters=counters, batch_probed=True
        )
        wrapper(("NCT001", {"feat_a": plan_a, "feat_b": plan_b}))
        assert counters.feature_lookups == 0
        assert counters.feature_hits == 0
        assert counters.groups_dispatched == 0
        assert counters.store_writes == 1
