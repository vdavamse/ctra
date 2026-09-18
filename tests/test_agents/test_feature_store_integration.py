"""Integration test: 2-branch feature store sharing.

Demonstrates that when two MCTS branches independently propose the same
feature plan for the same trials, the second branch pays zero LLM cost
because the first branch's results are cached in the shared feature store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

try:
    from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
    from ctra.agents.feature_builder import compute_features

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


def _make_plan(name: str = "feat_x", idea: str = "shared feature") -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=idea,
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[{"value": "1.0"}],
        possible_values={},
        feature_instructions="Extract value.",
    )


class TestTwoBranchesShareStore:
    """Branch A and Branch B both request the same (nctid, plan).
    The builder is invoked exactly once across both branches."""

    def test_shared_feature_store_reduces_llm_calls(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "feature_store"
        plan = _make_plan("feat_x", idea="shared feature")
        nctids = ["NCT001", "NCT002"]

        build_count = 0

        def fake_build(nctid: str, feature_plan_group: dict) -> tuple:
            nonlocal build_count
            build_count += 1
            return (
                {fn: {"value": "42"} for fn in feature_plan_group},
                {"builder_reasoning": "mock", "research_results": "mock"},
            )

        with (
            patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls,
            patch(
                "ctra.agents.feature_builder.ResettingRefine",
                side_effect=lambda module, **kw: module,
            ),
        ):
            mock_builder_cls.return_value = MagicMock(side_effect=fake_build)

            # Branch A: first time seeing this plan
            compute_features(
                grouper=MagicMock(),
                nctids=nctids,
                task_description="test task",
                plans={"feat_x": plan},
                feature_store_dir=store_dir,
                task_namespace="phase2",
                feature_store_enabled=True,
            )
            builds_after_a = build_count

            # Branch B: same plan, same nctids, independent invocation
            compute_features(
                grouper=MagicMock(),
                nctids=nctids,
                task_description="test task",
                plans={"feat_x": plan},
                feature_store_dir=store_dir,
                task_namespace="phase2",
                feature_store_enabled=True,
            )
            builds_after_b = build_count

        # Branch A should have built for all nctids
        assert builds_after_a == len(nctids), (
            f"Branch A should build {len(nctids)} features, but built {builds_after_a}"
        )

        # Branch B should build zero additional features (all from store)
        assert builds_after_b == builds_after_a, (
            f"Branch B should add zero builds, but added {builds_after_b - builds_after_a}"
        )

    def test_partial_cache_hit_rebuilds_only_misses(self, tmp_path: Path) -> None:
        """If store has feat_x for NCT001 but not NCT002, rebuild only NCT002."""
        store_dir = tmp_path / "feature_store"
        plan = _make_plan("feat_x")
        nctids = ["NCT001", "NCT002"]

        # Pre-populate store for NCT001 only
        from ctra.agents.feature_store import put_cached_feature

        put_cached_feature(
            store_dir,
            "phase2",
            "NCT001",
            "feat_x",
            plan,
            {"feat_x": {"value": "cached"}},
        )

        build_count = 0

        def fake_build(nctid: str, feature_plan_group: dict) -> tuple:
            nonlocal build_count
            build_count += 1
            return (
                {fn: {"value": f"built_for_{nctid}"} for fn in feature_plan_group},
                {"builder_reasoning": "mock"},
            )

        with (
            patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls,
            patch(
                "ctra.agents.feature_builder.ResettingRefine",
                side_effect=lambda module, **kw: module,
            ),
        ):
            mock_builder_cls.return_value = MagicMock(side_effect=fake_build)

            result, _, _ = compute_features(
                grouper=MagicMock(),
                nctids=nctids,
                task_description="test task",
                plans={"feat_x": plan},
                feature_store_dir=store_dir,
                task_namespace="phase2",
                feature_store_enabled=True,
            )

        # Should build only for NCT002 (NCT001 was pre-cached)
        assert build_count == 1, f"Expected 1 build (NCT002), got {build_count}"

        # Result should have both NCT001 and NCT002
        assert "NCT001" in result
        assert "NCT002" in result
        assert result["NCT001"]["feat_x"]["value"] == "cached"
        assert "built_for_NCT002" in str(result["NCT002"]["feat_x"]["value"])

    def test_fully_cached_trials_populate_builder_meta(self, tmp_path: Path) -> None:
        """Cached trials must not look like RESEARCHER failures to diagnostics.

        Regression: the upfront short-circuit accumulated feature values for
        fully-cached trials but skipped builder_meta, which drove
        research_coverage_score to 0 for cached features and flipped them
        into the RESEARCHER attribution bucket once the store warmed up.
        """
        from ctra.agents.feature_store import put_cached_feature

        store_dir = tmp_path / "feature_store"
        plan = _make_plan("feat_x")
        nctids = ["NCT001", "NCT002"]

        # Pre-populate store for both nctids so compute_features takes the
        # fully-cached short-circuit path for every trial.
        for nctid in nctids:
            put_cached_feature(
                store_dir,
                "phase2",
                nctid,
                "feat_x",
                plan,
                {"feat_x": {"value": "cached"}},
            )

        with (
            patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls,
            patch(
                "ctra.agents.feature_builder.ResettingRefine",
                side_effect=lambda module, **kw: module,
            ),
        ):
            mock_builder_cls.return_value = MagicMock(
                side_effect=AssertionError("builder should not be called"),
            )

            _, _, builder_meta = compute_features(
                grouper=MagicMock(),
                nctids=nctids,
                task_description="test task",
                plans={"feat_x": plan},
                feature_store_dir=store_dir,
                task_namespace="phase2",
                feature_store_enabled=True,
            )

        # Every cached trial must carry a non-empty research_results entry so
        # _build_builder_diagnostics computes research_coverage_score == 1.0.
        for nctid in nctids:
            assert nctid in builder_meta
            assert "feat_x" in builder_meta[nctid]
            assert builder_meta[nctid]["feat_x"]["research_results"]

        research_count = sum(
            1 for nctid in nctids if builder_meta[nctid]["feat_x"].get("research_results")
        )
        assert research_count / len(nctids) == 1.0


class TestExceptionSentinelDistinctness:
    """Test that exception and cached sentinels are distinct and don't collide."""

    def test_exception_sentinel_is_distinct_from_cached_sentinel(self) -> None:
        """BUILDER_EXCEPTION_RESEARCH_SENTINEL must not equal the cached sentinel."""
        from ctra.agents.data_models import BUILDER_EXCEPTION_RESEARCH_SENTINEL

        # The cached sentinel is hardcoded in compute_features
        cached_sentinel = "[cached]"

        assert cached_sentinel != BUILDER_EXCEPTION_RESEARCH_SENTINEL
        assert BUILDER_EXCEPTION_RESEARCH_SENTINEL == "[builder_exception]"
