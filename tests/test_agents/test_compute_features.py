"""Tests for compute_features aggregation logic.

Covers:
- Single plan skips grouper
- Multiple plans calls grouper
- Result aggregation (nctid merging via |=)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

try:
    from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
    from ctra.agents.feature_builder import compute_features
    from tests.test_agents.conftest import grouper_prediction

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan(name: str) -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[{"value": "1.0"}],
        possible_values={},
        feature_instructions=f"Extract {name}.",
    )


@pytest.fixture()
def mock_wrapper():
    """Patch WrappedFeatureBuilder to return deterministic results."""

    def _factory(feature_values_fn):
        """Create a mock wrapper whose __call__ uses feature_values_fn."""
        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            instance = MagicMock()
            instance.side_effect = feature_values_fn
            mock_cls.return_value = instance
            yield mock_cls, instance

    return _factory


# ======================================================================
# Single plan — skips grouper
# ======================================================================


class TestSinglePlanSkipsGrouper:
    def test_single_plan_does_not_call_grouper(self, tmp_path: Path) -> None:
        """When there's only one plan, grouper should NOT be called."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctids = ["NCT001", "NCT002"]

        grouper = MagicMock()

        # Mock the wrapper to return feature values
        def mock_call(arg):
            nctid, _plan_group = arg
            return (
                nctid,
                {"feat_a": {"value": 1.0}},
                {"none_feature_explanations": {}},
            )

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            raw_features, _none_explanations, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        # Grouper should NOT have been called
        grouper.assert_not_called()

        # Results should still be returned for each nctid
        assert "NCT001" in raw_features
        assert "NCT002" in raw_features


# ======================================================================
# Multiple plans — calls grouper
# ======================================================================


class TestMultiplePlansCallGrouper:
    def test_multiple_plans_calls_grouper(self, tmp_path: Path) -> None:
        """When there are multiple plans, grouper should be called."""
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        plans = {"feat_a": plan_a, "feat_b": plan_b}
        nctids = ["NCT001"]

        # Grouper returns a single group containing both features
        grouper = MagicMock(return_value=[plans])

        def mock_call(arg):
            nctid, plan_group = arg
            values = {name: {"value": 1.0} for name in plan_group}
            return (nctid, values, {"none_feature_explanations": {}})

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            _raw_features, _, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        # Grouper should have been called
        grouper.assert_called_once_with(feature_plans=plans, task="test task")

    def test_grouper_splits_into_two_groups(self, tmp_path: Path) -> None:
        """When grouper splits plans into two groups, both groups are processed."""
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        plans = {"feat_a": plan_a, "feat_b": plan_b}
        nctids = ["NCT001"]

        # Grouper splits into two groups
        group1 = {"feat_a": plan_a}
        group2 = {"feat_b": plan_b}
        grouper = MagicMock(return_value=[group1, group2])

        call_count = {"n": 0}

        def mock_call(arg):
            nctid, plan_group = arg
            call_count["n"] += 1
            values = {name: {"value": float(call_count["n"])} for name in plan_group}
            return (nctid, values, {})

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            _raw_features, _, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        # Should have been called twice (1 nctid x 2 groups)
        assert call_count["n"] == 2


# ======================================================================
# Site 3 — grouper partition fallback
#
# ``FeatureGrouper.forward()` no longer raises on a bad partition; it filters
# and returns whatever survives. Without a fallback here, an empty or partial
# partition would silently drop features from the build entirely.
# ======================================================================


class TestGrouperPartitionFallback:
    def _run(self, grouper, plans, nctids):
        """Returns (raw_features, built_feature_names, group_count).

        ``group_count`` is the number of builder invocations per trial, i.e. how
        many groups the build actually ran -- the signal that distinguishes a
        repaired partition from one rebuilt as one-feature-per-group.
        """
        built: list[str] = []
        groups_seen: list[tuple[str, ...]] = []

        def mock_call(arg):
            nctid, plan_group = arg
            built.extend(plan_group)
            groups_seen.append(tuple(sorted(plan_group)))
            return (nctid, {name: {"value": 1.0} for name in plan_group}, {})

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)
            raw_features, _, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )
        return raw_features, built, groups_seen

    def test_empty_partition_still_builds_every_feature(self) -> None:
        """Grouper returning [] must not silently drop every feature."""
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        grouper = MagicMock(return_value=[])

        raw_features, built, _groups = self._run(grouper, plans, ["NCT001"])

        assert sorted(built) == ["feat_a", "feat_b"]
        assert set(raw_features["NCT001"]) == {"feat_a", "feat_b"}

    def test_partial_partition_is_repaired_not_rebuilt(self) -> None:
        """A partition missing feat_c gains feat_c -- it does not lose its batching.

        Rebuilding as one-feature-per-group would cost 3 research passes here
        instead of 2, and ~5x at realistic feature counts.
        """
        plans = {
            "feat_a": _make_plan("feat_a"),
            "feat_b": _make_plan("feat_b"),
            "feat_c": _make_plan("feat_c"),
        }
        grouper = MagicMock(return_value=[{"feat_a": plans["feat_a"], "feat_b": plans["feat_b"]}])

        raw_features, built, groups = self._run(grouper, plans, ["NCT001"])

        assert sorted(built) == ["feat_a", "feat_b", "feat_c"]
        assert set(raw_features["NCT001"]) == {"feat_a", "feat_b", "feat_c"}
        # The good group survives intact; feat_c is appended as a singleton.
        assert groups == [("feat_a", "feat_b"), ("feat_c",)]

    def test_duplicate_partition_builds_nothing_twice(self) -> None:
        """A feature assigned to two groups must not be built twice."""
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        grouper = MagicMock(
            return_value=[
                {"feat_a": plans["feat_a"], "feat_b": plans["feat_b"]},
                {"feat_a": plans["feat_a"]},
            ]
        )

        _raw_features, built, groups = self._run(grouper, plans, ["NCT001"])

        assert sorted(built) == ["feat_a", "feat_b"]
        assert groups == [("feat_a", "feat_b")]

    def test_stray_name_in_partition_is_dropped(self) -> None:
        """A name the grouper invented is not passed to the builder."""
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        grouper = MagicMock(
            return_value=[{"feat_a": plans["feat_a"], "ghost": _make_plan("ghost")}]
        )

        _raw_features, built, groups = self._run(grouper, plans, ["NCT001"])

        assert "ghost" not in built
        assert sorted(built) == ["feat_a", "feat_b"]
        assert groups == [("feat_a",), ("feat_b",)]

    def test_valid_partition_keeps_its_grouping(self) -> None:
        """A well-formed partition must not be touched.

        Asserted on the group shape, not just the feature set: if the fallback
        fired, the same two features would arrive as two groups of one and the
        ~5x batching saving would be silently gone.
        """
        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}
        grouper = MagicMock(return_value=[{"feat_a": plans["feat_a"], "feat_b": plans["feat_b"]}])

        _raw_features, built, groups = self._run(grouper, plans, ["NCT001"])

        assert sorted(built) == ["feat_a", "feat_b"]
        assert groups == [("feat_a", "feat_b")], "fallback fired on a valid partition"


# ======================================================================
# Result aggregation (nctid merging via |=)
# ======================================================================


class TestResultAggregation:
    def test_features_merged_across_groups(self, tmp_path: Path) -> None:
        """Features from separate groups should be merged into one dict per nctid."""
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        plans = {"feat_a": plan_a, "feat_b": plan_b}
        nctids = ["NCT001"]

        group1 = {"feat_a": plan_a}
        group2 = {"feat_b": plan_b}
        grouper = MagicMock(return_value=[group1, group2])

        def mock_call(arg):
            nctid, plan_group = arg
            if "feat_a" in plan_group:
                return (nctid, {"feat_a": {"value": 1.0}}, {})
            else:
                return (nctid, {"feat_b": {"value": 2.0}}, {})

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            raw_features, _, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        assert "feat_a" in raw_features["NCT001"]
        assert "feat_b" in raw_features["NCT001"]
        assert raw_features["NCT001"]["feat_a"]["value"] == 1.0
        assert raw_features["NCT001"]["feat_b"]["value"] == 2.0

    def test_none_explanations_merged(self, tmp_path: Path) -> None:
        """None explanations from separate groups should be merged."""
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        plans = {"feat_a": plan_a, "feat_b": plan_b}
        nctids = ["NCT001"]

        group1 = {"feat_a": plan_a}
        group2 = {"feat_b": plan_b}
        grouper = MagicMock(return_value=[group1, group2])

        def mock_call(arg):
            nctid, plan_group = arg
            if "feat_a" in plan_group:
                return (
                    nctid,
                    {"feat_a": {"value": None}},
                    {"none_feature_explanations": {"feat_a": "No data found"}},
                )
            else:
                return (
                    nctid,
                    {"feat_b": {"value": None}},
                    {"none_feature_explanations": {"feat_b": "Ambiguous data"}},
                )

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            _, none_explanations, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        assert none_explanations["NCT001"]["feat_a"] == "No data found"
        assert none_explanations["NCT001"]["feat_b"] == "Ambiguous data"

    def test_multiple_nctids(self, tmp_path: Path) -> None:
        """Each nctid should get its own entry in the output."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctids = ["NCT001", "NCT002", "NCT003"]

        grouper = MagicMock()  # not called for single plan

        counter = {"n": 0}

        def mock_call(arg):
            nctid, _ = arg
            counter["n"] += 1
            return (nctid, {"feat_a": {"value": float(counter["n"])}}, {})

        with patch("ctra.agents.feature_builder.WrappedFeatureBuilder") as mock_cls:
            mock_cls.return_value = MagicMock(side_effect=mock_call)

            raw_features, _, _ = compute_features(
                grouper=grouper,
                nctids=nctids,
                task_description="test task",
                plans=plans,
            )

        assert len(raw_features) == 3
        assert "NCT001" in raw_features
        assert "NCT002" in raw_features
        assert "NCT003" in raw_features


class TestBuilderExceptionMetadata:
    """Tests for builder exception metadata handling."""

    def test_builder_exception_emits_none_explanations(self, tmp_path: Path) -> None:
        """Builder exception should emit none_explanations with sentinel."""
        from ctra.agents.data_models import BUILDER_EXCEPTION_PREFIX
        from ctra.agents.feature_builder import WrappedFeatureBuilder

        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}

        with patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls:
            # Raise exception on builder call
            mock_builder_cls.side_effect = RuntimeError("Test exception")

            builder = WrappedFeatureBuilder(
                task_description="test",
                feature_store_dir=None,
                feature_store_enabled=False,
            )

            nctid = "NCT001"
            _, _, meta = builder((nctid, plans))

            # Extract from meta (the third returned element from __call__)
            none_feature_explanations = meta.get("none_feature_explanations", {})

            # All uncached plans should have sentinel explanations
            assert "feat_a" in none_feature_explanations
            assert "feat_b" in none_feature_explanations
            assert none_feature_explanations["feat_a"].startswith(BUILDER_EXCEPTION_PREFIX)
            assert none_feature_explanations["feat_b"].startswith(BUILDER_EXCEPTION_PREFIX)
            assert meta["research_results"] == "[builder_exception]"

    def test_builder_exception_is_visible_to_diagnostics(self, tmp_path: Path) -> None:
        """Builder exception should surface as builder_exception reason in diagnostics."""
        from ctra.agents.feature_builder import WrappedFeatureBuilder
        from ctra.agents.orchestrator import _build_builder_diagnostics

        plans = {"feat_a": _make_plan("feat_a")}

        with patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls:
            mock_builder_cls.side_effect = RuntimeError("Test exception")

            builder = WrappedFeatureBuilder(
                task_description="test",
                feature_store_dir=None,
                feature_store_enabled=False,
            )

            nctid = "NCT001"
            _, _, meta = builder((nctid, plans))

            none_explanations = {nctid: meta.get("none_feature_explanations", {})}
            builder_meta = {nctid: meta}

            # Build diagnostics
            diagnostics = _build_builder_diagnostics(none_explanations, plans, builder_meta)

            # Check the diagnostic for feat_a
            diag = next(d for d in diagnostics.feature_diagnostics if d.feature_name == "feat_a")
            assert diag.none_rate == 1.0
            assert diag.dominant_failure_reason == "builder_exception"

            # format_for_llm should not report as "All features have low None rates"
            formatted = diagnostics.format_for_llm()
            assert "All features have low None rates" not in formatted
            assert "BUILDER" in formatted

    def test_builder_exception_preserves_cached_siblings(self, tmp_path: Path) -> None:
        """Exception on uncached features should not stamp exception metadata onto cached siblings."""
        from ctra.agents.feature_builder import compute_features
        from ctra.agents.feature_store import put_cached_feature

        plans = {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}

        # Pre-cache feat_a
        feature_store_dir = tmp_path / "feature_store"
        feature_store_dir.mkdir()

        nctid = "NCT001"
        nctids = [nctid]
        put_cached_feature(
            str(feature_store_dir),
            "phase2",
            nctid,
            "feat_a",
            plans["feat_a"],
            {"feat_a": {"value": 1.5}},
            metadata={"research_results": "cached"},
        )

        with patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls:
            # Raise on uncached plans
            mock_builder_cls.side_effect = RuntimeError("Test exception")

            # Two plans, so the grouper is genuinely called; this lambda keeps
            # both in one group so the cached/uncached split happens per group.
            _, _, builder_meta = compute_features(
                grouper=lambda feature_plans, task: [feature_plans],
                nctids=nctids,
                task_description="test",
                plans=plans,
                feature_store_dir=feature_store_dir,
                task_namespace="phase2",
                feature_store_enabled=True,
            )

            # feat_a (cached) should have "[cached]" sentinel, not exception metadata
            assert builder_meta[nctid]["feat_a"]["research_results"] == "[cached]"
            assert builder_meta[nctid]["feat_a"]["builder_reasoning"] == "[cached]"

            # feat_b (failed) should have exception metadata
            assert builder_meta[nctid]["feat_b"]["research_results"] == "[builder_exception]"
            assert "builder_exception:" in builder_meta[nctid]["feat_b"]["builder_reasoning"]

    def test_builder_exception_message_is_truncated(self, tmp_path: Path) -> None:
        """Long exception messages should be truncated to BUILDER_EXCEPTION_MSG_MAXLEN."""
        from ctra.agents.data_models import (
            BUILDER_EXCEPTION_MSG_MAXLEN,
            BUILDER_EXCEPTION_PREFIX,
        )
        from ctra.agents.feature_builder import WrappedFeatureBuilder

        plans = {"feat_a": _make_plan("feat_a")}

        with patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls:
            # Create a very long exception message
            long_msg = "X" * 5000
            mock_builder_cls.side_effect = RuntimeError(long_msg)

            builder = WrappedFeatureBuilder(
                task_description="test",
                feature_store_dir=None,
                feature_store_enabled=False,
            )

            _, _, meta = builder(("NCT001", plans))

            reason = meta.get("none_feature_explanations", {}).get("feat_a", "")
            # Reason format: f"{BUILDER_EXCEPTION_PREFIX} {detail}" where detail is
            # "RuntimeError: " + message, truncated to MAXLEN - 3 chars plus "...".
            assert reason.endswith("...")
            # Exact length: prefix + one space + MAXLEN (truncation is deterministic).
            expected_length = len(BUILDER_EXCEPTION_PREFIX) + 1 + BUILDER_EXCEPTION_MSG_MAXLEN
            assert len(reason) == expected_length

    def test_builder_exception_reason_names_the_exception_type(self, tmp_path: Path) -> None:
        """Exception type name should be included in the reason."""
        from ctra.agents.feature_builder import WrappedFeatureBuilder

        plans = {"feat_a": _make_plan("feat_a")}

        with patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls:
            mock_builder_cls.side_effect = ValueError("Test value error")

            builder = WrappedFeatureBuilder(
                task_description="test",
                feature_store_dir=None,
                feature_store_enabled=False,
            )

            _, _, meta = builder(("NCT001", plans))

            reason = meta.get("none_feature_explanations", {}).get("feat_a", "")
            assert "ValueError" in reason

    def test_builder_exception_writes_nothing_to_the_store(self, tmp_path: Path) -> None:
        """Exception path should not write to feature store (no negative caching)."""
        from ctra.agents.feature_builder import WrappedFeatureBuilder
        from ctra.agents.feature_store import get_cached_feature

        plans = {"feat_a": _make_plan("feat_a")}
        feature_store_dir = tmp_path / "feature_store"
        feature_store_dir.mkdir()

        with patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls:
            mock_builder_cls.side_effect = RuntimeError("Test exception")

            builder = WrappedFeatureBuilder(
                task_description="test",
                feature_store_dir=feature_store_dir,
                task_namespace="phase2",
                feature_store_enabled=True,
            )

            _, _, _ = builder(("NCT001", plans))

            # Check that no feature store entry was created using get_cached_feature
            cached = get_cached_feature(
                str(feature_store_dir), "phase2", "NCT001", "feat_a", plans["feat_a"]
            )
            assert cached is None


# ---------------------------------------------------------------------------
# Grouper Prediction shape tolerance
# ---------------------------------------------------------------------------


class TestGrouperPredictionUnwrap:
    """Grouper can return dspy.Prediction(groups=...) and it is unwrapped correctly."""

    def _make_plan(self, name: str) -> FeaturePlan:
        return FeaturePlan(
            feature_name=name,
            feature_idea=f"{name} idea",
            feature_type={"value": FeatureType.FLOAT},
            data_sources=[FeatureSource.PUBMED],
            example_values=[{"value": "1.0"}],
            possible_values={},
            feature_instructions=f"Extract {name}.",
        )

    @staticmethod
    def _patched_builder():
        """Patch the Refine-wrapped builder so each call echoes its group.

        Returns the patch context and the callable that stands in for the
        wrapped ``FeatureBuilder``; its ``call_args_list`` records the exact
        ``feature_plan_group`` dispatched for each (nctid, group) pair.
        """
        fake_builder = MagicMock(
            side_effect=lambda nctid, feature_plan_group: (
                {name: {"value": 1.0} for name in feature_plan_group},
                {"research_results": "r", "builder_reasoning": "b"},
            )
        )
        ctx = patch("ctra.agents.feature_builder.ResettingRefine", return_value=fake_builder)
        return ctx, fake_builder

    def test_grouper_prediction_is_unwrapped_to_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Prediction(groups=...) is unwrapped and each group reaches the builder intact.

        Discriminating on purpose: ``is_valid_grouper`` unwraps internally, so an
        un-unwrapped Prediction *passes* Site 3 and is then iterated as-is --
        ``dspy.Prediction`` yields its keys, so the builder receives the string
        ``'groups'`` as a "group" and ``dict(plans)`` raises inside the wrapper.
        The Site 3 repair cannot catch a missing unwrap; this test is the fence.
        """
        nctids = ["NCT001"]
        task = "Test task"
        plans = {name: self._make_plan(name) for name in ("feat_a", "feat_b", "feat_c")}

        groups_list = [
            {"feat_a": plans["feat_a"], "feat_b": plans["feat_b"]},
            {"feat_c": plans["feat_c"]},
        ]
        mock_grouper = MagicMock(return_value=grouper_prediction(groups_list))
        builder_patch, fake_builder = self._patched_builder()

        with builder_patch, caplog.at_level(logging.WARNING):
            raw_features, none_explanations, builder_meta = compute_features(
                grouper=mock_grouper,
                nctids=nctids,
                task_description=task,
                plans=plans,
                feature_store_enabled=False,
            )

        mock_grouper.assert_called_once_with(feature_plans=plans, task=task)
        assert "Grouper partition invalid" not in caplog.text
        # The builder was dispatched exactly the supplied groups, in order.
        dispatched = [call.kwargs["feature_plan_group"] for call in fake_builder.call_args_list]
        assert dispatched == groups_list
        assert set(raw_features["NCT001"]) == set(plans)
        assert none_explanations == {"NCT001": {}}
        assert set(builder_meta["NCT001"]) == set(plans)

    def test_empty_grouper_prediction_behaves_like_empty_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Prediction(groups=[]) is unwrapped to ``[]`` and repaired by Site 3.

        Two plans so the grouper is actually consulted (a single plan bypasses
        it). The empty partition assigns nothing, so the repair rebuilds it as
        one-feature-per-group and logs the warning.
        """
        nctids = ["NCT001"]
        task = "Test task"
        plans = {"feat_a": self._make_plan("feat_a"), "feat_b": self._make_plan("feat_b")}

        mock_grouper = MagicMock(return_value=grouper_prediction([]))
        builder_patch, fake_builder = self._patched_builder()

        with builder_patch, caplog.at_level(logging.WARNING):
            raw_features, _, _ = compute_features(
                grouper=mock_grouper,
                nctids=nctids,
                task_description=task,
                plans=plans,
                feature_store_enabled=False,
            )

        mock_grouper.assert_called_once_with(feature_plans=plans, task=task)
        assert "Grouper partition invalid" in caplog.text
        dispatched = [call.kwargs["feature_plan_group"] for call in fake_builder.call_args_list]
        assert dispatched == [{"feat_a": plans["feat_a"]}, {"feat_b": plans["feat_b"]}]
        assert set(raw_features["NCT001"]) == {"feat_a", "feat_b"}
