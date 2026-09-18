"""Tests for WrappedFeatureBuilder cache logic.

Covers:
- feature store integration: same plans -> same hash, different plans -> different hash
- __call__ store hit: write a JSON store file, verify it loads correctly
- __call__ store miss + exception: verify returns all-None dict
- __call__ partial build (issue #6): built features persisted, omitted ones
  filled with None + ``builder_omitted`` explanation and NOT persisted
- upfront batch query short-circuit for compute_features
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

try:
    from ctra.agents.data_models import (
        BUILDER_OMITTED_PREFIX,
        FeaturePlan,
        FeatureSource,
        FeatureType,
    )
    from ctra.agents.feature_builder import WrappedFeatureBuilder
    from ctra.agents.feature_store import (
        _plan_content_hash,
        get_cached_feature,
        put_cached_feature,
    )
    from tests.test_agents.conftest import builder_prediction

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan(name: str, idea: str = "test idea") -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=idea,
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[{"value": "1.0"}],
        possible_values={},
        feature_instructions="Extract value.",
    )


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / "feature_store"


@pytest.fixture()
def wrapper(store_dir: Path) -> WrappedFeatureBuilder:
    return WrappedFeatureBuilder(
        task_description="Predict trial success",
        feature_store_dir=store_dir,
        task_namespace="test",
        feature_store_enabled=True,
    )


# ======================================================================
# _plan_content_hash (per-feature)
# ======================================================================


class TestPlanContentHash:
    def test_same_plan_same_hash(self) -> None:
        plan_a = _make_plan("feat_a")
        hash1 = _plan_content_hash(plan_a)
        hash2 = _plan_content_hash(plan_a)
        assert hash1 == hash2

    def test_different_plans_different_hash(self) -> None:
        plan1 = _make_plan("feat_a")
        plan2 = _make_plan("feat_b")
        hash1 = _plan_content_hash(plan1)
        hash2 = _plan_content_hash(plan2)
        assert hash1 != hash2

    def test_different_idea_different_hash(self) -> None:
        plan1 = _make_plan("feat_a", idea="idea one")
        plan2 = _make_plan("feat_a", idea="idea two")
        hash1 = _plan_content_hash(plan1)
        hash2 = _plan_content_hash(plan2)
        assert hash1 != hash2

    def test_hash_is_16_char_hex(self) -> None:
        plan = _make_plan("feat_a")
        h = _plan_content_hash(plan)
        assert isinstance(h, str)
        # 16-char hex digest (64-bit key space)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# ======================================================================
# __call__ — store hit
# ======================================================================


class TestStoreHit:
    def test_loads_from_existing_store(
        self,
        wrapper: WrappedFeatureBuilder,
        store_dir: Path,
    ) -> None:
        """Pre-populate the store, then verify __call__ reads it."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctid = "NCT001"

        # Pre-populate store via put_cached_feature
        feature_values = {"feat_a": {"value": 1.5}}
        put_cached_feature(store_dir, "test", nctid, "feat_a", plan, feature_values)

        # Call the wrapper — should hit store
        result_nctid, result_values, result_meta = wrapper((nctid, plans))

        assert result_nctid == nctid
        # Note: result_values may be wrapped as {"feat_a": {...}} from the store hit
        assert result_values["feat_a"] == feature_values["feat_a"]
        assert result_meta == {}  # No meta on store hit

    def test_store_hit_does_not_call_builder(
        self,
        wrapper: WrappedFeatureBuilder,
        store_dir: Path,
    ) -> None:
        """On all-feature store hit, FeatureBuilder should not be instantiated."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctid = "NCT002"

        # Pre-populate store
        feature_values = {"feat_a": {"value": 2.0}}
        put_cached_feature(store_dir, "test", nctid, "feat_a", plan, feature_values)

        with patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls:
            wrapper((nctid, plans))
            mock_builder_cls.assert_not_called()


# ======================================================================
# __call__ — cache miss + exception
# ======================================================================


class TestCacheMissException:
    def test_exception_returns_all_none(
        self,
        wrapper: WrappedFeatureBuilder,
    ) -> None:
        """When FeatureBuilder raises, return all-None values."""
        plan = _make_plan("feat_a")
        plans = {"feat_a": plan}
        nctid = "NCT_FAIL"

        with (
            patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls,
            patch(
                "ctra.agents.feature_builder.ResettingRefine",
                side_effect=lambda module, **kw: module,
            ),
        ):
            mock_instance = MagicMock()
            mock_instance.side_effect = RuntimeError("LLM error")
            mock_builder_cls.return_value = mock_instance

            result_nctid, result_values, result_meta = wrapper((nctid, plans))

        assert result_nctid == nctid
        # All sub-feature keys should be None
        assert result_values == {"feat_a": {"value": None}}
        # Exception path now returns metadata with sentinels (not empty)
        assert "research_results" in result_meta
        assert result_meta["research_results"] == "[builder_exception]"
        assert "none_feature_explanations" in result_meta
        assert "feat_a" in result_meta["none_feature_explanations"]
        assert result_meta["none_feature_explanations"]["feat_a"].startswith("builder_exception:")

    def test_multi_plan_exception_returns_all_none(
        self,
        wrapper: WrappedFeatureBuilder,
    ) -> None:
        """Multiple plans with multi-valued types — all should be None on error."""
        plan_a = FeaturePlan(
            feature_name="drug_profile",
            feature_idea="Drug profile",
            feature_type={
                "mechanism": FeatureType.CATEGORICAL,
                "target_count": FeatureType.INTEGER,
            },
            data_sources=[FeatureSource.CHEMBL],
            example_values=[],
            possible_values={"mechanism": ["kinase", "antibody"]},
            feature_instructions="Look up drug profile.",
        )
        plan_b = _make_plan("safety_score")
        plans = {"drug_profile": plan_a, "safety_score": plan_b}
        nctid = "NCT_MULTI_FAIL"

        with (
            patch("ctra.agents.feature_builder.FeatureBuilder") as mock_builder_cls,
            patch(
                "ctra.agents.feature_builder.ResettingRefine",
                side_effect=lambda module, **kw: module,
            ),
        ):
            mock_instance = MagicMock()
            mock_instance.side_effect = RuntimeError("LLM error")
            mock_builder_cls.return_value = mock_instance

            result_nctid, result_values, _result_meta = wrapper((nctid, plans))

        assert result_nctid == nctid
        assert result_values["drug_profile"] == {
            "mechanism": None,
            "target_count": None,
        }
        assert result_values["safety_score"] == {"value": None}


# ======================================================================
# __call__ — cache miss + partial build (issue #6)
# ======================================================================


class TestPartialBuild:
    """The builder returns fewer features than planned and nothing raises.

    ``ResettingRefine`` is patched to identity so these cases target
    ``__call__``'s ordering (unwrap -> store writes -> omission fill -> merge),
    not Refine's retry loop (covered in test_refine_feedback_path.py).
    """

    @staticmethod
    def _two_plans() -> dict[str, FeaturePlan]:
        return {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}

    @staticmethod
    def _patched_builder(build_fn):
        """Patch FeatureBuilder with a double whose call runs ``build_fn``."""
        mock_instance = MagicMock(side_effect=build_fn)
        return (
            patch("ctra.agents.feature_builder.FeatureBuilder", return_value=mock_instance),
            patch(
                "ctra.agents.feature_builder.ResettingRefine",
                side_effect=lambda module, **kw: module,
            ),
            mock_instance,
        )

    def test_partial_build_persists_only_built_features(
        self, wrapper: WrappedFeatureBuilder, store_dir: Path
    ) -> None:
        """R9 + R11 + R12 in one call: feat_a persisted and real, feat_b filled
        with None, explained with the sentinel, and NOT written to the store."""
        plans = self._two_plans()
        nctid = "NCT_PARTIAL"
        builder_cls, refine_cls, _ = self._patched_builder(
            lambda **kw: builder_prediction(
                {"feat_a": {"value": 1.5}},
                {"research_results": "real research", "builder_reasoning": "r"},
            )
        )

        with builder_cls, refine_cls:
            result_nctid, values, meta = wrapper((nctid, plans))

        assert result_nctid == nctid
        # The built feature is the real value and was persisted.
        assert values["feat_a"] == {"value": 1.5}
        assert get_cached_feature(store_dir, "test", nctid, "feat_a", plans["feat_a"]) == {
            "feat_a": {"value": 1.5}
        }
        # R11: the omitted feature is NOT negatively cached.
        assert get_cached_feature(store_dir, "test", nctid, "feat_b", plans["feat_b"]) is None
        # R9: the omitted feature still has a real all-None dict.
        assert values["feat_b"] == {"value": None}
        # R12: the omission is visible to diagnostics via the sentinel entry.
        assert meta["none_feature_explanations"]["feat_b"].startswith(BUILDER_OMITTED_PREFIX)
        assert "feat_a" not in meta["none_feature_explanations"]
        # An omission is not a crash: research genuinely ran.
        assert meta["research_results"] == "real research"
        assert meta["research_results"] != "[builder_exception]"

    def test_five_plan_group_with_one_omission_keeps_the_other_four(
        self, wrapper: WrappedFeatureBuilder
    ) -> None:
        """The issue's acceptance case: 1 omission out of 5 no longer discards 4."""
        names = [f"feat_{i}" for i in range(5)]
        plans = {n: _make_plan(n) for n in names}
        built = {n: {"value": float(i)} for i, n in enumerate(names[:4])}
        builder_cls, refine_cls, _ = self._patched_builder(
            lambda **kw: builder_prediction(dict(built), {"research_results": "r"})
        )

        with builder_cls, refine_cls:
            _, values, meta = wrapper(("NCT_FIVE", plans))

        for n in names[:4]:
            assert values[n] == built[n]
            assert n not in meta["none_feature_explanations"]
        assert values["feat_4"] == {"value": None}
        assert meta["none_feature_explanations"]["feat_4"].startswith(BUILDER_OMITTED_PREFIX)
        assert meta["research_results"] == "r"

    def test_second_call_rebuilds_only_the_omitted_feature(
        self, wrapper: WrappedFeatureBuilder
    ) -> None:
        """Behavioural consequence of R11: an omission is retried next run."""
        plans = self._two_plans()
        builder_cls, refine_cls, mock_instance = self._patched_builder(
            lambda **kw: builder_prediction({"feat_a": {"value": 1.5}})
        )

        with builder_cls, refine_cls:
            wrapper(("NCT_TWICE", plans))
            wrapper(("NCT_TWICE", plans))

        assert mock_instance.call_count == 2
        first_group = mock_instance.call_args_list[0].kwargs["feature_plan_group"]
        second_group = mock_instance.call_args_list[1].kwargs["feature_plan_group"]
        assert set(first_group) == {"feat_a", "feat_b"}
        assert set(second_group) == {"feat_b"}

    def test_llm_explanation_is_preserved_after_the_sentinel(
        self, wrapper: WrappedFeatureBuilder
    ) -> None:
        """What the ``none_feature_explanations`` desc buys: the LLM's own reason
        survives behind the sentinel prefix."""
        plans = self._two_plans()
        llm_reason = "ChEMBL had no target data"
        builder_cls, refine_cls, _ = self._patched_builder(
            lambda **kw: builder_prediction(
                {"feat_a": {"value": 1.5}},
                {"none_feature_explanations": {"feat_b": llm_reason}},
            )
        )

        with builder_cls, refine_cls:
            _, _, meta = wrapper(("NCT_REASON", plans))

        reason = meta["none_feature_explanations"]["feat_b"]
        assert reason.startswith(BUILDER_OMITTED_PREFIX)
        assert llm_reason in reason

    def test_omission_without_explanation_gets_a_placeholder(
        self, wrapper: WrappedFeatureBuilder
    ) -> None:
        plans = self._two_plans()
        builder_cls, refine_cls, _ = self._patched_builder(
            lambda **kw: builder_prediction({"feat_a": {"value": 1.5}})
        )

        with builder_cls, refine_cls:
            _, _, meta = wrapper(("NCT_NOREASON", plans))

        assert meta["none_feature_explanations"]["feat_b"] == (
            f"{BUILDER_OMITTED_PREFIX} no explanation provided"
        )

    def test_legacy_tuple_double_still_builds(self, wrapper: WrappedFeatureBuilder) -> None:
        """Pins the pass-through tolerance the legacy ``(values, meta)`` doubles
        in test_feature_store_integration.py rely on: merge and fill both work."""
        plans = self._two_plans()
        builder_cls, refine_cls, _ = self._patched_builder(
            lambda **kw: ({"feat_a": {"value": 2.5}}, {"research_results": "legacy"})
        )

        with builder_cls, refine_cls:
            _, values, meta = wrapper(("NCT_LEGACY", plans))

        assert values == {"feat_a": {"value": 2.5}, "feat_b": {"value": None}}
        assert meta["research_results"] == "legacy"
        assert meta["none_feature_explanations"]["feat_b"].startswith(BUILDER_OMITTED_PREFIX)

    def test_complete_build_returns_metadata_untouched(
        self, wrapper: WrappedFeatureBuilder
    ) -> None:
        """Happy path allocates nothing: the exact metadata object comes back."""
        plans = self._two_plans()
        meta_in = {"research_results": "r", "none_feature_explanations": {}}
        builder_cls, refine_cls, _ = self._patched_builder(
            lambda **kw: builder_prediction(
                {"feat_a": {"value": 1.0}, "feat_b": {"value": 2.0}}, meta_in
            )
        )

        with builder_cls, refine_cls:
            _, values, meta_out = wrapper(("NCT_FULL", plans))

        assert values == {"feat_a": {"value": 1.0}, "feat_b": {"value": 2.0}}
        assert meta_out is meta_in
