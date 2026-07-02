"""Tests for build_feature_type_transformer.

Covers:
- CATEGORICAL -> OneHotEncoder
- MULTICATEGORICAL -> CountVectorizer pipeline
- INTEGER -> passthrough
- BOOLEAN -> passthrough
- FLOAT -> passthrough
- skip parameter
- empty plans
"""

from __future__ import annotations

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from ctra.agents.data_models import FeaturePlan, FeatureSource, FeatureType
    from ctra.agents.feature_utils import build_feature_type_transformer

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan(
    name: str,
    feature_type: dict[str, FeatureType] | None = None,
    possible_values: dict[str, list[str]] | None = None,
) -> FeaturePlan:
    ft = feature_type or {"value": FeatureType.FLOAT}
    pv = possible_values or {}
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type=ft,
        data_sources=[FeatureSource.PUBMED],
        example_values=[],
        possible_values=pv,
        feature_instructions=f"Extract {name}.",
    )


def _get_transformer_by_name(
    ct: ColumnTransformer, name_substring: str
) -> tuple[str, object, object]:
    """Find a transformer in the ColumnTransformer by name substring."""
    for tname, transformer, columns in ct.transformers:
        if name_substring in tname:
            return tname, transformer, columns
    raise KeyError(f"No transformer with name containing '{name_substring}'")


# ======================================================================
# CATEGORICAL -> OneHotEncoder
# ======================================================================


class TestCategorical:
    def test_categorical_uses_onehot_encoder(self) -> None:
        plan = _make_plan(
            "drug_type",
            feature_type={"value": FeatureType.CATEGORICAL},
            possible_values={"value": ["kinase", "antibody", "other"]},
        )
        ct = build_feature_type_transformer({"drug_type": plan}, model_type="xgb")

        _name, transformer, cols = _get_transformer_by_name(ct, "cat_onehot")
        assert isinstance(transformer, OneHotEncoder)
        assert cols == ["drug_type--value"]

    def test_categorical_categories_match_possible_values(self) -> None:
        pv = ["a", "b", "c"]
        plan = _make_plan(
            "cat_feat",
            feature_type={"value": FeatureType.CATEGORICAL},
            possible_values={"value": pv},
        )
        ct = build_feature_type_transformer({"cat_feat": plan}, model_type="xgb")
        _, transformer, _ = _get_transformer_by_name(ct, "cat_onehot")
        assert transformer.categories == [pv]


# ======================================================================
# MULTICATEGORICAL -> CountVectorizer pipeline
# ======================================================================


class TestMulticategorical:
    def test_multicategorical_uses_pipeline(self) -> None:
        plan = _make_plan(
            "targets",
            feature_type={"value": FeatureType.MULTICATEGORICAL},
            possible_values={"value": ["egfr", "vegf", "her2"]},
        )
        ct = build_feature_type_transformer({"targets": plan}, model_type="xgb")

        _name, transformer, col = _get_transformer_by_name(ct, "multicat")
        assert isinstance(transformer, Pipeline)
        # Should be a single column string (not list)
        assert col == "targets--value"

    def test_multicategorical_pipeline_has_countvectorizer(self) -> None:
        plan = _make_plan(
            "targets",
            feature_type={"value": FeatureType.MULTICATEGORICAL},
            possible_values={"value": ["egfr", "vegf"]},
        )
        ct = build_feature_type_transformer({"targets": plan}, model_type="xgb")
        _, transformer, _ = _get_transformer_by_name(ct, "multicat")

        # Check that the pipeline contains a CountVectorizer
        has_cv = any(isinstance(step, CountVectorizer) for _, step in transformer.steps)
        assert has_cv


# ======================================================================
# INTEGER — passthrough
# ======================================================================


class TestInteger:
    def test_integer_xgb_passthrough(self) -> None:
        plan = _make_plan(
            "count_feat",
            feature_type={"value": FeatureType.INTEGER},
        )
        ct = build_feature_type_transformer({"count_feat": plan}, model_type="xgb")
        _name, transformer, _cols = _get_transformer_by_name(ct, "int")
        assert transformer == "passthrough"

    def test_integer_tabpfn_passthrough(self) -> None:
        plan = _make_plan(
            "count_feat",
            feature_type={"value": FeatureType.INTEGER},
        )
        ct = build_feature_type_transformer({"count_feat": plan}, model_type="tabpfn")
        _name, transformer, _cols = _get_transformer_by_name(ct, "int")
        assert transformer == "passthrough"


# ======================================================================
# BOOLEAN — xgb passthrough
# ======================================================================


class TestBoolean:
    def test_boolean_xgb_passthrough(self) -> None:
        plan = _make_plan(
            "is_fda",
            feature_type={"value": FeatureType.BOOLEAN},
        )
        ct = build_feature_type_transformer({"is_fda": plan}, model_type="xgb")
        _name, transformer, _cols = _get_transformer_by_name(ct, "bool")
        assert transformer == "passthrough"


# ======================================================================
# FLOAT — xgb passthrough
# ======================================================================


class TestFloat:
    def test_float_xgb_passthrough(self) -> None:
        plan = _make_plan(
            "score",
            feature_type={"value": FeatureType.FLOAT},
        )
        ct = build_feature_type_transformer({"score": plan}, model_type="xgb")
        _name, transformer, _cols = _get_transformer_by_name(ct, "passthrough")
        assert transformer == "passthrough"


# ======================================================================
# Skip parameter
# ======================================================================


class TestSkipParameter:
    def test_skip_excludes_feature(self) -> None:
        plan_a = _make_plan("feat_a")
        plan_b = _make_plan("feat_b")
        plans = {"feat_a": plan_a, "feat_b": plan_b}

        ct = build_feature_type_transformer(plans, model_type="xgb", skip=["feat_a"])

        transformer_names = [name for name, _, _ in ct.transformers]
        # feat_a should not be present in any transformer name
        assert not any("feat_a" in name for name in transformer_names)
        # feat_b should be present
        assert any("feat_b" in name for name in transformer_names)

    def test_skip_multiple_features(self) -> None:
        plans = {
            "a": _make_plan("a"),
            "b": _make_plan("b"),
            "c": _make_plan("c"),
        }
        ct = build_feature_type_transformer(plans, model_type="xgb", skip=["a", "b"])

        transformer_names = [name for name, _, _ in ct.transformers]
        assert not any("a--" in name for name in transformer_names)
        assert not any("b--" in name for name in transformer_names)
        assert any("c--" in name for name in transformer_names)

    def test_skip_none_includes_all(self) -> None:
        plan = _make_plan("feat_a")
        ct = build_feature_type_transformer({"feat_a": plan}, model_type="xgb", skip=None)

        transformer_names = [name for name, _, _ in ct.transformers]
        assert any("feat_a" in name for name in transformer_names)


# ======================================================================
# Empty plans
# ======================================================================


class TestEmptyPlans:
    def test_empty_plans_returns_empty_transformer(self) -> None:
        ct = build_feature_type_transformer({}, model_type="xgb")
        assert isinstance(ct, ColumnTransformer)
        assert len(ct.transformers) == 0

    def test_empty_plans_remainder_is_drop(self) -> None:
        ct = build_feature_type_transformer({}, model_type="xgb")
        assert ct.remainder == "drop"


# ======================================================================
# Multi-valued features
# ======================================================================


class TestMultiValuedFeatures:
    def test_multi_valued_feature_creates_multiple_transformers(self) -> None:
        plan = _make_plan(
            "drug_profile",
            feature_type={
                "mechanism": FeatureType.CATEGORICAL,
                "target_count": FeatureType.INTEGER,
            },
            possible_values={"mechanism": ["kinase", "antibody"]},
        )
        ct = build_feature_type_transformer({"drug_profile": plan}, model_type="xgb")

        transformer_names = [name for name, _, _ in ct.transformers]
        assert any("mechanism" in name for name in transformer_names)
        assert any("target_count" in name for name in transformer_names)

    def test_column_names_use_double_dash_format(self) -> None:
        plan = _make_plan(
            "drug_profile",
            feature_type={
                "mechanism": FeatureType.CATEGORICAL,
            },
            possible_values={"mechanism": ["kinase"]},
        )
        ct = build_feature_type_transformer({"drug_profile": plan}, model_type="xgb")

        _, _, cols = _get_transformer_by_name(ct, "mechanism")
        assert cols == ["drug_profile--mechanism"]
