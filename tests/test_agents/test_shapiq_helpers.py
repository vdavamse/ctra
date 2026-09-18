"""Tests for shapiq-related helpers (Issue #18).

Covers:
- interaction_values_to_dict (feature_utils)
- format_interactions_for_llm (feature_utils)
- BuilderDiagnostics.format_for_llm (data_models)
- _build_builder_diagnostics (orchestrator)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers — mock shapiq InteractionValues-like object
# ---------------------------------------------------------------------------


def _make_mock_iv(
    order1: dict[tuple[int, ...], float],
    order2: dict[tuple[int, ...], float] | None = None,
) -> MagicMock:
    """Create a mock that behaves like shapiq.InteractionValues."""
    iv = MagicMock()

    def get_n_order(order: int = 1) -> SimpleNamespace:
        if order == 1:
            return SimpleNamespace(dict_values=order1)
        elif order == 2:
            return SimpleNamespace(dict_values=order2 or {})
        return SimpleNamespace(dict_values={})

    iv.get_n_order = get_n_order
    return iv


# ---------------------------------------------------------------------------
# interaction_values_to_dict
# ---------------------------------------------------------------------------


class TestInteractionValuesToDict:
    def test_basic_conversion(self) -> None:
        from ctra.agents.feature_utils import interaction_values_to_dict

        iv = _make_mock_iv(
            order1={(0,): 0.5, (1,): -0.3},
            order2={(0, 1): 0.15},
        )
        result = interaction_values_to_dict(iv, ["feat_a", "feat_b"], "k-SII", 2)

        assert result["main_effects"] == {"feat_a": 0.5, "feat_b": -0.3}
        assert result["interactions"] == {"(feat_a, feat_b)": 0.15}
        assert result["feature_names"] == ["feat_a", "feat_b"]
        assert result["index_type"] == "k-SII"
        assert result["max_order"] == 2

    def test_max_order_1_no_interactions(self) -> None:
        from ctra.agents.feature_utils import interaction_values_to_dict

        iv = _make_mock_iv(order1={(0,): 0.5, (1,): -0.3})
        result = interaction_values_to_dict(iv, ["feat_a", "feat_b"], "SV", 1)

        assert result["main_effects"] == {"feat_a": 0.5, "feat_b": -0.3}
        assert result["interactions"] == {}

    def test_out_of_bounds_index_uses_fallback_name(self) -> None:
        from ctra.agents.feature_utils import interaction_values_to_dict

        iv = _make_mock_iv(
            order1={(0,): 0.5, (99,): -0.1},
            order2={(0, 99): 0.05},
        )
        result = interaction_values_to_dict(iv, ["feat_a"], "k-SII", 2)

        assert "feat_a" in result["main_effects"]
        assert "f99" in result["main_effects"]
        assert "(feat_a, f99)" in result["interactions"]

    def test_fsii_index_type(self) -> None:
        from ctra.agents.feature_utils import interaction_values_to_dict

        iv = _make_mock_iv(order1={(0,): 0.2})
        result = interaction_values_to_dict(iv, ["x"], "FSII", 2)
        assert result["index_type"] == "FSII"


# ---------------------------------------------------------------------------
# format_interactions_for_llm
# ---------------------------------------------------------------------------


class TestFormatInteractionsForLlm:
    def test_empty_dict(self) -> None:
        from ctra.agents.feature_utils import format_interactions_for_llm

        assert format_interactions_for_llm({}) == "No interaction values available."

    def test_empty_main_effects(self) -> None:
        from ctra.agents.feature_utils import format_interactions_for_llm

        result = format_interactions_for_llm({"main_effects": {}, "interactions": {}})
        assert result == "No interaction values available."

    def test_basic_formatting(self) -> None:
        from ctra.agents.feature_utils import format_interactions_for_llm

        data = {
            "main_effects": {"feat_a": 0.5, "feat_b": -0.3},
            "interactions": {"(feat_a, feat_b)": 0.15},
            "index_type": "k-SII",
            "max_order": 2,
        }
        result = format_interactions_for_llm(data)

        assert "k-SII" in result
        assert "feat_a" in result
        assert "positive contribution" in result
        assert "feat_b" in result
        assert "negative contribution" in result
        assert "synergistic" in result

    def test_zero_value_labeled_neutral(self) -> None:
        from ctra.agents.feature_utils import format_interactions_for_llm

        data = {
            "main_effects": {"feat_a": 0.0},
            "interactions": {},
            "index_type": "k-SII",
            "max_order": 2,
        }
        result = format_interactions_for_llm(data)
        assert "neutral" in result

    def test_top_k_limiting(self) -> None:
        from ctra.agents.feature_utils import format_interactions_for_llm

        data = {
            "main_effects": {f"feat_{i}": float(i) for i in range(20)},
            "interactions": {},
            "index_type": "k-SII",
            "max_order": 2,
        }
        result = format_interactions_for_llm(data, top_k_main=3)
        # Should only contain top 3 by absolute value: feat_19, feat_18, feat_17
        assert "feat_19" in result
        assert "feat_0" not in result  # excluded by top_k

    def test_near_zero_interactions_filtered(self) -> None:
        from ctra.agents.feature_utils import format_interactions_for_llm

        data = {
            "main_effects": {"feat_a": 0.5},
            "interactions": {"(feat_a, feat_b)": 1e-8},
            "index_type": "k-SII",
            "max_order": 2,
        }
        result = format_interactions_for_llm(data)
        # Near-zero interaction should be filtered
        assert "feat_a, feat_b" not in result

    def test_redundant_interaction_labeled(self) -> None:
        from ctra.agents.feature_utils import format_interactions_for_llm

        data = {
            "main_effects": {"feat_a": 0.5},
            "interactions": {"(feat_a, feat_b)": -0.2},
            "index_type": "k-SII",
            "max_order": 2,
        }
        result = format_interactions_for_llm(data)
        assert "redundant/conflicting" in result


# ---------------------------------------------------------------------------
# BuilderDiagnostics.format_for_llm
# ---------------------------------------------------------------------------


class TestBuilderDiagnosticsFormatForLlm:
    def test_empty_diagnostics(self) -> None:
        from ctra.agents.data_models import BuilderDiagnostics

        diag = BuilderDiagnostics()
        assert diag.format_for_llm() == "No builder diagnostics available."

    def test_all_low_none_rates(self) -> None:
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        diag = BuilderDiagnostics(
            feature_diagnostics=[
                FeatureDiagnostic("feat_a", 0.01, "none", 0.9),
                FeatureDiagnostic("feat_b", 0.03, "none", 0.8),
            ]
        )
        assert diag.format_for_llm() == "All features have low None rates."

    def test_researcher_attribution(self) -> None:
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        diag = BuilderDiagnostics(
            feature_diagnostics=[
                FeatureDiagnostic("bad_feature", 0.9, "insufficient_data", 0.1),
            ]
        )
        result = diag.format_for_llm()
        assert "RESEARCHER" in result
        assert "bad_feature" in result

    def test_builder_attribution(self) -> None:
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        diag = BuilderDiagnostics(
            feature_diagnostics=[
                FeatureDiagnostic("broken_feature", 0.5, "extraction_error", 0.8),
            ]
        )
        result = diag.format_for_llm()
        assert "BUILDER" in result
        assert "broken_feature" in result

    def test_unclear_attribution(self) -> None:
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        diag = BuilderDiagnostics(
            feature_diagnostics=[
                FeatureDiagnostic("ambiguous_feature", 0.6, "other", 0.4),
            ]
        )
        result = diag.format_for_llm()
        assert "UNCLEAR" in result

    def test_sorted_by_none_rate_descending(self) -> None:
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        diag = BuilderDiagnostics(
            feature_diagnostics=[
                FeatureDiagnostic("low", 0.3, "other", 0.5),
                FeatureDiagnostic("high", 0.9, "insufficient_data", 0.1),
                FeatureDiagnostic("medium", 0.5, "extraction_error", 0.8),
            ]
        )
        result = diag.format_for_llm()
        # "high" should appear before "medium" before "low"
        assert result.index("high") < result.index("medium")


# ---------------------------------------------------------------------------
# _build_builder_diagnostics
# ---------------------------------------------------------------------------


class TestBuildBuilderDiagnostics:
    def test_basic_diagnostics(self) -> None:
        from ctra.agents.orchestrator import _build_builder_diagnostics

        none_explanations = {
            "NCT001": {"feat_a": "No data found for this drug"},
            "NCT002": {"feat_a": "Insufficient information available"},
            "NCT003": {},
        }
        feature_plans = {"feat_a": {}, "feat_b": {}}
        builder_meta = {
            "NCT001": {
                "feat_a": {"research_results": "Found drug info"},
                "feat_b": {"research_results": "Found trial"},
            },
            "NCT002": {
                "feat_a": {"research_results": ""},
                "feat_b": {"research_results": "Found data"},
            },
            "NCT003": {
                "feat_a": {"research_results": "Found data"},
                "feat_b": {"research_results": "Found data"},
            },
        }

        result = _build_builder_diagnostics(none_explanations, feature_plans, builder_meta)

        # feat_a: 2/3 None rate, feat_b: 0/3 None rate
        diag_a = next(d for d in result.feature_diagnostics if d.feature_name == "feat_a")
        diag_b = next(d for d in result.feature_diagnostics if d.feature_name == "feat_b")

        assert abs(diag_a.none_rate - 2 / 3) < 0.01
        assert diag_a.dominant_failure_reason == "insufficient_data"
        assert abs(diag_a.research_coverage_score - 2 / 3) < 0.01

        assert diag_b.none_rate == 0.0
        assert diag_b.dominant_failure_reason == "none"
        assert diag_b.research_coverage_score == 1.0

    def test_empty_inputs(self) -> None:
        from ctra.agents.orchestrator import _build_builder_diagnostics

        result = _build_builder_diagnostics({}, {}, {})
        assert result.feature_diagnostics == []

    def test_extraction_error_classification(self) -> None:
        from ctra.agents.orchestrator import _build_builder_diagnostics

        none_explanations = {
            "NCT001": {"feat_a": "API call failed with timeout"},
            "NCT002": {"feat_a": "Exception during extraction"},
        }
        result = _build_builder_diagnostics(none_explanations, {"feat_a": {}}, {})
        diag = result.feature_diagnostics[0]
        assert diag.dominant_failure_reason == "extraction_error"

    def test_ambiguity_classification(self) -> None:
        from ctra.agents.orchestrator import _build_builder_diagnostics

        none_explanations = {
            "NCT001": {"feat_a": "Ambiguous: multiple possible values"},
            "NCT002": {"feat_a": "Unclear which value to use"},
        }
        result = _build_builder_diagnostics(none_explanations, {"feat_a": {}}, {})
        diag = result.feature_diagnostics[0]
        assert diag.dominant_failure_reason == "ambiguity"

    def test_no_none_explanations_zero_rate(self) -> None:
        from ctra.agents.orchestrator import _build_builder_diagnostics

        builder_meta = {
            "NCT001": {"feat_a": {"research_results": "data"}},
            "NCT002": {"feat_a": {"research_results": "data"}},
        }
        result = _build_builder_diagnostics({}, {"feat_a": {}}, builder_meta)
        diag = result.feature_diagnostics[0]
        assert diag.none_rate == 0.0
        assert diag.research_coverage_score == 1.0
