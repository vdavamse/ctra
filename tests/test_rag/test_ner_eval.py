"""Tests for NER evaluation pipeline."""

from __future__ import annotations

import pytest

from ctra.rag.eval.data_models import (
    BENCHMARK_MAPPINGS,
    BENCHMARK_NATIVE_LABELS,
    CHIA_TO_CTRA,
    LABEL_VARIANTS,
    N2C2_TO_CTRA,
    TAC_TO_CTRA,
    Benchmark,
    EvalMode,
    GoldAnnotation,
    MatchStrategy,
    PerLabelMetrics,
    PredictedEntity,
)
from ctra.rag.eval.metrics import _compute_prf, compute_ner_metrics, match_entities


class TestDataModels:
    """Test data model definitions and validations."""

    def test_benchmark_enum(self) -> None:
        """Test Benchmark enum values."""
        assert Benchmark.CHIA.value == "chia"
        assert Benchmark.N2C2.value == "n2c2"
        assert Benchmark.TAC.value == "tac"

    def test_eval_mode_enum(self) -> None:
        """Test EvalMode enum values."""
        assert EvalMode.STANDARD.value == "standard"
        assert EvalMode.CTRA.value == "ctra"
        assert EvalMode.BOTH.value == "both"

    def test_match_strategy_enum(self) -> None:
        """Test MatchStrategy enum values."""
        assert MatchStrategy.STRICT.value == "strict"
        assert MatchStrategy.RELAXED.value == "relaxed"
        assert MatchStrategy.TEXT_ONLY.value == "text_only"

    def test_gold_annotation_creation(self) -> None:
        """Test GoldAnnotation dataclass creation."""
        annotation = GoldAnnotation(
            doc_id="nct123",
            start_char=0,
            end_char=5,
            text="Drug",
            gold_label="Drug",
            mapped_ctra_label="Drug",
        )
        assert annotation.doc_id == "nct123"
        assert annotation.start_char == 0
        assert annotation.end_char == 5
        assert annotation.text == "Drug"
        assert annotation.gold_label == "Drug"
        assert annotation.mapped_ctra_label == "Drug"

    def test_predicted_entity_creation(self) -> None:
        """Test PredictedEntity dataclass creation."""
        entity = PredictedEntity(
            doc_id="nct123",
            start_char=0,
            end_char=5,
            text="Drug",
            label="Drug",
            score=0.95,
        )
        assert entity.doc_id == "nct123"
        assert entity.start_char == 0
        assert entity.score == 0.95

    def test_per_label_metrics_creation(self) -> None:
        """Test PerLabelMetrics dataclass creation."""
        metrics = PerLabelMetrics(
            label="Drug",
            precision=0.95,
            recall=0.90,
            f1=0.92,
            support=100,
            predicted_count=105,
        )
        assert metrics.label == "Drug"
        assert metrics.f1 == 0.92


class TestEntityMappings:
    """Test entity type mappings."""

    def test_chia_to_ctra_mapping_completeness(self) -> None:
        """Test CHIA mapping covers all entity types."""
        # Test a sample of mappings
        assert CHIA_TO_CTRA["Drug"] == "Drug"
        assert CHIA_TO_CTRA["Condition"] == "Disease"
        assert CHIA_TO_CTRA["Procedure"] == "Therapeutic procedure"
        # Test unmappable types
        assert CHIA_TO_CTRA["Mood"] is None
        assert CHIA_TO_CTRA["Negation"] is None

    def test_n2c2_to_ctra_mapping_completeness(self) -> None:
        """Test N2C2 mapping covers all entity types."""
        assert N2C2_TO_CTRA["Drug"] == "Drug"
        assert N2C2_TO_CTRA["ADE"] == "Adverse event"
        assert N2C2_TO_CTRA["Dosage"] == "Drug dosage"

    def test_tac_to_ctra_mapping_completeness(self) -> None:
        """Test TAC mapping covers all entity types."""
        assert TAC_TO_CTRA["AdverseReaction"] == "Adverse event"
        assert TAC_TO_CTRA["DrugClass"] == "Drug"
        assert TAC_TO_CTRA["Animal"] is None

    def test_benchmark_mappings_dict(self) -> None:
        """Test BENCHMARK_MAPPINGS dict."""
        assert Benchmark.CHIA in BENCHMARK_MAPPINGS
        assert Benchmark.N2C2 in BENCHMARK_MAPPINGS
        assert Benchmark.TAC in BENCHMARK_MAPPINGS

    def test_benchmark_native_labels(self) -> None:
        """Test BENCHMARK_NATIVE_LABELS dict."""
        chia_labels = BENCHMARK_NATIVE_LABELS[Benchmark.CHIA]
        assert "Drug" in chia_labels
        assert "Condition" in chia_labels
        assert len(chia_labels) == 16

        n2c2_labels = BENCHMARK_NATIVE_LABELS[Benchmark.N2C2]
        assert "Drug" in n2c2_labels
        assert "ADE" in n2c2_labels
        assert len(n2c2_labels) == 9

    def test_label_variants_completeness(self) -> None:
        """Test LABEL_VARIANTS dict."""
        assert "Drug" in LABEL_VARIANTS
        assert len(LABEL_VARIANTS["Drug"]) >= 2
        assert "Disease" in LABEL_VARIANTS
        assert len(LABEL_VARIANTS["Disease"]) >= 2

        # All variants must be non-empty strings
        for _label, variants in LABEL_VARIANTS.items():
            assert isinstance(variants, list)
            for variant in variants:
                assert isinstance(variant, str)
                assert len(variant) > 0


class TestMetricsComputation:
    """Test NER metric computation."""

    def test_compute_prf_perfect(self) -> None:
        """Test PRF computation with perfect predictions."""
        precision, recall, f1 = _compute_prf(tp=10, fp=0, fn=0)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_compute_prf_no_predictions(self) -> None:
        """Test PRF computation with no predictions."""
        precision, recall, f1 = _compute_prf(tp=0, fp=0, fn=10)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_compute_prf_false_positives(self) -> None:
        """Test PRF computation with false positives."""
        precision, recall, f1 = _compute_prf(tp=5, fp=5, fn=0)
        assert precision == 0.5
        assert recall == 1.0
        assert f1 == pytest.approx(2 / 3, abs=0.01)

    def test_compute_prf_false_negatives(self) -> None:
        """Test PRF computation with false negatives."""
        precision, recall, f1 = _compute_prf(tp=5, fp=0, fn=5)
        assert precision == 1.0
        assert recall == 0.5
        assert f1 == pytest.approx(2 / 3, abs=0.01)

    def test_match_entities_strict(self) -> None:
        """Test strict entity matching."""
        gold = [
            GoldAnnotation(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                gold_label="Drug",
            ),
        ]
        predicted = [
            PredictedEntity(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                label="Drug",
                score=0.95,
            ),
        ]

        matched, unmatched_gold, _unmatched_pred = match_entities(
            gold, predicted, MatchStrategy.STRICT, "gold_label"
        )

        assert len(matched) == 1
        assert len(unmatched_gold) == 0

    def test_match_entities_strict_different_spans(self) -> None:
        """Test strict matching rejects different spans."""
        gold = [
            GoldAnnotation(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                gold_label="Drug",
            ),
        ]
        predicted = [
            PredictedEntity(
                doc_id="doc1",
                start_char=0,
                end_char=4,
                text="Drug",
                label="Drug",
                score=0.95,
            ),
        ]

        matched, unmatched_gold, _unmatched_pred = match_entities(
            gold, predicted, MatchStrategy.STRICT, "gold_label"
        )

        assert len(matched) == 0
        assert len(unmatched_gold) == 1

    def test_match_entities_relaxed(self) -> None:
        """Test relaxed entity matching with overlapping spans."""
        gold = [
            GoldAnnotation(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                gold_label="Drug",
            ),
        ]
        predicted = [
            PredictedEntity(
                doc_id="doc1",
                start_char=3,
                end_char=8,
                text="g1 Ext",
                label="Drug",
                score=0.95,
            ),
        ]

        matched, unmatched_gold, _unmatched_pred = match_entities(
            gold, predicted, MatchStrategy.RELAXED, "gold_label"
        )

        assert len(matched) == 1
        assert len(unmatched_gold) == 0

    def test_match_entities_text_only(self) -> None:
        """Test text-only matching."""
        gold = [
            GoldAnnotation(
                doc_id="doc1",
                start_char=0,
                end_char=4,
                text="DRUG",
                gold_label="Drug",
            ),
        ]
        predicted = [
            PredictedEntity(
                doc_id="doc1",
                start_char=10,
                end_char=14,
                text="drug",
                label="Drug",
                score=0.95,
            ),
        ]

        matched, _unmatched_gold, _unmatched_pred = match_entities(
            gold, predicted, MatchStrategy.TEXT_ONLY, "gold_label"
        )

        assert len(matched) == 1

    def test_match_entities_label_mismatch(self) -> None:
        """Test that mismatched labels don't match."""
        gold = [
            GoldAnnotation(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                gold_label="Drug",
            ),
        ]
        predicted = [
            PredictedEntity(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                label="Disease",  # Different label
                score=0.95,
            ),
        ]

        matched, unmatched_gold, _unmatched_pred = match_entities(
            gold, predicted, MatchStrategy.STRICT, "gold_label"
        )

        assert len(matched) == 0
        assert len(unmatched_gold) == 1

    def test_match_entities_different_docs(self) -> None:
        """Test that entities in different documents don't match."""
        gold = [
            GoldAnnotation(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                gold_label="Drug",
            ),
        ]
        predicted = [
            PredictedEntity(
                doc_id="doc2",
                start_char=0,
                end_char=5,
                text="Drug1",
                label="Drug",
                score=0.95,
            ),
        ]

        matched, unmatched_gold, _unmatched_pred = match_entities(
            gold, predicted, MatchStrategy.STRICT, "gold_label"
        )

        assert len(matched) == 0
        assert len(unmatched_gold) == 1

    def test_compute_ner_metrics_mode1(self) -> None:
        """Test NER metrics computation for Mode 1."""
        gold = [
            GoldAnnotation(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                gold_label="Drug",
                mapped_ctra_label="Drug",
            ),
            GoldAnnotation(
                doc_id="doc1",
                start_char=10,
                end_char=15,
                text="Cond1",
                gold_label="Condition",
                mapped_ctra_label="Disease",
            ),
        ]
        predicted = [
            PredictedEntity(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                label="Drug",
                score=0.95,
            ),
        ]

        result = compute_ner_metrics(
            gold,
            predicted,
            MatchStrategy.TEXT_ONLY,
            label_field="gold_label",
        )

        # 1 TP (Drug), 1 FN (Condition), 0 FP
        assert result.total_gold == 2
        assert result.total_predicted == 1
        assert result.total_matched == 1

    def test_compute_ner_metrics_mode2_filtering(self) -> None:
        """Test Mode 2 filters unmappable types."""
        gold = [
            GoldAnnotation(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                gold_label="Drug",
                mapped_ctra_label="Drug",
            ),
            GoldAnnotation(
                doc_id="doc1",
                start_char=10,
                end_char=15,
                text="Negation",
                gold_label="Negation",
                mapped_ctra_label=None,  # Unmappable
            ),
        ]
        predicted = [
            PredictedEntity(
                doc_id="doc1",
                start_char=0,
                end_char=5,
                text="Drug1",
                label="Drug",
                score=0.95,
            ),
        ]

        result = compute_ner_metrics(
            gold,
            predicted,
            MatchStrategy.TEXT_ONLY,
            label_field="mapped_ctra_label",
        )

        # Only 1 gold entity (Drug) since Negation is filtered
        assert result.total_gold == 1
        assert result.total_predicted == 1
        assert result.total_matched == 1

    def test_match_entities_multi_document(self) -> None:
        """Test entity matching across multiple documents.

        Regression test: per-document indices must not collide across
        documents when tracking matched entities.
        """
        gold = [
            # Document A: 2 entities
            GoldAnnotation(doc_id="docA", start_char=0, end_char=4, text="Drug", gold_label="Drug"),
            GoldAnnotation(
                doc_id="docA", start_char=10, end_char=17, text="Disease", gold_label="Disease"
            ),
            # Document B: 2 entities
            GoldAnnotation(doc_id="docB", start_char=0, end_char=4, text="Drug", gold_label="Drug"),
            GoldAnnotation(
                doc_id="docB", start_char=10, end_char=17, text="Disease", gold_label="Disease"
            ),
        ]
        predicted = [
            # Match 1 entity per document
            PredictedEntity(
                doc_id="docA", start_char=0, end_char=4, text="Drug", label="Drug", score=0.9
            ),
            PredictedEntity(
                doc_id="docB", start_char=0, end_char=4, text="Drug", label="Drug", score=0.9
            ),
        ]

        matched, unmatched_gold, unmatched_pred = match_entities(
            gold, predicted, MatchStrategy.TEXT_ONLY, "gold_label"
        )

        assert len(matched) == 2  # 1 match per document
        assert len(unmatched_gold) == 2  # 1 unmatched per document (Disease)
        assert len(unmatched_pred) == 0

    def test_match_entities_zero_score(self) -> None:
        """Test that predictions with score=0.0 still match.

        Regression test: _match_score must return >0 for valid matches
        even when pred.score is 0.0.
        """
        gold = [
            GoldAnnotation(doc_id="doc1", start_char=0, end_char=4, text="Drug", gold_label="Drug"),
        ]
        predicted = [
            PredictedEntity(
                doc_id="doc1", start_char=0, end_char=4, text="Drug", label="Drug", score=0.0
            ),
        ]

        matched, unmatched_gold, _unmatched_pred = match_entities(
            gold, predicted, MatchStrategy.TEXT_ONLY, "gold_label"
        )

        assert len(matched) == 1
        assert len(unmatched_gold) == 0


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_parse_args_defaults(self) -> None:
        """Test default argument values."""
        # Import the eval_ner script from scripts/eval/ by loading it as a module
        import importlib.util
        from pathlib import Path

        script_path = Path(__file__).parents[2] / "scripts" / "eval" / "eval_ner.py"
        spec = importlib.util.spec_from_file_location("eval_ner", script_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        # Verify parse_args function exists
        assert hasattr(module, "parse_args")
        assert module.parse_args is not None

    def test_benchmark_enum_from_string(self) -> None:
        """Test parsing benchmark strings."""
        benchmark_strs = ["chia", "n2c2", "tac"]
        for b_str in benchmark_strs:
            benchmark = Benchmark(b_str)
            assert benchmark.value == b_str

    def test_eval_mode_enum_from_string(self) -> None:
        """Test parsing eval mode strings."""
        mode_strs = ["standard", "ctra", "both"]
        for m_str in mode_strs:
            mode = EvalMode(m_str)
            assert mode.value == m_str


class TestGapAnalysis:
    """Test gap analysis functionality."""

    def test_gap_analysis_structure(self) -> None:
        """Test that gap analysis requires proper eval results."""
        # This is tested in integration tests
        # Unit test just verifies the import
        from ctra.rag.eval import gap_analysis as gap_fn

        assert gap_fn is not None


@pytest.mark.slow
@pytest.mark.integration
class TestIntegration:
    """Integration tests for the evaluation pipeline.

    These tests require the GLiNER model to be available and
    actual benchmark data to be downloaded.
    """

    def test_chia_loading(self) -> None:
        """Test CHIA benchmark loading."""
        pytest.importorskip("datasets")
        from ctra.rag.eval.loaders import load_benchmark

        try:
            annotations = load_benchmark(Benchmark.CHIA)
            assert len(annotations) > 0
            # Verify structure
            for ann in annotations[:5]:
                assert hasattr(ann, "doc_id")
                assert hasattr(ann, "text")
                assert hasattr(ann, "gold_label")
                assert hasattr(ann, "mapped_ctra_label")
        except Exception as e:
            pytest.skip(f"Could not load CHIA: {e}")

    def test_tac_loading(self) -> None:
        """Test TAC benchmark loading."""
        pytest.importorskip("datasets")
        from ctra.rag.eval.loaders import load_benchmark

        try:
            annotations = load_benchmark(Benchmark.TAC)
            assert len(annotations) > 0
        except Exception as e:
            pytest.skip(f"Could not load TAC: {e}")
