"""NER evaluation orchestrator.

Coordinates benchmark data loading, GLiNER-BioMed inference, and metric
computation for Mode 1 (standard benchmark) and Mode 2 (CTRA-mapped)
evaluation. Also implements threshold sweep and label phrasing ablation.

Reuses build_ner_pipeline() from ctra.rag.ner_config to construct the
GLiNER pipeline, ensuring consistency with production NER configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from ctra.config.settings import RAGConfig
from ctra.rag.eval.data_models import (
    BENCHMARK_NATIVE_LABELS,
    CHIA_CONDITION_VALID_LABELS,
    CTRA_LABELS,
    Benchmark,
    EvalResult,
    MatchStrategy,
    PredictedEntity,
)
from ctra.rag.eval.loaders import DatasetNotAvailableError, get_document_texts, load_benchmark
from ctra.rag.eval.metrics import compute_ner_metrics

if TYPE_CHECKING:
    from spacy.language import Language


logger = logging.getLogger(__name__)


class NERExperiment:
    """Runs NER evaluation experiments.

    Encapsulates the loaded GLiNER model to avoid reloading for each
    benchmark/mode/threshold combination.

    Args:
        gliner_model: Model identifier (default: "Ihor/gliner-biomed-large-v1.0").
        match_strategy: Entity matching strategy (default: text_only).
    """

    def __init__(
        self,
        gliner_model: str = "Ihor/gliner-biomed-large-v1.0",
        match_strategy: MatchStrategy = MatchStrategy.TEXT_ONLY,
    ) -> None:
        self._gliner_model = gliner_model
        self._match_strategy = match_strategy
        self._nlp: Language | None = None  # Lazy-loaded
        self._current_labels: list[str] | None = None
        self._current_threshold: float = 0.4

    def _get_or_build_pipeline(
        self,
        labels: list[str],
        threshold: float = 0.4,
    ) -> Language:
        """Build or reconfigure the GLiNER pipeline.

        Strategy for threshold/label changes without full reload:
        - gliner-spacy stores config on the pipeline component
        - We attempt to update labels/threshold on the existing component
        - If that fails, rebuild the full pipeline

        This is critical for threshold sweep performance: building
        the pipeline loads the 1.5GB model, which takes ~30s.
        """
        # Check if we can reuse the pipeline with label/threshold update
        if (
            self._nlp is not None
            and labels == self._current_labels
            and threshold == self._current_threshold
        ):
            return self._nlp

        # Lazy import: spacy is an optional dependency ([rag] extra).
        from ctra.rag.ner_config import build_ner_pipeline

        # If labels changed, must rebuild
        if labels != self._current_labels:
            logger.info(f"Building GLiNER pipeline with {len(labels)} labels")
            config = RAGConfig()
            config.ner_labels = labels
            config.ner_threshold = threshold
            self._nlp = build_ner_pipeline(config)
            self._current_labels = labels
            self._current_threshold = threshold
        else:
            # Only threshold changed; try to update in-place
            logger.info(f"Updating GLiNER threshold to {threshold}")
            assert self._nlp is not None
            try:
                gliner_pipe = self._nlp.get_pipe("gliner_spacy")
                gliner_pipe.cfg["threshold"] = threshold  # type: ignore[attr-defined,unused-ignore]
                self._current_threshold = threshold
            except Exception as e:
                # Fall back to rebuilding
                logger.warning(f"Failed to update threshold in-place: {e}. Rebuilding pipeline.")
                config = RAGConfig()
                config.ner_labels = labels
                config.ner_threshold = threshold
                self._nlp = build_ner_pipeline(config)
                self._current_threshold = threshold

        return self._nlp

    def run_mode1(
        self,
        benchmarks: list[Benchmark],
    ) -> list[EvalResult]:
        """Run Mode 1 (standard benchmark evaluation) on specified benchmarks.

        For each benchmark:
        1. Load gold annotations via loaders.load_benchmark()
        2. Get document texts via loaders.get_document_texts()
        3. Build GLiNER pipeline with the benchmark's native labels
        4. Run NER inference over all documents
        5. Compute metrics with match_entities() using gold_label field
        6. Return EvalResult with metadata (model, threshold, etc.)

        Returns list of EvalResult, one per benchmark.
        """
        results = []

        for benchmark in benchmarks:
            logger.info(f"Running Mode 1 evaluation on {benchmark.value}...")

            try:
                gold_annotations = load_benchmark(benchmark)
                doc_texts = get_document_texts(benchmark)
            except DatasetNotAvailableError as e:
                logger.warning(f"Skipping {benchmark.value}: {e}")
                continue

            # Get native labels for this benchmark
            native_labels = BENCHMARK_NATIVE_LABELS[benchmark]

            # Build pipeline with native labels
            self._get_or_build_pipeline(native_labels, threshold=0.4)

            # Run inference
            predictions = self._run_inference(doc_texts, native_labels, threshold=0.4)

            # Compute metrics
            eval_result = compute_ner_metrics(
                gold_annotations,
                predictions,
                self._match_strategy,
                label_field="gold_label",
            )
            eval_result.benchmark = benchmark.value
            eval_result.mode = "standard"
            eval_result.metadata = {
                "model": self._gliner_model,
                "threshold": "0.4",
                "benchmark_native_labels": str(len(native_labels)),
            }

            results.append(eval_result)
            logger.info(f"Mode 1 {benchmark.value}: micro_f1={eval_result.micro_f1:.3f}")

        return results

    def run_mode2(
        self,
        benchmarks: list[Benchmark],
        threshold: float = 0.4,
        labels: list[str] | None = None,
    ) -> list[EvalResult]:
        """Run Mode 2 (CTRA-mapped evaluation) on specified benchmarks.

        For each benchmark:
        1. Load gold annotations (with mapped_ctra_label populated)
        2. Filter out gold annotations where mapped_ctra_label is None
        3. Build GLiNER pipeline with CTRA's 16 labels (or custom labels)
        4. Run NER inference over all documents
        5. Compute metrics using mapped_ctra_label field
        6. Return EvalResult per benchmark

        Also computes combined metrics across all benchmarks.
        """
        if labels is None:
            labels = CTRA_LABELS

        results = []

        for benchmark in benchmarks:
            logger.info(f"Running Mode 2 evaluation on {benchmark.value}...")

            try:
                gold_annotations = load_benchmark(benchmark)
                doc_texts = get_document_texts(benchmark)
            except DatasetNotAvailableError as e:
                logger.warning(f"Skipping {benchmark.value}: {e}")
                continue

            # Build pipeline with CTRA labels
            self._get_or_build_pipeline(labels, threshold=threshold)

            # Run inference
            predictions = self._run_inference(doc_texts, labels, threshold=threshold)

            # Build valid_labels map for multi-label matching.
            # CHIA Condition can match Disease, Symptom, Adverse event, etc.
            valid_labels_map: dict[str, set[str]] | None = None
            if benchmark == Benchmark.CHIA:
                valid_labels_map = {
                    "Disease": CHIA_CONDITION_VALID_LABELS,
                }

            # Compute metrics
            eval_result = compute_ner_metrics(
                gold_annotations,
                predictions,
                self._match_strategy,
                label_field="mapped_ctra_label",
                valid_labels=valid_labels_map,
            )
            eval_result.benchmark = benchmark.value
            eval_result.mode = "ctra"
            eval_result.metadata = {
                "model": self._gliner_model,
                "threshold": str(threshold),
                "ctra_labels": str(len(labels)),
            }

            results.append(eval_result)
            logger.info(f"Mode 2 {benchmark.value}: micro_f1={eval_result.micro_f1:.3f}")

        return results

    def run_threshold_sweep(
        self,
        benchmark: Benchmark = Benchmark.CHIA,
        start: float = 0.1,
        stop: float = 0.9,
        step: float = 0.05,
    ) -> list[EvalResult]:
        """Sweep ner_threshold and run Mode 2 at each value.

        Returns list of EvalResult, one per threshold value.
        EvalResult.metadata["threshold"] contains the threshold used.

        Implementation note: Attempts to change threshold on existing
        pipeline component without reloading the model. The gliner-spacy
        component stores threshold in its config dict -- we try to update
        it in-place via nlp.get_pipe("gliner_spacy").cfg["threshold"].
        """
        logger.info(f"Running threshold sweep on {benchmark.value}...")

        try:
            gold_annotations = load_benchmark(benchmark)
            doc_texts = get_document_texts(benchmark)
        except DatasetNotAvailableError as e:
            logger.warning(f"Skipping threshold sweep: {e}")
            return []

        results = []
        thresholds = [round(start + i * step, 2) for i in range(int((stop - start) / step) + 1)]

        for threshold in tqdm(thresholds, desc="Threshold sweep"):
            # Build/update pipeline
            self._get_or_build_pipeline(CTRA_LABELS, threshold=threshold)

            # Run inference
            predictions = self._run_inference(doc_texts, CTRA_LABELS, threshold=threshold)

            # Compute metrics
            eval_result = compute_ner_metrics(
                gold_annotations,
                predictions,
                self._match_strategy,
                label_field="mapped_ctra_label",
            )
            eval_result.benchmark = benchmark.value
            eval_result.mode = "ctra"
            eval_result.metadata = {
                "model": self._gliner_model,
                "threshold": str(threshold),
            }

            results.append(eval_result)

        return results

    def run_label_ablation(
        self,
        benchmark: Benchmark = Benchmark.CHIA,
        variants: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Test alternative phrasings for each CTRA label.

        For each label and each variant:
        1. Replace the label in the label list with the variant
        2. Run Mode 2 on the specified benchmark
        3. Record the F1 delta vs. the original phrasing

        Returns list of dicts with keys:
        - original_label, variant, f1_original, f1_variant, f1_delta
        """
        if variants is None:
            from ctra.rag.eval.data_models import LABEL_VARIANTS

            variants = LABEL_VARIANTS

        logger.info(f"Running label ablation on {benchmark.value}...")

        try:
            gold_annotations = load_benchmark(benchmark)
            doc_texts = get_document_texts(benchmark)
        except DatasetNotAvailableError as e:
            logger.warning(f"Skipping label ablation: {e}")
            return []

        ablation_results = []

        # First, get baseline F1 for each label
        baseline_f1s: dict[str, float] = {}
        self._get_or_build_pipeline(CTRA_LABELS, threshold=0.4)
        predictions = self._run_inference(doc_texts, CTRA_LABELS, threshold=0.4)
        baseline_result = compute_ner_metrics(
            gold_annotations,
            predictions,
            self._match_strategy,
            label_field="mapped_ctra_label",
        )
        for metric in baseline_result.per_label:
            baseline_f1s[metric.label] = metric.f1

        # Test variants
        for label in CTRA_LABELS:
            if label not in variants:
                continue

            baseline_f1 = baseline_f1s.get(label, 0.0)

            for variant in variants[label]:
                # Build modified label list
                modified_labels = [variant if lbl == label else lbl for lbl in CTRA_LABELS]

                # Build/update pipeline with modified labels
                self._get_or_build_pipeline(modified_labels, threshold=0.4)

                # Run inference
                predictions = self._run_inference(doc_texts, modified_labels, threshold=0.4)

                # Normalize prediction labels: remap variant back to original
                # label so they can match gold annotations' mapped_ctra_label.
                # E.g., predictions labeled "Drug or medication" become "Drug"
                # to match gold mapped_ctra_label="Drug".
                normalized_predictions = [
                    PredictedEntity(
                        doc_id=p.doc_id,
                        start_char=p.start_char,
                        end_char=p.end_char,
                        text=p.text,
                        label=label if p.label == variant else p.label,
                        score=p.score,
                    )
                    for p in predictions
                ]

                # Compute metrics with normalized labels
                eval_result = compute_ner_metrics(
                    gold_annotations,
                    normalized_predictions,
                    self._match_strategy,
                    label_field="mapped_ctra_label",
                )

                # Find F1 for this label (now matches because we normalized)
                variant_f1 = 0.0
                for metric in eval_result.per_label:
                    if metric.label == label:
                        variant_f1 = metric.f1
                        break

                f1_delta = variant_f1 - baseline_f1

                ablation_results.append(
                    {
                        "original_label": label,
                        "variant": variant,
                        "f1_original": baseline_f1,
                        "f1_variant": variant_f1,
                        "f1_delta": f1_delta,
                    }
                )

        return ablation_results

    def _run_inference(
        self,
        doc_texts: dict[str, str],
        labels: list[str],
        threshold: float,
    ) -> list[PredictedEntity]:
        """Run GLiNER-BioMed NER inference over documents.

        Args:
            doc_texts: {doc_id: text} mapping.
            labels: Entity labels to use.
            threshold: Confidence threshold.

        Returns:
            List of PredictedEntity instances.

        Notes:
            Uses spaCy nlp.pipe() for batch processing.
            Logs progress with tqdm for large benchmark sets.
        """
        from spacy.tokens import Span

        nlp = self._get_or_build_pipeline(labels, threshold=threshold)

        predictions: list[PredictedEntity] = []

        # Process documents in batches
        doc_items = list(doc_texts.items())
        for doc_id, text in tqdm(doc_items, desc="Running inference", leave=False):
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    score = ent._.gliner_score if Span.has_extension("gliner_score") else 0.0

                    pred = PredictedEntity(
                        doc_id=doc_id,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        text=ent.text,
                        label=ent.label_,
                        score=score,
                    )
                    predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error processing {doc_id}: {e}")
                continue

        return predictions


def gap_analysis(
    eval_results: list[EvalResult],
    ctra_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Identify CTRA labels with no or insufficient benchmark coverage.

    Returns a list of dicts with keys:
    - label: CTRA label name
    - total_gold: total gold entities mapped to this label across benchmarks
    - status: "no_coverage" | "low_coverage" | "adequate"
    - recommendation: human-readable recommendation

    Expected no-coverage labels: Organization, Biomarker, Anatomical structure
    (no benchmark entities map to these).
    """
    if ctra_labels is None:
        ctra_labels = CTRA_LABELS

    # Aggregate support per label
    label_support: dict[str, int] = {label: 0 for label in ctra_labels}

    for result in eval_results:
        for metric in result.per_label:
            if metric.label in label_support:
                label_support[metric.label] += metric.support

    # Classify coverage
    gap_results = []
    for label in ctra_labels:
        support = label_support[label]

        if support == 0:
            status = "no_coverage"
            recommendation = f"{label} has no benchmark coverage. Consider expanding benchmarks or manual data collection."
        elif support < 100:
            status = "low_coverage"
            recommendation = f"{label} has low coverage ({support} entities). May need targeted benchmark expansion."
        else:
            status = "adequate"
            recommendation = f"{label} has adequate coverage ({support} entities)."

        gap_results.append(
            {
                "label": label,
                "total_gold": support,
                "status": status,
                "recommendation": recommendation,
            }
        )

    return gap_results
