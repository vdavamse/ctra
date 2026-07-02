"""NER evaluation metrics with configurable entity matching strategies.

Supports three matching strategies:
- strict: exact span match (start_char == gold.start_char AND end_char == gold.end_char AND label match)
- relaxed: overlapping span + correct label (max(start, gold_start) < min(end, gold_end) AND label match)
- text_only: surface form match (predicted.text.lower() == gold.text.lower() AND label match)

The text_only strategy is most relevant for LinearRAG, which only uses ent.text
for graph construction and never inspects spans.
"""

from __future__ import annotations

from ctra.rag.eval.data_models import (
    EvalResult,
    GoldAnnotation,
    MatchStrategy,
    PerLabelMetrics,
    PredictedEntity,
)


def compute_ner_metrics(
    gold: list[GoldAnnotation],
    predicted: list[PredictedEntity],
    strategy: MatchStrategy,
    label_field: str = "gold_label",
    valid_labels: dict[str, set[str]] | None = None,
) -> EvalResult:
    """Compute full NER evaluation metrics.

    Args:
        gold: Gold-standard annotations.
        predicted: Model predictions.
        strategy: Entity matching strategy.
        label_field: Which field on GoldAnnotation to use as the label
            for matching. "gold_label" for Mode 1 (native benchmark labels),
            "mapped_ctra_label" for Mode 2 (CTRA labels).
        valid_labels: Optional mapping from gold label to set of acceptable
            prediction labels. See match_entities() for details.

    Returns:
        EvalResult with per-label and aggregate metrics.

    Notes:
        - Gold annotations with mapped_ctra_label=None are excluded in Mode 2
          (unmappable types like Mood, Negation, Qualifier)
        - Each gold entity is matched to at most one prediction (greedy 1:1)
        - Each prediction is matched to at most one gold entity
    """
    # Filter gold annotations if using mapped_ctra_label
    filtered_gold = gold
    if label_field == "mapped_ctra_label":
        filtered_gold = [g for g in gold if g.mapped_ctra_label is not None]

    matched, unmatched_gold, unmatched_pred = match_entities(
        filtered_gold,
        predicted,
        strategy,
        label_field,
        valid_labels=valid_labels,
    )

    # Compute per-label metrics
    per_label_metrics: dict[str, dict[str, int]] = {}

    for gold_entity in filtered_gold:
        label = getattr(gold_entity, label_field)
        if label not in per_label_metrics:
            per_label_metrics[label] = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "support": 0,
                "predicted_count": 0,
            }
        per_label_metrics[label]["support"] += 1

    for pred_entity in predicted:
        label = pred_entity.label
        if label not in per_label_metrics:
            per_label_metrics[label] = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "support": 0,
                "predicted_count": 0,
            }
        per_label_metrics[label]["predicted_count"] += 1

    # Count matches
    for gold_entity, _pred_entity in matched:
        label = getattr(gold_entity, label_field)
        per_label_metrics[label]["tp"] += 1

    # Count false negatives
    for gold_entity in unmatched_gold:
        label = getattr(gold_entity, label_field)
        per_label_metrics[label]["fn"] += 1

    # Count false positives
    for pred_entity in unmatched_pred:
        label = pred_entity.label
        per_label_metrics[label]["fp"] += 1

    # Compute per-label P/R/F1
    per_label_results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in sorted(per_label_metrics.keys()):
        counts = per_label_metrics[label]
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        support = counts["support"]
        predicted_count = counts["predicted_count"]

        precision, recall, f1 = _compute_prf(tp, fp, fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_label_results.append(
            PerLabelMetrics(
                label=label,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
                predicted_count=predicted_count,
            )
        )

    # Compute aggregate metrics
    micro_precision, micro_recall, micro_f1 = _compute_prf(total_tp, total_fp, total_fn)

    # Macro F1 (average of per-label F1s)
    macro_f1 = (
        sum(m.f1 for m in per_label_results) / len(per_label_results) if per_label_results else 0.0
    )

    # Weighted F1 (weighted by support)
    total_support = sum(m.support for m in per_label_results)
    weighted_f1 = (
        sum(m.f1 * m.support for m in per_label_results) / total_support
        if total_support > 0
        else 0.0
    )

    return EvalResult(
        benchmark="",  # Will be set by evaluator
        mode="",  # Will be set by evaluator
        match_strategy=strategy.value,
        per_label=per_label_results,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        total_gold=len(filtered_gold),
        total_predicted=len(predicted),
        total_matched=len(matched),
    )


def match_entities(
    gold: list[GoldAnnotation],
    predicted: list[PredictedEntity],
    strategy: MatchStrategy,
    label_field: str = "gold_label",
    valid_labels: dict[str, set[str]] | None = None,
) -> tuple[
    list[tuple[GoldAnnotation, PredictedEntity]],
    list[GoldAnnotation],
    list[PredictedEntity],
]:
    """Match gold entities to predictions using the given strategy.

    Args:
        valid_labels: Optional mapping from gold label to set of prediction
            labels that should be accepted as matches. For example,
            {"Disease": {"Disease", "Symptom", "Adverse event"}} means a
            gold entity with mapped_ctra_label="Disease" can match predictions
            labeled "Symptom" or "Adverse event" in addition to "Disease".
            If None, only exact label matches are accepted.

    Returns:
        Tuple of:
        - matched: list of (gold, predicted) pairs
        - unmatched_gold: gold entities with no matching prediction (FN)
        - unmatched_pred: predictions with no matching gold entity (FP)

    Matching algorithm:
    1. Group gold and predicted by doc_id
    2. Within each document, attempt to match each gold entity to a prediction
    3. For strict/relaxed: sort by span position, use greedy matching
    4. For text_only: group by normalized text, match within same label
    5. Each entity participates in at most one match (1:1 greedy)
    """
    # Group by document
    gold_by_doc: dict[str, list[GoldAnnotation]] = {}
    for gold_entity in gold:
        if gold_entity.doc_id not in gold_by_doc:
            gold_by_doc[gold_entity.doc_id] = []
        gold_by_doc[gold_entity.doc_id].append(gold_entity)

    pred_by_doc: dict[str, list[PredictedEntity]] = {}
    for pred_entity in predicted:
        if pred_entity.doc_id not in pred_by_doc:
            pred_by_doc[pred_entity.doc_id] = []
        pred_by_doc[pred_entity.doc_id].append(pred_entity)

    matched: list[tuple[GoldAnnotation, PredictedEntity]] = []
    matched_gold_ids: set[int] = set()  # id() of matched GoldAnnotation objects
    matched_pred_ids: set[int] = set()  # id() of matched PredictedEntity objects

    # Match within each document
    for doc_id in gold_by_doc:
        doc_gold = gold_by_doc[doc_id]
        doc_pred = pred_by_doc.get(doc_id, [])

        if not doc_pred:
            continue

        # Sort by position for stable matching
        doc_gold_sorted = sorted(doc_gold, key=lambda x: (x.start_char, x.end_char))
        doc_pred_sorted = sorted(doc_pred, key=lambda x: (x.start_char, x.end_char))

        for gold_entity in doc_gold_sorted:
            gold_label = getattr(gold_entity, label_field)
            # Determine which prediction labels are acceptable
            acceptable = valid_labels.get(gold_label) if valid_labels else None
            best_pred: PredictedEntity | None = None
            best_score = 0.0

            for pred_entity in doc_pred_sorted:
                if id(pred_entity) in matched_pred_ids:
                    continue

                # Check label match (exact or via valid_labels set)
                if acceptable is not None:
                    if pred_entity.label not in acceptable:
                        continue
                elif pred_entity.label != gold_label:
                    continue

                # Apply strategy (returns >0 if match, 0 if no match)
                score = _match_score(gold_entity, pred_entity, strategy)
                if score > best_score:
                    best_score = score
                    best_pred = pred_entity

            if best_pred is not None and best_score > 0:
                matched.append((gold_entity, best_pred))
                matched_gold_ids.add(id(gold_entity))
                matched_pred_ids.add(id(best_pred))

    # Unmatched gold and predicted
    unmatched_gold = [e for e in gold if id(e) not in matched_gold_ids]
    unmatched_pred = [e for e in predicted if id(e) not in matched_pred_ids]

    return matched, unmatched_gold, unmatched_pred


def _match_score(
    gold: GoldAnnotation,
    pred: PredictedEntity,
    strategy: MatchStrategy,
) -> float:
    """Compute match score (0 if no match, >0 if match, used for tie-breaking).

    Returns max(pred.score, 1e-9) for valid matches to ensure the score is
    always positive (pred.score can be 0.0 when confidence is unavailable).
    Returns 0.0 for non-matches.
    """
    if strategy == MatchStrategy.STRICT:
        if gold.start_char == pred.start_char and gold.end_char == pred.end_char:
            return max(pred.score, 1e-9)
        return 0.0
    elif strategy == MatchStrategy.RELAXED:
        if _spans_overlap(gold.start_char, gold.end_char, pred.start_char, pred.end_char):
            return max(pred.score, 1e-9)
        return 0.0
    elif strategy == MatchStrategy.TEXT_ONLY:
        if gold.text.lower() == pred.text.lower():
            return max(pred.score, 1e-9)
        return 0.0
    else:
        msg = f"Unknown strategy: {strategy}"
        raise ValueError(msg)


def _spans_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two character spans overlap (for relaxed matching)."""
    return max(start1, start2) < min(end1, end2)


def _compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from counts. Returns (0, 0, 0) for edge cases."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def build_confusion_matrix(
    gold: list[GoldAnnotation],
    predicted: list[PredictedEntity],
    strategy: MatchStrategy,
    gold_labels: list[str],
    pred_labels: list[str],
) -> tuple[list[list[int]], list[str]]:
    """Build a confusion matrix between gold and predicted labels.

    Used in Mode 2 to see which CTRA labels get confused with
    which benchmark labels. Adds a "No prediction" column for
    false negatives (gold entities with no matching prediction).

    Args:
        gold: Gold annotations.
        predicted: Predicted entities.
        strategy: Matching strategy.
        gold_labels: List of gold label names (rows).
        pred_labels: List of predicted label names (columns).

    Returns:
        Tuple of:
        - 2D list where [i][j] = count of gold label i matched to pred label j.
          The last column is "No prediction" (false negatives).
        - Column labels (pred_labels + ["No prediction"]).
    """
    # Add explicit "No prediction" column for false negatives
    col_labels = [*pred_labels, "No prediction"]

    # Initialize matrix with extra column
    matrix = [[0] * len(col_labels) for _ in range(len(gold_labels))]

    gold_label_to_idx = {label: idx for idx, label in enumerate(gold_labels)}
    pred_label_to_idx = {label: idx for idx, label in enumerate(pred_labels)}
    no_pred_idx = len(pred_labels)  # Last column

    matched, unmatched_gold, _ = match_entities(gold, predicted, strategy, "gold_label")

    # Count matched pairs
    for gold_entity, pred_entity in matched:
        if gold_entity.gold_label in gold_label_to_idx and pred_entity.label in pred_label_to_idx:
            gold_idx = gold_label_to_idx[gold_entity.gold_label]
            pred_idx = pred_label_to_idx[pred_entity.label]
            matrix[gold_idx][pred_idx] += 1

    # Count false negatives in the dedicated "No prediction" column
    for gold_entity in unmatched_gold:
        if gold_entity.gold_label in gold_label_to_idx:
            gold_idx = gold_label_to_idx[gold_entity.gold_label]
            matrix[gold_idx][no_pred_idx] += 1

    return matrix, col_labels
