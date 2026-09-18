"""Shared utilities for the agent pipeline.

Contains helpers ported from AutoCT's ``agent.py``:
- ``soft_assert`` (line 84) — non-fatal validation (log + return None on failure)
- ``dump_as_json`` (line 76) — NamedTuple-aware JSON serialization
- ``parse_date`` — robust date parsing from various formats
- ``features_to_df`` — flatten multi-valued features to DataFrame
- ``build_feature_type_transformer`` (line 2365) — feature-type-aware ColumnTransformer
- ``eval_model`` (line 2602) — evaluate a pipeline and return ModelEvalResult

CTRA-specific additions (Issue #18):
- ``interaction_values_to_dict`` — convert shapiq InteractionValues to serializable dict
- ``compute_shapiq_for_pipeline`` — compute shapiq interactions for a trained sklearn pipeline
- ``format_interactions_for_llm`` — format interaction values as LLM-readable summary
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from ctra.agents.data_models import FeaturePlan, FeatureType, ModelEvalResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# parse_date — robust date parsing
# ---------------------------------------------------------------------------


def parse_date(value: Any) -> datetime.date | None:
    """Parse a date value from various formats.

    Handles None, NaN (float and numpy), ``datetime.date``,
    ``datetime.datetime``, and ISO-format strings (with optional time
    component, truncated to the first 10 characters).

    Returns ``datetime.date`` on success, ``None`` on failure.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if isinstance(value, datetime.datetime):
            return value.date()
        if isinstance(value, datetime.date):
            return value
        return datetime.date.fromisoformat(str(value)[:10])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# soft_assert — AutoCT agent.py line 84
# ---------------------------------------------------------------------------


def soft_assert(value: Any, condition: bool, message: str) -> Any:
    """Non-fatal validation that returns *value* on success, ``None`` on failure.

    Mirrors AutoCT's ``soft_assert`` (agent.py line 84).  When the condition
    fails, logs a warning and returns ``None`` instead of raising.
    """
    if not condition:
        logger.warning("soft_assert failed: %s", message)
        return None
    return value


# ---------------------------------------------------------------------------
# dump_as_json — AutoCT agent.py line 76
# ---------------------------------------------------------------------------


class _NamedTupleEncoder(json.JSONEncoder):
    """JSON encoder that converts NamedTuples via ``_asdict()`` and handles numpy."""

    def default(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization.

        Args:
            obj: Object to convert (checked in order: integer, floating, ndarray).

        Returns:
            Native Python type if obj is numpy, otherwise delegates to parent.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj: Any) -> str:
        return super().encode(self._convert(obj))

    def _convert(self, obj: Any) -> Any:
        if hasattr(obj, "_asdict"):
            return {k: self._convert(v) for k, v in obj._asdict().items()}
        if isinstance(obj, dict):
            return {k: self._convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)) and not hasattr(obj, "_asdict"):
            return [self._convert(item) for item in obj]
        return obj


def dump_as_json(obj: Any, pretty: bool = True) -> str:
    """Serialize *obj* to JSON, converting NamedTuples via ``_asdict()``."""
    return json.dumps(obj, cls=_NamedTupleEncoder, indent=2 if pretty else None)


# ---------------------------------------------------------------------------
# features_to_df
# ---------------------------------------------------------------------------


def features_to_df(
    features: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """Flatten multi-valued features to a DataFrame.

    Input shape::

        {nctid: {feature_name: {sub_feature: value, ...}, ...}, ...}

    Output columns use the format ``{feature_name}--{sub_feature_name}``.
    An ``id`` column is added with the nctid.
    """
    rows: list[dict[str, Any]] = []
    for nctid, feature_dict in features.items():
        row: dict[str, Any] = {"id": nctid}
        for feature_name, sub_features in feature_dict.items():
            if isinstance(sub_features, dict):
                for sub_name, value in sub_features.items():
                    row[f"{feature_name}--{sub_name}"] = value
            else:
                # Fallback for single-valued features stored as scalars
                row[f"{feature_name}--value"] = sub_features
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_feature_type_transformer — AutoCT agent.py line 2365
# ---------------------------------------------------------------------------


def build_feature_type_transformer(
    plans: dict[str, FeaturePlan],
    model_type: Literal["xgb", "tabpfn"],
    skip: list[str] | None = None,
) -> ColumnTransformer:
    """Build a feature-type-aware ColumnTransformer (AutoCT line 2365).

    XGBoost/TabPFN both handle NaN natively, so numeric types use passthrough.

    Args:
        plans: Feature plans keyed by feature name.
        model_type: Classifier type (``"xgb"`` or ``"tabpfn"``).
        skip: Feature names to exclude.

    Returns:
        Configured ``ColumnTransformer`` with ``remainder="drop"``.
    """
    skip_set = set(skip or [])
    transformers: list[tuple[str, Any, Any]] = []

    for fp in plans.values():
        if fp.feature_name in skip_set:
            continue

        for sub_name, sub_type in fp.feature_type.items():
            col = f"{fp.feature_name}--{sub_name}"

            if sub_type == FeatureType.CATEGORICAL:
                transformers.append(
                    (
                        f"{col}--cat_onehot",
                        OneHotEncoder(
                            categories=[fp.possible_values[sub_name]],
                            handle_unknown="ignore",
                        ),
                        [col],
                    )
                )
            elif sub_type == FeatureType.MULTICATEGORICAL:
                transformers.append(
                    (
                        f"{col}--multicat",
                        make_pipeline(
                            FunctionTransformer(
                                lambda x: x.apply(lambda c: [] if c is None else c),
                                feature_names_out="one-to-one",
                            ),
                            CountVectorizer(
                                analyzer=lambda lst: lst,
                                vocabulary=fp.possible_values[sub_name],
                                lowercase=False,
                            ),
                        ),
                        col,  # CountVectorizer expects a single column, not a list
                    )
                )
            elif sub_type == FeatureType.INTEGER:
                transformers.append((f"{col}--int", "passthrough", [col]))
            elif sub_type == FeatureType.BOOLEAN:
                transformers.append((f"{col}--bool", "passthrough", [col]))
            else:
                # FLOAT or unknown — default
                transformers.append((f"{col}--passthrough", "passthrough", [col]))

    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# eval_model — AutoCT agent.py line 2602
# ---------------------------------------------------------------------------


def eval_model(
    pipeline: Any,
    df: pd.DataFrame,
    y_true: pd.Series | NDArray[Any],
) -> ModelEvalResult:
    """Evaluate a trained pipeline on a dataset (AutoCT line 2602).

    Returns:
        ``ModelEvalResult`` with metrics, interaction values (populated by
        orchestrator after model-specific explainer runs), and wrong prediction
        details.

    Note:
        ``wrong_idxs`` are **positions in ``df``**; ``wrong_df`` is ``df.iloc[wrong_idxs]``
        and therefore retains ``df``'s index labels. ``wrong_idxs``, ``wrong_preds``
        and ``wrong_df`` are positionally aligned with each other; consumers must
        index all three positionally and must not use ``wrong_idxs`` values as ``.loc``
        labels.
    """
    y_true_arr = np.asarray(y_true)
    y_pred = pipeline.predict(df)
    y_prob = pipeline.predict_proba(df)[:, 1]

    # Metrics
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob))
        if np.isnan(roc_auc):
            raise ValueError("ROC-AUC returned NaN")
    except ValueError:
        logger.warning("ROC-AUC undefined (single class in y_true); setting to 0.0")
        roc_auc = 0.0
    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob))
        if np.isnan(pr_auc):
            raise ValueError("PR-AUC returned NaN")
    except ValueError:
        logger.warning("PR-AUC undefined; setting to 0.0")
        pr_auc = 0.0
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0.0))

    # Wrong predictions
    wrong_mask = y_pred != y_true_arr
    wrong_idxs = np.where(wrong_mask)[0].tolist()
    wrong_preds = y_pred[wrong_mask].tolist()
    wrong_df = df.iloc[wrong_idxs].copy()

    return ModelEvalResult(
        roc_auc=roc_auc,
        f1=f1,
        pr_auc=pr_auc,
        interaction_values={},  # Populated by orchestrator after model-specific explainer runs
        wrong_idxs=wrong_idxs,
        wrong_preds=wrong_preds,
        wrong_df=wrong_df,
        pipeline=pipeline,
    )


# ---------------------------------------------------------------------------
# shapiq interaction value helpers
# ---------------------------------------------------------------------------


def interaction_values_to_dict(
    iv: Any,  # shapiq.InteractionValues
    feature_names: list[str],
    index_type: str,
    max_order: int,
) -> dict[str, Any]:
    """Convert a shapiq InteractionValues object to a serializable dict.

    Args:
        iv: A shapiq.InteractionValues object (from TreeExplainer or TabPFNExplainer).
        feature_names: List of feature names matching the transformed feature indices.
        index_type: The index used ("k-SII" or "FSII").
        max_order: Maximum interaction order computed.

    Returns:
        Dict with 'main_effects', 'interactions', 'feature_names', 'index_type', 'max_order'.
    """
    # Extract order-1 main effects
    order1 = iv.get_n_order(order=1)
    main_effects: dict[str, float] = {}
    for key, val in order1.dict_values.items():
        if len(key) == 1:
            idx = key[0]
            name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
            main_effects[name] = float(val)

    # Extract higher-order interactions (order 2 through max_order)
    interactions: dict[str, float] = {}
    for order in range(2, max_order + 1):
        order_n = iv.get_n_order(order=order)
        for key, val in order_n.dict_values.items():
            names = [feature_names[idx] if idx < len(feature_names) else f"f{idx}" for idx in key]
            interactions[f"({', '.join(names)})"] = float(val)

    return {
        "main_effects": main_effects,
        "interactions": interactions,
        "feature_names": feature_names,
        "index_type": index_type,
        "max_order": max_order,
    }


def compute_shapiq_for_pipeline(
    pipeline: Any,
    model_type: str,
    val_df: pd.DataFrame,
    train_cols: list[str],
    y_train: Any = None,
    train_df: pd.DataFrame | None = None,
    max_order: int = 2,
    max_samples: int = 50,
    budget: int = 2048,
) -> dict[str, Any]:
    """Compute shapiq interaction values for a trained sklearn pipeline.

    Uses TreeExplainer (k-SII, exact) for XGBoost and TabPFNExplainer
    (FSII, faithful) for TabPFN.  Returns an empty dict on any failure.

    Args:
        pipeline: Trained sklearn Pipeline with 'preprocessor' and 'classifier' steps.
        model_type: Either ``"xgb"`` or ``"tabpfn"``.
        val_df: Validation DataFrame (used to compute explanations).
        train_cols: Column names to use from val_df.
        y_train: Training labels (required for TabPFN recontextualization).
        train_df: Training DataFrame (required for TabPFN recontextualization).
        max_order: Maximum interaction order (default 2 = pairwise).
        max_samples: Maximum number of validation samples to explain.
        budget: Approximation budget for TabPFN (ignored for XGBoost).

    Returns:
        Serializable dict with 'main_effects', 'interactions', 'feature_names',
        'index_type', 'max_order'.  Empty dict on failure.
    """
    empty: dict[str, Any] = {}
    try:
        import shapiq
    except ImportError:
        logger.warning("shapiq not available, returning empty interaction values")
        return empty

    try:
        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        if model_type == "xgb":
            X_val_transformed = preprocessor.transform(val_df[train_cols].head(max_samples))

            explainer = shapiq.TreeExplainer(
                model=classifier,
                max_order=max_order,
                index="k-SII",
            )

            all_ivs = []
            for i in range(min(max_samples, len(X_val_transformed))):
                if hasattr(X_val_transformed, "toarray"):
                    x_row = np.asarray(X_val_transformed[i].toarray()).ravel()
                else:
                    x_row = np.asarray(X_val_transformed[i]).ravel()
                iv = explainer.explain(x=x_row)
                all_ivs.append(iv)

            if not all_ivs:
                return empty
            aggregated = all_ivs[0].aggregate(all_ivs[1:], aggregation="mean")
            feature_names = (
                list(preprocessor.get_feature_names_out())
                if hasattr(preprocessor, "get_feature_names_out")
                else train_cols
            )
            return interaction_values_to_dict(aggregated, feature_names, "k-SII", max_order)

        elif model_type == "tabpfn":
            if train_df is None or y_train is None:
                logger.warning("TabPFN requires train_df and y_train for interaction values")
                return empty

            explainer = shapiq.TabPFNExplainer(
                model=classifier,
                data=np.asarray(train_df[train_cols]),
                labels=np.asarray(y_train),
                index="FSII",
                max_order=max_order,
            )

            X_val_sub = val_df[train_cols].head(max_samples)
            all_ivs = []
            for i in range(len(X_val_sub)):
                iv = explainer.explain(x=np.asarray(X_val_sub.iloc[i]), budget=budget)
                all_ivs.append(iv)

            if not all_ivs:
                return empty
            aggregated = all_ivs[0].aggregate(all_ivs[1:], aggregation="mean")
            return interaction_values_to_dict(aggregated, train_cols, "FSII", max_order)

        else:
            logger.warning("Unsupported model type '%s' for shapiq computation", model_type)
            return empty

    except Exception:
        logger.warning("shapiq interaction computation failed", exc_info=True)
        return empty


def format_interactions_for_llm(
    interaction_values: dict[str, Any],
    top_k_main: int = 10,
    top_k_interactions: int = 10,
) -> str:
    """Format interaction values into a concise LLM-readable summary.

    Args:
        interaction_values: Dict from interaction_values_to_dict().
        top_k_main: Number of top main effects to include.
        top_k_interactions: Number of top pairwise interactions to include.

    Returns:
        Formatted string for LLM consumption.
    """
    if not interaction_values or not interaction_values.get("main_effects"):
        return "No interaction values available."

    lines = [f"## Feature Analysis ({interaction_values.get('index_type', 'shapiq')})\n"]

    # Main effects (order 1) — sorted by absolute value
    main = interaction_values["main_effects"]
    sorted_main = sorted(main.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k_main]

    lines.append("### Main Effects (Shapley values, higher |value| = more important):")
    for name, val in sorted_main:
        direction = "positive" if val > 0 else ("negative" if val < 0 else "neutral")
        lines.append(f"  - {name}: {val:+.4f} ({direction} contribution)")

    # Feature interactions (order >= 2) — sorted by absolute value
    interactions = interaction_values.get("interactions", {})
    if interactions:
        sorted_inter = sorted(interactions.items(), key=lambda x: abs(x[1]), reverse=True)[
            :top_k_interactions
        ]

        max_order = interaction_values.get("max_order", 2)
        header = (
            "Pairwise Interactions"
            if max_order == 2
            else f"Feature Interactions (up to order {max_order})"
        )
        lines.append(f"\n### {header} (positive = synergy, negative = redundancy):")
        for pair, val in sorted_inter:
            if abs(val) < 1e-6:
                continue
            synergy = "synergistic" if val > 0 else "redundant/conflicting"
            lines.append(f"  - {pair}: {val:+.4f} ({synergy})")

    return "\n".join(lines)
