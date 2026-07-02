"""Feature Importance page -- global and per-trial SHAP analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from ctra.dashboard.api_client import CTRAClient


def _collect_prediction_history() -> list[dict[str, Any]]:
    """Retrieve all previous predictions from Streamlit session state.

    Predictions are cached in ``st.session_state["prediction_history"]``
    as a dict mapping prediction IDs to prediction records.

    Returns:
        List of prediction dicts from the session history.
    """
    return list(st.session_state.get("prediction_history", {}).values())


def _compute_global_importance(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate feature importance across all predictions using mean |SHAP|.

    For each feature, computes the mean absolute SHAP value and the number
    of predictions in which the feature appeared. Results are sorted by
    descending importance.

    Args:
        predictions: List of prediction dicts, each with a ``features`` key
            containing a list of ``{name, shap_value}`` dicts.

    Returns:
        DataFrame with columns: ``Feature``, ``Mean |SHAP|``, ``Count``,
        sorted by ``Mean |SHAP|`` descending.
    """
    accum: dict[str, list[float]] = defaultdict(list)

    for pred in predictions:
        for feat in pred.get("features", []):
            accum[feat["name"]].append(abs(feat["shap_value"]))

    if not accum:
        return pd.DataFrame(columns=["Feature", "Mean |SHAP|", "Count"])

    rows = [
        {
            "Feature": name,
            "Mean |SHAP|": float(np.mean(values)),
            "Count": len(values),
        }
        for name, values in accum.items()
    ]
    df = pd.DataFrame(rows).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    return df


def _build_global_importance_chart(df: pd.DataFrame) -> go.Figure:
    """Build a Plotly horizontal bar chart of global feature importance.

    Displays the top 20 most important features (by mean |SHAP|) in
    ascending order (most important at the top).

    Args:
        df: DataFrame from ``_compute_global_importance`` with columns
            ``Feature``, ``Mean |SHAP|``, ``Count``.

    Returns:
        A ``plotly.graph_objects.Figure`` horizontal bar chart.
    """
    df_plot = df.head(20).sort_values("Mean |SHAP|", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_plot["Mean |SHAP|"],
            y=df_plot["Feature"],
            orientation="h",
            marker_color="#3498db",
            text=[f"{v:.4f}" for v in df_plot["Mean |SHAP|"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Global Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        height=max(400, len(df_plot) * 30 + 100),
        margin={"l": 200, "r": 80, "t": 50, "b": 40},
    )
    return fig


def _build_per_trial_chart(features: list[dict[str, Any]]) -> go.Figure:
    """Build a Plotly horizontal bar chart of a trial's SHAP values.

    Shows each feature's SHAP contribution to the prediction for a single
    trial. Green bars indicate positive contributions (toward success),
    red bars indicate negative contributions (toward failure).

    Args:
        features: List of dicts with keys: ``name`` (str), ``shap_value`` (float).

    Returns:
        A ``plotly.graph_objects.Figure`` horizontal bar chart.
    """
    if not features:
        fig = go.Figure()
        fig.add_annotation(text="No features available", showarrow=False)
        fig.update_layout(height=200)
        return fig

    sorted_feats = sorted(features, key=lambda f: abs(f["shap_value"]))
    names = [f["name"] for f in sorted_feats]
    vals = [f["shap_value"] for f in sorted_feats]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Per-Trial SHAP Contributions",
        xaxis_title="SHAP Value",
        yaxis_title="",
        height=max(300, len(features) * 35 + 100),
        margin={"l": 200, "r": 60, "t": 50, "b": 40},
        xaxis={"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "#7f8c8d"},
    )
    return fig


def _build_correlation_heatmap(predictions: list[dict[str, Any]]) -> go.Figure | None:
    """Build a Plotly heatmap of feature correlations from predictions.

    Extracts numeric feature values from all predictions, computes the
    Pearson correlation matrix, and visualizes it as a heatmap. Returns
    ``None`` if there are too few predictions or numeric features.

    Args:
        predictions: List of prediction dicts with ``features`` key.

    Returns:
        A ``plotly.graph_objects.Figure`` heatmap, or ``None`` if
        insufficient data (< 2 predictions or < 2 numeric features).
    """
    # Build a trial x feature matrix of raw values
    rows = []
    for pred in predictions:
        row: dict[str, float | None] = {}
        for feat in pred.get("features", []):
            val = feat.get("value")
            if isinstance(val, (int, float)):
                row[feat["name"]] = float(val)
        if row:
            rows.append(row)

    if len(rows) < 2:
        return None

    df = pd.DataFrame(rows)
    # Keep only columns with at least 2 non-null values
    df = df.dropna(axis=1, thresh=2)

    if df.shape[1] < 2:
        return None

    corr = df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap",
    )
    fig.update_layout(
        height=max(400, len(corr.columns) * 30 + 100),
        margin={"t": 50, "b": 40},
    )
    return fig


def render_feature_importance_page(client: CTRAClient) -> None:
    """Render the feature importance analysis page."""
    st.header("Feature Importance Analysis")
    st.markdown(
        "Explore which features drive model predictions.  "
        "Analysis is based on SHAP values from cached predictions in this session."
    )

    predictions = _collect_prediction_history()

    if not predictions:
        st.info(
            "No predictions cached yet.  Go to **Trial Prediction** or "
            "**Trial Portfolio** and run predictions first.  Predictions with "
            "SHAP explanations will appear here automatically."
        )

        # Allow a quick single-trial lookup to seed data
        st.markdown("---")
        st.subheader("Quick prediction")
        trial_id = st.text_input(
            "Trial ID",
            placeholder="e.g. NCT04280705",
            key="fi_trial_id",
        )
        if st.button("Predict and analyze", type="primary", disabled=not trial_id.strip()):
            with st.spinner("Running prediction..."):
                try:
                    result = client.predict(trial_id=trial_id.strip(), include_shap=True)
                    if "prediction_history" not in st.session_state:
                        st.session_state["prediction_history"] = {}
                    st.session_state["prediction_history"][result["trial_id"]] = result
                    st.rerun()
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
        return

    # ------------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------------
    st.subheader("Global Feature Importance")
    st.caption(f"Aggregated across {len(predictions)} prediction(s)")

    df_importance = _compute_global_importance(predictions)
    if df_importance.empty:
        st.warning("No SHAP data available in cached predictions.")
    else:
        fig_global = _build_global_importance_chart(df_importance)
        st.plotly_chart(fig_global, use_container_width=True)

        with st.expander("Importance table", expanded=False):
            st.dataframe(df_importance, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Per-trial contributions
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Per-Trial Feature Contributions")

    trial_ids = [p["trial_id"] for p in predictions]
    selected_trial = st.selectbox("Select trial", trial_ids, key="fi_select_trial")

    if selected_trial:
        selected = st.session_state["prediction_history"][selected_trial]
        features = selected.get("features", [])
        fig_trial = _build_per_trial_chart(features)
        st.plotly_chart(fig_trial, use_container_width=True)

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Feature Correlation")

    fig_corr = _build_correlation_heatmap(predictions)
    if fig_corr is not None:
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Need at least 2 predictions with 2+ numeric features to compute correlations.")
