"""Trial Prediction page -- predict a single trial's outcome."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ctra.dashboard.api_client import CTRAClient, CTRAClientError


def _build_probability_gauge(prob_success: float) -> go.Figure:
    """Build a Plotly gauge chart showing trial success probability.

    Displays the probability as a percentage (0-100%) with color zones:
    - Green (60-100%): High probability of success
    - Yellow (40-60%): Moderate probability
    - Red (0-40%): Low probability of success

    Args:
        prob_success: Probability value in [0, 1].

    Returns:
        A ``plotly.graph_objects.Figure`` gauge chart.
    """
    color = "#2ecc71" if prob_success >= 0.6 else "#f39c12" if prob_success >= 0.4 else "#e74c3c"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_success * 100,
            number={"suffix": "%", "font": {"size": 48}},
            title={"text": "Probability of Success", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 40], "color": "#fadbd8"},
                    {"range": [40, 60], "color": "#fdebd0"},
                    {"range": [60, 100], "color": "#d5f5e3"},
                ],
                "threshold": {
                    "line": {"color": "#2c3e50", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin={"t": 60, "b": 20, "l": 40, "r": 40},
    )
    return fig


def _build_shap_waterfall(features: list[dict[str, Any]]) -> go.Figure:
    """Build a horizontal bar chart showing SHAP feature contributions.

    Each bar represents one feature's SHAP value (impact on the prediction):
    - Green bars: positive contribution (toward success)
    - Red bars: negative contribution (toward failure)

    Bars are sorted by absolute SHAP value so the most impactful features
    are at the top, mimicking the layout of a traditional SHAP waterfall.

    Args:
        features: List of dicts with keys: ``name`` (str), ``shap_value`` (float).

    Returns:
        A ``plotly.graph_objects.Figure`` horizontal bar chart.
    """
    if not features:
        fig = go.Figure()
        fig.add_annotation(text="No SHAP explanations available", showarrow=False)
        fig.update_layout(height=200)
        return fig

    # Sort by absolute SHAP value ascending so the largest bar is at the top
    sorted_feats = sorted(features, key=lambda f: abs(f["shap_value"]))

    names = [f["name"] for f in sorted_feats]
    shap_values = [f["shap_value"] for f in sorted_feats]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in shap_values]

    fig = go.Figure(
        go.Bar(
            x=shap_values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in shap_values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="",
        height=max(300, len(features) * 35 + 100),
        margin={"l": 200, "r": 60, "t": 50, "b": 40},
        xaxis={"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "#7f8c8d"},
    )
    return fig


def render_prediction_page(client: CTRAClient) -> None:
    """Render the single-trial prediction page."""
    st.header("Trial Outcome Prediction")
    st.markdown(
        "Enter a ClinicalTrials.gov NCT ID or internal trial ID to predict "
        "the probability of success or failure."
    )

    # ------------------------------------------------------------------
    # Input form
    # ------------------------------------------------------------------
    col_input, col_options = st.columns([2, 1])

    with col_input:
        trial_id = st.text_input(
            "Trial ID",
            placeholder="e.g. NCT04280705",
            help="ClinicalTrials.gov NCT ID or internal trial identifier",
        )

    with col_options:
        model_choice = st.selectbox(
            "Model",
            options=["Best available", "xgboost", "tabpfn"],
            index=0,
            help="Choose the classifier or let the system pick the best one",
        )
        include_shap = st.checkbox("Include SHAP explanations", value=True)

    predict_clicked = st.button("Predict", type="primary", disabled=not trial_id.strip())

    # ------------------------------------------------------------------
    # Run prediction
    # ------------------------------------------------------------------
    if predict_clicked and trial_id.strip():
        model_param = None if model_choice == "Best available" else model_choice

        with st.spinner(f"Predicting outcome for {trial_id.strip()}..."):
            try:
                result = client.predict(
                    trial_id=trial_id.strip(),
                    model=model_param,
                    include_shap=include_shap,
                )
            except CTRAClientError as exc:
                st.error(f"Prediction failed: {exc.message}")
                if exc.detail:
                    st.caption(exc.detail)
                return
            except ConnectionError as exc:
                st.error(str(exc))
                return

        # Store in session state for persistence across reruns
        st.session_state["last_prediction"] = result

    # ------------------------------------------------------------------
    # Display results (from session state so they survive reruns)
    # ------------------------------------------------------------------
    result = st.session_state.get("last_prediction")
    if result is None:
        return

    st.markdown("---")

    # Prediction label
    prediction = result["prediction"]
    prob_success = result["probability_success"]
    prob_failure = result["probability_failure"]
    model_used = result["model_used"]

    if prediction == "success":
        st.success(
            f"**Prediction: SUCCESS** -- {prob_success:.1%} confidence (model: {model_used})"
        )
    else:
        st.error(f"**Prediction: FAILURE** -- {prob_failure:.1%} confidence (model: {model_used})")

    # Two-column layout: gauge + metadata
    col_gauge, col_meta = st.columns([2, 1])

    with col_gauge:
        fig_gauge = _build_probability_gauge(prob_success)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_meta:
        st.subheader("Prediction Details")
        st.metric("Trial ID", result["trial_id"])
        st.metric("Model", model_used)

        metadata = result.get("metadata", {})
        if metadata:
            st.metric("Model Version", metadata.get("model_version", "N/A"))
            st.metric(
                "Inference Time",
                f"{metadata.get('inference_time_ms', 0):.0f} ms",
            )
            st.metric("Features Used", metadata.get("n_features", "N/A"))
            st.caption(f"Feature set: {metadata.get('feature_set_id', 'N/A')}")

    # ------------------------------------------------------------------
    # SHAP explanations
    # ------------------------------------------------------------------
    features = result.get("features", [])
    if features:
        st.markdown("---")
        st.subheader("Feature Explanations (SHAP)")

        fig_shap = _build_shap_waterfall(features)
        st.plotly_chart(fig_shap, use_container_width=True)

        # Feature detail table
        with st.expander("Feature detail table", expanded=False):
            df = pd.DataFrame(features)
            df = df[["name", "value", "shap_value"]]
            df.columns = ["Feature", "Value", "SHAP Contribution"]
            df = df.sort_values("SHAP Contribution", key=abs, ascending=False)
            df = df.reset_index(drop=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
    elif include_shap:
        st.info("No SHAP explanations were returned for this prediction.")
