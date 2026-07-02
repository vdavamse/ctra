"""Trial Portfolio page -- batch analysis of multiple clinical trials."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ctra.dashboard.api_client import CTRAClient, CTRAClientError


def _parse_trial_ids(text: str) -> list[str]:
    """Parse trial IDs from newline or comma-separated input text.

    Handles both formats (\n or ,) and strips whitespace from each ID.

    Args:
        text: Multi-line or comma-separated trial ID string.

    Returns:
        List of parsed trial ID strings (empty if input is empty).
    """
    ids: list[str] = []
    for line in text.replace(",", "\n").splitlines():
        tid = line.strip()
        if tid:
            ids.append(tid)
    return ids


def _build_risk_heatmap(df: pd.DataFrame) -> go.Figure:
    """Build a Plotly table showing portfolio risk predictions with color coding.

    Displays one row per trial with columns: Trial ID, Prediction (Success/Failure),
    Probability, and Top Risk Factor. Cells are color-coded by success probability:
    - Green (60-100%): Low risk
    - Yellow (40-60%): Moderate risk
    - Red (0-40%): High risk

    Args:
        df: DataFrame with columns: ``Trial ID``, ``Prediction``,
            ``Prob. Success``, ``Top Risk Factor``.

    Returns:
        A ``plotly.graph_objects.Figure`` table.
    """
    # Color by probability: lower success probability = higher risk (redder)
    colors = [
        "#d5f5e3" if p >= 0.6 else "#fdebd0" if p >= 0.4 else "#fadbd8" for p in df["Prob. Success"]
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": list(df.columns),
                    "fill_color": "#2c3e50",
                    "font": {"color": "white", "size": 13},
                    "align": "left",
                },
                cells={
                    "values": [df[col] for col in df.columns],
                    "fill_color": [
                        ["white"] * len(df),  # Trial ID
                        colors,  # Prediction
                        colors,  # Prob. Success
                        ["white"] * len(df),  # Top Risk Factor
                    ],
                    "font": {"size": 12},
                    "align": "left",
                    "height": 30,
                },
            )
        ]
    )
    fig.update_layout(
        title="Portfolio Risk Summary",
        height=max(300, len(df) * 35 + 100),
        margin={"t": 50, "b": 20, "l": 20, "r": 20},
    )
    return fig


def _build_risk_distribution(probs: list[float]) -> go.Figure:
    """Build a histogram showing the distribution of success probabilities.

    Displays the count of trials in each probability bucket (0-5%, 5-10%, etc.),
    with a reference line at 50% to highlight the pass/fail boundary.

    Args:
        probs: List of success probabilities for all trials in [0, 1].

    Returns:
        A ``plotly.express`` Figure with histogram bars.
    """
    fig = px.histogram(
        x=probs,
        nbins=20,
        labels={"x": "Probability of Success"},
        title="Risk Distribution",
        color_discrete_sequence=["#3498db"],
    )
    fig.update_layout(
        xaxis={"range": [0, 1]},
        yaxis_title="Number of Trials",
        height=350,
        margin={"t": 50, "b": 40},
    )
    # Add a 50% threshold line
    fig.add_vline(x=0.5, line_dash="dash", line_color="#e74c3c", annotation_text="50%")
    return fig


def _build_phase_distribution(trial_ids: list[str]) -> go.Figure | None:
    """Build a phase distribution chart from trial metadata.

    Currently returns ``None`` because the API response does not include
    trial phase directly. In production, this would extract phase from
    prediction metadata or trial data and display a bar chart showing
    the count of trials in each phase.

    Args:
        trial_ids: List of trial IDs to analyze.

    Returns:
        A chart Figure, or ``None`` if phase data is unavailable.
    """
    # We cannot reliably infer phase from the trial ID alone.
    # Return None and let the caller handle it.
    return None


def _extract_top_risk_factor(features: list[dict[str, Any]]) -> str:
    """Extract the single most important risk factor from SHAP contributions.

    Finds the feature with the most negative SHAP value (strongest negative
    impact on success probability). If all SHAP values are positive, returns
    ``"None (all positive)"`` to indicate no risk factors.

    Args:
        features: List of dicts with keys: ``name`` (str), ``shap_value`` (float).

    Returns:
        String describing the top risk factor (e.g., ``"dropout_rate (-0.234)"``)
        or ``"N/A"`` if no features available.
    """
    if not features:
        return "N/A"
    # Most negative SHAP = strongest push toward failure
    worst = min(features, key=lambda f: f["shap_value"])
    if worst["shap_value"] >= 0:
        return "None (all positive)"
    return f"{worst['name']} ({worst['shap_value']:+.3f})"


def render_portfolio_page(client: CTRAClient) -> None:
    """Render the trial portfolio analysis page."""
    st.header("Trial Portfolio Analysis")
    st.markdown(
        "Analyze multiple clinical trials at once.  Enter trial IDs below or upload a file."
    )

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    tab_text, tab_upload = st.tabs(["Enter IDs", "Upload file"])

    with tab_text:
        ids_text = st.text_area(
            "Trial IDs (one per line or comma-separated)",
            height=150,
            placeholder="NCT04280705\nNCT03461562\nNCT02793414",
            key="portfolio_ids_text",
        )

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a text or CSV file with trial IDs (one per line)",
            type=["txt", "csv"],
            key="portfolio_upload",
        )
        if uploaded is not None:
            file_content = uploaded.read().decode("utf-8")
            ids_text = file_content  # override
            st.text_area("Parsed from file", value=file_content, height=100, disabled=True)

    # Options
    col_model, col_shap = st.columns(2)
    with col_model:
        model_choice = st.selectbox(
            "Model",
            options=["Best available", "xgboost", "tabpfn"],
            index=0,
            key="portfolio_model",
        )
    with col_shap:
        include_shap = st.checkbox(
            "Include SHAP explanations",
            value=True,
            key="portfolio_shap",
        )

    trial_ids = _parse_trial_ids(ids_text or "")

    analyze_clicked = st.button(
        f"Analyze Portfolio ({len(trial_ids)} trials)",
        type="primary",
        disabled=len(trial_ids) == 0,
    )

    # ------------------------------------------------------------------
    # Run batch prediction
    # ------------------------------------------------------------------
    if analyze_clicked and trial_ids:
        if len(trial_ids) > 100:
            st.error("Maximum 100 trials per batch. Please reduce the list.")
            return

        model_param = None if model_choice == "Best available" else model_choice

        with st.spinner(f"Analyzing {len(trial_ids)} trials..."):
            try:
                result = client.predict_batch(
                    trial_ids=trial_ids,
                    model=model_param,
                    include_shap=include_shap,
                )
            except CTRAClientError as exc:
                st.error(f"Batch prediction failed: {exc.message}")
                if exc.detail:
                    st.caption(exc.detail)
                return
            except ConnectionError as exc:
                st.error(str(exc))
                return

        st.session_state["portfolio_result"] = result

        # Also store individual predictions in history for feature importance page
        if "prediction_history" not in st.session_state:
            st.session_state["prediction_history"] = {}
        for pred in result.get("predictions", []):
            st.session_state["prediction_history"][pred["trial_id"]] = pred

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    result = st.session_state.get("portfolio_result")
    if result is None:
        return

    predictions = result.get("predictions", [])
    batch_meta = result.get("batch_metadata", {})

    if not predictions:
        st.warning("No predictions were returned.")
        return

    st.markdown("---")

    # Batch summary
    col_total, col_success, col_failed, col_time = st.columns(4)
    with col_total:
        st.metric("Total Trials", batch_meta.get("total_trials", len(predictions)))
    with col_success:
        st.metric("Successful Predictions", batch_meta.get("successful_predictions", "N/A"))
    with col_failed:
        st.metric("Failed Predictions", batch_meta.get("failed_predictions", "N/A"))
    with col_time:
        time_ms = batch_meta.get("total_time_ms", 0)
        st.metric("Total Time", f"{time_ms:.0f} ms")

    # ------------------------------------------------------------------
    # Risk summary table (heatmap-style)
    # ------------------------------------------------------------------
    st.markdown("---")

    summary_rows = []
    for pred in predictions:
        summary_rows.append(
            {
                "Trial ID": pred["trial_id"],
                "Prediction": pred["prediction"].upper(),
                "Prob. Success": round(pred["probability_success"], 3),
                "Top Risk Factor": _extract_top_risk_factor(pred.get("features", [])),
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    fig_table = _build_risk_heatmap(df_summary)
    st.plotly_chart(fig_table, use_container_width=True)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    col_risk, col_pred = st.columns(2)

    with col_risk:
        probs = [p["probability_success"] for p in predictions]
        fig_risk = _build_risk_distribution(probs)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_pred:
        # Prediction outcome distribution (success vs failure counts)
        outcome_counts = pd.DataFrame(predictions)["prediction"].value_counts()
        fig_outcome = px.pie(
            values=outcome_counts.values,
            names=outcome_counts.index,
            title="Predicted Outcome Distribution",
            color=outcome_counts.index,
            color_discrete_map={"success": "#2ecc71", "failure": "#e74c3c"},
        )
        fig_outcome.update_layout(height=350, margin={"t": 50, "b": 20})
        st.plotly_chart(fig_outcome, use_container_width=True)

    # ------------------------------------------------------------------
    # Full sortable table
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("All Predictions")

    detail_rows = []
    for pred in predictions:
        detail_rows.append(
            {
                "Trial ID": pred["trial_id"],
                "Prediction": pred["prediction"],
                "P(Success)": pred["probability_success"],
                "P(Failure)": pred["probability_failure"],
                "Model": pred["model_used"],
                "Features": len(pred.get("features", [])),
                "Inference (ms)": pred.get("metadata", {}).get("inference_time_ms", None),
            }
        )

    df_detail = pd.DataFrame(detail_rows)
    st.dataframe(
        df_detail.style.format(
            {
                "P(Success)": "{:.3f}",
                "P(Failure)": "{:.3f}",
                "Inference (ms)": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
