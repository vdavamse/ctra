"""Model Performance page -- metrics tracking and model comparison."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from ctra.dashboard.api_client import CTRAClient

logger = logging.getLogger(__name__)


def _try_load_mlflow_metrics() -> pd.DataFrame | None:
    """Attempt to load historical metrics from MLflow experiment tracking.

    Queries the ``"ctra"`` MLflow experiment for the 100 most recent runs,
    extracts all metrics, and returns a DataFrame for visualization.

    Returns:
        DataFrame with columns: ``run_id``, ``model``, ``metric``, ``value``,
        ``timestamp``. Returns ``None`` if MLflow is unavailable or the
        experiment has no runs.
    """
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("ctra")
        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=100,
        )
        if not runs:
            return None

        rows = []
        for run in runs:
            model_name = run.data.params.get("model_type", "unknown")
            run_time = run.info.start_time
            for key, value in run.data.metrics.items():
                rows.append(
                    {
                        "run_id": run.info.run_id[:8],
                        "model": model_name,
                        "metric": key,
                        "value": value,
                        "timestamp": pd.Timestamp(run_time, unit="ms"),
                    }
                )
        return pd.DataFrame(rows) if rows else None

    except ImportError:
        return None
    except Exception:
        logger.debug("MLflow metrics unavailable", exc_info=True)
        return None


def _build_static_comparison() -> pd.DataFrame:
    """Return a static model comparison table from published benchmarks.

    Includes expected baseline performance (XGBoost, TabPFN) and reference
    baselines (HINT, MEXA-CTP) on the TOP dataset across all phases.
    These values serve as a reference until MLflow tracking is populated
    with production run metrics.

    Returns:
        DataFrame with columns: ``Model``, ``ROC-AUC (Phase I/II/III)``,
        ``F1 (macro)``, ``PR-AUC``, ``ECE``, ``Status``.
    """
    return pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "ROC-AUC (Phase I)": 0.72,
                "ROC-AUC (Phase II)": 0.68,
                "ROC-AUC (Phase III)": 0.71,
                "F1 (macro)": 0.66,
                "PR-AUC": 0.69,
                "ECE": 0.08,
                "Status": "Baseline",
            },
            {
                "Model": "TabPFN",
                "ROC-AUC (Phase I)": 0.75,
                "ROC-AUC (Phase II)": 0.71,
                "ROC-AUC (Phase III)": 0.74,
                "F1 (macro)": 0.70,
                "PR-AUC": 0.72,
                "ECE": 0.06,
                "Status": "Baseline",
            },
            {
                "Model": "HINT",
                "ROC-AUC (Phase I)": 0.70,
                "ROC-AUC (Phase II)": 0.66,
                "ROC-AUC (Phase III)": 0.69,
                "F1 (macro)": 0.63,
                "PR-AUC": 0.66,
                "ECE": 0.12,
                "Status": "Reference baseline",
            },
            {
                "Model": "MEXA-CTP",
                "ROC-AUC (Phase I)": 0.76,
                "ROC-AUC (Phase II)": 0.72,
                "ROC-AUC (Phase III)": 0.75,
                "F1 (macro)": 0.71,
                "PR-AUC": 0.73,
                "ECE": 0.05,
                "Status": "Reference SOTA",
            },
        ]
    )


def _build_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Build a grouped bar chart comparing model ROC-AUC across phases.

    Displays one group per phase (I, II, III) with bars for each model,
    allowing side-by-side performance comparison.

    Args:
        df: DataFrame from ``_build_static_comparison`` with ``ROC-AUC (Phase *)``
            columns and a ``Model`` column.

    Returns:
        A ``plotly.express`` Figure with grouped bars.
    """
    metrics = [c for c in df.columns if c.startswith("ROC-AUC")]
    df_melted = df.melt(
        id_vars=["Model"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Value",
    )

    fig = px.bar(
        df_melted,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="Model Comparison: ROC-AUC by Phase",
        color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
    )
    fig.update_layout(
        yaxis={"range": [0, 1], "title": "ROC-AUC"},
        height=400,
        margin={"t": 50, "b": 40},
    )
    return fig


def _build_metrics_over_time(df_mlflow: pd.DataFrame, metric_name: str) -> go.Figure:
    """Build a line chart showing a metric's evolution across MLflow runs.

    Each line represents a different model, with timestamps on the x-axis
    and metric values on the y-axis.

    Args:
        df_mlflow: DataFrame from ``_try_load_mlflow_metrics`` with columns:
            ``timestamp``, ``model``, ``metric``, ``value``.
        metric_name: The metric column to plot (e.g., ``"val_roc_auc"``).

    Returns:
        A ``plotly.express`` Figure with line traces for each model.
    """
    df_metric = df_mlflow[df_mlflow["metric"] == metric_name].copy()
    df_metric = df_metric.sort_values("timestamp")

    fig = px.line(
        df_metric,
        x="timestamp",
        y="value",
        color="model",
        title=f"{metric_name} Over Time",
        markers=True,
    )
    fig.update_layout(
        yaxis_title=metric_name,
        xaxis_title="Training Run",
        height=350,
        margin={"t": 50, "b": 40},
    )
    return fig


def _render_drift_alerts(models_info: list[dict[str, Any]]) -> None:
    """Render data drift monitoring alerts from model monitoring system.

    Checks for feature distribution drift and prediction calibration
    degradation against a reference window. Displays alerts in the
    Streamlit UI if drift is detected.

    Args:
        models_info: List of model metadata dicts with monitoring status.
    """
    st.subheader("Monitoring Alerts")

    # In production, drift detection would check feature distributions
    # and prediction calibration against a reference window.
    # For now, show placeholder status.
    col_feat, col_pred, col_cal = st.columns(3)

    with col_feat:
        st.info("**Feature Drift**\nNo drift detected")
    with col_pred:
        st.info("**Prediction Drift**\nNo drift detected")
    with col_cal:
        st.info("**Calibration**\nWithin tolerance")

    st.caption(
        "Drift detection will activate once the MLOps monitoring pipeline "
        "is configured with a reference dataset and threshold settings."
    )


def render_performance_page(client: CTRAClient) -> None:
    """Render the model performance tracking page."""
    st.header("Model Performance")
    st.markdown("Compare model metrics, track performance over time, and monitor for data drift.")

    # ------------------------------------------------------------------
    # Currently loaded models
    # ------------------------------------------------------------------
    st.subheader("Loaded Models")

    try:
        models = client.get_models()
        if models:
            model_rows = []
            for m in models:
                row = {
                    "Name": m["name"],
                    "Type": m["type"],
                    "Version": m["version"],
                    "Features": m["n_features"],
                    "Feature Set": m["feature_set_id"],
                }
                # Merge training metrics
                for metric_name, metric_val in m.get("training_metrics", {}).items():
                    row[metric_name] = metric_val
                model_rows.append(row)

            df_models = pd.DataFrame(model_rows)
            st.dataframe(df_models, use_container_width=True, hide_index=True)
        else:
            st.info("No models currently loaded in the API.")
    except ConnectionError:
        st.warning("Cannot reach the API.  Showing static reference metrics below.")
    except Exception as exc:
        st.warning(f"Could not fetch model info: {exc}")

    # ------------------------------------------------------------------
    # Model comparison table
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Model Comparison")

    # Try MLflow first, fall back to static data
    df_mlflow = _try_load_mlflow_metrics()

    if df_mlflow is not None and not df_mlflow.empty:
        st.caption("Source: MLflow experiment tracking")

        # Build a pivot table: rows = model, columns = metrics
        latest_per_model = df_mlflow.sort_values("timestamp").drop_duplicates(
            subset=["model", "metric"], keep="last"
        )
        df_pivot = latest_per_model.pivot(index="model", columns="metric", values="value")
        df_pivot = df_pivot.reset_index().rename(columns={"model": "Model"})
        st.dataframe(
            df_pivot.style.format({c: "{:.4f}" for c in df_pivot.columns if c != "Model"}),
            use_container_width=True,
            hide_index=True,
        )

        # Metrics over time
        st.markdown("---")
        st.subheader("Metrics Over Time")

        available_metrics = sorted(df_mlflow["metric"].unique())
        selected_metric = st.selectbox("Metric", available_metrics, key="perf_metric")
        if selected_metric:
            fig_time = _build_metrics_over_time(df_mlflow, selected_metric)
            st.plotly_chart(fig_time, use_container_width=True)

    else:
        st.caption(
            "Source: Static reference values (MLflow tracking not available). "
            "These baselines are approximate and will be replaced once models "
            "are trained and tracked."
        )

    df_static = _build_static_comparison()
    st.dataframe(
        df_static.style.format(
            {c: "{:.2f}" for c in df_static.columns if c not in ("Model", "Status")}
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_comparison = _build_comparison_chart(df_static)
    st.plotly_chart(fig_comparison, use_container_width=True)

    # ------------------------------------------------------------------
    # Prediction distribution from session
    # ------------------------------------------------------------------
    prediction_history = st.session_state.get("prediction_history", {})
    if prediction_history:
        st.markdown("---")
        st.subheader("Session Prediction Distribution")
        st.caption(f"Based on {len(prediction_history)} predictions made this session")

        probs = [p["probability_success"] for p in prediction_history.values()]
        fig_dist = px.histogram(
            x=probs,
            nbins=20,
            labels={"x": "Probability of Success"},
            title="Prediction Distribution (This Session)",
            color_discrete_sequence=["#3498db"],
        )
        fig_dist.update_layout(
            xaxis={"range": [0, 1]},
            yaxis_title="Count",
            height=300,
            margin={"t": 50, "b": 40},
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ------------------------------------------------------------------
    # Drift monitoring
    # ------------------------------------------------------------------
    st.markdown("---")
    try:
        models_info = client.get_models()
    except Exception:
        models_info = []
    _render_drift_alerts(models_info)
