"""CTRA Dashboard -- Clinical Trial Risk Assessment.

Main Streamlit application with sidebar navigation across four pages:
Trial Prediction, Feature Importance, Trial Portfolio, and Model Performance.

Launch with::

    streamlit run src/ctra/dashboard/app.py

Or via the entry point::

    ctra-dashboard
"""

from __future__ import annotations

import streamlit as st

from ctra.dashboard.api_client import CTRAClient
from ctra.dashboard.pages.feature_importance import render_feature_importance_page
from ctra.dashboard.pages.performance import render_performance_page
from ctra.dashboard.pages.portfolio import render_portfolio_page
from ctra.dashboard.pages.prediction import render_prediction_page


def _get_client() -> CTRAClient:
    """Return a cached API client stored in Streamlit session state.

    Streamlit reruns the entire script on every interaction, so we cache
    the API client in ``st.session_state`` to avoid recreating it.
    The client handles connection pooling and bearer token management.

    Returns:
        The ``CTRAClient`` instance for making requests to the CTRA API.
    """
    if "api_client" not in st.session_state:
        st.session_state.api_client = CTRAClient()
    return st.session_state.api_client  # type: ignore[no-any-return]


def _render_sidebar() -> str:
    """Render the navigation sidebar with API health status.

    Displays:
    - Page selector (Trial Prediction, Feature Importance, Portfolio, Performance)
    - API connection status (healthy/degraded/offline)
    - Loaded model names
    - RAG index status per source

    Returns:
        The name of the page selected by the user.
    """
    with st.sidebar:
        st.title("CTRA")
        st.caption("Clinical Trial Risk Assessment")
        st.markdown("---")

        page = st.selectbox(
            "Navigation",
            [
                "Trial Prediction",
                "Feature Importance",
                "Trial Portfolio",
                "Model Performance",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Connection status indicator
        client = _get_client()
        try:
            health = client.get_health()
            status = health.get("status", "unknown")
            version = health.get("version", "?")
            if status == "healthy":
                st.success(f"API connected (v{version})")
            else:
                st.warning(f"API degraded (v{version})")

            models_loaded = health.get("models_loaded", [])
            if models_loaded:
                st.caption(f"Models: {', '.join(models_loaded)}")

            index_status = health.get("index_status", {})
            if index_status:
                for source, state in index_status.items():
                    if state == "ready":
                        st.caption(f"  {source}: ready")
                    elif state == "stale":
                        st.caption(f"  {source}: stale")
                    else:
                        st.caption(f"  {source}: {state}")
        except (ConnectionError, Exception):
            st.error("API offline")
            st.caption(f"Endpoint: {client.base_url}")

    return page  # type: ignore[no-any-return]


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="CTRA -- Clinical Trial Risk Assessment",
        page_icon="\U0001f52c",  # microscope
        layout="wide",
        initial_sidebar_state="expanded",
    )

    page = _render_sidebar()
    client = _get_client()

    if page == "Trial Prediction":
        render_prediction_page(client)
    elif page == "Feature Importance":
        render_feature_importance_page(client)
    elif page == "Trial Portfolio":
        render_portfolio_page(client)
    elif page == "Model Performance":
        render_performance_page(client)


if __name__ == "__main__":
    main()
