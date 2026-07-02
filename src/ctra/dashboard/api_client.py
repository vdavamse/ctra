"""Thin HTTP client for the CTRA REST API.

Used by the Streamlit dashboard to communicate with the prediction service.
All methods are synchronous because Streamlit runs in a synchronous event loop.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class CTRAClientError(Exception):
    """Raised when an API call fails."""

    def __init__(self, status_code: int, message: str, detail: str | None = None) -> None:
        self.status_code = status_code
        self.message = message
        self.detail = detail
        super().__init__(
            f"[{status_code}] {message}: {detail}" if detail else f"[{status_code}] {message}"
        )


class CTRAClient:
    """Synchronous client for the CTRA prediction API.

    Args:
        base_url: Root URL of the CTRA API (e.g. ``http://localhost:8000``).
            Falls back to the ``CTRA_API_URL`` environment variable, then
            ``http://localhost:8000``.
        api_key: Optional bearer token. Falls back to ``CTRA_API_KEY`` env var.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        url = base_url or os.getenv("CTRA_API_URL") or "http://localhost:8000"
        self.base_url = url.rstrip("/")
        self._api_key = api_key or os.getenv("CTRA_API_KEY")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Execute an HTTP request and return parsed JSON.

        Raises :class:`CTRAClientError` on non-2xx responses and
        :class:`ConnectionError` when the API is unreachable.
        """
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.request(method, url, headers=self._headers(), **kwargs)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach CTRA API at {self.base_url}. Ensure the API server is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(f"Request to {url} timed out after {self._timeout}s.") from exc

        if response.status_code >= 400:
            body = (
                response.json()
                if response.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            raise CTRAClientError(
                status_code=response.status_code,
                message=body.get("error", response.reason_phrase or "Unknown error"),
                detail=body.get("detail"),
            )

        return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        trial_id: str,
        model: str | None = None,
        include_shap: bool = True,
    ) -> dict[str, Any]:
        """Predict outcome for a single clinical trial.

        Calls ``POST /api/v1/predict``.
        """
        payload: dict[str, Any] = {
            "trial_id": trial_id,
            "include_shap": include_shap,
        }
        if model is not None:
            payload["model"] = model
        return self._request("POST", "/api/v1/predict", json=payload)

    def predict_batch(
        self,
        trial_ids: list[str],
        model: str | None = None,
        include_shap: bool = True,
    ) -> dict[str, Any]:
        """Predict outcomes for multiple clinical trials.

        Calls ``POST /api/v1/predict/batch``.
        """
        payload: dict[str, Any] = {
            "trial_ids": trial_ids,
            "include_shap": include_shap,
        }
        if model is not None:
            payload["model"] = model
        return self._request("POST", "/api/v1/predict/batch", json=payload)

    def get_models(self) -> list[dict[str, Any]]:
        """List loaded models.

        Calls ``GET /api/v1/models``.
        """
        return self._request("GET", "/api/v1/models")  # type: ignore[return-value]

    def get_health(self) -> dict[str, Any]:
        """Health check.

        Calls ``GET /api/v1/health``.
        """
        return self._request("GET", "/api/v1/health")
