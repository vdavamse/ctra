"""ClinicalTrials.gov API v2 async client.

Provides paginated search over the ClinicalTrials.gov API v2 with rate
limiting and incremental update support (fetch studies modified since a date).

API reference: https://clinicaltrials.gov/data-api/api

Key design decisions:
    - ``pageToken``-based pagination (NOT offset/limit -- v2 deprecated that)
    - Rate limiting via asyncio.Semaphore + sleep (10 req/sec default)
    - Only ``protocolSection`` fields are fetched by default to prevent
      label leakage from ``resultsSection``
    - All HTTP uses ``httpx.AsyncClient`` for non-blocking I/O
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

# Default fields to request -- protocolSection only (no resultsSection).
_DEFAULT_FIELDS = [
    "protocolSection",
]


class CTGApiClient:
    """Async client for ClinicalTrials.gov API v2.

    Provides paginated search with rate limiting and incremental update support
    (fetch studies modified since a date). Uses pageToken-based pagination
    (API v2 deprecated offset/limit). Only fetches protocolSection to prevent
    label leakage from resultsSection.

    Args:
        base_url: API v2 base URL.
        rate_limit: Maximum requests per second.
        timeout: HTTP request timeout in seconds.
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(
        self,
        base_url: str | None = None,
        rate_limit: float = 10.0,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the CTG API client.

        Args:
            base_url: API v2 base URL (uses default if None).
            rate_limit: Maximum requests per second (default 10).
            timeout: HTTP request timeout in seconds (default 30).
        """
        self._base_url = base_url or self.BASE_URL
        self._rate_limit = rate_limit
        self._timeout = timeout
        # Semaphore limits concurrency; sleep enforces per-request spacing.
        self._semaphore = asyncio.Semaphore(max(1, int(rate_limit)))

    async def _get(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Rate-limited GET request with retry."""
        async with self._semaphore:
            response = await self._get_with_retry(client, url, params)
            # Enforce minimum interval between requests.
            if self._rate_limit > 0:
                await asyncio.sleep(1.0 / self._rate_limit)
            return response

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _get_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """GET with tenacity retries on transient errors."""
        resp = await client.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def search(
        self,
        query: str | None = None,
        filter_status: list[str] | None = None,
        filter_advanced: str | None = None,
        fields: list[str] | None = None,
        page_size: int = 100,
        max_pages: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Search studies with pageToken pagination.

        Args:
            query: Free-text search (``query.term``).
            filter_status: List of ``overallStatus`` values to filter by
                (e.g., ``["COMPLETED", "TERMINATED"]``).
            filter_advanced: Raw ``filter.advanced`` query string.
            fields: Fields to return (default: ``protocolSection`` only).
            page_size: Number of studies per page (max 1000).
            max_pages: Maximum number of pages to fetch (``None`` = all).

        Yields:
            Study dicts from each page.
        """
        params: dict[str, Any] = {
            "pageSize": min(page_size, 1000),
            "fields": "|".join(fields or _DEFAULT_FIELDS),
        }

        if query:
            params["query.term"] = query

        if filter_status:
            params["filter.overallStatus"] = ",".join(filter_status)

        if filter_advanced:
            params["filter.advanced"] = filter_advanced

        pages_fetched = 0
        next_page_token: str | None = None

        async with httpx.AsyncClient() as client:
            while True:
                if next_page_token:
                    params["pageToken"] = next_page_token
                elif "pageToken" in params:
                    del params["pageToken"]

                try:
                    data = await self._get(client, self._base_url, params)
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    logger.error("CTG API request failed after retries: %s", exc)
                    return

                studies = data.get("studies", [])
                for study in studies:
                    yield study

                pages_fetched += 1

                # Check for next page
                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break

                if max_pages is not None and pages_fetched >= max_pages:
                    logger.info("Reached max_pages=%d, stopping pagination", max_pages)
                    break

        logger.info("CTG search completed: %d pages fetched", pages_fetched)

    async def get_study(self, nct_id: str, fields: list[str] | None = None) -> dict[str, Any]:
        """Get a single study by NCT ID.

        Args:
            nct_id: NCT identifier (e.g., ``NCT00110279``).
            fields: Fields to return (default: ``protocolSection``).

        Returns:
            Study dict.

        Raises:
            httpx.HTTPStatusError: If the study is not found (404) or other HTTP error.
        """
        url = f"{self._base_url}/{nct_id}"
        params: dict[str, Any] = {
            "fields": "|".join(fields or _DEFAULT_FIELDS),
        }

        async with httpx.AsyncClient() as client:
            data = await self._get(client, url, params)
        return data

    async def get_updated_since(
        self,
        since_date: str,
        fields: list[str] | None = None,
        page_size: int = 100,
        max_pages: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Get studies updated since a given date.

        Uses the ``filter.advanced`` parameter with ``AREA[LastUpdatePostDate]``
        to find studies modified after the specified date.

        Args:
            since_date: Date in ``YYYY-MM-DD`` format.
            fields: Fields to return.
            page_size: Studies per page.
            max_pages: Maximum number of pages.

        Yields:
            Study dicts modified since ``since_date``.
        """
        # CTG API v2 advanced filter syntax for date range.
        # AREA[LastUpdatePostDate]RANGE[since_date, MAX]
        filter_query = f"AREA[LastUpdatePostDate]RANGE[{since_date}, MAX]"

        async for study in self.search(
            filter_advanced=filter_query,
            fields=fields,
            page_size=page_size,
            max_pages=max_pages,
        ):
            yield study

    async def get_study_count(
        self,
        query: str | None = None,
        filter_status: list[str] | None = None,
        filter_advanced: str | None = None,
    ) -> int:
        """Get the total count of studies matching a query.

        Uses ``countTotal=true`` and ``pageSize=0`` to efficiently get
        only the total count without downloading study data.

        Args:
            query: Free-text search term.
            filter_status: Status filter list.
            filter_advanced: Advanced filter query.

        Returns:
            Total number of matching studies.
        """
        params: dict[str, Any] = {
            "pageSize": 0,
            "countTotal": "true",
        }

        if query:
            params["query.term"] = query

        if filter_status:
            params["filter.overallStatus"] = ",".join(filter_status)

        if filter_advanced:
            params["filter.advanced"] = filter_advanced

        async with httpx.AsyncClient() as client:
            data = await self._get(client, self._base_url, params)

        return data.get("totalCount", 0)  # type: ignore[no-any-return]
