"""FAERS/OpenFDA API client for adverse event data.

Queries the OpenFDA drug/event endpoint to retrieve post-market safety
signals.  Results are cached locally as Parquet for indexing by LinearRAG.

The client supports:
    - Single-drug adverse event queries with date filtering
    - Drug safety profile summarization as text passages
    - Batch pre-caching for a list of drugs
    - Rate limiting to respect OpenFDA API quotas

Date filtering uses the ``receivedate`` field from FAERS reports to ensure
temporal alignment with the trial's start date (label-leakage prevention).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import polars as pl

from ctra.config.settings import DataConfig, get_settings

logger = logging.getLogger(__name__)


class FAERSLoader:
    """Query and cache FAERS adverse event data from OpenFDA.

    Queries the OpenFDA drug/event endpoint to retrieve post-market safety
    signals. Results are cached locally as Parquet for indexing by LinearRAG.
    Supports single-drug adverse event queries with date filtering for
    temporal alignment with trial start dates (label-leakage prevention).

    Args:
        config: Data configuration. If ``None``, uses global settings.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initialize the FAERS loader.

        Args:
            config: Optional DataConfig to override global settings.
        """
        self._config = config or get_settings().data
        self._df: pl.DataFrame | None = None

    def load_parquet(self, path: Path | None = None) -> pl.DataFrame:
        """Load pre-cached FAERS data from Parquet.

        Expected columns: drug_name, event, count, seriousness, report_quarter.
        """
        parquet_path = Path(path or self._config.faers_parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"FAERS parquet not found at {parquet_path}.  "
                "Run `python scripts/data/fetch_faers.py` first."
            )

        logger.info("Loading FAERS data from %s", parquet_path)
        self._df = pl.read_parquet(parquet_path)
        logger.info("Loaded %d adverse event records", len(self._df))
        return self._df

    def query_drug_events(
        self,
        drug_name: str,
        before_date: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query OpenFDA for adverse events associated with a drug.

        Args:
            drug_name: Generic drug name.
            before_date: ISO date string (YYYY-MM-DD).  If provided, only
                events with ``receivedate`` before this date are counted.
                Also stored as ``report_quarter`` on each output record so
                the indexer can assign temporal metadata.
            limit: Maximum number of event types to return.

        Returns:
            List of event records with keys: ``drug_name``, ``event``,
            ``count``, ``report_quarter``.  The ``report_quarter`` field is
            set to ``before_date`` (the temporal boundary used for the query)
            so that the indexer can assign a date to each passage for
            downstream temporal filtering.
        """
        import httpx

        api_key = os.environ.get(self._config.openfda_api_key_env_var, "")
        base_url = self._config.openfda_api_base

        # Build search query with optional date filter (C2 fix)
        search_parts = [f'patient.drug.medicinalproduct:"{drug_name}"']
        if before_date is not None:
            date_clean = before_date.replace("-", "")
            search_parts.append(f"receivedate:[19700101+TO+{date_clean}]")
        search_query = "+AND+".join(search_parts)

        params: dict[str, str | int] = {
            "search": search_query,
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": limit,
        }
        if api_key:
            params["api_key"] = api_key

        try:
            response = httpx.get(base_url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError:
            logger.warning("OpenFDA query failed for drug: %s", drug_name)
            return []

        results = data.get("results", [])
        events = []
        for item in results:
            events.append(
                {
                    "drug_name": drug_name,
                    "event": item.get("term", ""),
                    "count": item.get("count", 0),
                    "report_quarter": before_date or "",
                }
            )

        # Rate limiting
        rate_limit = self._config.openfda_rate_limit
        if rate_limit > 0:
            time.sleep(1.0 / rate_limit)

        return events

    def fetch_and_cache(
        self,
        drug_names: list[str],
        output_path: Path | None = None,
        before_date: str | None = None,
    ) -> pl.DataFrame:
        """Fetch FAERS data for multiple drugs and save to Parquet.

        Args:
            drug_names: List of drug names to query.
            output_path: Where to save the Parquet file.
            before_date: ISO date string (YYYY-MM-DD).  If provided, only
                events reported before this date are included (label-leakage
                prevention).

        Returns:
            DataFrame with all adverse event records.
        """
        all_events: list[dict[str, Any]] = []

        for i, drug in enumerate(drug_names):
            logger.info("Querying FAERS for %s (%d/%d)", drug, i + 1, len(drug_names))
            events = self.query_drug_events(drug, before_date=before_date)
            all_events.extend(events)

        df = pl.DataFrame(all_events) if all_events else pl.DataFrame()
        if len(df) > 0:
            out = Path(output_path or self._config.faers_parquet)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out)
            logger.info("Saved %d FAERS records to %s", len(df), out)

        self._df = df
        return df


# ---------------------------------------------------------------------------
# FAERSClient -- date-aware API client
# ---------------------------------------------------------------------------


class FAERSClient:
    """Client for OpenFDA drug adverse event API with date filtering.

    Provides date-filtered queries suitable for label-leakage prevention:
    only adverse events reported *before* a given date are returned.

    Args:
        api_key: OpenFDA API key (optional; increases rate limit from 40/min
            to 240/min).
        rate_limit: Maximum requests per second.
    """

    BASE_URL = "https://api.fda.gov/drug/event.json"

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: float = 4.0,
    ) -> None:
        """Initialize the OpenFDA client.

        Args:
            api_key: OpenFDA API key (from environment if not provided).
            rate_limit: Maximum requests per second (default 4).
        """
        self._api_key = api_key or os.environ.get("OPENFDA_API_KEY", "")
        self._rate_limit = rate_limit
        self._last_request_time: float = 0.0

    def _throttle(self) -> None:
        """Enforce rate limiting between requests."""
        if self._rate_limit <= 0:
            return
        min_interval = 1.0 / self._rate_limit
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def search_drug_events(
        self,
        drug_name: str,
        before_date: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search adverse events for a drug before a given date.

        Args:
            drug_name: Generic drug name (e.g., ``"aspirin"``).
            before_date: ISO date string (YYYY-MM-DD).  If provided, only
                events with ``receivedate`` before this date are counted.
            limit: Maximum number of event types to return.

        Returns:
            List of event dicts with keys: ``drug_name``, ``event``, ``count``.
        """
        import httpx

        self._throttle()

        # Build the search query with optional date filter
        search_parts = [f'patient.drug.medicinalproduct:"{drug_name}"']
        if before_date is not None:
            # OpenFDA date format: YYYYMMDD
            date_clean = before_date.replace("-", "")
            search_parts.append(f"receivedate:[19700101+TO+{date_clean}]")

        search_query = "+AND+".join(search_parts)

        params: dict[str, str | int] = {
            "search": search_query,
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": limit,
        }
        if self._api_key:
            params["api_key"] = self._api_key

        try:
            response = httpx.get(self.BASE_URL, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            logger.warning("OpenFDA query failed for %s: %s", drug_name, exc)
            return []

        results = data.get("results", [])
        events: list[dict[str, Any]] = []
        for item in results:
            events.append(
                {
                    "drug_name": drug_name,
                    "event": item.get("term", ""),
                    "count": item.get("count", 0),
                }
            )

        return events

    def get_drug_safety_profile(
        self,
        drug_name: str,
        before_date: str | None = None,
        top_n: int = 20,
    ) -> str:
        """Get a text summary of a drug's adverse event profile.

        Returns a human-readable summary suitable for LinearRAG indexing
        or direct use by the feature builder agent.

        Args:
            drug_name: Generic drug name.
            before_date: ISO date (YYYY-MM-DD) cutoff for temporal filtering.
            top_n: Number of top adverse events to include.

        Returns:
            Multi-sentence text summary of the drug's safety profile.
        """
        events = self.search_drug_events(
            drug_name=drug_name,
            before_date=before_date,
            limit=top_n,
        )

        if not events:
            return f"No adverse event data found for {drug_name} in FAERS."

        lines: list[str] = [
            f"FAERS adverse event profile for {drug_name}"
            + (f" (reports before {before_date})" if before_date else "")
            + ":"
        ]

        total_reports = sum(e["count"] for e in events)
        lines.append(f"Total reports across top {len(events)} events: {total_reports}.")

        for event in events[:top_n]:
            pct = (event["count"] / total_reports * 100) if total_reports > 0 else 0
            lines.append(f"- {event['event']}: {event['count']} reports ({pct:.1f}%)")

        return "\n".join(lines)

    def cache_drug_events(
        self,
        drug_names: list[str],
        output_dir: str | Path,
        before_date: str | None = None,
    ) -> pl.DataFrame:
        """Pre-cache adverse event data for a list of drugs.

        Queries OpenFDA for each drug and saves results as a Parquet file
        for later indexing.

        Args:
            drug_names: List of generic drug names to query.
            output_dir: Directory to save the cache file.
            before_date: Optional temporal cutoff for all queries.

        Returns:
            DataFrame with all adverse event records.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cache_path = out / "faers_cache.parquet"

        all_events: list[dict[str, Any]] = []

        for i, drug in enumerate(drug_names):
            logger.info("Caching FAERS for %s (%d/%d)", drug, i + 1, len(drug_names))
            events = self.search_drug_events(
                drug_name=drug,
                before_date=before_date,
            )
            all_events.extend(events)

        df = pl.DataFrame(all_events) if all_events else pl.DataFrame()
        if len(df) > 0:
            df.write_parquet(cache_path)
            logger.info("Cached %d FAERS records to %s", len(df), cache_path)

        return df
