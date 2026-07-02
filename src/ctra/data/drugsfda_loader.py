"""Drugs@FDA API client for FDA drug approval history.

Queries the openFDA drug/drugsfda endpoint to retrieve FDA approval records
including application numbers, sponsor names, active ingredients, approval
dates, and review priority designations.  Results are cached locally as
Parquet for indexing by LinearRAG.

The client supports:
    - Paginated fetching of all FDA drug applications (~28,961 total)
    - Flattening of nested JSON into a tabular format
    - Sponsor-based and drug-name-based lookups
    - Rate limiting to respect OpenFDA API quotas (shared with FAERS)
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

# Maximum results per openFDA API request
_MAX_PAGE_SIZE = 1000


class DrugsFDALoader:
    """Query and cache Drugs@FDA approval data from OpenFDA.

    Fetches FDA drug application records (NDAs, ANDAs, BLAs) from the openFDA
    drugsfda endpoint, flattens the nested JSON, and caches the result as Parquet
    for use by LinearRAG.  Supports sponsor-based and drug-name lookups.

    Args:
        config: Data configuration. If ``None``, uses global settings.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initialize the Drugs@FDA loader.

        Args:
            config: Optional DataConfig to override global settings.
        """
        self._config = config or get_settings().data
        self._df: pl.DataFrame | None = None

    def load_parquet(self, path: Path | None = None) -> pl.DataFrame:
        """Load pre-cached Drugs@FDA data from Parquet.

        Expected columns: application_number, sponsor_name, brand_name,
        active_ingredients, application_type, approval_date, review_priority,
        submission_class_description, is_orphan, marketing_status.
        """
        parquet_path = Path(path or self._config.drugsfda_parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Drugs@FDA parquet not found at {parquet_path}.  "
                "Run `python scripts/data/fetch_drugsfda.py` first."
            )

        logger.info("Loading Drugs@FDA data from %s", parquet_path)
        self._df = pl.read_parquet(parquet_path)
        logger.info("Loaded %d drug application records", len(self._df))
        return self._df

    def fetch_approvals(
        self,
        limit: int = 1000,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """Fetch a page of drug applications from the openFDA API.

        Args:
            limit: Number of results per page (max 1000).
            skip: Number of results to skip (for pagination).

        Returns:
            List of raw application dicts from the API response.
        """
        import httpx

        api_key = os.environ.get(self._config.openfda_api_key_env_var, "")
        base_url = self._config.drugsfda_api_base

        params: dict[str, str | int] = {
            "limit": min(limit, _MAX_PAGE_SIZE),
            "skip": skip,
        }
        if api_key:
            params["api_key"] = api_key

        try:
            response = httpx.get(base_url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            logger.warning("Drugs@FDA query failed (skip=%d): %s", skip, exc)
            return []

        results = data.get("results", [])

        # Rate limiting (shared openFDA quota with FAERS)
        rate_limit = self._config.openfda_rate_limit
        if rate_limit > 0:
            time.sleep(1.0 / rate_limit)

        return results  # type: ignore[no-any-return]

    def fetch_and_cache(
        self,
        output_path: Path | None = None,
    ) -> pl.DataFrame:
        """Paginate through all Drugs@FDA applications and save to Parquet.

        Fetches all ~28,961 applications in pages of 1000, flattens the
        nested JSON structure into tabular columns, and writes the result
        as a Parquet file.

        Args:
            output_path: Where to save the Parquet file.  If ``None``, uses
                the configured ``drugsfda_parquet`` path.

        Returns:
            DataFrame with all flattened application records.
        """
        all_records: list[dict[str, Any]] = []
        skip = 0

        while True:
            logger.info("Fetching Drugs@FDA applications (skip=%d)", skip)
            results = self.fetch_approvals(limit=_MAX_PAGE_SIZE, skip=skip)

            if not results:
                break

            for app in results:
                record = _flatten_application(app)
                if record is not None:
                    all_records.append(record)

            if len(results) < _MAX_PAGE_SIZE:
                break

            skip += _MAX_PAGE_SIZE

        logger.info("Fetched %d total application records", len(all_records))

        df = pl.DataFrame(all_records) if all_records else pl.DataFrame()
        if len(df) > 0:
            out = Path(output_path or self._config.drugsfda_parquet)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out)
            logger.info("Saved %d Drugs@FDA records to %s", len(df), out)

        self._df = df
        return df

    def get_sponsor_approvals(self, sponsor_name: str) -> list[dict[str, Any]]:
        """Filter cached data by sponsor name (case-insensitive substring).

        Args:
            sponsor_name: Sponsor name or partial name.

        Returns:
            List of matching application records as dicts.
        """
        if self._df is None:
            self.load_parquet()
        assert self._df is not None

        mask = self._df.get_column("sponsor_name").str.contains(f"(?i){sponsor_name}")
        return self._df.filter(mask).to_dicts()

    def get_drug_approval(self, drug_name: str) -> dict[str, Any] | None:
        """Search for a drug by active ingredient name (case-insensitive).

        Returns the first matching application record, or ``None`` if no
        match is found.

        Args:
            drug_name: Active ingredient name or partial name.

        Returns:
            Application record dict, or ``None``.
        """
        if self._df is None:
            self.load_parquet()
        assert self._df is not None

        mask = self._df.get_column("active_ingredients").str.contains(f"(?i){drug_name}")
        matches = self._df.filter(mask)
        if len(matches) == 0:
            return None
        return matches.row(0, named=True)


# ---------------------------------------------------------------------------
# JSON flattening helpers
# ---------------------------------------------------------------------------


def _flatten_application(app: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten a single Drugs@FDA application JSON into a tabular record.

    Extracts the original (ORIG) submission to determine approval date,
    review priority, and orphan designation.

    Args:
        app: Raw application dict from the openFDA API.

    Returns:
        Flattened record dict, or ``None`` if the application number is
        missing.
    """
    application_number = app.get("application_number", "")
    if not application_number:
        return None

    sponsor_name = app.get("sponsor_name", "")

    # Derive application type from the prefix (NDA, ANDA, BLA)
    application_type = _extract_application_type(application_number)

    # Extract product information (first product entry)
    products = app.get("products", [])
    brand_name = ""
    active_ingredients_list: list[str] = []
    marketing_status = ""

    for product in products:
        if not brand_name:
            brand_name = product.get("brand_name", "")
        if not marketing_status:
            marketing_status = product.get("marketing_status", "")

        for ingredient in product.get("active_ingredients", []):
            name = ingredient.get("name", "")
            if name and name not in active_ingredients_list:
                active_ingredients_list.append(name)

    # Fallback to openfda fields if products are empty
    openfda = app.get("openfda", {})
    if not brand_name:
        openfda_brands = openfda.get("brand_name", [])
        if openfda_brands:
            brand_name = openfda_brands[0]
    if not active_ingredients_list:
        openfda_generics = openfda.get("generic_name", [])
        if openfda_generics:
            active_ingredients_list = list(openfda_generics)

    active_ingredients = "|".join(active_ingredients_list)

    # Find the original (ORIG) submission for approval date and metadata
    submissions = app.get("submissions", [])
    approval_date = ""
    review_priority = ""
    submission_class_description = ""
    is_orphan = False

    orig_submission = _find_original_submission(submissions)
    if orig_submission is not None:
        # Parse YYYYMMDD date to ISO format YYYY-MM-DD
        raw_date = orig_submission.get("submission_status_date", "")
        approval_date = _parse_fda_date(raw_date)

        review_priority = orig_submission.get("review_priority", "")
        submission_class_description = orig_submission.get("submission_class_code_description", "")

        # Check for orphan designation
        for prop in orig_submission.get("submission_property_type", []):
            if prop.get("code", "") == "Orphan":
                is_orphan = True
                break

    return {
        "application_number": application_number,
        "sponsor_name": sponsor_name,
        "brand_name": brand_name,
        "active_ingredients": active_ingredients,
        "application_type": application_type,
        "approval_date": approval_date,
        "review_priority": review_priority,
        "submission_class_description": submission_class_description,
        "is_orphan": is_orphan,
        "marketing_status": marketing_status,
    }


def _find_original_submission(
    submissions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the original (ORIG) submission with approval status.

    Searches for a submission where ``submission_type == "ORIG"`` and
    ``submission_status == "AP"`` (approved).

    Args:
        submissions: List of submission dicts from the API.

    Returns:
        The original approved submission dict, or ``None``.
    """
    for sub in submissions:
        if sub.get("submission_type") == "ORIG" and sub.get("submission_status") == "AP":
            return sub

    # Fallback: any ORIG submission regardless of status
    for sub in submissions:
        if sub.get("submission_type") == "ORIG":
            return sub

    return None


def _extract_application_type(application_number: str) -> str:
    """Extract the application type prefix (NDA, ANDA, BLA).

    Args:
        application_number: e.g., ``"NDA125514"`` or ``"ANDA078432"``.

    Returns:
        Application type string (``"NDA"``, ``"ANDA"``, ``"BLA"``), or
        empty string if unrecognized.
    """
    upper = application_number.upper()
    for prefix in ("NDA", "ANDA", "BLA"):
        if upper.startswith(prefix):
            return prefix
    return ""


def _parse_fda_date(raw_date: str) -> str:
    """Parse FDA date format (YYYYMMDD) to ISO format (YYYY-MM-DD).

    Args:
        raw_date: Date string in ``YYYYMMDD`` format.

    Returns:
        ISO date string ``YYYY-MM-DD``, or empty string if parsing fails.
    """
    if not raw_date or len(raw_date) < 8:
        return ""

    try:
        year = raw_date[:4]
        month = raw_date[4:6]
        day = raw_date[6:8]
        # Basic validation
        int(year)
        int(month)
        int(day)
        return f"{year}-{month}-{day}"
    except (ValueError, IndexError):
        logger.warning("Failed to parse FDA date: %s", raw_date)
        return ""
