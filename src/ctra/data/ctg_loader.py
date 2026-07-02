"""ClinicalTrials.gov data loading and parsing.

Handles both:
    - Local Parquet files (pre-downloaded via fetch_ctg.py)
    - Live API v2 queries for single-trial lookup
    - Polars-based queries for efficient trial selection
    - Protocol flattening into prose sentences for LinearRAG indexing
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import polars as pl

from ctra.config.settings import DataConfig, get_settings

logger = logging.getLogger(__name__)


class CTGLoader:
    """Load and query ClinicalTrials.gov trial records.

    Supports loading pre-cached trials from Parquet (via fetch_ctg.py) and
    live API v2 queries.  Can flatten protocol sections into prose sentences
    for LinearRAG indexing.

    Args:
        config: Data configuration. If ``None``, uses global settings.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initialize the CTG loader.

        Args:
            config: Optional DataConfig to override global settings.
        """
        self._config = config or get_settings().data
        self._df: pl.DataFrame | None = None

    def load_parquet(self, path: Path | None = None) -> pl.DataFrame:
        """Load trials from a local Parquet file.

        Args:
            path: Override path to the Parquet file.

        Returns:
            DataFrame of trial records.
        """
        parquet_path = Path(path or self._config.ctg_parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"CTG parquet not found at {parquet_path}.  "
                "Run `python scripts/data/fetch_ctg.py` first."
            )

        logger.info("Loading CTG data from %s", parquet_path)
        self._df = pl.read_parquet(parquet_path)
        logger.info("Loaded %d trials", len(self._df))
        return self._df

    def get_trial(self, nct_id: str) -> dict[str, Any] | None:
        """Look up a single trial by NCT ID.

        First checks the loaded Parquet.  Falls back to the CTG API v2.
        """
        # Local lookup
        if self._df is not None:
            id_col = "nct_id" if "nct_id" in self._df.columns else "nctId"
            if id_col in self._df.columns:
                match = self._df.filter(pl.col(id_col) == nct_id)
                if len(match) > 0:
                    return match.row(0, named=True)

        # API fallback
        return self._fetch_from_api(nct_id)

    def _fetch_from_api(self, nct_id: str) -> dict[str, Any] | None:
        """Fetch a single trial from the ClinicalTrials.gov API v2.

        Returns only the ``protocolSection`` to prevent label leakage
        from ``resultsSection``.
        """
        import httpx

        url = f"{self._config.ctg_api_base}/studies/{nct_id}"
        params = {"fields": "protocolSection"}

        try:
            response = httpx.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data.get("protocolSection", data)  # type: ignore[no-any-return]
        except httpx.HTTPError:
            logger.warning("Failed to fetch %s from CTG API", nct_id)
            return None

    def load_benchmark_splits(
        self,
        tasks_dir: Path | None = None,
        task: str = "trial_approval",
        phase: int = 2,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load train/val/test splits for a benchmark task.

        Args:
            tasks_dir: Directory containing task splits.
            task: Task name (e.g., "trial_approval").
            phase: Trial phase (1, 2, or 3).

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        base = Path(tasks_dir or self._config.tasks_dir) / task
        prefix = f"phase{phase}_"

        train = pl.read_parquet(base / f"{prefix}train_data.parquet")
        val = pl.read_parquet(base / f"{prefix}val_data.parquet")
        test = pl.read_parquet(base / f"{prefix}test_data.parquet")

        logger.info(
            "Loaded %s phase %d splits: train=%d, val=%d, test=%d",
            task,
            phase,
            len(train),
            len(val),
            len(test),
        )
        return train, val, test


# ---------------------------------------------------------------------------
# Polars-based loading functions
# ---------------------------------------------------------------------------


def get_trial_info(nctid: str, parquet_path: str | None = None) -> dict[str, Any]:
    """Get a single trial's protocolSection as a dict.

    Uses polars for efficient single-row lookup.

    Args:
        nctid: NCT ID of the trial (e.g., ``NCT00110279``).
        parquet_path: Path to the CTG parquet file.  Defaults to settings.

    Returns:
        Dict of trial fields.  Includes ``startDate`` derived from
        ``statusModule.startDateStruct.date`` for use as the temporal cutoff.

    Raises:
        KeyError: If the trial is not found.
    """
    settings = get_settings()
    path = parquet_path or str(settings.data.ctg_parquet)

    if not Path(path).exists():
        raise FileNotFoundError(f"CTG parquet not found at {path}")

    df = pl.read_parquet(path)

    # Try to find the trial using either column name convention
    if "nctId" in df.columns:
        result = df.filter(pl.col("nctId") == nctid)
    elif "nct_id" in df.columns:
        result = df.filter(pl.col("nct_id") == nctid)
    else:
        raise KeyError(f"No NCT ID column found in {path}")

    if len(result) == 0:
        raise KeyError(f"Trial {nctid} not found in {path}")

    row = result.row(0, named=True)
    record = {k: v for k, v in row.items() if v is not None}

    # Ensure startDate is present for the AutoCT tool interface.
    # Prefer statusModule.startDateStruct.date, fall back to start_date column.
    if "startDate" not in record:
        for col in ("start_date", "startDateStruct_date"):
            if col in record:
                record["startDate"] = str(record[col])
                break

    return record


def flatten_protocol_to_sentences(protocol: dict[str, Any]) -> list[str]:
    """Flatten a protocolSection dict to prose sentences.

    This is a general-purpose utility for converting CTG protocolSection
    JSON into readable text.  It is **not** used by the current LinearRAG
    indexer, which uses ``_ctg_to_passages()`` in ``indexer.py`` for
    passage-level indexing.  This function remains available for ad-hoc
    inspection, debugging, and downstream consumers that need sentence-
    level protocol text.

    Extracts text from all relevant modules in the CTG protocolSection
    and converts structured fields into readable sentences.

    The ``resultsSection`` is never processed.

    Args:
        protocol: A CTG API v2 ``protocolSection`` dict.

    Returns:
        List of sentence strings.
    """
    sentences: list[str] = []

    def _get(d: dict[str, Any], *keys: str, default: str = "") -> Any:
        """Nested dict access with fallback."""
        current: Any = d
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, {})
            else:
                return default
        return current if current != {} else default

    # -- Identification --
    id_mod = protocol.get("identificationModule", {})
    brief_title = id_mod.get("briefTitle", "")
    if brief_title:
        sentences.append(brief_title)

    official_title = id_mod.get("officialTitle", "")
    if official_title and official_title != brief_title:
        sentences.append(official_title)

    # -- Description --
    desc_mod = protocol.get("descriptionModule", {})
    brief_summary = desc_mod.get("briefSummary", "")
    if brief_summary:
        sentences.extend(_split_text(brief_summary))

    detailed = desc_mod.get("detailedDescription", "")
    if detailed:
        sentences.extend(_split_text(detailed))

    # -- Design --
    design_mod = protocol.get("designModule", {})
    study_type = design_mod.get("studyType", "")
    phases = design_mod.get("phases", [])
    design_info = design_mod.get("designInfo", {})

    if study_type or phases:
        parts = []
        if study_type:
            parts.append(f"Study type: {study_type}")
        if phases:
            parts.append(f"Phase: {', '.join(phases)}")
        allocation = design_info.get("allocation", "")
        if allocation:
            parts.append(f"Allocation: {allocation}")
        masking = _get(design_info, "maskingInfo", "masking")
        if masking:
            parts.append(f"Masking: {masking}")
        purpose = design_info.get("primaryPurpose", "")
        if purpose:
            parts.append(f"Purpose: {purpose}")
        sentences.append(". ".join(parts) + ".")

    # -- Enrollment --
    enrollment = design_mod.get("enrollmentInfo", {})
    if enrollment:
        count = enrollment.get("count", "")
        etype = enrollment.get("type", "")
        if count:
            sentences.append(f"Enrollment: {count} ({etype}).")

    # -- Conditions --
    cond_mod = protocol.get("conditionsModule", {})
    conditions = cond_mod.get("conditions", [])
    if conditions:
        sentences.append(f"Conditions: {', '.join(conditions)}.")

    keywords = cond_mod.get("keywords", [])
    if keywords:
        sentences.append(f"Keywords: {', '.join(keywords)}.")

    # -- Interventions --
    arms_mod = protocol.get("armsInterventionsModule", {})
    for iv in arms_mod.get("interventions", []):
        iv_type = iv.get("type", "")
        iv_name = iv.get("name", "")
        iv_desc = iv.get("description", "")
        text = f"Intervention ({iv_type}): {iv_name}."
        if iv_desc:
            text += f" {iv_desc}"
        sentences.append(text)

    # -- Eligibility --
    elig_mod = protocol.get("eligibilityModule", {})
    criteria = elig_mod.get("eligibilityCriteria", "")
    if criteria:
        sentences.extend(_split_text(criteria))

    min_age = elig_mod.get("minimumAge", "")
    max_age = elig_mod.get("maximumAge", "")
    sex = elig_mod.get("sex", "")
    if min_age or max_age or sex:
        parts = []
        if min_age:
            parts.append(f"Minimum age: {min_age}")
        if max_age:
            parts.append(f"Maximum age: {max_age}")
        if sex and sex != "ALL":
            parts.append(f"Sex: {sex}")
        sentences.append(". ".join(parts) + ".")

    # -- Outcomes --
    outcomes_mod = protocol.get("outcomesModule", {})
    for outcome in outcomes_mod.get("primaryOutcomes", []):
        measure = outcome.get("measure", "")
        if measure:
            sentences.append(f"Primary outcome: {measure}.")

    for outcome in outcomes_mod.get("secondaryOutcomes", []):
        measure = outcome.get("measure", "")
        if measure:
            sentences.append(f"Secondary outcome: {measure}.")

    # -- Sponsor --
    sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})
    lead = sponsor_mod.get("leadSponsor", {})
    if lead:
        name = lead.get("name", "")
        if name:
            sentences.append(f"Lead sponsor: {name}.")

    return sentences


def _split_text(text: str, min_length: int = 10) -> list[str]:
    """Split text on sentence boundaries or newlines."""
    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= min_length]
