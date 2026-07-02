"""AACT (Aggregate Analysis of ClinicalTrials.gov) data loader.

AACT provides the full ClinicalTrials.gov database as pipe-delimited CSV
files.  This loader handles:
    - Pre-processed Parquet loading
    - Raw pipe-delimited CSV ingestion (studies, sponsors, interventions,
      conditions, calculated_values) with joins on ``nct_id``
    - Sponsor-level and condition-level aggregate statistics for feature
      engineering by the AutoCT agent

The AACT download is available at https://aact.ctti-clinicaltrials.org/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ctra.config.settings import DataConfig, get_settings

logger = logging.getLogger(__name__)

# Columns to read from each AACT table (avoids loading all 70+ columns)
_STUDIES_COLS = [
    "nct_id",
    "overall_status",
    "phase",
    "enrollment",
    "start_date",
    "completion_date",
    "brief_title",
    "official_title",
    "why_stopped",
    "source",
]

_SPONSORS_COLS = [
    "id",
    "nct_id",
    "agency_class",
    "lead_or_collaborator",
    "name",
]

_INTERVENTIONS_COLS = [
    "id",
    "nct_id",
    "intervention_type",
    "name",
    "description",
]

_CONDITIONS_COLS = [
    "id",
    "nct_id",
    "name",
    "downcase_name",
]

_CALCULATED_VALUES_COLS = [
    "id",
    "nct_id",
    "number_of_facilities",
    "were_results_reported",
    "actual_duration",
]


class AACTLoader:
    """Load and query AACT ClinicalTrials.gov data.

    The Aggregate Analysis of ClinicalTrials.gov (AACT) database provides
    the full CTG dataset as CSV tables. This loader reads pre-processed
    Parquet files or raw CSVs, handles joins on nct_id, and computes
    sponsor/condition aggregate statistics for feature engineering.

    Args:
        config: Data configuration. If ``None``, uses global settings.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initialize the AACT loader.

        Args:
            config: Optional DataConfig to override global settings.
        """
        self._config = config or get_settings().data
        self._df: pl.DataFrame | None = None

    def load_parquet(self, path: Path | None = None) -> pl.DataFrame:
        """Load pre-processed AACT data from Parquet.

        Expected columns: nct_id, overall_status, phase, enrollment,
        start_date, completion_date, brief_title, official_title,
        why_stopped, source, sponsor_name, agency_class, intervention_type,
        intervention_name, condition_name, number_of_facilities,
        were_results_reported, actual_duration.
        """
        parquet_path = Path(path or self._config.aact_parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"AACT parquet not found at {parquet_path}.  "
                "Run `python scripts/data/fetch_aact.py` first."
            )

        logger.info("Loading AACT data from %s", parquet_path)
        self._df = pl.read_parquet(parquet_path)
        logger.info("Loaded %d AACT study records", len(self._df))
        return self._df

    def load_from_csv(self, csv_dir: Path | None = None) -> pl.DataFrame:
        """Read pipe-delimited AACT CSV files and join into a single DataFrame.

        Reads ``studies.txt``, ``sponsors.txt``, ``interventions.txt``,
        ``conditions.txt``, and ``calculated_values.txt`` from *csv_dir*,
        joins them on ``nct_id``, and returns a unified DataFrame.

        Args:
            csv_dir: Directory containing the AACT pipe-delimited text files.
                If ``None``, uses ``config.aact_csv_dir``.

        Returns:
            DataFrame with merged AACT data.
        """
        csv_path = Path(csv_dir or self._config.aact_csv_dir)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"AACT CSV directory not found at {csv_path}.  "
                "Download from https://aact.ctti-clinicaltrials.org/."
            )

        logger.info("Reading AACT CSVs from %s", csv_path)

        # --- studies.txt ---
        studies_path = csv_path / "studies.txt"
        logger.info("Reading %s", studies_path)
        studies = pl.read_csv(
            studies_path,
            separator="|",
            null_values=[""],
            try_parse_dates=True,
            columns=_STUDIES_COLS,
        )
        logger.info("  studies: %d rows", len(studies))

        # --- sponsors.txt (lead sponsors only) ---
        sponsors_path = csv_path / "sponsors.txt"
        logger.info("Reading %s", sponsors_path)
        sponsors = pl.read_csv(
            sponsors_path,
            separator="|",
            null_values=[""],
            try_parse_dates=True,
            columns=_SPONSORS_COLS,
        )
        sponsors = sponsors.filter(pl.col("lead_or_collaborator") == "lead")
        sponsors = sponsors.rename({"name": "sponsor_name"}).drop(["id", "lead_or_collaborator"])
        logger.info("  sponsors (lead): %d rows", len(sponsors))

        # --- interventions.txt ---
        interventions_path = csv_path / "interventions.txt"
        logger.info("Reading %s", interventions_path)
        interventions = pl.read_csv(
            interventions_path,
            separator="|",
            null_values=[""],
            try_parse_dates=True,
            columns=_INTERVENTIONS_COLS,
        )
        interventions = interventions.rename(
            {
                "name": "intervention_name",
                "description": "intervention_description",
            }
        ).drop("id")
        logger.info("  interventions: %d rows", len(interventions))

        # --- conditions.txt ---
        conditions_path = csv_path / "conditions.txt"
        logger.info("Reading %s", conditions_path)
        conditions = pl.read_csv(
            conditions_path,
            separator="|",
            null_values=[""],
            try_parse_dates=True,
            columns=_CONDITIONS_COLS,
        )
        conditions = conditions.rename({"name": "condition_name"}).drop(["id", "downcase_name"])
        logger.info("  conditions: %d rows", len(conditions))

        # --- calculated_values.txt ---
        calc_path = csv_path / "calculated_values.txt"
        logger.info("Reading %s", calc_path)
        calculated = pl.read_csv(
            calc_path,
            separator="|",
            null_values=[""],
            try_parse_dates=True,
            columns=_CALCULATED_VALUES_COLS,
        )
        calculated = calculated.drop("id")
        logger.info("  calculated_values: %d rows", len(calculated))

        # --- Join all tables on nct_id ---
        logger.info("Joining AACT tables on nct_id")
        df = studies.join(sponsors, on="nct_id", how="left")
        df = df.join(interventions, on="nct_id", how="left")
        df = df.join(conditions, on="nct_id", how="left")
        df = df.join(calculated, on="nct_id", how="left")

        logger.info("Merged AACT DataFrame: %d rows, %d columns", len(df), len(df.columns))
        self._df = df
        return self._df

    def compute_sponsor_stats(self) -> pl.DataFrame:
        """Compute per-sponsor trial statistics for feature engineering.

        Groups by lead sponsor and computes:
            - ``total_trials``: total number of trials
            - ``phase1_count``, ``phase2_count``, ``phase3_count``
            - ``completed_count``: trials with overall_status == "Completed"
            - ``terminated_count``: trials with status "Terminated" or "Withdrawn"
            - ``success_rate``: completed / (completed + terminated + withdrawn)

        Returns:
            DataFrame with one row per sponsor.
        """
        if self._df is None:
            self.load_parquet()
        assert self._df is not None

        # Deduplicate to study level (joins may have created duplicates)
        studies = self._df.select("nct_id", "overall_status", "phase", "sponsor_name").unique(
            subset=["nct_id"]
        )

        stats = studies.group_by("sponsor_name").agg(
            pl.col("nct_id").count().alias("total_trials"),
            (pl.col("phase").str.contains("(?i)phase 1").sum()).alias("phase1_count"),
            (pl.col("phase").str.contains("(?i)phase 2").sum()).alias("phase2_count"),
            (pl.col("phase").str.contains("(?i)phase 3").sum()).alias("phase3_count"),
            (pl.col("overall_status") == "Completed").sum().alias("completed_count"),
            (pl.col("overall_status").is_in(["Terminated", "Withdrawn"]))
            .sum()
            .alias("terminated_count"),
        )

        stats = stats.with_columns(
            (
                pl.col("completed_count") / (pl.col("completed_count") + pl.col("terminated_count"))
            ).alias("success_rate")
        )

        logger.info("Computed sponsor stats for %d sponsors", len(stats))
        return stats

    def compute_condition_stats(self) -> pl.DataFrame:
        """Compute per-condition trial statistics for feature engineering.

        Groups by condition name and computes:
            - ``trial_count``: number of trials for this condition
            - ``success_rate``: completed / (completed + terminated + withdrawn)
            - ``avg_enrollment``: mean enrollment across trials

        Returns:
            DataFrame with one row per condition.
        """
        if self._df is None:
            self.load_parquet()
        assert self._df is not None

        # Deduplicate to study-condition level
        studies = self._df.select(
            "nct_id", "overall_status", "enrollment", "condition_name"
        ).unique(subset=["nct_id", "condition_name"])

        stats = studies.group_by("condition_name").agg(
            pl.col("nct_id").count().alias("trial_count"),
            (pl.col("overall_status") == "Completed").sum().alias("completed_count"),
            (pl.col("overall_status").is_in(["Terminated", "Withdrawn"]))
            .sum()
            .alias("terminated_count"),
            pl.col("enrollment").mean().alias("avg_enrollment"),
        )

        stats = stats.with_columns(
            (
                pl.col("completed_count") / (pl.col("completed_count") + pl.col("terminated_count"))
            ).alias("success_rate")
        )

        stats = stats.drop(["completed_count", "terminated_count"])

        logger.info("Computed condition stats for %d conditions", len(stats))
        return stats
