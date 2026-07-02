"""Unified data ingestion service for CTRA.

Orchestrates incremental (and full) updates across all seven data sources
(ClinicalTrials.gov, PubMed, ChEMBL, FAERS/OpenFDA, AACT, PrimeKG,
Drugs@FDA) and triggers LinearRAG index rebuilds when data changes.

Design:
    - All HTTP I/O uses ``httpx.AsyncClient`` (non-blocking).
    - Rate limiting per source via ``asyncio.Semaphore`` + per-request sleep.
    - Persistent ``last_updated`` timestamps stored in a JSON state file so
      incremental runs only fetch new/modified records.
    - Each source writes to its own Parquet file; ``rebuild_indexes()`` then
      delegates to :class:`~ctra.rag.indexer.IndexBuilder`.
    - Transient network errors are retried via ``tenacity``.

Usage::

    from ctra.data.ingestion import IngestionService

    svc = IngestionService()
    report = await svc.update_all(incremental=True)
    print(report)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ctra.config.settings import DataConfig, DataSource, IngestionConfig, get_settings
from ctra.data.ctg_api import CTGApiClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SourceReport:
    """Report for a single data source update.

    Attributes:
        source: Name of the data source (e.g., "ctg", "pubmed").
        records_added: Count of new records ingested.
        records_updated: Count of existing records updated with new data.
        duration_seconds: Wall-clock time spent on this source.
        errors: List of error messages (empty if successful).
    """

    source: str
    records_added: int = 0
    records_updated: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True if the update completed without errors."""
        return len(self.errors) == 0


@dataclass
class IndexReport:
    """Report for LinearRAG index rebuilds.

    Attributes:
        sources_rebuilt: Names of sources that were re-indexed.
        duration_seconds: Wall-clock time spent rebuilding indexes.
        errors: List of error messages (empty if successful).
    """

    sources_rebuilt: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class IngestionReport:
    """Aggregate report for a full ingestion run.

    Attributes:
        sources: Per-source reports keyed by source name.
        index_report: LinearRAG index rebuild report (None if not rebuilt).
        total_duration_seconds: Total elapsed time for the entire ingestion.
    """

    sources: dict[str, SourceReport] = field(default_factory=dict)
    index_report: IndexReport | None = None
    total_duration_seconds: float = 0.0

    @property
    def index_rebuilt(self) -> bool:
        """Return True if the LinearRAG indexes were successfully rebuilt."""
        return self.index_report is not None and len(self.index_report.errors) == 0


@dataclass
class SourceStatus:
    """Status snapshot for a single data source.

    Attributes:
        source: Data source name.
        last_updated: ISO date of the last successful ingestion, or None.
        record_count: Number of records in the cached Parquet file.
        parquet_path: Path to the Parquet file.
        parquet_exists: Whether the Parquet file exists on disk.
    """

    source: str
    last_updated: str | None
    record_count: int
    parquet_path: str
    parquet_exists: bool


# ---------------------------------------------------------------------------
# Ingestion state persistence
# ---------------------------------------------------------------------------


class _IngestionState:
    """Reads/writes the JSON state file that tracks per-source timestamps.

    Used by IngestionService to store and retrieve the last-updated timestamp
    for each data source, enabling incremental fetches.
    """

    def __init__(self, path: Path) -> None:
        """Initialize the state manager with a file path.

        Args:
            path: Path to the JSON state file (created if missing).
        """
        self._path = path

    def load(self) -> dict[str, Any]:
        """Load the current state from the JSON file.

        Returns:
            State dict mapping source names to metadata dicts, or empty dict
            if the file does not exist.
        """
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return {}

    def save(self, state: dict[str, Any]) -> None:
        """Write the state dict to the JSON file.

        Creates parent directories as needed.

        Args:
            state: State dict to persist.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def get_last_updated(self, source: str) -> str | None:
        """Retrieve the last-updated timestamp for a single data source.

        Args:
            source: Data source name (e.g., "ctg", "pubmed").

        Returns:
            ISO date string (e.g., "2025-04-05"), or None if not found.
        """
        state = self.load()
        return state.get(source, {}).get("last_updated")  # type: ignore[no-any-return]

    def set_last_updated(self, source: str, ts: str) -> None:
        """Update the last-updated timestamp for a single data source.

        Args:
            source: Data source name.
            ts: ISO date string to store.
        """
        state = self.load()
        if source not in state:
            state[source] = {}
        state[source]["last_updated"] = ts
        self.save(state)


# ---------------------------------------------------------------------------
# Retry decorator for HTTP calls
# ---------------------------------------------------------------------------

_http_retry = retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)


# ---------------------------------------------------------------------------
# IngestionService
# ---------------------------------------------------------------------------


class IngestionService:
    """Orchestrates data source updates and LinearRAG re-indexing.

    Coordinates concurrent fetches from all seven data sources (ClinicalTrials.gov,
    PubMed, ChEMBL, FAERS/OpenFDA, AACT, PrimeKG, Drugs@FDA), handles incremental
    updates via persistent state, and triggers LinearRAG index rebuilds when data
    changes.  All HTTP I/O is non-blocking (httpx.AsyncClient).

    Args:
        config: Data configuration override (uses global settings if None).
        ingestion_config: Ingestion-specific settings override.
    """

    def __init__(
        self,
        config: DataConfig | None = None,
        ingestion_config: IngestionConfig | None = None,
    ) -> None:
        """Initialize the ingestion service.

        Args:
            config: Optional DataConfig to override global settings.
            ingestion_config: Optional IngestionConfig to override global settings.
        """
        settings = get_settings()
        self._data_cfg = config or settings.data
        self._ing_cfg = ingestion_config or settings.ingestion
        self._state = _IngestionState(self._ing_cfg.state_file)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update_all(
        self,
        incremental: bool = True,
        rebuild: bool = True,
    ) -> IngestionReport:
        """Update all data sources and optionally rebuild indexes.

        Args:
            incremental: If ``True``, only fetch records newer than the
                last successful update.  If ``False``, re-download everything.
            rebuild: Whether to rebuild LinearRAG indexes after updating.

        Returns:
            Aggregate ingestion report.
        """
        t0 = time.monotonic()
        report = IngestionReport()

        # Run all source updates concurrently.
        results = await asyncio.gather(
            self.update_ctg(incremental=incremental),
            self.update_pubmed(incremental=incremental),
            self.update_faers(incremental=incremental),
            self.update_aact(incremental=incremental),
            self.update_chembl(),
            self.update_primekg(),
            self.update_drugsfda(incremental=incremental),
            return_exceptions=True,
        )

        source_names = [
            "ctg",
            "pubmed",
            "faers",
            "aact",
            "chembl",
            "primekg",
            "drugsfda",
        ]
        for name, result in zip(source_names, results, strict=True):
            if isinstance(result, BaseException):
                report.sources[name] = SourceReport(
                    source=name,
                    errors=[f"Unhandled exception: {result!r}"],
                )
            else:
                report.sources[name] = result

        # Rebuild indexes if any source got new data.
        if rebuild:
            updated_sources = [
                name
                for name, sr in report.sources.items()
                if sr.ok and (sr.records_added > 0 or sr.records_updated > 0)
            ]
            if updated_sources:
                report.index_report = await self.rebuild_indexes(sources=updated_sources)
            else:
                logger.info("No sources updated; skipping index rebuild")

        report.total_duration_seconds = time.monotonic() - t0
        logger.info("Ingestion completed in %.1fs", report.total_duration_seconds)
        return report

    async def update_ctg(self, incremental: bool = True) -> SourceReport:
        """Fetch new/updated trials from ClinicalTrials.gov API v2.

        Incremental mode uses ``AREA[LastUpdatePostDate]`` to fetch only
        trials modified since the last ingestion run.

        Args:
            incremental: Fetch only modified trials if ``True``.

        Returns:
            Source-level ingestion report.
        """
        t0 = time.monotonic()
        report = SourceReport(source="ctg")
        parquet_path = Path(self._data_cfg.ctg_parquet)

        try:
            client = CTGApiClient(
                base_url=self._data_cfg.ctg_api_base + "/studies",
                rate_limit=self._ing_cfg.ctg_rate_limit,
            )

            existing_df: pl.DataFrame | None = None
            if parquet_path.exists():
                existing_df = pl.read_parquet(parquet_path)

            since_date = None
            if incremental:
                since_date = self._state.get_last_updated("ctg")

            new_studies: list[dict[str, Any]] = []
            if since_date:
                logger.info("CTG incremental update since %s", since_date)
                async for study in client.get_updated_since(
                    since_date=since_date,
                    page_size=self._ing_cfg.ctg_page_size,
                ):
                    flat = _flatten_ctg_study(study)
                    if flat:
                        new_studies.append(flat)
            else:
                logger.info("CTG full download")
                async for study in client.search(
                    page_size=self._ing_cfg.ctg_page_size,
                ):
                    flat = _flatten_ctg_study(study)
                    if flat:
                        new_studies.append(flat)

            if new_studies:
                new_df = pl.DataFrame(new_studies)

                if existing_df is not None and len(existing_df) > 0:
                    # Determine the NCT ID column name in existing data.
                    nct_col = "nctId" if "nctId" in existing_df.columns else "nct_id"
                    new_nct_col = "nctId" if "nctId" in new_df.columns else "nct_id"

                    # Separate new records from updated records.
                    existing_ids = set(existing_df.get_column(nct_col).to_list())
                    incoming_ids = set(new_df.get_column(new_nct_col).to_list())
                    updated_ids = existing_ids & incoming_ids
                    added_ids = incoming_ids - existing_ids

                    report.records_added = len(added_ids)
                    report.records_updated = len(updated_ids)

                    # Remove updated rows from existing, then concat.
                    if updated_ids:
                        existing_df = existing_df.filter(~pl.col(nct_col).is_in(list(updated_ids)))
                    merged = pl.concat([existing_df, new_df])
                else:
                    merged = new_df
                    report.records_added = len(new_df)

                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                merged.write_parquet(parquet_path)
                logger.info(
                    "CTG: %d added, %d updated -> %d total in %s",
                    report.records_added,
                    report.records_updated,
                    len(merged),
                    parquet_path,
                )
            else:
                logger.info("CTG: no new studies found")

            # Update state timestamp.
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._state.set_last_updated("ctg", now_str)

        except Exception as exc:
            logger.error("CTG update failed: %s", exc, exc_info=True)
            report.errors.append(str(exc))

        report.duration_seconds = time.monotonic() - t0
        return report

    async def update_pubmed(self, incremental: bool = True) -> SourceReport:
        """Fetch new PubMed abstracts via Entrez eSearch + eFetch.

        Incremental mode uses ``mindate`` to restrict the search to articles
        published or indexed since the last run.

        Args:
            incremental: Restrict to recent articles if ``True``.

        Returns:
            Source-level ingestion report.
        """
        t0 = time.monotonic()
        report = SourceReport(source="pubmed")
        parquet_path = Path(self._data_cfg.pubmed_parquet)

        try:
            api_key = os.environ.get(self._data_cfg.pubmed_api_key_env_var, "")
            base_url = self._data_cfg.pubmed_api_base

            # Determine date range for incremental mode.
            min_date: str | None = None
            if incremental:
                min_date = self._state.get_last_updated("pubmed")

            # Step 1: eSearch to find PMIDs.
            pmids = await self._pubmed_esearch(
                base_url=base_url,
                api_key=api_key,
                min_date=min_date,
            )

            if not pmids:
                logger.info("PubMed: no new articles found")
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                self._state.set_last_updated("pubmed", now_str)
                report.duration_seconds = time.monotonic() - t0
                return report

            logger.info("PubMed: fetching %d articles", len(pmids))

            # Step 2: eFetch in batches.
            articles = await self._pubmed_efetch(
                base_url=base_url,
                api_key=api_key,
                pmids=pmids,
                batch_size=self._ing_cfg.pubmed_batch_size,
            )

            if articles:
                new_df = pl.DataFrame(articles)

                existing_df_pm: pl.DataFrame | None = None
                if parquet_path.exists():
                    existing_df_pm = pl.read_parquet(parquet_path)

                if existing_df_pm is not None and len(existing_df_pm) > 0:
                    existing_pmids = set(existing_df_pm.get_column("pmid").to_list())
                    incoming_pmids = set(new_df.get_column("pmid").to_list())
                    report.records_added = len(incoming_pmids - existing_pmids)
                    report.records_updated = len(incoming_pmids & existing_pmids)

                    # Remove updated rows, then concat.
                    if incoming_pmids & existing_pmids:
                        existing_df_pm = existing_df_pm.filter(
                            ~pl.col("pmid").is_in(list(incoming_pmids))
                        )
                    merged = pl.concat([existing_df_pm, new_df])
                else:
                    merged = new_df
                    report.records_added = len(new_df)

                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                merged.write_parquet(parquet_path)
                logger.info(
                    "PubMed: %d added, %d updated -> %d total",
                    report.records_added,
                    report.records_updated,
                    len(merged),
                )

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._state.set_last_updated("pubmed", now_str)

        except Exception as exc:
            logger.error("PubMed update failed: %s", exc, exc_info=True)
            report.errors.append(str(exc))

        report.duration_seconds = time.monotonic() - t0
        return report

    async def update_faers(self, incremental: bool = True) -> SourceReport:
        """Fetch FAERS adverse event data from OpenFDA.

        Queries the OpenFDA drug/event endpoint for the drugs present in the
        ChEMBL parquet.  In incremental mode, only drugs that have not been
        queried since the last update are re-fetched.

        Args:
            incremental: If ``True``, only re-query drugs with stale data.

        Returns:
            Source-level ingestion report.
        """
        t0 = time.monotonic()
        report = SourceReport(source="faers")
        parquet_path = Path(self._data_cfg.faers_parquet)

        try:
            # Get the list of drug names to query from ChEMBL.
            chembl_path = Path(self._data_cfg.chembl_parquet)
            if not chembl_path.exists():
                msg = f"ChEMBL parquet not found at {chembl_path}. Run ChEMBL ingestion first."
                logger.warning(msg)
                report.errors.append(msg)
                report.duration_seconds = time.monotonic() - t0
                return report

            chembl_df = pl.read_parquet(chembl_path, columns=["pref_name"])
            drug_names = chembl_df.get_column("pref_name").drop_nulls().unique().to_list()

            # In incremental mode, load existing data and skip already-cached drugs.
            existing_drugs: set[str] = set()
            existing_df_faers: pl.DataFrame | None = None
            if incremental and parquet_path.exists():
                existing_df_faers = pl.read_parquet(parquet_path)
                if "drug_name" in existing_df_faers.columns:
                    existing_drugs = set(
                        existing_df_faers.get_column("drug_name").unique().to_list()
                    )

                last_updated = self._state.get_last_updated("faers")
                if last_updated:
                    # Only re-query drugs we haven't seen before.
                    drug_names = [d for d in drug_names if d not in existing_drugs]

            if not drug_names:
                logger.info("FAERS: all drugs already cached; skipping")
                report.duration_seconds = time.monotonic() - t0
                return report

            logger.info("FAERS: querying %d drugs via OpenFDA", len(drug_names))

            # Fetch adverse events for each drug via OpenFDA.
            all_events: list[dict[str, Any]] = []
            api_key = os.environ.get(self._data_cfg.openfda_api_key_env_var, "")
            rate_limit = self._ing_cfg.faers_rate_limit
            semaphore = asyncio.Semaphore(max(1, int(rate_limit)))

            async with httpx.AsyncClient() as client:
                for i, drug in enumerate(drug_names):
                    async with semaphore:
                        events = await self._fetch_openfda_events(
                            client=client,
                            drug_name=drug,
                            api_key=api_key,
                        )
                        all_events.extend(events)
                        if rate_limit > 0:
                            await asyncio.sleep(1.0 / rate_limit)

                    if (i + 1) % 50 == 0:
                        logger.info(
                            "FAERS progress: %d/%d drugs queried",
                            i + 1,
                            len(drug_names),
                        )

            if all_events:
                new_df = pl.DataFrame(all_events)
                report.records_added = len(new_df)

                if existing_df_faers is not None and len(existing_df_faers) > 0:
                    merged = pl.concat([existing_df_faers, new_df])
                else:
                    merged = new_df

                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                merged.write_parquet(parquet_path)
                logger.info(
                    "FAERS: %d new event records -> %d total",
                    report.records_added,
                    len(merged),
                )
            else:
                logger.info("FAERS: no events returned from OpenFDA")

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._state.set_last_updated("faers", now_str)

        except Exception as exc:
            logger.error("FAERS update failed: %s", exc, exc_info=True)
            report.errors.append(str(exc))

        report.duration_seconds = time.monotonic() - t0
        return report

    async def update_aact(self, incremental: bool = True) -> SourceReport:
        """Update AACT data from CSV snapshots.

        AACT CSVs must be downloaded manually from
        https://aact.ctti-clinicaltrials.org/pipe_files and placed in
        the configured ``aact_csv_dir``.  This method parses the CSVs
        and writes the aggregated parquet.
        """
        t0 = time.monotonic()
        report = SourceReport(source="aact")

        try:
            from ctra.data.aact_loader import AACTLoader

            loader = AACTLoader(config=self._data_cfg)
            csv_dir = Path(self._data_cfg.aact_csv_dir)

            if not csv_dir.exists():
                report.errors.append(
                    f"AACT CSV directory not found at {csv_dir}.  "
                    "Download from https://aact.ctti-clinicaltrials.org/pipe_files"
                )
                report.duration_seconds = time.monotonic() - t0
                return report

            df = loader.load_from_csv(csv_dir)

            logger.info("AACT: processed %d records", len(df))

        except Exception as exc:
            logger.exception("AACT update failed")
            report.errors.append(str(exc))

        report.duration_seconds = time.monotonic() - t0
        return report

    async def update_chembl(self) -> SourceReport:
        """Update ChEMBL data from SQLite database.

        The ChEMBL SQLite file must be downloaded manually from
        https://www.ebi.ac.uk/chembl/ and placed at the configured
        ``chembl_sqlite_path``.  This method extracts targeted tables
        and writes the parquet.
        """
        t0 = time.monotonic()
        report = SourceReport(source="chembl")

        try:
            from ctra.data.chembl_loader import ChEMBLLoader

            loader = ChEMBLLoader(config=self._data_cfg)
            sqlite_path = Path(self._data_cfg.chembl_sqlite_path)

            if not sqlite_path.exists():
                report.errors.append(
                    f"ChEMBL SQLite not found at {sqlite_path}.  "
                    "Download from https://www.ebi.ac.uk/chembl/"
                )
                report.duration_seconds = time.monotonic() - t0
                return report

            df = loader.extract_from_sqlite(sqlite_path)

            logger.info("ChEMBL: extracted %d drug records", len(df))

        except Exception as exc:
            logger.exception("ChEMBL update failed")
            report.errors.append(str(exc))

        report.duration_seconds = time.monotonic() - t0
        return report

    async def update_primekg(self) -> SourceReport:
        """Update PrimeKG knowledge graph data.

        Downloads PrimeKG CSV from Harvard Dataverse (or PyTDC) and
        writes to parquet.
        """
        t0 = time.monotonic()
        report = SourceReport(source="primekg")

        try:
            from ctra.data.primekg_loader import PrimeKGLoader

            loader = PrimeKGLoader(config=self._data_cfg)

            # Try PyTDC first, fall back to CSV
            try:
                df = loader.load_from_tdc()
            except (ImportError, Exception):
                csv_path = Path(self._data_cfg.primekg_parquet).with_suffix(".csv")
                if csv_path.exists():
                    df = loader.load_from_csv(csv_path)
                else:
                    report.errors.append(
                        "PrimeKG: PyTDC not installed and CSV not found.  "
                        "Install PyTDC (`pip install PyTDC`) or download kg.csv "
                        "from Harvard Dataverse."
                    )
                    report.duration_seconds = time.monotonic() - t0
                    return report

            logger.info("PrimeKG: loaded %d relationships", len(df))

        except Exception as exc:
            logger.exception("PrimeKG update failed")
            report.errors.append(str(exc))

        report.duration_seconds = time.monotonic() - t0
        return report

    async def update_drugsfda(self, incremental: bool = True) -> SourceReport:
        """Fetch FDA drug approval history from openFDA.

        Uses the same openFDA API key and rate limiting as FAERS.
        """
        t0 = time.monotonic()
        report = SourceReport(source="drugsfda")

        try:
            from ctra.data.drugsfda_loader import DrugsFDALoader

            loader = DrugsFDALoader(config=self._data_cfg)
            df = loader.fetch_and_cache()

            logger.info("Drugs@FDA: cached %d applications", len(df))

        except Exception as exc:
            logger.exception("Drugs@FDA update failed")
            report.errors.append(str(exc))

        report.duration_seconds = time.monotonic() - t0
        return report

    async def rebuild_indexes(self, sources: list[str] | None = None) -> IndexReport:
        """Rebuild LinearRAG indexes for specified sources (or all).

        Delegates to :class:`~ctra.rag.indexer.IndexBuilder` which reads
        the Parquet files and produces passage-level indexes.

        Args:
            sources: List of source names (``"ctg"``, ``"pubmed"``, etc.).
                If ``None``, rebuilds all sources.

        Returns:
            Index rebuild report.
        """
        t0 = time.monotonic()
        report = IndexReport()

        try:
            from ctra.rag.indexer import IndexBuilder

            source_enums = None
            if sources:
                source_enums = [DataSource(s) for s in sources]
                report.sources_rebuilt = sources
            else:
                report.sources_rebuilt = [s.value for s in DataSource]

            builder = IndexBuilder(data_config=self._data_cfg)

            # IndexBuilder.build_all is synchronous; run in thread pool.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, builder.build_all, source_enums, None)

            logger.info(
                "LinearRAG indexes rebuilt for sources: %s",
                report.sources_rebuilt,
            )

        except Exception as exc:
            logger.error("Index rebuild failed: %s", exc, exc_info=True)
            report.errors.append(str(exc))

        report.duration_seconds = time.monotonic() - t0
        return report

    def get_status(self) -> dict[str, SourceStatus]:
        """Get last update timestamps and record counts per source.

        Returns:
            Dict mapping source name to its current status.
        """
        source_parquets = {
            "ctg": Path(self._data_cfg.ctg_parquet),
            "pubmed": Path(self._data_cfg.pubmed_parquet),
            "faers": Path(self._data_cfg.faers_parquet),
            "aact": Path(self._data_cfg.aact_parquet),
            "chembl": Path(self._data_cfg.chembl_parquet),
            "primekg": Path(self._data_cfg.primekg_parquet),
            "drugsfda": Path(self._data_cfg.drugsfda_parquet),
        }

        statuses: dict[str, SourceStatus] = {}
        for name, pq_path in source_parquets.items():
            last_updated = self._state.get_last_updated(name)
            record_count = 0
            exists = pq_path.exists()
            if exists:
                try:
                    df = pl.read_parquet(pq_path)
                    record_count = len(df)
                except Exception:
                    record_count = -1

            statuses[name] = SourceStatus(
                source=name,
                last_updated=last_updated,
                record_count=record_count,
                parquet_path=str(pq_path),
                parquet_exists=exists,
            )

        return statuses

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _pubmed_esearch(
        self,
        base_url: str,
        api_key: str,
        min_date: str | None = None,
        max_results: int = 10_000,
    ) -> list[str]:
        """Run PubMed eSearch to find PMIDs for clinical trial articles.

        Args:
            base_url: Entrez E-utilities base URL.
            api_key: NCBI API key (optional).
            min_date: Only find articles indexed after this date (YYYY-MM-DD).
            max_results: Maximum number of PMIDs to return.

        Returns:
            List of PMID strings.
        """
        url = f"{base_url}/esearch.fcgi"
        # Search for clinical trial-related articles.
        query = "clinical trial[pt] OR clinical trial outcome[tiab]"

        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "date",
        }
        if api_key:
            params["api_key"] = api_key
        if min_date:
            # Entrez date format: YYYY/MM/DD
            params["mindate"] = min_date.replace("-", "/")
            params["datetype"] = "edat"  # Entrez date (index date)

        async with httpx.AsyncClient() as client:
            resp = await self._http_get_with_retry(client, url, params)

        result = resp.get("esearchresult", {})
        pmids = result.get("idlist", [])
        logger.info("PubMed eSearch returned %d PMIDs", len(pmids))
        return pmids  # type: ignore[no-any-return]

    async def _pubmed_efetch(
        self,
        base_url: str,
        api_key: str,
        pmids: list[str],
        batch_size: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch PubMed article metadata via eFetch in batches.

        Args:
            base_url: Entrez E-utilities base URL.
            api_key: NCBI API key.
            pmids: List of PubMed IDs to fetch.
            batch_size: Number of PMIDs per request.

        Returns:
            List of article dicts.
        """
        from ctra.data.pubmed_loader import PubMedLoader

        url = f"{base_url}/efetch.fcgi"
        rate = 10.0 if api_key else self._data_cfg.pubmed_rate_limit
        all_articles: list[dict[str, Any]] = []

        async with httpx.AsyncClient() as client:
            for i in range(0, len(pmids), batch_size):
                batch = pmids[i : i + batch_size]
                params: dict[str, Any] = {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "xml",
                    "retmode": "xml",
                }
                if api_key:
                    params["api_key"] = api_key

                try:
                    resp = await client.get(url, params=params, timeout=60.0)
                    resp.raise_for_status()
                    articles = PubMedLoader._parse_pubmed_xml(resp.text)
                    all_articles.extend(articles)
                except httpx.HTTPError as exc:
                    logger.warning(
                        "PubMed eFetch batch %d-%d failed: %s",
                        i,
                        i + len(batch),
                        exc,
                    )

                if rate > 0:
                    await asyncio.sleep(1.0 / rate)

                if (i + batch_size) % 2000 == 0:
                    logger.info(
                        "PubMed eFetch progress: %d/%d",
                        min(i + batch_size, len(pmids)),
                        len(pmids),
                    )

        return all_articles

    @staticmethod
    @_http_retry
    async def _http_get_with_retry(
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """HTTP GET with tenacity retry on transient failures."""
        resp = await client.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    @staticmethod
    @_http_retry
    async def _fetch_openfda_events(
        client: httpx.AsyncClient,
        drug_name: str,
        api_key: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch adverse events for a single drug from OpenFDA.

        Args:
            client: Shared async HTTP client.
            drug_name: Generic drug name.
            api_key: OpenFDA API key (optional).
            limit: Max event types to return.

        Returns:
            List of event dicts.
        """
        base_url = "https://api.fda.gov/drug/event.json"
        search_query = f'patient.drug.medicinalproduct:"{drug_name}"'

        params: dict[str, Any] = {
            "search": search_query,
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": limit,
        }
        if api_key:
            params["api_key"] = api_key

        try:
            resp = await client.get(base_url, params=params, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # No data for this drug -- not an error.
                return []
            raise

        results = data.get("results", [])
        events: list[dict[str, Any]] = []
        for item in results:
            events.append(
                {
                    "drug_name": drug_name,
                    "event": item.get("term", ""),
                    "count": item.get("count", 0),
                    "report_quarter": "",
                }
            )
        return events


# ---------------------------------------------------------------------------
# Helper: flatten CTG study dict to a flat row for Parquet
# ---------------------------------------------------------------------------


def _flatten_ctg_study(study: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten a CTG API v2 study dict into a flat dict for Parquet storage.

    Only processes ``protocolSection`` to prevent label leakage from
    ``resultsSection``.

    Args:
        study: A study dict from the CTG API v2 response.

    Returns:
        Flat dict with common columns, or ``None`` if the study is malformed.
    """
    protocol = study.get("protocolSection")
    if protocol is None:
        logger.warning("Study missing protocolSection; skipping")
        return None

    id_mod = protocol.get("identificationModule", {})
    nct_id = id_mod.get("nctId", "")
    if not nct_id:
        return None

    status_mod = protocol.get("statusModule", {})
    desc_mod = protocol.get("descriptionModule", {})
    design_mod = protocol.get("designModule", {})
    cond_mod = protocol.get("conditionsModule", {})
    elig_mod = protocol.get("eligibilityModule", {})
    sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})

    # Extract start date from statusModule.
    start_date_struct = status_mod.get("startDateStruct", {})
    start_date = start_date_struct.get("date", "")

    # Extract last update post date for incremental tracking.
    last_update = status_mod.get("lastUpdatePostDateStruct", {})
    last_update_date = last_update.get("date", "")

    # Extract phases.
    phases = design_mod.get("phases", [])
    phase_str = ",".join(phases) if isinstance(phases, list) else str(phases)

    # Extract conditions.
    conditions = cond_mod.get("conditions", [])
    conditions_str = "|".join(conditions) if isinstance(conditions, list) else str(conditions)

    # Extract interventions.
    arms_mod = protocol.get("armsInterventionsModule", {})
    interventions = arms_mod.get("interventions", [])
    intervention_names: list[str] = []
    for iv in interventions:
        name = iv.get("name", "")
        if name:
            intervention_names.append(name)

    # Sponsor.
    lead_sponsor = sponsor_mod.get("leadSponsor", {})

    return {
        "nctId": nct_id,
        "briefTitle": id_mod.get("briefTitle", ""),
        "officialTitle": id_mod.get("officialTitle", ""),
        "startDate": start_date,
        "lastUpdatePostDate": last_update_date,
        "briefSummary": desc_mod.get("briefSummary", ""),
        "detailedDescription": desc_mod.get("detailedDescription", ""),
        "phases": phase_str,
        "studyType": design_mod.get("studyType", ""),
        "conditions": conditions_str,
        "interventions": "|".join(intervention_names),
        "eligibilityCriteria": elig_mod.get("eligibilityCriteria", ""),
        "minimumAge": elig_mod.get("minimumAge", ""),
        "maximumAge": elig_mod.get("maximumAge", ""),
        "sex": elig_mod.get("sex", ""),
        "leadSponsor": lead_sponsor.get("name", ""),
    }
