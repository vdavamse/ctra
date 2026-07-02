"""ChEMBL data loading from SQLite database.

ChEMBL provides drug mechanism, target, and clinical phase data.  This loader
handles:
    - Pre-processed Parquet loading
    - SQLite extraction: molecule dictionary, drug mechanisms with targets,
      indications, synonyms, and ATC classifications
    - Drug lookup and synonym resolution for entity normalization

The ChEMBL SQLite download is available at
https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from ctra.config.settings import DataConfig, get_settings

logger = logging.getLogger(__name__)


class ChEMBLLoader:
    """Load and query ChEMBL drug/target data.

    ChEMBL provides drug mechanism, target, and clinical phase data.
    Handles pre-processed Parquet loading and SQLite extraction of molecule
    dictionary, mechanisms, indications, synonyms, and ATC classifications.

    Args:
        config: Data configuration. If ``None``, uses global settings.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initialize the ChEMBL loader.

        Args:
            config: Optional DataConfig to override global settings.
        """
        self._config = config or get_settings().data
        self._df: pl.DataFrame | None = None

    def load_parquet(self, path: Path | None = None) -> pl.DataFrame:
        """Load pre-processed ChEMBL data from Parquet.

        Expected columns: chembl_id, pref_name, molecule_type, max_phase,
        first_approval, mechanism_of_action, action_type, target_names,
        target_organism, indication_mesh_headings, atc_codes, synonyms,
        black_box_warning, withdrawn_flag, first_in_class.
        """
        parquet_path = Path(path or self._config.chembl_parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"ChEMBL parquet not found at {parquet_path}.  "
                "Run `python scripts/data/fetch_chembl.py` first."
            )

        logger.info("Loading ChEMBL data from %s", parquet_path)
        self._df = pl.read_parquet(parquet_path)
        logger.info("Loaded %d drug records", len(self._df))
        return self._df

    def extract_from_sqlite(self, sqlite_path: Path | None = None) -> pl.DataFrame:
        """Extract drug data from ChEMBL SQLite database.

        Runs sequential queries against the ChEMBL SQLite file, merges
        results in polars, and saves the output as Parquet.

        Args:
            sqlite_path: Path to the ChEMBL SQLite file (e.g.,
                ``chembl_35.db``).  If ``None``, uses
                ``config.chembl_sqlite_path``.

        Returns:
            DataFrame with one row per molecule (multi-valued fields are
            pipe-separated).
        """
        db_path = Path(sqlite_path or self._config.chembl_sqlite_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"ChEMBL SQLite not found at {db_path}.  "
                "Download from https://ftp.ebi.ac.uk/pub/databases/chembl/"
                "ChEMBLdb/latest/."
            )

        logger.info("Extracting ChEMBL data from %s", db_path)
        import sqlite3

        conn = sqlite3.connect(str(db_path))

        try:
            # --- Query 1: molecule_dictionary ---
            logger.info("Query 1: molecule_dictionary")
            q1 = """
                SELECT
                    molecule_chembl_id,
                    pref_name,
                    molecule_type,
                    max_phase,
                    first_approval,
                    black_box_warning,
                    withdrawn_flag,
                    first_in_class
                FROM molecule_dictionary
            """
            molecules = pl.read_database(q1, conn)
            logger.info("  molecule_dictionary: %d rows", len(molecules))

            # Parse max_phase from string "4.0" to float
            molecules = molecules.with_columns(pl.col("max_phase").cast(pl.Float64, strict=False))

            # Filter to molecules that reached at least phase 1
            molecules = molecules.filter(pl.col("max_phase") >= 1.0)
            logger.info("  after max_phase >= 1.0 filter: %d rows", len(molecules))

            # Cast boolean-like integer columns
            molecules = molecules.with_columns(
                pl.col("black_box_warning").cast(pl.Boolean, strict=False),
                pl.col("withdrawn_flag").cast(pl.Boolean, strict=False),
                pl.col("first_in_class").cast(pl.Boolean, strict=False),
            )

            # --- Query 2: drug_mechanism JOIN target_dictionary ---
            logger.info("Query 2: drug_mechanism + target_dictionary")
            q2 = """
                SELECT
                    dm.molecule_chembl_id,
                    dm.mechanism_of_action,
                    dm.action_type,
                    td.pref_name AS target_name,
                    td.target_type,
                    td.organism
                FROM drug_mechanism dm
                JOIN target_dictionary td
                    ON dm.target_chembl_id = td.chembl_id
            """
            mechanisms = pl.read_database(q2, conn)
            logger.info("  drug_mechanism: %d rows", len(mechanisms))

            # Aggregate mechanisms per molecule
            mech_agg = mechanisms.group_by("molecule_chembl_id").agg(
                pl.col("mechanism_of_action")
                .drop_nulls()
                .unique()
                .str.concat("|")
                .alias("mechanism_of_action"),
                pl.col("action_type").drop_nulls().unique().str.concat("|").alias("action_type"),
                pl.col("target_name").drop_nulls().unique().str.concat("|").alias("target_names"),
                pl.col("organism").first().alias("target_organism"),
            )

            # --- Query 3: drug_indication ---
            logger.info("Query 3: drug_indication")
            q3 = """
                SELECT molecule_chembl_id, mesh_heading
                FROM drug_indication
            """
            indications = pl.read_database(q3, conn)
            logger.info("  drug_indication: %d rows", len(indications))

            ind_agg = indications.group_by("molecule_chembl_id").agg(
                pl.col("mesh_heading")
                .drop_nulls()
                .unique()
                .str.concat("|")
                .alias("indication_mesh_headings"),
            )

            # --- Query 4: molecule_synonyms ---
            logger.info("Query 4: molecule_synonyms")
            q4 = """
                SELECT molecule_chembl_id, synonyms
                FROM molecule_synonyms
            """
            synonyms = pl.read_database(q4, conn)
            logger.info("  molecule_synonyms: %d rows", len(synonyms))

            syn_agg = synonyms.group_by("molecule_chembl_id").agg(
                pl.col("synonyms").drop_nulls().unique().str.concat("|").alias("synonyms"),
            )

            # --- Query 5: molecule_atc_classification ---
            logger.info("Query 5: molecule_atc_classification")
            q5 = """
                SELECT molecule_chembl_id, level5 AS atc_code
                FROM molecule_atc_classification
            """
            atc = pl.read_database(q5, conn)
            logger.info("  molecule_atc_classification: %d rows", len(atc))

            atc_agg = atc.group_by("molecule_chembl_id").agg(
                pl.col("atc_code").drop_nulls().unique().str.concat("|").alias("atc_codes"),
            )

        finally:
            conn.close()

        # --- Merge all tables in polars ---
        logger.info("Merging ChEMBL tables on molecule_chembl_id")
        df = molecules.join(mech_agg, on="molecule_chembl_id", how="left")
        df = df.join(ind_agg, on="molecule_chembl_id", how="left")
        df = df.join(syn_agg, on="molecule_chembl_id", how="left")
        df = df.join(atc_agg, on="molecule_chembl_id", how="left")

        # Rename primary key column
        df = df.rename({"molecule_chembl_id": "chembl_id"})

        logger.info(
            "Merged ChEMBL DataFrame: %d rows, %d columns",
            len(df),
            len(df.columns),
        )

        # Save to parquet
        out_path = Path(self._config.chembl_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        logger.info("Saved ChEMBL parquet to %s", out_path)

        self._df = df
        return self._df

    def get_drug_info(self, drug_name: str) -> dict[str, Any] | None:
        """Look up a drug by name (case-insensitive substring match).

        Args:
            drug_name: Drug name or partial name to search.

        Returns:
            First matching drug record as a dict, or ``None`` if not found.
        """
        if self._df is None:
            self.load_parquet()
        assert self._df is not None

        mask = self._df.get_column("pref_name").str.contains(f"(?i){drug_name}")
        matches = self._df.filter(mask)

        if len(matches) == 0:
            return None
        return matches.row(0, named=True)

    def get_synonyms(self, drug_name: str) -> list[str]:
        """Get all synonyms for a drug name.

        Searches the ``pref_name`` column (case-insensitive) and returns
        the pipe-separated synonyms as a list.

        Args:
            drug_name: Drug name to look up.

        Returns:
            List of synonym strings.  Returns ``[drug_name]`` if no match.
        """
        if self._df is None:
            self.load_parquet()
        assert self._df is not None

        matches = self._df.filter(pl.col("pref_name").str.to_lowercase() == drug_name.lower())

        if len(matches) == 0:
            return [drug_name]

        row = matches.row(0, named=True)
        synonyms_str = row.get("synonyms", "")
        if not synonyms_str:
            return [str(row["pref_name"])]

        synonym_list = [s.strip() for s in str(synonyms_str).split("|") if s.strip()]
        pref = str(row["pref_name"])
        if pref not in synonym_list:
            synonym_list.insert(0, pref)
        return synonym_list
