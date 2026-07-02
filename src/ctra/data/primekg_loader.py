"""PrimeKG biological knowledge graph loader.

PrimeKG is a precision medicine knowledge graph containing 4M+ relationships
across 10 biological entity types (gene/protein, drug, disease, etc.).  This
loader handles:
    - Pre-processed Parquet loading
    - Raw CSV parsing from the PrimeKG ``kg.csv`` distribution
    - Loading via PyTDC (Therapeutics Data Commons) API
    - Entity relationship queries for feature engineering

Drug nodes use PrimeKG internal IDs and disease nodes use MONDO IDs
(e.g., MONDO:0005301).  Drug names can be matched to ChEMBL via the
entity resolver.  The graph stores directed edges both ways, yielding
~8.1M rows total.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from ctra.config.settings import DataConfig, get_settings

logger = logging.getLogger(__name__)

# The 12 validated columns in PrimeKG's kg.csv
_EXPECTED_COLUMNS = [
    "relation",
    "display_relation",
    "x_index",
    "x_id",
    "x_type",
    "x_name",
    "x_source",
    "y_index",
    "y_id",
    "y_type",
    "y_name",
    "y_source",
]

# The 10 known node types in PrimeKG
_NODE_TYPES = frozenset(
    {
        "biological_process",
        "gene/protein",
        "disease",
        "effect/phenotype",
        "anatomy",
        "molecular_function",
        "drug",
        "cellular_component",
        "pathway",
        "exposure",
    }
)


class PrimeKGLoader:
    """Load and query the PrimeKG biological knowledge graph.

    PrimeKG contains 4M+ relationships across 10 biological entity types.
    Supports pre-processed Parquet loading, raw CSV parsing, and PyTDC
    retrieval. Drug nodes use PrimeKG internal IDs; disease nodes use
    MONDO IDs (e.g., MONDO:0005301).

    Args:
        config: Data configuration. If ``None``, uses global settings.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initialize the PrimeKG loader.

        Args:
            config: Optional DataConfig to override global settings.
        """
        self._config = config or get_settings().data
        self._df: pl.DataFrame | None = None

    def load_parquet(self, path: Path | None = None) -> pl.DataFrame:
        """Load pre-processed PrimeKG data from Parquet.

        Expected columns: relation, display_relation, x_index, x_id, x_type,
        x_name, x_source, y_index, y_id, y_type, y_name, y_source.
        """
        parquet_path = Path(path or self._config.primekg_parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"PrimeKG parquet not found at {parquet_path}.  "
                "Run `python scripts/data/fetch_primekg.py` first."
            )

        logger.info("Loading PrimeKG data from %s", parquet_path)
        self._df = pl.read_parquet(parquet_path)
        logger.info("Loaded %d edge records", len(self._df))
        return self._df

    def load_from_csv(self, csv_path: Path | None = None) -> pl.DataFrame:
        """Parse PrimeKG from the raw ``kg.csv`` distribution file.

        The CSV contains 12 columns and ~8.1M rows (directed edges stored
        both ways).  After loading, the DataFrame is saved to Parquet for
        faster subsequent access.

        Args:
            csv_path: Path to the ``kg.csv`` file.  If ``None``, looks for
                ``datasets/primekg/kg.csv`` under the configured base dir.

        Returns:
            DataFrame with 12 validated columns.
        """
        if csv_path is None:
            csv_path = self._config.base_dir / "primekg" / "kg.csv"
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"PrimeKG CSV not found at {csv_path}.  "
                "Download from https://dataverse.harvard.edu/dataset.xhtml"
                "?persistentId=doi:10.7910/DVN/IXA7BM"
            )

        logger.info("Parsing PrimeKG CSV from %s (this may take a minute)", csv_path)
        df = pl.read_csv(csv_path, infer_schema_length=10_000)

        # Validate expected columns
        missing = set(_EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"PrimeKG CSV is missing expected columns: {sorted(missing)}")

        # Keep only the expected columns in canonical order
        df = df.select(_EXPECTED_COLUMNS)

        # Cast index columns to integers for consistency
        df = df.with_columns(
            pl.col("x_index").cast(pl.Int64),
            pl.col("y_index").cast(pl.Int64),
        )

        logger.info("Parsed %d edges from PrimeKG CSV", len(df))

        # Save to parquet for faster future loads
        parquet_path = Path(self._config.primekg_parquet)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_path)
        logger.info("Saved PrimeKG parquet to %s", parquet_path)

        self._df = df
        return df

    def load_from_tdc(self) -> pl.DataFrame:
        """Load PrimeKG via the PyTDC (Therapeutics Data Commons) API.

        Requires the ``PyTDC`` package to be installed.  The returned
        DataFrame has the same 12-column schema as :meth:`load_from_csv`.

        Returns:
            DataFrame with 12 validated columns.

        Raises:
            ImportError: If ``PyTDC`` is not installed.
        """
        try:
            from tdc.resource import PrimeKG
        except ImportError as exc:
            raise ImportError(
                "PyTDC is required to load PrimeKG via TDC.  Install with: pip install PyTDC"
            ) from exc

        logger.info("Loading PrimeKG via PyTDC (downloading if needed)")
        data = PrimeKG(path="./data")
        pandas_df = data.get_data()

        df = pl.from_pandas(pandas_df)

        # Validate expected columns
        missing = set(_EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"TDC PrimeKG is missing expected columns: {sorted(missing)}")

        df = df.select(_EXPECTED_COLUMNS)
        df = df.with_columns(
            pl.col("x_index").cast(pl.Int64),
            pl.col("y_index").cast(pl.Int64),
        )

        logger.info("Loaded %d edges from PrimeKG via TDC", len(df))

        # Save to parquet for faster future loads
        parquet_path = Path(self._config.primekg_parquet)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_path)
        logger.info("Saved PrimeKG parquet to %s", parquet_path)

        self._df = df
        return df  # type: ignore[no-any-return]

    def get_entity_relationships(
        self,
        entity_name: str,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find all relationships involving a named entity.

        Searches both ``x_name`` and ``y_name`` columns for the entity.
        Optionally filters by entity type (e.g., ``"drug"``, ``"disease"``).

        Args:
            entity_name: Name of the entity to search for (case-insensitive).
            entity_type: If provided, restrict to edges where the matched
                side has this type (one of the 10 PrimeKG node types).

        Returns:
            List of edge records as dicts.
        """
        if self._df is None:
            self.load_parquet()
        assert self._df is not None

        entity_lower = entity_name.lower()

        # Match on x_name or y_name (case-insensitive)
        x_match = self._df.get_column("x_name").str.to_lowercase() == entity_lower
        y_match = self._df.get_column("y_name").str.to_lowercase() == entity_lower

        if entity_type is not None:
            type_lower = entity_type.lower()
            x_type_match = self._df.get_column("x_type").str.to_lowercase() == type_lower
            y_type_match = self._df.get_column("y_type").str.to_lowercase() == type_lower
            mask = (x_match & x_type_match) | (y_match & y_type_match)
        else:
            mask = x_match | y_match

        return self._df.filter(mask).to_dicts()
