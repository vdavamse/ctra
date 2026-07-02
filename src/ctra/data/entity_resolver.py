"""Drug synonym resolution via ChEMBL + PubChem synonym lists.

Provides drug name synonym lookup for query-time entity expansion in
the LinearRAG retrieval pipeline.  Uses ChEMBL molecule_synonyms as the
primary synonym source.

**ChEMBL is the primary synonym source** because it covers both small-
molecule drugs AND biologics (antibodies like pembrolizumab, trastuzumab).
PubChem is supplementary for small molecules only — its PUG-REST API
returns "No CID found" for biologics.

The resolver is used by :class:`~ctra.rag.linearrag_wrapper.LinearRAGWrapper`
at **query time** to expand drug entities with synonyms before seed
matching in LinearRAG's Personalized PageRank.  For example, when an
agent queries "mechanism of pembrolizumab", the resolver expands the
query entities to include ["pembrolizumab", "keytruda", "mk-3475"],
so PPR seeds from ALL graph nodes for that drug — not just the one
that matches the query text most closely.

This approach avoids fragile regex-based text replacement at index time.
Passage text is left unchanged; synonym expansion happens at retrieval.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import polars as pl

from ctra.config.settings import DataConfig, get_settings

logger = logging.getLogger(__name__)

# PubChem PUG-REST rate limit
_PUBCHEM_RATE_LIMIT = 5.0  # requests/sec


class EntityResolver:
    """Drug name normalization via ChEMBL + PubChem synonym lists.

    Provides query-time drug synonym expansion for LinearRAG entity matching.
    When a query mentions "keytruda", the resolver expands it to include
    ["pembrolizumab", "keytruda", "mk-3475"] so graph retrieval seeds from all
    matching nodes. ChEMBL is the primary source (covers biologics); PubChem
    supplements small molecules.

    Args:
        config: Data configuration. If ``None``, uses global settings.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initialize the entity resolver.

        Args:
            config: Optional DataConfig to override global settings.
        """
        self._config = config or get_settings().data
        self._synonym_to_canonical: dict[str, str] = {}
        self._canonical_to_synonyms: dict[str, list[str]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_synonyms(self, path: Path | None = None) -> None:
        """Load pre-cached synonym mappings from Parquet.

        Expected columns: ``canonical_name``, ``synonym``.

        Args:
            path: Override path to the synonyms Parquet file.
        """
        parquet_path = Path(path or self._config.pubchem_synonyms_parquet)
        if not parquet_path.exists():
            logger.warning(
                "Synonym cache not found at %s.  "
                "Run EntityResolver.build_synonym_cache() first.  "
                "Entity normalization will be a no-op.",
                parquet_path,
            )
            self._loaded = True
            return

        df = pl.read_parquet(parquet_path)
        logger.info("Loading %d synonym mappings from %s", len(df), parquet_path)

        for row in df.iter_rows(named=True):
            canonical = str(row["canonical_name"])
            synonym = str(row["synonym"])
            self._synonym_to_canonical[synonym.lower()] = canonical
            self._canonical_to_synonyms.setdefault(canonical.lower(), []).append(synonym)

        self._loaded = True
        logger.info(
            "EntityResolver loaded: %d synonyms → %d canonical names",
            len(self._synonym_to_canonical),
            len(self._canonical_to_synonyms),
        )

    def _ensure_loaded(self) -> None:
        """Load synonyms if not already loaded.

        Lazy-loads the synonym cache on first use.
        """
        if not self._loaded:
            self.load_synonyms()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, drug_name: str) -> str:
        """Return the canonical name for a drug, or the original if unknown.

        The canonical name uses Title Case (e.g., "Pembrolizumab").

        Args:
            drug_name: Any drug name variant (brand, generic, research code).

        Returns:
            Canonical drug name in Title Case, or the input unchanged.
        """
        self._ensure_loaded()
        return self._synonym_to_canonical.get(drug_name.lower(), drug_name)

    def get_synonyms(self, drug_name: str) -> list[str]:
        """Get all known synonyms for a drug name.

        Uses ChEMBL molecule_synonyms as the primary synonym source.

        Args:
            drug_name: Drug name to look up.

        Returns:
            List of synonym strings (including the canonical name).
        """
        self._ensure_loaded()
        canonical = self.normalize(drug_name)
        synonyms = self._canonical_to_synonyms.get(canonical.lower(), [])
        if not synonyms:
            return [drug_name]
        if canonical not in synonyms:
            synonyms = [canonical, *synonyms]
        return synonyms

    # ------------------------------------------------------------------
    # Cache building
    # ------------------------------------------------------------------

    def build_synonym_cache(
        self,
        chembl_parquet: Path | None = None,
        supplement_with_pubchem: bool = True,
        pubchem_drug_names: list[str] | None = None,
        output_path: Path | None = None,
    ) -> pl.DataFrame:
        """Build the synonym cache from ChEMBL (+ optional PubChem).

        Step 1: Extract all synonyms from ChEMBL parquet.
        Step 2: Optionally supplement small-molecule drugs with PubChem.
        Step 3: Merge, deduplicate, save to Parquet.

        Args:
            chembl_parquet: Path to ChEMBL parquet (from ChEMBLLoader).
            supplement_with_pubchem: Whether to fetch PubChem synonyms
                for small-molecule drugs.
            pubchem_drug_names: Explicit list of drug names to look up
                in PubChem.  If ``None``, uses small-molecule names from
                ChEMBL.
            output_path: Where to save the cache Parquet.

        Returns:
            DataFrame with columns ``canonical_name``, ``synonym``.
        """
        chembl_path = Path(chembl_parquet or self._config.chembl_parquet)
        out_path = Path(output_path or self._config.pubchem_synonyms_parquet)

        records: list[dict[str, str]] = []

        # -- Step 1: ChEMBL synonyms --
        if chembl_path.exists():
            records.extend(self._extract_chembl_synonyms(chembl_path))
            logger.info("Extracted %d synonym pairs from ChEMBL", len(records))
        else:
            logger.warning("ChEMBL parquet not found at %s; skipping", chembl_path)

        # -- Step 2: PubChem supplementary synonyms --
        if supplement_with_pubchem:
            small_mol_names = pubchem_drug_names
            if small_mol_names is None and chembl_path.exists():
                cdf = pl.read_parquet(chembl_path)
                if "molecule_type" in cdf.columns and "pref_name" in cdf.columns:
                    small_mol_names = (
                        cdf.filter(pl.col("molecule_type") == "Small molecule")
                        .get_column("pref_name")
                        .drop_nulls()
                        .unique()
                        .to_list()
                    )

            if small_mol_names:
                pubchem_records = self._fetch_pubchem_synonyms(small_mol_names)
                records.extend(pubchem_records)
                logger.info("Fetched %d synonym pairs from PubChem", len(pubchem_records))

        # -- Step 3: Merge and save --
        if not records:
            logger.warning("No synonyms collected; writing empty cache")
            df = pl.DataFrame({"canonical_name": [], "synonym": []})
        else:
            df = pl.DataFrame(records).unique(subset=["canonical_name", "synonym"])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        logger.info("Wrote %d synonym mappings to %s", len(df), out_path)

        # Reload into memory
        self._loaded = False
        self.load_synonyms(out_path)

        return df

    @staticmethod
    def _extract_chembl_synonyms(chembl_parquet: Path) -> list[dict[str, str]]:
        """Extract synonym pairs from ChEMBL parquet.

        ChEMBL stores synonyms as pipe-separated strings in a ``synonyms``
        column.  The canonical name is ``pref_name`` in Title Case.
        """
        df = pl.read_parquet(chembl_parquet)
        records: list[dict[str, str]] = []

        if "pref_name" not in df.columns:
            return records

        for row in df.iter_rows(named=True):
            pref_name = row.get("pref_name")
            if not pref_name:
                continue

            # Canonical name in Title Case
            canonical = str(pref_name).strip().title()

            # Add pref_name itself as a synonym
            records.append({"canonical_name": canonical, "synonym": canonical})

            # Add all pipe-separated synonyms
            synonyms_str = row.get("synonyms", "")
            if synonyms_str:
                for syn in str(synonyms_str).split("|"):
                    syn = syn.strip()
                    if syn:
                        records.append({"canonical_name": canonical, "synonym": syn})

        return records

    @staticmethod
    def _fetch_pubchem_synonyms(drug_names: list[str]) -> list[dict[str, str]]:
        """Fetch synonyms from PubChem PUG-REST for small-molecule drugs.

        Rate limited to 5 requests/second per PubChem policy.
        Gracefully handles "No CID found" for biologics/unknown drugs.
        """
        try:
            import pubchempy as pcp
        except ImportError:
            logger.warning("pubchempy not installed; skipping PubChem synonyms")
            return []

        records: list[dict[str, str]] = []

        for i, name in enumerate(drug_names):
            try:
                results = pcp.get_synonyms(name, "name")
                if results:
                    # PubChem response: [{"CID": int, "Synonym": [str, ...]}]
                    canonical = name.strip().title()
                    for result in results:
                        for syn in result.get("Synonym", [])[:50]:
                            records.append({"canonical_name": canonical, "synonym": syn})
            except Exception:
                # PubChem raises PubChemHTTPError for "No CID found"
                pass

            # Rate limiting
            time.sleep(1.0 / _PUBCHEM_RATE_LIMIT)

            if (i + 1) % 100 == 0:
                logger.info("PubChem synonyms: %d/%d drugs fetched", i + 1, len(drug_names))

        return records
