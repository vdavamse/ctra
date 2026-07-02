"""PubMed Entrez client for fetching biomedical literature.

Uses the NCBI E-utilities API to fetch article metadata (title, abstract,
MeSH terms, publication date) by PMID or search query.

Supports two retrieval modes:
    - **By PMID list**: ``fetch_by_pmids()`` for known articles
    - **By search query**: ``fetch_pubmed_abstracts()`` for topic-based search

Both modes produce dicts with a consistent schema that can be converted to
text passages for LinearRAG indexing via ``abstract_to_passage()``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import polars as pl

from ctra.config.settings import DataConfig, get_settings

logger = logging.getLogger(__name__)

# Month name to number mapping for PubMed dates
_MONTH_MAP: dict[str, str] = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


class PubMedLoader:
    """Fetch and parse PubMed articles via the NCBI E-utilities API.

    Supports two retrieval modes: by PMID list (fetch_by_pmids) or by
    search query (fetch_pubmed_abstracts). Both produce dicts with consistent
    schema for conversion to LinearRAG text passages.

    Args:
        config: Data configuration. If ``None``, uses global settings.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initialize the PubMed loader.

        Args:
            config: Optional DataConfig to override global settings.
        """
        self._config = config or get_settings().data
        self._df: pl.DataFrame | None = None

    def load_parquet(self, path: Path | None = None) -> pl.DataFrame:
        """Load pre-fetched PubMed data from Parquet.

        Expected columns: pmid, title, abstract, authors, journal,
        publication_date, mesh_terms.
        """
        parquet_path = Path(path or self._config.pubmed_parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"PubMed parquet not found at {parquet_path}.  "
                "Run `python scripts/data/fetch_pubmed.py` first."
            )

        logger.info("Loading PubMed data from %s", parquet_path)
        self._df = pl.read_parquet(parquet_path)
        logger.info("Loaded %d PubMed articles", len(self._df))
        return self._df

    def fetch_by_pmids(
        self,
        pmids: list[str],
        batch_size: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch article metadata from PubMed by PMID list.

        Args:
            pmids: List of PubMed IDs.
            batch_size: Number of PMIDs per API request.

        Returns:
            List of article dicts.
        """
        import httpx

        api_key = os.environ.get(self._config.pubmed_api_key_env_var, "")
        base_url = f"{self._config.pubmed_api_base}/efetch.fcgi"

        all_articles: list[dict[str, Any]] = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params: dict[str, str] = {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "xml",
                "retmode": "xml",
            }
            if api_key:
                params["api_key"] = api_key

            try:
                response = httpx.get(base_url, params=params, timeout=60.0)
                response.raise_for_status()
                articles = self._parse_pubmed_xml(response.text)
                all_articles.extend(articles)
            except httpx.HTTPError:
                logger.warning("Failed to fetch PMIDs batch %d-%d", i, i + len(batch))

            # Rate limiting: 3/sec without key, 10/sec with key
            rate = 10.0 if api_key else self._config.pubmed_rate_limit
            time.sleep(1.0 / rate)

            logger.info(
                "Fetched %d/%d articles", len(all_articles), min(i + batch_size, len(pmids))
            )

        return all_articles

    @staticmethod
    def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
        """Parse PubMed efetch XML response into article dicts."""
        articles: list[dict[str, Any]] = []

        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            logger.warning("Failed to parse PubMed XML response")
            return articles

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                # PMID
                pmid_elem = article_elem.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""

                # Title
                title_elem = article_elem.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""

                # Abstract
                abstract_parts = []
                for abs_text in article_elem.findall(".//AbstractText"):
                    if abs_text.text:
                        label = abs_text.get("Label", "")
                        prefix = f"{label}: " if label else ""
                        abstract_parts.append(f"{prefix}{abs_text.text}")
                abstract = " ".join(abstract_parts)

                # Journal
                journal_elem = article_elem.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""

                # Authors (first 5)
                authors: list[str] = []
                for author_elem in article_elem.findall(".//Author")[:5]:
                    last = author_elem.findtext("LastName", "")
                    first = author_elem.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last} {first}".strip())

                # Publication date
                pub_date_elem = article_elem.find(".//PubDate")
                pub_date = ""
                if pub_date_elem is not None:
                    year = pub_date_elem.findtext("Year", "")
                    month = pub_date_elem.findtext("Month", "01")
                    day = pub_date_elem.findtext("Day", "01")
                    if year:
                        # Normalize month names (e.g., "Mar" -> "03")
                        month = _MONTH_MAP.get(month, month)
                        # Ensure two-digit month/day
                        try:
                            month = f"{int(month):02d}"
                        except ValueError:
                            month = "01"
                        try:
                            day = f"{int(day):02d}"
                        except ValueError:
                            day = "01"
                        pub_date = f"{year}-{month}-{day}"

                # MeSH terms
                mesh_terms = [
                    mesh.text
                    for mesh in article_elem.findall(".//MeshHeading/DescriptorName")
                    if mesh.text
                ]

                articles.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "authors": "|".join(authors),
                        "journal": journal,
                        "publication_date": pub_date,
                        "mesh_terms": "|".join(mesh_terms),
                    }
                )
            except Exception:
                logger.warning("Failed to parse article element")

        return articles

    def fetch_and_cache(
        self,
        pmids: list[str],
        output_path: Path | None = None,
    ) -> pl.DataFrame:
        """Fetch PubMed articles and save to Parquet.

        Args:
            pmids: List of PubMed IDs.
            output_path: Where to save the Parquet file.

        Returns:
            DataFrame with article records.
        """
        articles = self.fetch_by_pmids(pmids)
        df = pl.DataFrame(articles) if articles else pl.DataFrame()

        if len(df) > 0:
            out = Path(output_path or self._config.pubmed_parquet)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out)
            logger.info("Saved %d PubMed articles to %s", len(df), out)

        self._df = df
        return df


# ---------------------------------------------------------------------------
# Search-based retrieval (ESearch + EFetch)
# ---------------------------------------------------------------------------


def fetch_pubmed_abstracts(
    query: str,
    max_results: int = 1000,
    config: DataConfig | None = None,
) -> list[dict[str, Any]]:
    """Fetch PubMed abstracts via the Entrez ESearch + EFetch APIs.

    Unlike ``PubMedLoader.fetch_by_pmids()`` which requires known PMIDs,
    this function performs a keyword search first, then fetches the results.

    Args:
        query: PubMed search query (supports MeSH terms and Boolean operators).
        max_results: Maximum number of articles to return.
        config: Data configuration override.

    Returns:
        List of article dicts with keys: pmid, title, abstract,
        authors, journal, publication_date, mesh_terms.
    """
    import httpx

    cfg = config or get_settings().data
    api_key = os.environ.get(cfg.pubmed_api_key_env_var, "")

    # -- Step 1: ESearch to get PMIDs --
    esearch_url = f"{cfg.pubmed_api_base}/esearch.fcgi"
    search_params: dict[str, str | int] = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        search_params["api_key"] = api_key

    try:
        response = httpx.get(esearch_url, params=search_params, timeout=30.0)
        response.raise_for_status()
        search_result = response.json()
    except httpx.HTTPError:
        logger.warning("PubMed ESearch failed for query: %s", query)
        return []

    pmid_list = search_result.get("esearchresult", {}).get("idlist", [])
    if not pmid_list:
        logger.info("No PubMed results for query: %s", query)
        return []

    logger.info("PubMed search returned %d PMIDs for: %s", len(pmid_list), query)

    # Rate limit between search and fetch
    rate = 10.0 if api_key else cfg.pubmed_rate_limit
    time.sleep(1.0 / rate)

    # -- Step 2: EFetch to get article details --
    loader = PubMedLoader(config=cfg)
    return loader.fetch_by_pmids(pmid_list)


# ---------------------------------------------------------------------------
# Passage generation for LinearRAG
# ---------------------------------------------------------------------------


def abstract_to_passage(abstract: dict[str, Any]) -> str:
    """Convert a PubMed abstract dict to a text passage for LinearRAG indexing.

    Produces a structured text passage that preserves the key entities
    (drug names, diseases, outcomes) for LinearRAG's NER pipeline.

    Args:
        abstract: A dict with keys: pmid, title, abstract, authors,
            journal, publication_date, mesh_terms.

    Returns:
        A text passage string.
    """
    parts: list[str] = []

    title = abstract.get("title", "")
    if title:
        parts.append(str(title))

    abstract_text = abstract.get("abstract", "")
    if abstract_text:
        parts.append(str(abstract_text))

    # Append MeSH terms as a sentence for entity extraction
    mesh = abstract.get("mesh_terms", "")
    if mesh and isinstance(mesh, str):
        terms = mesh.split("|")
        if terms:
            parts.append(f"MeSH terms: {', '.join(terms)}.")

    journal = abstract.get("journal", "")
    pub_date = abstract.get("publication_date", "")
    if journal or pub_date:
        meta_parts = []
        if journal:
            meta_parts.append(f"Published in {journal}")
        if pub_date:
            meta_parts.append(f"({pub_date})")
        parts.append(" ".join(meta_parts) + ".")

    return " ".join(parts)
