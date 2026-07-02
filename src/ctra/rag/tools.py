"""RAG tool factory functions for DSPy ReAct agents.

Each factory returns a ``(query: str) -> str`` callable matching the AutoCT
tool interface (see ``CogComp/autoct src/lfe/impl/tools.py``).  The callable
is bound to a specific trial's start date via closure, ensuring that every
retrieval request respects the temporal cutoff for label-leakage prevention.

Output format: YAML blocks separated by ``"\\n\\n---------\\n\\n"`` (matching
the original AutoCT separator).

Usage in a DSPy ReAct agent::

    from ctra.rag.tools import make_pubmed_search, make_nct_search
    from ctra.rag.tools import get_trial_info_dict

    nct_info = get_trial_info_dict("NCT00110279")  # returns dict
    tools = [
        make_pubmed_search(nct_info),
        make_nct_search(nct_info),
        make_chembl_search(nct_info),
        make_faers_search(nct_info),
        get_detailed_nct_info,  # returns YAML str (for agent use)
    ]
    agent = dspy.ReAct(signature, tools=tools)
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ctra.config.settings import DataSource, get_settings
from ctra.rag.linearrag_wrapper import LinearRAGWrapper, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Separator between YAML result blocks (matches AutoCT convention)
_RESULT_SEPARATOR = "\n\n---------\n\n"

# Maximum number of characters per tool response (prevents token budget blowout)
_MAX_RESPONSE_CHARS = 8000

# ---------------------------------------------------------------------------
# Disk cache (memoization across agent invocations)
# ---------------------------------------------------------------------------

_cache = None


def _get_cache() -> Any:
    """Lazy-initialize the diskcache FanoutCache for tool response memoization.

    The cache is stored in ``output_dir/tool_cache``. Repeated queries that
    produce the same results are returned from disk, reducing redundant
    RAG retrievals and LLM calls during multi-rollout MCTS runs.

    Returns:
        A ``diskcache.FanoutCache`` instance, or ``None`` if diskcache is
        not installed (tool caching is disabled in that case).
    """
    global _cache
    if _cache is not None:
        return _cache

    try:
        from diskcache import FanoutCache
    except ImportError:
        logger.warning("diskcache not installed; tool caching disabled")
        return None

    settings = get_settings()
    cache_dir = str(settings.output_dir / "tool_cache")
    _cache = FanoutCache(directory=cache_dir, tag_index=True)
    return _cache


def _cache_memoize(tag: str) -> Callable[..., Any]:
    """Decorator for memoizing tool call results via diskcache.

    Wraps a tool function so that identical queries return cached results
    from previous runs. The cache key is built from function name, arguments,
    and kwargs. Tagged entries are grouped by ``tag`` for bulk cache management.

    Args:
        tag: A cache tag for grouping related memoizations (e.g., ``"pm-v0"``
            for PubMed searches version 0).

    Returns:
        A decorator function that wraps the target tool function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = _get_cache()
            if cache is None:
                return func(*args, **kwargs)

            # Build a cache key from function name + arguments
            key = f"{tag}:{func.__name__}:{args}:{kwargs}"
            result = cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.set(key, result, tag=tag)
            return result

        # Preserve name/doc for DSPy tool registration
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Shared wrapper instance (lazy singleton)
# ---------------------------------------------------------------------------

_rag_instance: LinearRAGWrapper | None = None


def _get_rag() -> LinearRAGWrapper:
    """Return the shared LinearRAGWrapper singleton instance.

    The instance is created on first call and reused for all subsequent
    tool invocations, avoiding redundant index loading. The index is lazy-loaded
    when first needed, so this function itself is fast.

    Returns:
        The ``LinearRAGWrapper`` instance (may be loading on first call).
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LinearRAGWrapper()
    return _rag_instance


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def _parse_start_date(nct_info: dict[str, Any]) -> str:
    """Extract and normalize the trial's start date for temporal filtering.

    AutoCT trial dicts may store dates in ``YYYY-MM-DD`` or ``YYYY-MM``
    format. This function normalizes ``YYYY-MM`` → ``YYYY-MM-01`` so it
    can be safely parsed by ``datetime.date.fromisoformat``.

    Args:
        nct_info: Trial info dict with ``startDate`` or ``start_date`` field.

    Returns:
        Normalized date string in ``YYYY-MM-DD`` format.

    Raises:
        ValueError: If no ``startDate`` field is present.
    """
    start_date = str(nct_info.get("startDate", nct_info.get("start_date", "")))
    if not start_date:
        raise ValueError("nct_info must contain 'startDate' or 'start_date'")
    # Normalize partial dates: "2015-03" -> "2015-03-01"
    if len(start_date) < 10:
        start_date = start_date + "-01"
    return start_date


def _date_str_to_date(date_str: str) -> datetime.date:
    """Parse an ISO date string to a ``datetime.date`` object.

    Args:
        date_str: A date string in ``YYYY-MM-DD`` format.

    Returns:
        A ``datetime.date`` object.

    Raises:
        ValueError: If the string is not a valid ISO date.
    """
    return datetime.date.fromisoformat(date_str)


def _chunks_to_yaml_blocks(
    chunks: list[RetrievedChunk],
    extra_fields: Callable[[RetrievedChunk], dict[str, Any]] | None = None,
) -> str:
    """Format retrieved chunks as YAML blocks matching AutoCT output format.

    Each ``RetrievedChunk`` is serialized as a YAML dict with keys:
    ``text``, ``source``, ``doc_id``, ``date`` (if available), ``score``,
    and any metadata fields. Multiple chunks are separated by the AutoCT
    standard ``\n\n---------\n\n`` separator.

    Respects ``_MAX_RESPONSE_CHARS`` to prevent token budget overrun.
    Results are truncated with ``... (results truncated)`` if they exceed
    the maximum character limit.

    Args:
        chunks: List of retrieved passages to format.
        extra_fields: Optional callback to add source-specific fields to
            each chunk before serialization.

    Returns:
        YAML-formatted string with results, or a message if no chunks found.
    """
    if not chunks:
        return "No relevant documents found."

    blocks: list[str] = []
    total_chars = 0

    for chunk in chunks:
        block_data: dict[str, Any] = {
            "text": chunk.text,
            "source": chunk.source,
            "doc_id": chunk.doc_id,
        }
        if chunk.date is not None:
            block_data["date"] = str(chunk.date)
        if chunk.score > 0:
            block_data["score"] = round(chunk.score, 4)

        # Merge metadata
        if chunk.metadata:
            block_data.update(chunk.metadata)

        # Add source-specific extra fields
        if extra_fields is not None:
            block_data.update(extra_fields(chunk))

        block_str = yaml.dump(block_data, default_flow_style=False, sort_keys=False)

        if total_chars + len(block_str) > _MAX_RESPONSE_CHARS:
            blocks.append("... (results truncated)")
            break

        blocks.append(block_str)
        total_chars += len(block_str)

    return _RESULT_SEPARATOR.join(blocks)


# ---------------------------------------------------------------------------
# Factory functions (match AutoCT interface)
# ---------------------------------------------------------------------------


def make_pubmed_search(nct_info: dict[str, Any]) -> Callable[[str], str]:
    """Create a PubMed search tool bound to a trial's start date.

    Args:
        nct_info: Trial info dict with at least ``startDate`` (YYYY-MM-DD).

    Returns:
        A callable ``(query: str) -> str`` that searches PubMed abstracts
        published before the trial's start date and returns YAML-formatted
        results.
    """
    start_date = _parse_start_date(nct_info)
    date_cutoff = _date_str_to_date(start_date)
    rag = _get_rag()

    @_cache_memoize(tag="pm-v0")
    def pubmed_search(query: str) -> str:
        """Search PubMed biomedical literature.

        Returns abstracts and metadata of the most relevant articles published
        before the current trial's start date.

        Repeated searches for the same query will yield the same results.
        """
        chunks = rag.search(
            query=query,
            before_date=date_cutoff,
            sources=[DataSource.PUBMED],
            top_k=10,
        )
        return _chunks_to_yaml_blocks(chunks)

    pubmed_search.__name__ = "pubmed_search"
    return pubmed_search  # type: ignore[no-any-return]


def make_nct_search(nct_info: dict[str, Any]) -> Callable[[str], str]:
    """Create a related-trials search tool bound to a trial's start date.

    Args:
        nct_info: Trial info dict with at least ``startDate`` (YYYY-MM-DD).

    Returns:
        A callable ``(query: str) -> str`` that searches ClinicalTrials.gov
        for related trials that started before the current trial.
    """
    start_date = _parse_start_date(nct_info)
    date_cutoff = _date_str_to_date(start_date)
    rag = _get_rag()

    @_cache_memoize(tag="nct-v0")
    def related_trials_nct_search(query: str) -> str:
        """Search ClinicalTrials.gov for related trials.

        Returns summaries of the most relevant trials that took place before
        the current trial's start date.

        Repeated searches for the same query will yield the same results.
        """
        chunks = rag.search(
            query=query,
            before_date=date_cutoff,
            sources=[DataSource.CTG],
            top_k=10,
        )
        return _chunks_to_yaml_blocks(chunks)

    related_trials_nct_search.__name__ = "related_trials_nct_search"
    return related_trials_nct_search  # type: ignore[no-any-return]


def make_faers_search(nct_info: dict[str, Any]) -> Callable[[str], str]:
    """Create a FAERS adverse event search tool bound to a trial's start date.

    Args:
        nct_info: Trial info dict with at least ``startDate`` (YYYY-MM-DD).

    Returns:
        A callable ``(query: str) -> str`` that searches FAERS for adverse
        event reports filed before the trial's start date.
    """
    start_date = _parse_start_date(nct_info)
    date_cutoff = _date_str_to_date(start_date)
    rag = _get_rag()

    @_cache_memoize(tag="faers-v0")
    def faers_search(query: str) -> str:
        """Search FAERS for adverse event reports.

        Returns post-market safety signals and adverse event data reported
        before the current trial's start date.

        Repeated searches for the same query will yield the same results.
        """
        chunks = rag.search(
            query=query,
            before_date=date_cutoff,
            sources=[DataSource.FAERS],
            top_k=10,
        )
        return _chunks_to_yaml_blocks(chunks)

    faers_search.__name__ = "faers_search"
    return faers_search  # type: ignore[no-any-return]


def make_chembl_search(nct_info: dict[str, Any]) -> Callable[[str], str]:
    """Create a ChEMBL drug mechanism search tool bound to a trial's start date.

    Args:
        nct_info: Trial info dict with at least ``startDate`` (YYYY-MM-DD).

    Returns:
        A callable ``(query: str) -> str`` that searches ChEMBL for drug
        mechanisms, targets, and clinical phase data.
    """
    start_date = _parse_start_date(nct_info)
    date_cutoff = _date_str_to_date(start_date)
    rag = _get_rag()

    @_cache_memoize(tag="chembl-v0")
    def chembl_search(query: str) -> str:
        """Search ChEMBL for drug mechanism, target, and phase data.

        Returns drug mechanisms of action, protein targets, clinical phase
        progression, and ATC classification available before the trial.

        Repeated searches for the same query will yield the same results.
        """
        chunks = rag.search(
            query=query,
            before_date=date_cutoff,
            sources=[DataSource.CHEMBL],
            top_k=10,
        )
        return _chunks_to_yaml_blocks(chunks)

    chembl_search.__name__ = "chembl_search"
    return chembl_search  # type: ignore[no-any-return]


def make_aact_search(nct_info: dict[str, Any]) -> Callable[[str], str]:
    """Create an AACT population-level search tool bound to a trial's start date.

    Args:
        nct_info: Trial info dict with at least ``startDate`` (YYYY-MM-DD).

    Returns:
        A callable ``(query: str) -> str`` that searches AACT for sponsor
        track records, disease success rates, and trial pipeline history.
    """
    start_date = _parse_start_date(nct_info)
    date_cutoff = _date_str_to_date(start_date)
    rag = _get_rag()

    @_cache_memoize(tag="aact-v0")
    def aact_search(query: str) -> str:
        """Search AACT for clinical trial population statistics.

        Returns sponsor track records, historical success rates by disease
        and phase, and trial pipeline history before the current trial.

        Repeated searches for the same query will yield the same results.
        """
        chunks = rag.search(
            query=query,
            before_date=date_cutoff,
            sources=[DataSource.AACT],
            top_k=10,
        )
        return _chunks_to_yaml_blocks(chunks)

    aact_search.__name__ = "aact_search"
    return aact_search  # type: ignore[no-any-return]


def make_primekg_search(nct_info: dict[str, Any]) -> Callable[[str], str]:
    """Create a PrimeKG biological knowledge search tool bound to a trial's start date.

    Args:
        nct_info: Trial info dict with at least ``startDate`` (YYYY-MM-DD).

    Returns:
        A callable ``(query: str) -> str`` that searches PrimeKG for
        biological relationships between drugs, targets, pathways, and diseases.
    """
    start_date = _parse_start_date(nct_info)
    date_cutoff = _date_str_to_date(start_date)
    rag = _get_rag()

    @_cache_memoize(tag="primekg-v0")
    def primekg_search(query: str) -> str:
        """Search PrimeKG for biological knowledge.

        Returns drug-target relationships, disease-gene associations,
        pathway memberships, side effect mechanisms, and drug interactions
        from the PrimeKG biological knowledge graph.

        Repeated searches for the same query will yield the same results.
        """
        chunks = rag.search(
            query=query,
            before_date=date_cutoff,
            sources=[DataSource.PRIMEKG],
            top_k=10,
        )
        return _chunks_to_yaml_blocks(chunks)

    primekg_search.__name__ = "primekg_search"
    return primekg_search  # type: ignore[no-any-return]


def make_drugsfda_search(nct_info: dict[str, Any]) -> Callable[[str], str]:
    """Create a Drugs@FDA approval history search tool bound to a trial's start date.

    Args:
        nct_info: Trial info dict with at least ``startDate`` (YYYY-MM-DD).

    Returns:
        A callable ``(query: str) -> str`` that searches FDA approval
        records for drug approval history and sponsor track records.
    """
    start_date = _parse_start_date(nct_info)
    date_cutoff = _date_str_to_date(start_date)
    rag = _get_rag()

    @_cache_memoize(tag="drugsfda-v0")
    def drugsfda_search(query: str) -> str:
        """Search Drugs@FDA for drug approval history.

        Returns FDA approval records, sponsor track records, review
        priority designations, and regulatory classifications before
        the current trial's start date.

        Repeated searches for the same query will yield the same results.
        """
        chunks = rag.search(
            query=query,
            before_date=date_cutoff,
            sources=[DataSource.DRUGSFDA],
            top_k=10,
        )
        return _chunks_to_yaml_blocks(chunks)

    drugsfda_search.__name__ = "drugsfda_search"
    return drugsfda_search  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Direct functions (not factories)
# ---------------------------------------------------------------------------


# Fields that reveal trial outcome or post-completion status.
# These MUST be stripped from any trial info returned to agents to prevent
# label leakage.  See ARCHITECT_REVIEW.md C1.
_LEAKAGE_FIELD_PATTERNS = {
    # Status fields that reveal outcome
    "overallstatus",
    "overall_status",
    "laststatus",
    "last_known_status",
    "lastknownstatus",
    "whystopped",
    "why_stopped",
    # Results section (any column derived from resultsSection)
    "resultssection",
    "results_section",
    "resultsposted",
    "results_posted",
    "resultsposteddate",
    "results_posted_date",
    "resultsfirstposted",
    "results_first_posted",
    "resultsfirstsubmitted",
    "results_first_submitted",
    # Completion info that correlates with outcome
    "completiondate",
    "completion_date",
    "primarycompletiondate",
    "primary_completion_date",
    "studycompletiondate",
    "study_completion_date",
    # Disposition and withdrawal (outcome signals)
    "dispositionmodule",
    "disposition_module",
    # Status-verified date (implies monitoring of outcome)
    "statusverifieddate",
    "status_verified_date",
}


def _is_leakage_field(field_name: str) -> bool:
    """Check if a column name could reveal trial outcome (label leakage prevention).

    Matches against known leakage patterns like ``overallStatus``,
    ``resultsSection``, ``completionDate``, etc. Used by ``get_detailed_nct_info``
    and ``get_trial_info_dict`` to strip outcome-revealing fields before
    returning trial data to agents.

    Args:
        field_name: A column name to check.

    Returns:
        ``True`` if the field could reveal trial outcome, ``False`` otherwise.
    """
    normalized = field_name.lower().replace("-", "").replace("_", "")
    return any(pattern.replace("_", "") in normalized for pattern in _LEAKAGE_FIELD_PATTERNS)


def get_detailed_nct_info(nctid: str) -> str:
    """Get detailed clinical trial info from parquet via polars.

    Queries the local ClinicalTrials.gov parquet file for the
    ``protocolSection`` of the specified trial.  Returns YAML-formatted
    trial metadata.

    **Leakage prevention:** All columns that could reveal trial outcome
    (``overallStatus``, ``resultsSection``, ``completionDate``, etc.) are
    stripped before returning.  See ``_LEAKAGE_FIELD_PATTERNS``.

    Args:
        nctid: The NCT ID (e.g., ``NCT00110279``).

    Returns:
        YAML-formatted trial information, or an error message.
    """
    import polars as pl

    settings = get_settings()
    ctg_path = str(settings.data.ctg_parquet)

    if not Path(ctg_path).exists():
        return f"Error: CTG data file not found at {ctg_path}"

    cache = _get_cache()
    cache_key = f"nctinfo-v1:{nctid}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

    try:
        df = pl.read_parquet(ctg_path)

        # Try to find the trial using either column name convention
        if "nctId" in df.columns:
            result = df.filter(pl.col("nctId") == nctid)
        elif "nct_id" in df.columns:
            result = df.filter(pl.col("nct_id") == nctid)
        else:
            return f"No NCT ID column found in {ctg_path}"

        if len(result) == 0:
            return f"No trial found with NCT ID: {nctid}"

        row = result.row(0, named=True)
        record = {k: v for k, v in row.items() if v is not None}

        # Strip outcome-revealing fields to prevent label leakage (C1 fix).
        clean_record: dict[str, Any] = {}
        stripped_fields: list[str] = []
        for k, v in record.items():
            if _is_leakage_field(k):
                stripped_fields.append(k)
                continue
            if hasattr(v, "item"):
                clean_record[k] = v.item()
            else:
                clean_record[k] = v

        if stripped_fields:
            logger.debug(
                "Stripped %d leakage fields from %s: %s",
                len(stripped_fields),
                nctid,
                stripped_fields,
            )

        output = yaml.dump(clean_record, default_flow_style=False, sort_keys=False)

        if cache is not None:
            cache.set(cache_key, output, tag="nctinfo-v1")

        return output

    except Exception as exc:
        logger.warning("Failed to look up %s: %s", nctid, exc)
        return f"Error looking up {nctid}: {exc}"


def get_trial_info_dict(nctid: str) -> dict[str, Any]:
    """Get trial info as a dict (for programmatic use by factory functions).

    Unlike :func:`get_detailed_nct_info` which returns a YAML string suitable
    for agent consumption, this function returns a Python dict that can be
    passed directly to ``make_pubmed_search(nct_info)``, etc.

    Outcome-revealing fields are stripped (same leakage prevention as
    ``get_detailed_nct_info``).

    Args:
        nctid: The NCT ID (e.g., ``NCT00110279``).

    Returns:
        Dict of trial fields with ``startDate`` guaranteed present.

    Raises:
        KeyError: If the trial is not found.
        FileNotFoundError: If the CTG parquet file does not exist.
    """
    # Delegate to ctg_loader.get_trial_info() which returns a dict
    from ctra.data.ctg_loader import get_trial_info

    record = get_trial_info(nctid)

    # Strip leakage fields (same filter as get_detailed_nct_info)
    clean: dict[str, Any] = {}
    for k, v in record.items():
        if _is_leakage_field(k):
            continue
        if hasattr(v, "item"):
            clean[k] = v.item()
        else:
            clean[k] = v

    # Ensure startDate is present (required by factory functions)
    if "startDate" not in clean:
        for col in ("start_date", "startDateStruct_date"):
            if col in clean:
                clean["startDate"] = str(clean[col])
                break

    if "startDate" not in clean:
        raise KeyError(
            f"Trial {nctid} has no startDate field.  "
            "Cannot create date-filtered tools without a temporal cutoff."
        )

    return clean
