"""Index builder for LinearRAG across CTG, PubMed, ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA.

Reads cleaned Parquet files for each data source, converts structured records
into natural-language passages suitable for LinearRAG's NER + graph pipeline,
and writes the combined index to disk.

**Important:** LinearRAG handles sentence splitting and NER internally via
spaCy.  The indexer's job is to produce coherent *passage-level* chunks —
one passage per logical record (trial, abstract, drug, adverse event profile).
LinearRAG then segments each passage into sentences, extracts entities, and
builds its tri-graph (entity ↔ sentence ↔ passage).

The indexer enforces two leakage-prevention rules at index time:
    1. ``resultsSection`` from ClinicalTrials.gov is **never** indexed.
    2. Every passage carries a date so the retrieval layer can filter by time.

Usage::

    python scripts/build_rag_index.py build --sources ctg,pubmed --output datasets/linearrag-index

    # Or programmatically:
    from ctra.rag.indexer import IndexBuilder
    builder = IndexBuilder()
    builder.build_all(output_dir="datasets/linearrag-index")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from ctra.config.settings import DataConfig, DataSource, RAGConfig, get_settings

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Passage generators per source
#
# Each generator yields one dict per PASSAGE (not per sentence).
# LinearRAG internally splits passages into sentences via spaCy and
# extracts entities for graph construction.
# ---------------------------------------------------------------------------


def _ctg_to_passages(df: pl.DataFrame) -> Iterator[dict[str, str | None]]:
    """Convert ClinicalTrials.gov records into passage-level dicts.

    Each trial becomes ONE passage containing all its protocol text.
    The ``resultsSection`` is deliberately excluded to prevent label leakage.
    """
    for row in df.iter_rows(named=True):
        nct_id = str(row.get("nct_id", row.get("nctId", "")))
        start_date = row.get("start_date", row.get("startDate"))
        date_str = str(start_date) if start_date is not None else None

        parts: list[str] = []

        # Identification
        title = row.get("brief_title", row.get("briefTitle", ""))
        if title:
            parts.append(str(title))

        official = row.get("official_title", row.get("officialTitle", ""))
        if official and official != title:
            parts.append(str(official))

        # Description
        summary = row.get("brief_summary", row.get("briefSummary", ""))
        if summary:
            parts.append(str(summary))

        detail = row.get("detailed_description", row.get("detailedDescription", ""))
        if detail:
            parts.append(str(detail))

        # Design
        study_type = row.get("study_type", row.get("studyType", ""))
        phases = row.get("phases", row.get("phase", ""))
        if study_type or phases:
            design_parts = []
            if study_type:
                design_parts.append(f"Study type: {study_type}")
            if phases:
                phase_str = phases if isinstance(phases, str) else ", ".join(phases)
                design_parts.append(f"Phase: {phase_str}")
            allocation = row.get("allocation", "")
            if allocation:
                design_parts.append(f"Allocation: {allocation}")
            masking = row.get("masking", "")
            if masking:
                design_parts.append(f"Masking: {masking}")
            purpose = row.get("primary_purpose", row.get("primaryPurpose", ""))
            if purpose:
                design_parts.append(f"Purpose: {purpose}")
            parts.append(". ".join(design_parts) + ".")

        # Enrollment
        enrollment = row.get("enrollment", row.get("enrollmentCount", ""))
        enrollment_type = row.get("enrollment_type", row.get("enrollmentType", ""))
        if enrollment:
            parts.append(f"Enrollment: {enrollment} ({enrollment_type}).")

        # Conditions
        conditions = row.get("conditions", "")
        if conditions:
            cond_str = conditions if isinstance(conditions, str) else ", ".join(conditions)
            if cond_str.strip():
                parts.append(f"Conditions: {cond_str}.")

        # Keywords
        keywords = row.get("keywords", "")
        if keywords:
            kw_str = keywords if isinstance(keywords, str) else ", ".join(keywords)
            if kw_str.strip():
                parts.append(f"Keywords: {kw_str}.")

        # Interventions
        interventions = row.get("interventions", row.get("intervention_name", ""))
        if interventions:
            if isinstance(interventions, list):
                for iv in interventions:
                    if isinstance(iv, dict):
                        iv_text = f"Intervention ({iv.get('type', '')}): {iv.get('name', '')}."
                        desc = iv.get("description", "")
                        if desc:
                            iv_text += f" {desc}"
                        parts.append(iv_text)
                    else:
                        parts.append(f"Intervention: {iv}.")
            elif isinstance(interventions, str) and interventions.strip():
                parts.append(f"Intervention: {interventions}.")

        # Eligibility
        criteria = row.get("eligibility_criteria", row.get("eligibilityCriteria", ""))
        if criteria:
            parts.append(str(criteria))

        min_age = row.get("minimum_age", row.get("minimumAge", ""))
        max_age = row.get("maximum_age", row.get("maximumAge", ""))
        sex = row.get("sex", row.get("eligibility_sex", ""))
        if min_age or max_age or (sex and str(sex).upper() != "ALL"):
            elig_parts = []
            if min_age:
                elig_parts.append(f"Minimum age: {min_age}")
            if max_age:
                elig_parts.append(f"Maximum age: {max_age}")
            if sex and str(sex).upper() != "ALL":
                elig_parts.append(f"Sex: {sex}")
            parts.append(". ".join(elig_parts) + ".")

        # Outcomes
        for outcome_key in ["primary_outcomes", "primaryOutcomes"]:
            outcomes = row.get(outcome_key, "")
            if outcomes:
                if isinstance(outcomes, list):
                    for o in outcomes:
                        measure = o.get("measure", o) if isinstance(o, dict) else str(o)
                        if measure:
                            parts.append(f"Primary outcome: {measure}.")
                elif isinstance(outcomes, str) and outcomes.strip():
                    parts.append(f"Primary outcome: {outcomes}.")
                break

        for outcome_key in ["secondary_outcomes", "secondaryOutcomes"]:
            outcomes = row.get(outcome_key, "")
            if outcomes:
                if isinstance(outcomes, list):
                    for o in outcomes:
                        measure = o.get("measure", o) if isinstance(o, dict) else str(o)
                        if measure:
                            parts.append(f"Secondary outcome: {measure}.")
                break

        # Sponsor
        sponsor = row.get("lead_sponsor", row.get("leadSponsor", ""))
        if sponsor:
            sponsor_name = (
                sponsor.get("name", sponsor) if isinstance(sponsor, dict) else str(sponsor)
            )
            if sponsor_name:
                parts.append(f"Lead sponsor: {sponsor_name}.")

        if not parts:
            continue

        yield {
            "text": " ".join(parts),
            "source": DataSource.CTG.value,
            "doc_id": nct_id,
            "date": date_str,
        }


def _pubmed_to_passages(df: pl.DataFrame) -> Iterator[dict[str, str | None]]:
    """Convert PubMed records into passage-level dicts.

    Each article becomes ONE passage: title + abstract.
    """
    for row in df.iter_rows(named=True):
        pmid = str(row.get("pmid", ""))
        pub_date = row.get("publication_date", row.get("pubDate"))
        date_str = str(pub_date) if pub_date is not None else None

        parts: list[str] = []

        title = row.get("title", "")
        if title:
            parts.append(str(title))

        abstract = row.get("abstract", "")
        if abstract:
            parts.append(str(abstract))

        if not parts:
            continue

        yield {
            "text": " ".join(parts),
            "source": DataSource.PUBMED.value,
            "doc_id": pmid,
            "date": date_str,
        }


def _faers_to_passages(df: pl.DataFrame) -> Iterator[dict[str, str | None]]:
    """Convert FAERS adverse event records into passage-level dicts.

    Groups events by drug and produces ONE passage per drug summarizing
    its adverse event profile.
    """
    # Group FAERS records by drug to produce one passage per drug
    drug_events: dict[str, list[dict[str, Any]]] = {}
    drug_dates: dict[str, str | None] = {}

    for row in df.iter_rows(named=True):
        drug = str(row.get("drug_name", row.get("medicinalproduct", "")))
        if not drug:
            continue

        if drug not in drug_events:
            drug_events[drug] = []
            quarter = row.get("report_quarter", row.get("quarter", ""))
            drug_dates[drug] = str(quarter) if quarter is not None else None

        drug_events[drug].append(row)

    for drug, events in drug_events.items():
        parts: list[str] = [f"Adverse event profile for {drug}."]

        for ev in events:
            event = str(ev.get("event", ev.get("reaction", "")))
            count = ev.get("count", ev.get("event_count", ""))
            seriousness = ev.get("seriousness", "")

            if not event:
                continue

            line = f"{event}"
            if count is not None:
                line += f" ({count} reports)"
            if seriousness is not None and seriousness:
                line += f", seriousness: {seriousness}"
            parts.append(line + ".")

        if len(parts) <= 1:
            continue

        yield {
            "text": " ".join(parts),
            "source": DataSource.FAERS.value,
            "doc_id": f"faers-{drug}".replace(" ", "_").lower(),
            "date": drug_dates[drug],
        }


def _aact_to_passages(df: pl.DataFrame) -> Iterator[dict[str, str | None]]:
    """Convert AACT aggregated records into population-level passages.

    Groups by sponsor + condition to produce ONE passage per pair.
    Embeds actual drug names and disease names in the text so that
    scispaCy/GLiNER NER can extract biomedical entities for graph
    connectivity (sponsor names alone are NOT extracted by biomedical NER).
    """
    # Build sponsor-condition groups
    sponsor_col = "sponsor_name"
    condition_col = "conditions"

    if sponsor_col not in df.columns or condition_col not in df.columns:
        logger.warning("AACT DataFrame missing required columns; skipping")
        return

    for key, group_df in df.group_by([sponsor_col, condition_col]):
        sponsor = str(key[0]) if key[0] is not None else ""
        condition = str(key[1]) if key[1] is not None else ""
        if not sponsor or not condition:
            continue

        n_trials = len(group_df)

        # Collect unique interventions for entity extraction
        interventions_col = "interventions"
        drug_names: list[str] = []
        if interventions_col in group_df.columns:
            for row in group_df.iter_rows(named=True):
                ivs = row.get(interventions_col, "")
                if ivs:
                    drug_names.extend(str(ivs).split("|"))
        unique_drugs = list(dict.fromkeys(d.strip() for d in drug_names if d.strip()))[:10]

        # Phase breakdown
        phase_counts: dict[str, int] = {}
        if "phase" in group_df.columns:
            for row in group_df.iter_rows(named=True):
                phase = str(row.get("phase", ""))
                if phase:
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1

        # Success proxy
        completed = 0
        failed = 0
        if "overall_status" in group_df.columns:
            for row in group_df.iter_rows(named=True):
                status = str(row.get("overall_status", "")).lower()
                if status == "completed":
                    completed += 1
                elif status in ("terminated", "withdrawn"):
                    failed += 1
        total_outcome = completed + failed
        success_rate = f"{completed / total_outcome * 100:.0f}%" if total_outcome > 0 else "unknown"

        # Latest date for temporal filtering
        latest_date: str | None = None
        if "start_date" in group_df.columns:
            dates = group_df.get_column("start_date").drop_nulls()
            if len(dates) > 0:
                latest_date = str(dates.max())

        # Build passage text with embedded drug/disease names for NER
        parts: list[str] = [
            f"Clinical trial history for {condition} by {sponsor}.",
            f"{sponsor} has conducted {n_trials} clinical trials for {condition}.",
        ]

        if phase_counts:
            phase_str = ", ".join(f"{k}: {v}" for k, v in sorted(phase_counts.items()))
            parts.append(f"Phase breakdown: {phase_str}.")

        parts.append(f"Success rate: {success_rate}.")

        if unique_drugs:
            parts.append(f"Drug interventions studied: {', '.join(unique_drugs)}.")

        yield {
            "text": " ".join(parts),
            "source": DataSource.AACT.value,
            "doc_id": f"aact-{sponsor}-{condition}".replace(" ", "_").lower(),
            "date": latest_date,
        }


def _chembl_to_passages(df: pl.DataFrame) -> Iterator[dict[str, str | None]]:
    """Convert ChEMBL drug records into passage-level dicts.

    One passage per drug containing mechanism, targets, phase, ATC
    classification, and indications as natural-language text optimized
    for biomedical NER entity extraction.
    """
    for row in df.iter_rows(named=True):
        chembl_id = str(row.get("chembl_id", ""))
        pref_name = str(row.get("pref_name", ""))
        if not pref_name or not chembl_id:
            continue

        mol_type = row.get("molecule_type", "")
        max_phase = row.get("max_phase")
        first_approval = row.get("first_approval")
        date_str = f"{first_approval}-01-01" if first_approval is not None else None

        parts: list[str] = [f"{pref_name} ({chembl_id})."]

        if mol_type:
            parts.append(f"Type: {mol_type}.")

        if max_phase is not None:
            phase_int = int(float(max_phase)) if max_phase else 0
            phase_label = {4: "Approved", 3: "Phase III", 2: "Phase II", 1: "Phase I"}.get(
                phase_int, f"Phase {phase_int}"
            )
            parts.append(f"Max clinical phase: {phase_int} ({phase_label}).")

        if first_approval is not None:
            parts.append(f"First approved: {first_approval}.")

        moa = row.get("mechanism_of_action", "")
        if moa:
            parts.append(f"Mechanism of action: {moa}.")

        action_type = row.get("action_type", "")
        if action_type:
            parts.append(f"Action type: {action_type}.")

        targets = row.get("target_names", "")
        if targets:
            parts.append(f"Targets: {targets}.")

        atc = row.get("atc_codes", "")
        if atc:
            parts.append(f"ATC classification: {atc}.")

        indications = row.get("indication_mesh_headings", "")
        if indications:
            parts.append(f"Indications: {indications}.")

        bbw = row.get("black_box_warning")
        if bbw:
            parts.append("Black box warning: yes.")

        fic = row.get("first_in_class")
        if fic:
            parts.append("First in class: yes.")

        if len(parts) <= 1:
            continue

        yield {
            "text": " ".join(parts),
            "source": DataSource.CHEMBL.value,
            "doc_id": chembl_id,
            "date": date_str,
        }


def _primekg_to_passages(df: pl.DataFrame) -> Iterator[dict[str, str | None]]:
    """Convert PrimeKG relationships into entity-centered passages.

    Filters to edges involving drug or disease nodes, groups by source
    entity, and produces ONE passage per entity with up to 30
    relationships rendered as natural-language sentences.

    Uses ``display_relation`` (human-readable) and full entity names
    for optimal NER recall.  Date is set to ``"2000-01-01"`` because
    biological facts are static knowledge that cannot leak trial outcomes.
    """
    # Filter to edges involving drugs or diseases
    type_col_x = "x_type"
    type_col_y = "y_type"
    if type_col_x not in df.columns or type_col_y not in df.columns:
        logger.warning("PrimeKG DataFrame missing x_type/y_type columns; skipping")
        return

    relevant = df.filter(
        pl.col(type_col_x).is_in(["drug", "disease"])
        | pl.col(type_col_y).is_in(["drug", "disease"])
    )

    if len(relevant) == 0:
        return

    # Group by source entity
    entity_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for row in relevant.iter_rows(named=True):
        x_name = str(row.get("x_name", ""))
        x_type = str(row.get("x_type", ""))
        x_id = str(row.get("x_id", ""))
        if not x_name:
            continue

        key = (x_name, x_type, x_id)
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(row)

    for (x_name, x_type, x_id), edges in entity_groups.items():
        # Cap at 30 relationships per passage
        edges = edges[:30]

        # Build natural-language sentences for each relationship
        parts: list[str] = [f"Biological knowledge for {x_name} ({x_type})."]

        # Group relationships by display_relation for natural sentences
        rel_groups: dict[str, list[str]] = {}
        for edge in edges:
            display_rel = str(edge.get("display_relation", edge.get("relation", "")))
            y_name = str(edge.get("y_name", ""))
            if y_name and display_rel:
                rel_groups.setdefault(display_rel, []).append(y_name)

        for rel, targets in rel_groups.items():
            target_str = ", ".join(targets[:10])
            if rel in ("target", "carrier", "enzyme", "transporter"):
                parts.append(f"{x_name} {rel}s {target_str}.")
            elif rel == "indication":
                parts.append(f"{x_name} is indicated for {target_str}.")
            elif rel == "contraindication":
                parts.append(f"{x_name} is contraindicated in {target_str}.")
            elif rel == "side effect":
                parts.append(f"Known side effects of {x_name} include {target_str}.")
            elif rel == "off-label use":
                parts.append(f"{x_name} has off-label use for {target_str}.")
            elif rel in ("associated with", "linked to"):
                parts.append(f"{x_name} is {rel} {target_str}.")
            elif rel == "interacts with":
                parts.append(f"{x_name} interacts with {target_str}.")
            elif rel in ("phenotype present", "phenotype absent"):
                polarity = "present" if "present" in rel else "absent"
                parts.append(f"Phenotype {polarity} in {x_name}: {target_str}.")
            elif rel in ("expression present", "expression absent"):
                polarity = "expressed" if "present" in rel else "not expressed"
                parts.append(f"{x_name} is {polarity} in {target_str}.")
            else:
                parts.append(f"{x_name} {rel} {target_str}.")

        if len(parts) <= 1:
            continue

        yield {
            "text": " ".join(parts),
            "source": DataSource.PRIMEKG.value,
            "doc_id": f"primekg-{x_type}-{x_id}".replace(" ", "_").replace("/", "-").lower(),
            "date": "2000-01-01",
        }


def _drugsfda_to_passages(df: pl.DataFrame) -> Iterator[dict[str, str | None]]:
    """Convert Drugs@FDA approval records into passage-level dicts.

    One passage per FDA application containing drug name, sponsor,
    approval date, review priority, and classification.
    """
    for row in df.iter_rows(named=True):
        app_num = str(row.get("application_number", ""))
        if not app_num:
            continue

        sponsor = row.get("sponsor_name", "")
        brand = row.get("brand_name", "")
        ingredients = row.get("active_ingredients", "")
        approval_date = row.get("approval_date")
        date_str = str(approval_date) if approval_date is not None else None
        app_type = row.get("application_type", "")
        review = row.get("review_priority", "")
        classification = row.get("submission_class_description", "")
        is_orphan = row.get("is_orphan", False)
        marketing = row.get("marketing_status", "")

        parts: list[str] = []

        if brand and ingredients:
            parts.append(f"FDA approval: {brand} ({ingredients}) by {sponsor}.")
        elif ingredients:
            parts.append(f"FDA approval: {ingredients} by {sponsor}.")
        else:
            parts.append(f"FDA application {app_num} by {sponsor}.")

        parts.append(f"Application {app_num}.")

        if app_type:
            parts.append(f"Application type: {app_type}.")

        if date_str:
            parts.append(f"Approved: {date_str}.")

        if review:
            parts.append(f"Review priority: {review}.")

        if classification:
            parts.append(f"Classification: {classification}.")

        if is_orphan:
            parts.append("Orphan drug designation: yes.")

        if marketing:
            parts.append(f"Marketing status: {marketing}.")

        if len(parts) <= 1:
            continue

        yield {
            "text": " ".join(parts),
            "source": DataSource.DRUGSFDA.value,
            "doc_id": app_num,
            "date": date_str,
        }


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

_PASSAGE_GENERATORS = {
    DataSource.CTG: _ctg_to_passages,
    DataSource.PUBMED: _pubmed_to_passages,
    DataSource.FAERS: _faers_to_passages,
    DataSource.AACT: _aact_to_passages,
    DataSource.CHEMBL: _chembl_to_passages,
    DataSource.PRIMEKG: _primekg_to_passages,
    DataSource.DRUGSFDA: _drugsfda_to_passages,
}


# ---------------------------------------------------------------------------
# IndexBuilder class
# ---------------------------------------------------------------------------


class IndexBuilder:
    """Builds LinearRAG indexes from CTRA data sources.

    Provides both per-source and combined index building methods.  Each method
    reads a Parquet file, converts records into passage-level text chunks,
    and writes the index via :meth:`LinearRAGWrapper.build_index`.

    LinearRAG handles sentence splitting and NER internally — the indexer
    only needs to produce coherent passage-level chunks.

    Args:
        data_config: Override data paths.
        rag_config: Override RAG settings.
    """

    def __init__(
        self,
        data_config: DataConfig | None = None,
        rag_config: RAGConfig | None = None,
        entity_resolver: Any | None = None,
    ) -> None:
        settings = get_settings()
        self._data_cfg = data_config or settings.data
        self._rag_cfg = rag_config or settings.rag
        self._resolver = entity_resolver

    def build_all(
        self,
        sources: list[DataSource] | None = None,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Build a combined index from all specified data sources.

        Args:
            sources: Data sources to include.  Defaults to all configured.
            output_dir: Directory to write the index.  Defaults to ``rag.index_dir``.

        Returns:
            Path to the output directory.
        """
        sources = sources or list(self._rag_cfg.sources)
        out = Path(output_dir or self._rag_cfg.index_dir)

        all_passages = self._collect_passages(sources)

        if not all_passages:
            raise ValueError("No passages generated from any source.  Check data files.")

        logger.info("Total: %d passages from %d sources", len(all_passages), len(sources))

        from ctra.rag.linearrag_wrapper import LinearRAGWrapper

        return LinearRAGWrapper.build_index(
            passages=all_passages,
            output_dir=out,
            rag_config=self._rag_cfg,
        )

    def build_ctg_index(
        self, ctg_parquet_path: str | None = None, output_dir: str | None = None
    ) -> Path:
        """Build a LinearRAG index from ClinicalTrials.gov parquet data."""
        path = Path(ctg_parquet_path or self._data_cfg.ctg_parquet)
        out = Path(output_dir or str(self._rag_cfg.index_dir) + "-ctg")
        return self._build_source_index(DataSource.CTG, path, out)

    def build_pubmed_index(
        self, pubmed_parquet_path: str | None = None, output_dir: str | None = None
    ) -> Path:
        """Build a LinearRAG index from PubMed abstracts."""
        path = Path(pubmed_parquet_path or self._data_cfg.pubmed_parquet)
        out = Path(output_dir or str(self._rag_cfg.index_dir) + "-pubmed")
        return self._build_source_index(DataSource.PUBMED, path, out)

    def build_faers_index(
        self, faers_parquet_path: str | None = None, output_dir: str | None = None
    ) -> Path:
        """Build a LinearRAG index from cached FAERS/OpenFDA data."""
        path = Path(faers_parquet_path or self._data_cfg.faers_parquet)
        out = Path(output_dir or str(self._rag_cfg.index_dir) + "-faers")
        return self._build_source_index(DataSource.FAERS, path, out)

    def add_to_index(
        self,
        new_passages: list[dict[str, str | None]],
        index_dir: Path | str | None = None,
    ) -> Path:
        """Add new passages to an existing index without re-downloading raw data.

        Appends the new passages to ``passages.parquet`` in the index directory,
        then triggers a full LinearRAG graph rebuild.  LinearRAG has no native
        incremental graph update support, so the graph must be rebuilt from
        scratch — but this still saves significant time because the raw data
        has already been downloaded and parsed.

        Args:
            new_passages: List of passage dicts with keys:
                ``text``, ``source``, ``doc_id``, ``date``.
            index_dir: Directory containing the existing index.

        Returns:
            Path to the index directory.
        """
        out = Path(index_dir or self._rag_cfg.index_dir)
        passages_path = out / "passages.parquet"

        if not new_passages:
            logger.warning("add_to_index called with no passages; skipping")
            return out

        new_df = pl.DataFrame(new_passages)

        if passages_path.exists():
            existing_df = pl.read_parquet(passages_path)
            merged_df = pl.concat([existing_df, new_df])

            dedup_cols = ["source", "doc_id", "text"]
            available_cols = [c for c in dedup_cols if c in merged_df.columns]
            if available_cols:
                merged_df = merged_df.unique(subset=available_cols, keep="last")

            logger.info(
                "Merging %d new passages with %d existing -> %d total",
                len(new_df),
                len(existing_df),
                len(merged_df),
            )
        else:
            merged_df = new_df
            logger.info("No existing index; creating with %d passages", len(merged_df))

        out.mkdir(parents=True, exist_ok=True)
        merged_df.write_parquet(passages_path)

        all_passages = merged_df.to_dicts()

        from ctra.rag.linearrag_wrapper import LinearRAGWrapper

        return LinearRAGWrapper.build_index(
            passages=all_passages,
            output_dir=out,
            rag_config=self._rag_cfg,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_source_index(self, source: DataSource, path: Path, output_dir: Path) -> Path:
        """Build a LinearRAG index for a single data source.

        Reads passages from a single source parquet file and invokes
        ``LinearRAGWrapper.build_index`` to construct the full graph-based
        index (embeddings, entity co-occurrence graph, NER results).

        Args:
            source: The data source (CTG, PUBMED, etc.).
            path: Path to the source's parquet file.
            output_dir: Directory to write the index artifacts.

        Returns:
            Path to the output directory.

        Raises:
            ValueError: If the source parquet produces no passages.
        """
        passages = self._collect_passages([source], {source: path})

        if not passages:
            raise ValueError(f"No passages generated from {source.value} at {path}")

        from ctra.rag.linearrag_wrapper import LinearRAGWrapper

        return LinearRAGWrapper.build_index(
            passages=passages,
            output_dir=output_dir,
            rag_config=self._rag_cfg,
        )

    def build_aact_index(
        self, aact_parquet_path: str | None = None, output_dir: str | None = None
    ) -> Path:
        """Build a LinearRAG index from AACT parquet data."""
        path = Path(aact_parquet_path or self._data_cfg.aact_parquet)
        out = Path(output_dir or str(self._rag_cfg.index_dir) + "-aact")
        return self._build_source_index(DataSource.AACT, path, out)

    def build_chembl_index(
        self, chembl_parquet_path: str | None = None, output_dir: str | None = None
    ) -> Path:
        """Build a LinearRAG index from ChEMBL parquet data."""
        path = Path(chembl_parquet_path or self._data_cfg.chembl_parquet)
        out = Path(output_dir or str(self._rag_cfg.index_dir) + "-chembl")
        return self._build_source_index(DataSource.CHEMBL, path, out)

    def build_primekg_index(
        self, primekg_parquet_path: str | None = None, output_dir: str | None = None
    ) -> Path:
        """Build a LinearRAG index from PrimeKG parquet data."""
        path = Path(primekg_parquet_path or self._data_cfg.primekg_parquet)
        out = Path(output_dir or str(self._rag_cfg.index_dir) + "-primekg")
        return self._build_source_index(DataSource.PRIMEKG, path, out)

    def build_drugsfda_index(
        self, drugsfda_parquet_path: str | None = None, output_dir: str | None = None
    ) -> Path:
        """Build a LinearRAG index from Drugs@FDA parquet data."""
        path = Path(drugsfda_parquet_path or self._data_cfg.drugsfda_parquet)
        out = Path(output_dir or str(self._rag_cfg.index_dir) + "-drugsfda")
        return self._build_source_index(DataSource.DRUGSFDA, path, out)

    def _collect_passages(
        self,
        sources: list[DataSource],
        path_overrides: dict[DataSource, Path] | None = None,
    ) -> list[dict[str, str | None]]:
        """Collect passages from the specified data sources via passage generators.

        For each source, reads its parquet file and applies the corresponding
        passage generator (_ctg_to_passages, _pubmed_to_passages, etc.) to
        convert structured records into passage-level text chunks. This
        prepares the data for LinearRAG indexing.

        Args:
            sources: List of data sources to process.
            path_overrides: Optional override paths for specific sources
                (used by per-source index builders).

        Returns:
            Combined list of passage dicts with keys:
            ``text``, ``source``, ``doc_id``, ``date``.
        """
        source_files: dict[DataSource, Path] = {
            DataSource.CTG: Path(self._data_cfg.ctg_parquet),
            DataSource.PUBMED: Path(self._data_cfg.pubmed_parquet),
            DataSource.FAERS: Path(self._data_cfg.faers_parquet),
            DataSource.AACT: Path(self._data_cfg.aact_parquet),
            DataSource.CHEMBL: Path(self._data_cfg.chembl_parquet),
            DataSource.PRIMEKG: Path(self._data_cfg.primekg_parquet),
            DataSource.DRUGSFDA: Path(self._data_cfg.drugsfda_parquet),
        }

        if path_overrides:
            source_files.update(path_overrides)

        all_passages: list[dict[str, str | None]] = []

        for src in sources:
            path = source_files.get(src)
            if path is None or not path.exists():
                logger.warning("Skipping %s: data file not found at %s", src.value, path)
                continue

            logger.info("Processing %s from %s", src.value, path)
            df = pl.read_parquet(path)
            generator = _PASSAGE_GENERATORS.get(src)
            if generator is None:
                logger.warning("No passage generator for source %s", src.value)
                continue

            count_before = len(all_passages)
            all_passages.extend(generator(df))
            logger.info("  -> %d passages from %s", len(all_passages) - count_before, src.value)

        return all_passages


# ---------------------------------------------------------------------------
# Standalone build_index function (backward compatibility)
# ---------------------------------------------------------------------------


def build_index(
    sources: list[DataSource] | None = None,
    output_dir: Path | None = None,
    data_config: DataConfig | None = None,
    rag_config: RAGConfig | None = None,
) -> Path:
    """Build a LinearRAG index from the specified data sources.

    This is a convenience wrapper around :class:`IndexBuilder`.

    Args:
        sources: Data sources to include.  Defaults to all configured.
        output_dir: Directory to write the index.  Defaults to ``rag.index_dir``.
        data_config: Override data paths.
        rag_config: Override RAG settings.

    Returns:
        Path to the output directory.
    """
    builder = IndexBuilder(data_config=data_config, rag_config=rag_config)
    return builder.build_all(sources=sources, output_dir=output_dir)
