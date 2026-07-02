"""Internal trial data alignment layer.

Maps internal (e.g. Merck) trial records to ClinicalTrials.gov
``protocolSection`` format so the same LLM agents, polars queries, and
LinearRAG indexes can process both public and proprietary trials.

The alignment is a *mapping*, not a transformation -- it renames fields and
restructures the nesting but does not alter content.  Missing fields produce
warnings, not errors, because AutoCT's FeatureBuilder already handles
``None`` gracefully.

Typical usage::

    from ctra.data.alignment import TrialAligner

    aligner = TrialAligner()  # uses default field mapping
    protocol = aligner.align(internal_record)
    warnings = aligner.validate(protocol)
    aligner.to_parquet([internal_record], "datasets/internal-trials.parquet")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from ctra.data.trial_schema import (
    ArmGroup,
    ArmsInterventionsModule,
    ConditionsModule,
    ContactsLocationsModule,
    DescriptionModule,
    DesignInfo,
    DesignModule,
    EligibilityModule,
    EnrollmentInfo,
    IdentificationModule,
    Intervention,
    Location,
    OutcomeMeasure,
    OutcomesModule,
    ProtocolSection,
    SponsorCollaboratorsModule,
    StatusModule,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default field mapping: internal field name -> protocolSection dot-path
# ---------------------------------------------------------------------------

DEFAULT_FIELD_MAPPING: dict[str, str] = {
    # Identification
    "trial_id": "identificationModule.nctId",
    "trial_title": "identificationModule.briefTitle",
    "official_title": "identificationModule.officialTitle",
    "organization": "identificationModule.organization",
    "org_study_id": "identificationModule.orgStudyId",
    # Status
    "start_date": "statusModule.startDate",
    "start_dt": "statusModule.startDate",
    "first_post_date": "statusModule.studyFirstPostDate",
    # Design
    "trial_phase": "designModule.phases",
    "phase": "designModule.phases",
    "study_type": "designModule.studyType",
    "allocation": "designModule.designInfo.allocation",
    "intervention_model": "designModule.designInfo.interventionModel",
    "primary_purpose": "designModule.designInfo.primaryPurpose",
    "masking": "designModule.designInfo.masking",
    "enrollment": "designModule.enrollmentInfo.count",
    "enrollment_count": "designModule.enrollmentInfo.count",
    "enrollment_type": "designModule.enrollmentInfo.type",
    "number_of_arms": "designModule.numberOfArms",
    # Conditions
    "conditions": "conditionsModule.conditions",
    "indication": "conditionsModule.conditions",
    "therapeutic_area": "conditionsModule.keywords",
    # Eligibility
    "eligibility_criteria": "eligibilityModule.eligibilityCriteria",
    "inclusion_exclusion": "eligibilityModule.eligibilityCriteria",
    "sex": "eligibilityModule.sex",
    "gender": "eligibilityModule.sex",
    "minimum_age": "eligibilityModule.minimumAge",
    "min_age": "eligibilityModule.minimumAge",
    "maximum_age": "eligibilityModule.maximumAge",
    "max_age": "eligibilityModule.maximumAge",
    # Arms & Interventions
    "arm_groups": "armsInterventionsModule.armGroups",
    "treatment_arms": "armsInterventionsModule.armGroups",
    "interventions": "armsInterventionsModule.interventions",
    "drug_name": "armsInterventionsModule.interventions",
    # Outcomes
    "primary_endpoints": "outcomesModule.primaryOutcomes",
    "primary_outcomes": "outcomesModule.primaryOutcomes",
    "secondary_endpoints": "outcomesModule.secondaryOutcomes",
    "secondary_outcomes": "outcomesModule.secondaryOutcomes",
    # Locations
    "sites": "contactsLocationsModule.locations",
    "locations": "contactsLocationsModule.locations",
    # Description
    "brief_summary": "descriptionModule.briefSummary",
    "protocol_synopsis": "descriptionModule.briefSummary",
    "detailed_description": "descriptionModule.detailedDescription",
    # Sponsor
    "sponsor": "sponsorCollaboratorsModule.leadSponsorName",
    "lead_sponsor": "sponsorCollaboratorsModule.leadSponsorName",
    "collaborators": "sponsorCollaboratorsModule.collaboratorNames",
}


class TrialAligner:
    """Aligns internal trial records to ClinicalTrials.gov protocolSection format.

    Transforms internal (e.g., Merck) trial data into ClinicalTrials.gov's canonical
    nested structure so the same LLM agents, polars queries, and LinearRAG indexes
    can process both public and proprietary trials.  The alignment is a *mapping*,
    not a transformation -- it renames and restructures without altering content.

    Args:
        field_mapping: Maps internal field names to protocolSection dot-paths.
            If ``None``, uses :data:`DEFAULT_FIELD_MAPPING`.
    """

    def __init__(self, field_mapping: dict[str, str] | None = None) -> None:
        """Initialize the aligner with a field mapping.

        Args:
            field_mapping: Custom field mapping dict, or None to use the default.
        """
        self.field_mapping = field_mapping or DEFAULT_FIELD_MAPPING

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self, internal_record: dict[str, Any]) -> ProtocolSection:
        """Convert a single internal trial record to ProtocolSection format.

        Args:
            internal_record: Flat dict with internal field names.

        Returns:
            A validated :class:`ProtocolSection`.

        Raises:
            ValueError: If the record is missing the trial ID entirely.
        """
        mapped = self._apply_mapping(internal_record)
        return self._build_protocol_section(mapped)

    def align_batch(self, records: list[dict[str, Any]] | pl.DataFrame) -> pl.DataFrame:
        """Align a batch of internal trial records.

        Args:
            records: List of dicts or a DataFrame with internal field names.

        Returns:
            DataFrame where each row is a JSON-serialized CTG-compatible
            ``protocolSection`` dict, plus a ``trial_id`` column for indexing.
        """
        if isinstance(records, pl.DataFrame):
            records = records.to_dicts()

        rows: list[dict[str, Any]] = []
        for i, rec in enumerate(records):
            try:
                protocol = self.align(rec)
                ctg_dict = protocol.to_ctg_dict()
                rows.append(
                    {
                        "trial_id": protocol.identification_module.nct_id,
                        "protocolSection": json.dumps(ctg_dict),
                        **self._flatten_for_parquet(ctg_dict),
                    }
                )
            except Exception:
                logger.warning("Failed to align record %d: %s", i, rec, exc_info=True)

        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def validate(self, aligned: ProtocolSection) -> list[str]:
        """Validate an aligned record and return a list of warnings.

        Warnings indicate missing or unusual values that may affect
        downstream feature extraction quality.  They do **not** prevent
        the record from being used.

        Args:
            aligned: A :class:`ProtocolSection` instance.

        Returns:
            List of human-readable warning strings.
        """
        warnings: list[str] = []
        tid = aligned.identification_module.nct_id

        if not aligned.identification_module.brief_title:
            warnings.append(f"[{tid}] Missing briefTitle")

        if not aligned.status_module.start_date:
            warnings.append(f"[{tid}] Missing startDate — temporal RAG filtering will not work")

        if not aligned.design_module.phases:
            warnings.append(
                f"[{tid}] Missing phases — phase-stratified evaluation will skip this trial"
            )

        if not aligned.design_module.study_type:
            warnings.append(f"[{tid}] Missing studyType")

        if not aligned.eligibility_module.eligibility_criteria:
            warnings.append(
                f"[{tid}] Missing eligibilityCriteria — eligibility-based features will be empty"
            )

        if not aligned.arms_interventions_module.interventions:
            warnings.append(f"[{tid}] No interventions — drug-based features will be empty")

        if not aligned.outcomes_module.primary_outcomes:
            warnings.append(f"[{tid}] No primary outcomes defined")

        if not aligned.description_module.brief_summary:
            warnings.append(f"[{tid}] Missing briefSummary")

        if not aligned.conditions_module.conditions:
            warnings.append(f"[{tid}] No conditions listed")

        return warnings

    def to_parquet(
        self,
        records: list[dict[str, Any]] | pl.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """Align records and write to Parquet in CTG-compatible format.

        The output Parquet can be queried by the same polars queries that
        load public CTG data.

        Args:
            records: Internal trial records.
            output_path: Destination Parquet file path.

        Returns:
            The resolved output path.
        """
        df = self.align_batch(records)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        logger.info("Wrote %d aligned trial records to %s", len(df), out)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_mapping(self, internal_record: dict[str, Any]) -> dict[str, Any]:
        """Apply the field mapping to rearrange an internal record into
        a nested dict whose keys correspond to protocolSection paths.

        Returns a nested dict keyed by dot-path segments.
        """
        mapped: dict[str, Any] = {}
        unmapped_keys: list[str] = []

        for internal_key, value in internal_record.items():
            dot_path = self.field_mapping.get(internal_key)
            if dot_path is None:
                unmapped_keys.append(internal_key)
                continue
            self._set_nested(mapped, dot_path, value)

        if unmapped_keys:
            logger.debug("Unmapped internal fields (ignored): %s", unmapped_keys)

        return mapped

    @staticmethod
    def _set_nested(d: dict[str, Any], dot_path: str, value: Any) -> None:
        """Set a value in a nested dict using a dot-separated path.

        If the path already has a value, the new value is kept (first
        mapping wins).
        """
        keys = dot_path.split(".")
        current = d
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        leaf = keys[-1]
        if leaf not in current:
            current[leaf] = value

    def _build_protocol_section(self, mapped: dict[str, Any]) -> ProtocolSection:
        """Build a :class:`ProtocolSection` from the nested mapped dict."""

        id_raw = mapped.get("identificationModule", {})
        status_raw = mapped.get("statusModule", {})
        design_raw = mapped.get("designModule", {})
        di_raw = design_raw.get("designInfo", {})
        ei_raw = design_raw.get("enrollmentInfo", {})
        cond_raw = mapped.get("conditionsModule", {})
        elig_raw = mapped.get("eligibilityModule", {})
        arms_raw = mapped.get("armsInterventionsModule", {})
        outcomes_raw = mapped.get("outcomesModule", {})
        contacts_raw = mapped.get("contactsLocationsModule", {})
        desc_raw = mapped.get("descriptionModule", {})
        sponsor_raw = mapped.get("sponsorCollaboratorsModule", {})

        trial_id = id_raw.get("nctId", "")
        if not trial_id:
            raise ValueError(
                "Internal record has no trial ID.  Ensure the mapping "
                "includes a field that maps to 'identificationModule.nctId'."
            )

        # Normalize phases: accept a single string or a list
        phases_val = design_raw.get("phases", [])
        if isinstance(phases_val, str):
            phases_val = [self._normalize_phase(phases_val)]
        elif isinstance(phases_val, list):
            phases_val = [self._normalize_phase(p) for p in phases_val]

        return ProtocolSection(
            identification_module=IdentificationModule(
                nct_id=trial_id,
                brief_title=id_raw.get("briefTitle", ""),
                official_title=id_raw.get("officialTitle"),
                organization=id_raw.get("organization"),
                org_study_id=id_raw.get("orgStudyId"),
            ),
            status_module=StatusModule(
                start_date=self._to_str(status_raw.get("startDate")),
                study_first_post_date=self._to_str(status_raw.get("studyFirstPostDate")),
            ),
            design_module=DesignModule(
                phases=phases_val,
                study_type=self._normalize_study_type(design_raw.get("studyType", "")),
                design_info=DesignInfo(
                    allocation=di_raw.get("allocation"),
                    intervention_model=di_raw.get("interventionModel"),
                    primary_purpose=di_raw.get("primaryPurpose"),
                    masking=di_raw.get("masking"),
                )
                if di_raw
                else None,
                enrollment_info=EnrollmentInfo(
                    count=self._to_int(ei_raw.get("count")),
                    type=ei_raw.get("type"),
                )
                if ei_raw
                else None,
                number_of_arms=self._to_int(design_raw.get("numberOfArms")),
            ),
            conditions_module=ConditionsModule(
                conditions=self._ensure_list(cond_raw.get("conditions", [])),
                keywords=self._ensure_list(cond_raw.get("keywords", [])),
            ),
            eligibility_module=EligibilityModule(
                eligibility_criteria=elig_raw.get("eligibilityCriteria", ""),
                sex=elig_raw.get("sex"),
                minimum_age=self._to_str(elig_raw.get("minimumAge")),
                maximum_age=self._to_str(elig_raw.get("maximumAge")),
            ),
            arms_interventions_module=ArmsInterventionsModule(
                arm_groups=self._parse_arm_groups(arms_raw.get("armGroups", [])),
                interventions=self._parse_interventions(arms_raw.get("interventions", [])),
            ),
            outcomes_module=OutcomesModule(
                primary_outcomes=self._parse_outcomes(outcomes_raw.get("primaryOutcomes", [])),
                secondary_outcomes=self._parse_outcomes(outcomes_raw.get("secondaryOutcomes", [])),
            ),
            contacts_locations_module=ContactsLocationsModule(
                locations=self._parse_locations(contacts_raw.get("locations", [])),
            ),
            description_module=DescriptionModule(
                brief_summary=desc_raw.get("briefSummary", ""),
                detailed_description=desc_raw.get("detailedDescription"),
            ),
            sponsor_collaborators_module=SponsorCollaboratorsModule(
                lead_sponsor_name=sponsor_raw.get("leadSponsorName"),
                collaborator_names=self._ensure_list(sponsor_raw.get("collaboratorNames", [])),
            ),
        )

    # ------------------------------------------------------------------
    # Value normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_phase(raw: str) -> str:
        """Normalize phase strings to CTG format (e.g. 'Phase 2' -> 'PHASE2')."""
        if not raw:
            return ""
        cleaned = raw.upper().replace(" ", "").replace("-", "").replace("_", "")
        # Handle common patterns
        mapping = {
            "PHASE1": "PHASE1",
            "PHASEI": "PHASE1",
            "P1": "PHASE1",
            "1": "PHASE1",
            "PHASE2": "PHASE2",
            "PHASEII": "PHASE2",
            "P2": "PHASE2",
            "2": "PHASE2",
            "PHASE3": "PHASE3",
            "PHASEIII": "PHASE3",
            "P3": "PHASE3",
            "3": "PHASE3",
            "PHASE4": "PHASE4",
            "PHASEIV": "PHASE4",
            "P4": "PHASE4",
            "4": "PHASE4",
            "EARLYP1": "EARLY_PHASE1",
            "EARLYPHASE1": "EARLY_PHASE1",
        }
        return mapping.get(cleaned, raw.upper())

    @staticmethod
    def _normalize_study_type(raw: str) -> str:
        """Normalize study type to CTG format."""
        if not raw:
            return ""
        cleaned = raw.upper().strip()
        mapping = {
            "INTERVENTIONAL": "INTERVENTIONAL",
            "OBSERVATIONAL": "OBSERVATIONAL",
            "EXPANDED_ACCESS": "EXPANDED_ACCESS",
            "INT": "INTERVENTIONAL",
            "OBS": "OBSERVATIONAL",
        }
        return mapping.get(cleaned, cleaned)

    @staticmethod
    def _to_str(value: Any) -> str | None:
        """Convert a value to string, returning None for empty/None."""
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    @staticmethod
    def _to_int(value: Any) -> int | None:
        """Convert a value to int, returning None on failure."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _ensure_list(value: Any) -> list[Any]:
        """Ensure a value is a list (wrap scalars)."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _parse_arm_groups(raw: Any) -> list[ArmGroup]:
        """Parse arm groups from various internal formats."""
        if not raw:
            return []
        items = raw if isinstance(raw, list) else [raw]
        result: list[ArmGroup] = []
        for item in items:
            if isinstance(item, dict):
                result.append(
                    ArmGroup(
                        label=item.get("label", item.get("name", "")),
                        type=item.get("type"),
                        description=item.get("description"),
                    )
                )
            elif isinstance(item, str):
                result.append(ArmGroup(label=item))
        return result

    @staticmethod
    def _parse_interventions(raw: Any) -> list[Intervention]:
        """Parse interventions from various internal formats.

        Accepts:
        - list of dicts with type/name/description
        - list of strings (treated as drug names)
        - single string (treated as a single drug name)
        """
        if not raw:
            return []
        items = raw if isinstance(raw, list) else [raw]
        result: list[Intervention] = []
        for item in items:
            if isinstance(item, dict):
                result.append(
                    Intervention(
                        type=item.get("type", "DRUG"),
                        name=item.get("name", ""),
                        description=item.get("description"),
                    )
                )
            elif isinstance(item, str):
                result.append(Intervention(type="DRUG", name=item))
        return result

    @staticmethod
    def _parse_outcomes(raw: Any) -> list[OutcomeMeasure]:
        """Parse outcome measures from various internal formats."""
        if not raw:
            return []
        items = raw if isinstance(raw, list) else [raw]
        result: list[OutcomeMeasure] = []
        for item in items:
            if isinstance(item, dict):
                result.append(
                    OutcomeMeasure(
                        measure=item.get("measure", item.get("name", "")),
                        description=item.get("description"),
                        time_frame=item.get("time_frame", item.get("timeFrame")),
                    )
                )
            elif isinstance(item, str):
                result.append(OutcomeMeasure(measure=item))
        return result

    @staticmethod
    def _parse_locations(raw: Any) -> list[Location]:
        """Parse locations from various internal formats."""
        if not raw:
            return []
        items = raw if isinstance(raw, list) else [raw]
        result: list[Location] = []
        for item in items:
            if isinstance(item, dict):
                result.append(
                    Location(
                        facility=item.get("facility", item.get("site_name")),
                        city=item.get("city"),
                        state=item.get("state"),
                        country=item.get("country"),
                    )
                )
            elif isinstance(item, str):
                result.append(Location(facility=item))
        return result

    @staticmethod
    def _flatten_for_parquet(ctg_dict: dict[str, Any]) -> dict[str, Any]:
        """Extract commonly-queried scalar fields from a CTG dict.

        These flat columns sit alongside the full JSON ``protocolSection``
        column so that polars queries can filter efficiently without parsing
        the JSON blob.
        """
        id_mod = ctg_dict.get("identificationModule", {})
        status_mod = ctg_dict.get("statusModule", {})
        design_mod = ctg_dict.get("designModule", {})

        phases = design_mod.get("phases", [])

        return {
            "nctId": id_mod.get("nctId", ""),
            "briefTitle": id_mod.get("briefTitle", ""),
            "startDate": (status_mod.get("startDateStruct", {}).get("date", "")),
            "studyType": design_mod.get("studyType", ""),
            "phases": json.dumps(phases),
        }
