"""ClinicalTrials.gov protocolSection schema and alignment.

Defines the canonical trial record schema used throughout CTRA.  Internal
Merck trial data and ClinicalTrials.gov records are aligned to this schema
before entering the pipeline.

The schema maps to ClinicalTrials.gov API v2's ``protocolSection`` structure.
The ``resultsSection`` is deliberately excluded to prevent label leakage.

Two schema families coexist:

1. **TrialRecord** (dataclass) -- flat representation used by the feature
   builder and RAG layer.  All data sources are flattened into this format.

2. **ProtocolSection** (Pydantic) -- nested representation that mirrors the
   ClinicalTrials.gov API v2 JSON structure.  Used by the alignment layer to
   validate internal trial data before flattening to ``TrialRecord``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models -- ClinicalTrials.gov protocolSection (nested)
# ---------------------------------------------------------------------------


class IdentificationModule(BaseModel):
    """``protocolSection.identificationModule``."""

    nct_id: str = Field(
        ...,
        description="NCT ID for public trials or internal trial ID.",
    )
    brief_title: str = Field(default="", description="Short public title.")
    official_title: str | None = Field(default=None, description="Full scientific title.")
    organization: str | None = Field(default=None, description="Responsible organization name.")
    org_study_id: str | None = Field(default=None, description="Organization study identifier.")


class StatusModule(BaseModel):
    """``protocolSection.statusModule``.

    Only temporal fields are retained.  ``overallStatus`` and
    ``completionDate`` are excluded to prevent label leakage.
    """

    start_date: str | None = Field(
        default=None,
        description="Trial start date (YYYY-MM-DD or partial, e.g. 2023-01).",
    )
    study_first_post_date: str | None = Field(
        default=None,
        description="Date the study record was first posted on CTG.",
    )


class DesignInfo(BaseModel):
    """``protocolSection.designModule.designInfo``."""

    allocation: str | None = Field(default=None, description="RANDOMIZED | NON_RANDOMIZED")
    intervention_model: str | None = Field(
        default=None,
        description="PARALLEL | CROSSOVER | SEQUENTIAL | SINGLE_GROUP | FACTORIAL",
    )
    primary_purpose: str | None = Field(
        default=None,
        description="TREATMENT | PREVENTION | DIAGNOSTIC | etc.",
    )
    masking: str | None = Field(
        default=None, description="NONE | SINGLE | DOUBLE | TRIPLE | QUADRUPLE"
    )


class EnrollmentInfo(BaseModel):
    """``protocolSection.designModule.enrollmentInfo``."""

    count: int | None = Field(default=None, description="Number of participants.")
    type: str | None = Field(default=None, description="ACTUAL | ESTIMATED")


class DesignModule(BaseModel):
    """``protocolSection.designModule``."""

    phases: list[str] = Field(
        default_factory=list,
        description='Trial phases, e.g. ["PHASE2"].',
    )
    study_type: str = Field(
        default="",
        description="INTERVENTIONAL | OBSERVATIONAL",
    )
    design_info: DesignInfo | None = Field(default=None)
    enrollment_info: EnrollmentInfo | None = Field(default=None)
    number_of_arms: int | None = Field(default=None)


class EligibilityModule(BaseModel):
    """``protocolSection.eligibilityModule``."""

    eligibility_criteria: str = Field(
        default="", description="Free-text inclusion/exclusion criteria."
    )
    sex: str | None = Field(default=None, description="ALL | FEMALE | MALE")
    minimum_age: str | None = Field(default=None, description='E.g. "18 Years".')
    maximum_age: str | None = Field(default=None, description='E.g. "65 Years".')


class ArmGroup(BaseModel):
    """A single arm in ``armsInterventionsModule.armGroups``."""

    label: str = Field(default="", description="Arm label.")
    type: str | None = Field(
        default=None,
        description="EXPERIMENTAL | ACTIVE_COMPARATOR | PLACEBO_COMPARATOR | etc.",
    )
    description: str | None = Field(default=None)


class Intervention(BaseModel):
    """A single intervention in ``armsInterventionsModule.interventions``."""

    type: str = Field(default="", description="DRUG | BIOLOGICAL | DEVICE | etc.")
    name: str = Field(default="", description="Intervention name.")
    description: str | None = Field(default=None)


class ArmsInterventionsModule(BaseModel):
    """``protocolSection.armsInterventionsModule``."""

    arm_groups: list[ArmGroup] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)


class OutcomeMeasure(BaseModel):
    """A single outcome measure."""

    measure: str = Field(default="", description="Outcome measure text.")
    description: str | None = Field(default=None)
    time_frame: str | None = Field(default=None)


class OutcomesModule(BaseModel):
    """``protocolSection.outcomesModule``."""

    primary_outcomes: list[OutcomeMeasure] = Field(default_factory=list)
    secondary_outcomes: list[OutcomeMeasure] = Field(default_factory=list)


class Location(BaseModel):
    """A single study site."""

    facility: str | None = Field(default=None, description="Site name.")
    city: str | None = Field(default=None)
    state: str | None = Field(default=None)
    country: str | None = Field(default=None)


class ContactsLocationsModule(BaseModel):
    """``protocolSection.contactsLocationsModule``."""

    locations: list[Location] = Field(default_factory=list)


class ConditionsModule(BaseModel):
    """``protocolSection.conditionsModule``."""

    conditions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class DescriptionModule(BaseModel):
    """``protocolSection.descriptionModule``."""

    brief_summary: str = Field(default="", description="Protocol synopsis.")
    detailed_description: str | None = Field(default=None)


class SponsorCollaboratorsModule(BaseModel):
    """``protocolSection.sponsorCollaboratorsModule``."""

    lead_sponsor_name: str | None = Field(default=None)
    collaborator_names: list[str] = Field(default_factory=list)


class ProtocolSection(BaseModel):
    """ClinicalTrials.gov ``protocolSection`` schema.

    This is the canonical nested format for all trial data in CTRA.  Both
    public ClinicalTrials.gov records and internal Merck trials are aligned
    to this structure before entering the pipeline.

    The ``resultsSection`` is deliberately excluded to prevent label leakage.
    """

    identification_module: IdentificationModule
    status_module: StatusModule = Field(default_factory=StatusModule)
    design_module: DesignModule = Field(default_factory=DesignModule)
    conditions_module: ConditionsModule = Field(default_factory=ConditionsModule)
    eligibility_module: EligibilityModule = Field(default_factory=EligibilityModule)
    arms_interventions_module: ArmsInterventionsModule = Field(
        default_factory=ArmsInterventionsModule
    )
    outcomes_module: OutcomesModule = Field(default_factory=OutcomesModule)
    contacts_locations_module: ContactsLocationsModule = Field(
        default_factory=ContactsLocationsModule
    )
    description_module: DescriptionModule = Field(default_factory=DescriptionModule)
    sponsor_collaborators_module: SponsorCollaboratorsModule = Field(
        default_factory=SponsorCollaboratorsModule
    )

    # --- conversion helpers ---

    def to_trial_record(self, label: int | None = None) -> TrialRecord:
        """Flatten this nested ProtocolSection into a :class:`TrialRecord`.

        Args:
            label: Optional outcome label (0=failure, 1=success).

        Returns:
            A flat :class:`TrialRecord` suitable for the feature builder.
        """
        dm = self.design_module
        di = dm.design_info or DesignInfo()
        ei = dm.enrollment_info or EnrollmentInfo()

        def _parse_date(val: str | None) -> date | None:
            """Parse an ISO date string to a date object.

            Parameters:
                val: ISO date string (e.g., "2025-04-05") or None.

            Returns:
                date object, or None if parsing fails.
            """
            if not val:
                return None
            try:
                return date.fromisoformat(str(val)[:10])
            except Exception:
                return None

        return TrialRecord(
            nct_id=self.identification_module.nct_id,
            brief_title=self.identification_module.brief_title,
            official_title=self.identification_module.official_title or "",
            org_study_id=self.identification_module.org_study_id or "",
            start_date=_parse_date(self.status_module.start_date),
            completion_date=None,  # excluded — leakage risk
            study_first_post_date=_parse_date(self.status_module.study_first_post_date),
            study_type=dm.study_type,
            phase=dm.phases[0] if dm.phases else "",
            allocation=di.allocation or "",
            masking=di.masking or "",
            primary_purpose=di.primary_purpose or "",
            enrollment_count=ei.count,
            enrollment_type=ei.type or "",
            number_of_arms=dm.number_of_arms,
            conditions=self.conditions_module.conditions,
            interventions=[
                {
                    "type": iv.type,
                    "name": iv.name,
                    "description": iv.description or "",
                }
                for iv in self.arms_interventions_module.interventions
            ],
            eligibility_criteria=self.eligibility_module.eligibility_criteria,
            min_age=self.eligibility_module.minimum_age or "",
            max_age=self.eligibility_module.maximum_age or "",
            sex=self.eligibility_module.sex or "ALL",
            primary_outcomes=[o.measure for o in self.outcomes_module.primary_outcomes],
            secondary_outcomes=[o.measure for o in self.outcomes_module.secondary_outcomes],
            lead_sponsor=self.sponsor_collaborators_module.lead_sponsor_name or "",
            collaborators=self.sponsor_collaborators_module.collaborator_names,
            brief_summary=self.description_module.brief_summary,
            detailed_description=self.description_module.detailed_description or "",
            label=label,
        )

    def to_ctg_dict(self) -> dict[str, Any]:
        """Serialize to a ClinicalTrials.gov API v2-compatible nested dict.

        Useful for writing aligned internal trial data to Parquet in a format
        that polars queries (and ``flatten_protocol_to_sentences``) can
        consume identically to public CTG data.
        """
        dm = self.design_module
        di = dm.design_info
        ei = dm.enrollment_info

        result: dict[str, Any] = {
            "identificationModule": {
                "nctId": self.identification_module.nct_id,
                "briefTitle": self.identification_module.brief_title,
                "officialTitle": self.identification_module.official_title or "",
                "orgStudyIdInfo": {"id": self.identification_module.org_study_id or ""},
            },
            "statusModule": {
                "startDateStruct": {"date": self.status_module.start_date or ""},
                "studyFirstPostDateStruct": {
                    "date": self.status_module.study_first_post_date or ""
                },
            },
            "designModule": {
                "phases": dm.phases,
                "studyType": dm.study_type,
                "numberOfArms": dm.number_of_arms,
                "designInfo": {
                    "allocation": di.allocation or "" if di else "",
                    "primaryPurpose": di.primary_purpose or "" if di else "",
                    "maskingInfo": {
                        "masking": di.masking or "" if di else "",
                    },
                    "interventionModel": (di.intervention_model or "" if di else ""),
                },
                "enrollmentInfo": {
                    "count": ei.count if ei else None,
                    "type": ei.type or "" if ei else "",
                },
            },
            "conditionsModule": {
                "conditions": self.conditions_module.conditions,
                "keywords": self.conditions_module.keywords,
            },
            "eligibilityModule": {
                "eligibilityCriteria": (self.eligibility_module.eligibility_criteria),
                "sex": self.eligibility_module.sex or "ALL",
                "minimumAge": self.eligibility_module.minimum_age or "",
                "maximumAge": self.eligibility_module.maximum_age or "",
            },
            "armsInterventionsModule": {
                "armGroups": [
                    {
                        "label": ag.label,
                        "type": ag.type or "",
                        "description": ag.description or "",
                    }
                    for ag in self.arms_interventions_module.arm_groups
                ],
                "interventions": [
                    {
                        "type": iv.type,
                        "name": iv.name,
                        "description": iv.description or "",
                    }
                    for iv in self.arms_interventions_module.interventions
                ],
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": o.measure,
                        "description": o.description or "",
                        "timeFrame": o.time_frame or "",
                    }
                    for o in self.outcomes_module.primary_outcomes
                ],
                "secondaryOutcomes": [
                    {
                        "measure": o.measure,
                        "description": o.description or "",
                        "timeFrame": o.time_frame or "",
                    }
                    for o in self.outcomes_module.secondary_outcomes
                ],
            },
            "contactsLocationsModule": {
                "locations": [
                    {
                        "facility": loc.facility or "",
                        "city": loc.city or "",
                        "state": loc.state or "",
                        "country": loc.country or "",
                    }
                    for loc in self.contacts_locations_module.locations
                ],
            },
            "descriptionModule": {
                "briefSummary": self.description_module.brief_summary,
                "detailedDescription": (self.description_module.detailed_description or ""),
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {
                    "name": (self.sponsor_collaborators_module.lead_sponsor_name or ""),
                },
                "collaborators": [
                    {"name": c} for c in self.sponsor_collaborators_module.collaborator_names
                ],
            },
        }
        return result

    @classmethod
    def from_ctg_dict(cls, raw: dict[str, Any]) -> ProtocolSection:
        """Parse a ClinicalTrials.gov API v2 ``protocolSection`` dict.

        This is the inverse of :meth:`to_ctg_dict`.  It tolerates missing
        keys and returns a validated ``ProtocolSection`` instance.

        Args:
            raw: Nested dict from the CTG API v2 ``protocolSection``.

        Returns:
            A validated :class:`ProtocolSection`.
        """

        def _get(d: dict[str, Any], *keys: str, default: Any = "") -> Any:
            """Recursively get a nested value from a dict by key path.

            Parameters:
                d: Dictionary to traverse.
                *keys: Sequence of keys to follow (e.g., "a", "b", "c").
                default: Value to return if any key is missing.

            Returns:
                The nested value or the default if traversal fails.
            """
            current: Any = d
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key, {})
                else:
                    return default
            return current if current != {} else default

        id_mod = raw.get("identificationModule", {})
        status_mod = raw.get("statusModule", {})
        design_mod = raw.get("designModule", {})
        design_info_raw = design_mod.get("designInfo", {})
        enrollment_raw = design_mod.get("enrollmentInfo", {})
        conditions_mod = raw.get("conditionsModule", {})
        elig_mod = raw.get("eligibilityModule", {})
        arms_mod = raw.get("armsInterventionsModule", {})
        outcomes_mod = raw.get("outcomesModule", {})
        contacts_mod = raw.get("contactsLocationsModule", {})
        desc_mod = raw.get("descriptionModule", {})
        sponsor_mod = raw.get("sponsorCollaboratorsModule", {})

        return cls(
            identification_module=IdentificationModule(
                nct_id=id_mod.get("nctId", ""),
                brief_title=id_mod.get("briefTitle", ""),
                official_title=id_mod.get("officialTitle"),
                organization=None,
                org_study_id=_get(id_mod, "orgStudyIdInfo", "id") or None,
            ),
            status_module=StatusModule(
                start_date=_get(status_mod, "startDateStruct", "date") or None,
                study_first_post_date=(
                    _get(status_mod, "studyFirstPostDateStruct", "date") or None
                ),
            ),
            design_module=DesignModule(
                phases=design_mod.get("phases", []),
                study_type=design_mod.get("studyType", ""),
                design_info=DesignInfo(
                    allocation=design_info_raw.get("allocation"),
                    intervention_model=design_info_raw.get("interventionModel"),
                    primary_purpose=design_info_raw.get("primaryPurpose"),
                    masking=_get(design_info_raw, "maskingInfo", "masking") or None,
                )
                if design_info_raw
                else None,
                enrollment_info=EnrollmentInfo(
                    count=enrollment_raw.get("count"),
                    type=enrollment_raw.get("type"),
                )
                if enrollment_raw
                else None,
                number_of_arms=design_mod.get("numberOfArms"),
            ),
            conditions_module=ConditionsModule(
                conditions=conditions_mod.get("conditions", []),
                keywords=conditions_mod.get("keywords", []),
            ),
            eligibility_module=EligibilityModule(
                eligibility_criteria=elig_mod.get("eligibilityCriteria", ""),
                sex=elig_mod.get("sex"),
                minimum_age=elig_mod.get("minimumAge"),
                maximum_age=elig_mod.get("maximumAge"),
            ),
            arms_interventions_module=ArmsInterventionsModule(
                arm_groups=[
                    ArmGroup(
                        label=ag.get("label", ""),
                        type=ag.get("type"),
                        description=ag.get("description"),
                    )
                    for ag in arms_mod.get("armGroups", [])
                ],
                interventions=[
                    Intervention(
                        type=iv.get("type", ""),
                        name=iv.get("name", ""),
                        description=iv.get("description"),
                    )
                    for iv in arms_mod.get("interventions", [])
                ],
            ),
            outcomes_module=OutcomesModule(
                primary_outcomes=[
                    OutcomeMeasure(
                        measure=o.get("measure", ""),
                        description=o.get("description"),
                        time_frame=o.get("timeFrame"),
                    )
                    for o in outcomes_mod.get("primaryOutcomes", [])
                ],
                secondary_outcomes=[
                    OutcomeMeasure(
                        measure=o.get("measure", ""),
                        description=o.get("description"),
                        time_frame=o.get("timeFrame"),
                    )
                    for o in outcomes_mod.get("secondaryOutcomes", [])
                ],
            ),
            contacts_locations_module=ContactsLocationsModule(
                locations=[
                    Location(
                        facility=loc.get("facility"),
                        city=loc.get("city"),
                        state=loc.get("state"),
                        country=loc.get("country"),
                    )
                    for loc in contacts_mod.get("locations", [])
                ],
            ),
            description_module=DescriptionModule(
                brief_summary=desc_mod.get("briefSummary", ""),
                detailed_description=desc_mod.get("detailedDescription"),
            ),
            sponsor_collaborators_module=SponsorCollaboratorsModule(
                lead_sponsor_name=_get(sponsor_mod, "leadSponsor", "name") or None,
                collaborator_names=[
                    c.get("name", "") for c in sponsor_mod.get("collaborators", [])
                ],
            ),
        )


@dataclass
class TrialRecord:
    """Canonical clinical trial record aligned to CTG protocolSection.

    This is the internal representation used by the feature builder and
    RAG layer.  All data sources (CTG, internal trials) are converted to
    this format.
    """

    # Identification
    nct_id: str
    brief_title: str = ""
    official_title: str = ""
    org_study_id: str = ""

    # Status (dates only -- no outcome status to prevent leakage)
    start_date: date | None = None
    completion_date: date | None = None  # primary completion date
    study_first_post_date: date | None = None

    # Design
    study_type: str = ""  # "INTERVENTIONAL" | "OBSERVATIONAL"
    phase: str = ""  # "PHASE1" | "PHASE2" | "PHASE3" | etc.
    allocation: str = ""  # "RANDOMIZED" | "NON_RANDOMIZED"
    masking: str = ""  # "DOUBLE" | "SINGLE" | "NONE"
    primary_purpose: str = ""  # "TREATMENT" | "PREVENTION" | etc.
    enrollment_count: int | None = None
    enrollment_type: str = ""  # "ACTUAL" | "ESTIMATED"
    number_of_arms: int | None = None

    # Conditions and interventions
    conditions: list[str] = field(default_factory=list)
    interventions: list[dict[str, str]] = field(default_factory=list)
    # Each intervention: {"type": "DRUG", "name": "...", "description": "..."}

    # Eligibility
    eligibility_criteria: str = ""
    min_age: str = ""
    max_age: str = ""
    sex: str = ""  # "ALL" | "FEMALE" | "MALE"

    # Outcomes
    primary_outcomes: list[str] = field(default_factory=list)
    secondary_outcomes: list[str] = field(default_factory=list)

    # Sponsor
    lead_sponsor: str = ""
    collaborators: list[str] = field(default_factory=list)

    # Description
    brief_summary: str = ""
    detailed_description: str = ""

    # Label (for training data only -- NOT from the trial record itself)
    label: int | None = None  # 0 = failure, 1 = success


def align_to_protocol_section(raw: dict[str, Any]) -> TrialRecord:
    """Convert a raw CTG API v2 ``protocolSection`` dict to a TrialRecord.

    This handles the nested JSON structure of the CTG API response and
    normalizes it into a flat dataclass.

    Args:
        raw: Raw ``protocolSection`` dict from the CTG API v2.

    Returns:
        A normalized :class:`TrialRecord`.
    """

    def _get(d: dict[str, Any], *keys: str, default: Any = "") -> Any:
        """Nested dict access with fallback."""
        current = d
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, {})
            else:
                return default
        return current if current != {} else default

    def _parse_date(val: Any) -> date | None:
        """Parse a date from a date object or ISO string.

        Args:
            val: date object or ISO date string (e.g., "2025-04-05"), or None.

        Returns:
            date object, or None if val is None or parsing fails.
        """
        if not val:
            return None
        try:
            if isinstance(val, date):
                return val
            return date.fromisoformat(str(val)[:10])
        except Exception:
            return None

    # Identification
    id_mod = raw.get("identificationModule", {})
    nct_id = id_mod.get("nctId", "")

    # Status
    status_mod = raw.get("statusModule", {})
    start_date_raw = _get(status_mod, "startDateStruct", "date")
    # NOTE: primaryCompletionDateStruct is deliberately NOT read here.
    # Populating completion_date from raw CTG data constitutes label leakage
    # because this date is only known after the trial ends and is correlated
    # with outcome (completed vs. terminated).  See ARCHITECT_REVIEW_P2.md C1.
    first_post_raw = _get(status_mod, "studyFirstPostDateStruct", "date")

    # Design
    design_mod = raw.get("designModule", {})
    design_info = design_mod.get("designInfo", {})
    enrollment_info = design_mod.get("enrollmentInfo", {})

    # Conditions
    conditions_mod = raw.get("conditionsModule", {})
    conditions_list = conditions_mod.get("conditions", [])

    # Interventions
    arms_mod = raw.get("armsInterventionsModule", {})
    interventions_raw = arms_mod.get("interventions", [])
    interventions = [
        {
            "type": iv.get("type", ""),
            "name": iv.get("name", ""),
            "description": iv.get("description", ""),
        }
        for iv in interventions_raw
    ]

    # Eligibility
    elig_mod = raw.get("eligibilityModule", {})

    # Outcomes
    outcomes_mod = raw.get("outcomesModule", {})
    primary_outcomes = [o.get("measure", "") for o in outcomes_mod.get("primaryOutcomes", [])]
    secondary_outcomes = [o.get("measure", "") for o in outcomes_mod.get("secondaryOutcomes", [])]

    # Sponsor
    sponsor_mod = raw.get("sponsorCollaboratorsModule", {})
    lead_sponsor_info = sponsor_mod.get("leadSponsor", {})
    collaborators_raw = sponsor_mod.get("collaborators", [])

    # Description
    desc_mod = raw.get("descriptionModule", {})

    phases = design_mod.get("phases", [])
    phase_str = phases[0] if phases else ""

    return TrialRecord(
        nct_id=nct_id,
        brief_title=id_mod.get("briefTitle", ""),
        official_title=id_mod.get("officialTitle", ""),
        org_study_id=_get(id_mod, "orgStudyIdInfo", "id"),
        start_date=_parse_date(start_date_raw),
        completion_date=None,  # excluded — leakage risk (see ARCHITECT_REVIEW_P2.md C1)
        study_first_post_date=_parse_date(first_post_raw),
        study_type=design_mod.get("studyType", ""),
        phase=phase_str,
        allocation=design_info.get("allocation", ""),
        masking=_get(design_info, "maskingInfo", "masking"),
        primary_purpose=design_info.get("primaryPurpose", ""),
        enrollment_count=enrollment_info.get("count"),
        enrollment_type=enrollment_info.get("type", ""),
        number_of_arms=design_mod.get("numberOfArms"),
        conditions=conditions_list,
        interventions=interventions,
        eligibility_criteria=elig_mod.get("eligibilityCriteria", ""),
        min_age=elig_mod.get("minimumAge", ""),
        max_age=elig_mod.get("maximumAge", ""),
        sex=elig_mod.get("sex", "ALL"),
        primary_outcomes=primary_outcomes,
        secondary_outcomes=secondary_outcomes,
        lead_sponsor=lead_sponsor_info.get("name", ""),
        collaborators=[c.get("name", "") for c in collaborators_raw],
        brief_summary=desc_mod.get("briefSummary", ""),
        detailed_description=desc_mod.get("detailedDescription", ""),
    )
