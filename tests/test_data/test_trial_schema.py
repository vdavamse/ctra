"""Tests for the ProtocolSection schema and conversion logic.

All tests use in-memory Pydantic models — no disk I/O, no external services.
"""

from __future__ import annotations

import pytest

from ctra.data.trial_schema import (
    ArmsInterventionsModule,
    DescriptionModule,
    DesignInfo,
    DesignModule,
    EligibilityModule,
    EnrollmentInfo,
    IdentificationModule,
    Intervention,
    OutcomeMeasure,
    OutcomesModule,
    ProtocolSection,
    StatusModule,
    TrialRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_protocol() -> ProtocolSection:
    """A fully populated ProtocolSection for round-trip and flattening tests."""
    return ProtocolSection(
        identification_module=IdentificationModule(
            nct_id="NCT12345678",
            brief_title="A Phase 2 Study of Drug Z",
            official_title="Randomised Double-Blind Study of Drug Z",
            org_study_id="ORG-001",
        ),
        status_module=StatusModule(
            start_date="2024-06-15",
            study_first_post_date="2024-05-01",
        ),
        design_module=DesignModule(
            phases=["PHASE2"],
            study_type="INTERVENTIONAL",
            design_info=DesignInfo(
                allocation="RANDOMIZED",
                intervention_model="PARALLEL",
                primary_purpose="TREATMENT",
                masking="DOUBLE",
            ),
            enrollment_info=EnrollmentInfo(count=200, type="ESTIMATED"),
            number_of_arms=2,
        ),
        eligibility_module=EligibilityModule(
            eligibility_criteria="Age >= 18 years. No prior treatment.",
            sex="ALL",
            minimum_age="18 Years",
            maximum_age="65 Years",
        ),
        arms_interventions_module=ArmsInterventionsModule(
            interventions=[
                Intervention(type="DRUG", name="Drug Z", description="50 mg BID"),
                Intervention(type="DRUG", name="Placebo"),
            ],
        ),
        outcomes_module=OutcomesModule(
            primary_outcomes=[
                OutcomeMeasure(
                    measure="Overall Response Rate",
                    time_frame="24 weeks",
                ),
            ],
            secondary_outcomes=[
                OutcomeMeasure(measure="Duration of Response"),
            ],
        ),
        description_module=DescriptionModule(
            brief_summary="Evaluates Drug Z in advanced cancer.",
            detailed_description="Full protocol details here.",
        ),
    )


# ---------------------------------------------------------------------------
# Round-trip: to_ctg_dict -> from_ctg_dict
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_roundtrip_preserves_nct_id(self, full_protocol: ProtocolSection) -> None:
        ctg = full_protocol.to_ctg_dict()
        restored = ProtocolSection.from_ctg_dict(ctg)
        assert restored.identification_module.nct_id == "NCT12345678"

    def test_roundtrip_preserves_title(self, full_protocol: ProtocolSection) -> None:
        ctg = full_protocol.to_ctg_dict()
        restored = ProtocolSection.from_ctg_dict(ctg)
        assert restored.identification_module.brief_title == "A Phase 2 Study of Drug Z"

    def test_roundtrip_preserves_phases(self, full_protocol: ProtocolSection) -> None:
        ctg = full_protocol.to_ctg_dict()
        restored = ProtocolSection.from_ctg_dict(ctg)
        assert restored.design_module.phases == ["PHASE2"]

    def test_roundtrip_preserves_interventions(self, full_protocol: ProtocolSection) -> None:
        ctg = full_protocol.to_ctg_dict()
        restored = ProtocolSection.from_ctg_dict(ctg)
        ivs = restored.arms_interventions_module.interventions
        assert len(ivs) == 2
        assert ivs[0].name == "Drug Z"

    def test_roundtrip_preserves_outcomes(self, full_protocol: ProtocolSection) -> None:
        ctg = full_protocol.to_ctg_dict()
        restored = ProtocolSection.from_ctg_dict(ctg)
        assert len(restored.outcomes_module.primary_outcomes) == 1
        assert restored.outcomes_module.primary_outcomes[0].measure == "Overall Response Rate"


# ---------------------------------------------------------------------------
# to_trial_record() — flat representation
# ---------------------------------------------------------------------------


class TestToTrialRecord:
    def test_produces_flat_record(self, full_protocol: ProtocolSection) -> None:
        rec = full_protocol.to_trial_record()
        assert isinstance(rec, TrialRecord)
        assert rec.nct_id == "NCT12345678"

    def test_completion_date_is_none(self, full_protocol: ProtocolSection) -> None:
        rec = full_protocol.to_trial_record()
        assert rec.completion_date is None, "completion_date must be None to prevent leakage"

    def test_label_forwarded(self, full_protocol: ProtocolSection) -> None:
        rec = full_protocol.to_trial_record(label=1)
        assert rec.label == 1

    def test_start_date_parsed(self, full_protocol: ProtocolSection) -> None:
        rec = full_protocol.to_trial_record()
        assert rec.start_date is not None
        assert rec.start_date.isoformat() == "2024-06-15"


# ---------------------------------------------------------------------------
# StatusModule — no overallStatus field
# ---------------------------------------------------------------------------


class TestStatusModule:
    def test_no_overall_status_field(self) -> None:
        """StatusModule must not expose overallStatus (leakage vector)."""
        fields = set(StatusModule.model_fields.keys())
        assert "overall_status" not in fields
        assert "overallStatus" not in fields

    def test_no_completion_date_field(self) -> None:
        fields = set(StatusModule.model_fields.keys())
        assert "completion_date" not in fields
        assert "completionDate" not in fields


# ---------------------------------------------------------------------------
# Missing modules tolerated
# ---------------------------------------------------------------------------


class TestMissingModules:
    def test_from_ctg_dict_empty_dict(self) -> None:
        """Parsing a nearly empty CTG dict should not raise."""
        ps = ProtocolSection.from_ctg_dict({"identificationModule": {"nctId": "NCT00000000"}})
        assert ps.identification_module.nct_id == "NCT00000000"
        # All other modules should have safe defaults
        assert ps.design_module.phases == []
        assert ps.eligibility_module.eligibility_criteria == ""

    def test_missing_design_info_tolerated(self) -> None:
        ps = ProtocolSection.from_ctg_dict(
            {
                "identificationModule": {"nctId": "NCT00000001"},
                "designModule": {"phases": ["PHASE3"], "studyType": "INTERVENTIONAL"},
            }
        )
        assert ps.design_module.phases == ["PHASE3"]
        assert ps.design_module.design_info is None
