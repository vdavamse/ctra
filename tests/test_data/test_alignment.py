"""Tests for the TrialAligner data transformation layer.

Validates field mapping, phase/study-type normalisation, intervention
parsing, and validation warnings — all using synthetic dicts, no external
services.
"""

from __future__ import annotations

import pytest

from ctra.data.alignment import TrialAligner
from ctra.data.trial_schema import ProtocolSection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def aligner() -> TrialAligner:
    return TrialAligner()


@pytest.fixture()
def complete_record() -> dict:
    """An internal record with every commonly mapped field populated."""
    return {
        "trial_id": "MK-0001",
        "trial_title": "Study of Drug Y in Hypertension",
        "official_title": "Phase 3 Randomised Controlled Trial of Drug Y",
        "start_date": "2024-03-01",
        "trial_phase": "Phase III",
        "study_type": "Interventional",
        "allocation": "RANDOMIZED",
        "primary_purpose": "TREATMENT",
        "masking": "DOUBLE",
        "enrollment": 500,
        "enrollment_type": "ESTIMATED",
        "conditions": "Essential Hypertension",
        "eligibility_criteria": "Age >= 21. No prior MI.",
        "sex": "ALL",
        "minimum_age": "21 Years",
        "maximum_age": "75 Years",
        "interventions": [{"type": "DRUG", "name": "Drug Y", "description": "10 mg QD"}],
        "primary_endpoints": [{"measure": "Change in SBP", "time_frame": "12 weeks"}],
        "secondary_endpoints": [{"measure": "Change in DBP"}],
        "brief_summary": "This trial evaluates Drug Y.",
        "sponsor": "Pharma Corp",
    }


@pytest.fixture()
def minimal_record() -> dict:
    """Only the absolute minimum to avoid ValueError (trial_id present)."""
    return {"trial_id": "INTERNAL-0042"}


# ---------------------------------------------------------------------------
# align() tests
# ---------------------------------------------------------------------------


class TestAlign:
    def test_complete_record_round_trips(
        self, aligner: TrialAligner, complete_record: dict
    ) -> None:
        ps = aligner.align(complete_record)
        assert isinstance(ps, ProtocolSection)
        assert ps.identification_module.nct_id == "MK-0001"
        assert ps.identification_module.brief_title == "Study of Drug Y in Hypertension"

    def test_minimal_record_succeeds(self, aligner: TrialAligner, minimal_record: dict) -> None:
        ps = aligner.align(minimal_record)
        assert ps.identification_module.nct_id == "INTERNAL-0042"

    def test_missing_trial_id_raises(self, aligner: TrialAligner) -> None:
        with pytest.raises(ValueError, match="trial ID"):
            aligner.align({"conditions": "Diabetes"})

    def test_phase_normalized_in_output(self, aligner: TrialAligner, complete_record: dict) -> None:
        ps = aligner.align(complete_record)
        assert ps.design_module.phases == ["PHASE3"]

    def test_study_type_normalized(self, aligner: TrialAligner, complete_record: dict) -> None:
        ps = aligner.align(complete_record)
        assert ps.design_module.study_type == "INTERVENTIONAL"

    def test_interventions_parsed(self, aligner: TrialAligner, complete_record: dict) -> None:
        ps = aligner.align(complete_record)
        ivs = ps.arms_interventions_module.interventions
        assert len(ivs) == 1
        assert ivs[0].name == "Drug Y"
        assert ivs[0].type == "DRUG"


# ---------------------------------------------------------------------------
# _normalize_phase() tests
# ---------------------------------------------------------------------------


class TestNormalizePhase:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Phase II", "PHASE2"),
            ("2", "PHASE2"),
            ("P3", "PHASE3"),
            ("PHASE1", "PHASE1"),
            ("Phase III", "PHASE3"),
            ("phase 1", "PHASE1"),
            ("P4", "PHASE4"),
            ("", ""),
        ],
    )
    def test_normalize_phase(self, raw: str, expected: str) -> None:
        assert TrialAligner._normalize_phase(raw) == expected

    def test_already_normalized_passthrough(self) -> None:
        assert TrialAligner._normalize_phase("PHASE2") == "PHASE2"


# ---------------------------------------------------------------------------
# _normalize_study_type() tests
# ---------------------------------------------------------------------------


class TestNormalizeStudyType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("INT", "INTERVENTIONAL"),
            ("OBS", "OBSERVATIONAL"),
            ("Interventional", "INTERVENTIONAL"),
            ("", ""),
        ],
    )
    def test_normalize_study_type(self, raw: str, expected: str) -> None:
        assert TrialAligner._normalize_study_type(raw) == expected


# ---------------------------------------------------------------------------
# validate() tests
# ---------------------------------------------------------------------------


class TestValidate:
    def test_complete_record_no_warnings(
        self, aligner: TrialAligner, complete_record: dict
    ) -> None:
        ps = aligner.align(complete_record)
        warnings = aligner.validate(ps)
        assert warnings == []

    def test_missing_start_date_warns(self, aligner: TrialAligner) -> None:
        ps = aligner.align(
            {
                "trial_id": "T-1",
                "trial_title": "X",
                "trial_phase": "2",
                "study_type": "INT",
                "eligibility_criteria": "None",
                "interventions": ["DrugZ"],
                "primary_endpoints": ["EP1"],
                "brief_summary": "Summary",
                "conditions": "Cancer",
            }
        )
        warnings = aligner.validate(ps)
        assert any("startDate" in w for w in warnings)

    def test_missing_phases_warns(self, aligner: TrialAligner) -> None:
        ps = aligner.align(
            {
                "trial_id": "T-2",
                "trial_title": "X",
                "start_date": "2024-01-01",
                "study_type": "INT",
                "eligibility_criteria": "None",
                "interventions": ["DrugZ"],
                "primary_endpoints": ["EP1"],
                "brief_summary": "Summary",
                "conditions": "Cancer",
            }
        )
        warnings = aligner.validate(ps)
        assert any("phases" in w for w in warnings)

    def test_missing_summary_warns(self, aligner: TrialAligner) -> None:
        ps = aligner.align({"trial_id": "T-3"})
        warnings = aligner.validate(ps)
        assert any("briefSummary" in w for w in warnings)


# ---------------------------------------------------------------------------
# _parse_interventions() tests
# ---------------------------------------------------------------------------


class TestParseInterventions:
    def test_string_input(self) -> None:
        result = TrialAligner._parse_interventions("Aspirin")
        assert len(result) == 1
        assert result[0].name == "Aspirin"
        assert result[0].type == "DRUG"

    def test_dict_list_input(self) -> None:
        raw = [
            {"type": "BIOLOGICAL", "name": "mAb-X", "description": "Monoclonal antibody"},
            {"type": "DRUG", "name": "Placebo"},
        ]
        result = TrialAligner._parse_interventions(raw)
        assert len(result) == 2
        assert result[0].type == "BIOLOGICAL"
        assert result[1].name == "Placebo"

    def test_empty_input(self) -> None:
        assert TrialAligner._parse_interventions([]) == []
        assert TrialAligner._parse_interventions("") == []
        assert TrialAligner._parse_interventions(None) == []
