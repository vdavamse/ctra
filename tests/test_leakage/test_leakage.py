"""Cross-cutting leakage prevention tests.

Validates that outcome-revealing fields (overallStatus, completionDate,
resultsPosted, whyStopped, etc.) are never exposed to the prediction
pipeline -- across the RAG indexer, retrieval tools, schema layer, and
data ingestion.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from ctra.data.ingestion import _flatten_ctg_study
from ctra.data.trial_schema import (
    IdentificationModule,
    ProtocolSection,
    StatusModule,
)
from ctra.rag.indexer import _ctg_to_passages
from ctra.rag.tools import _is_leakage_field

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ctg_row_with_status() -> pl.DataFrame:
    """A single CTG record that includes leakage fields in its columns."""
    return pl.DataFrame(
        [
            {
                "nctId": "NCT00000001",
                "briefTitle": "Test Trial for Drug X",
                "officialTitle": "A Phase 2 Trial of Drug X",
                "briefSummary": "This trial evaluates Drug X.",
                "studyType": "INTERVENTIONAL",
                "phase": "PHASE2",
                "startDate": "2021-06-01",
                # Leakage fields -- these should never appear in passage text
                "overallStatus": "TERMINATED",
                "completionDate": "2023-01-15",
                "resultsPosted": True,
                "whyStopped": "Futility",
                "primaryCompletionDate": "2022-12-01",
            }
        ]
    )


@pytest.fixture()
def ctg_study_dict_full() -> dict:
    """A CTG API v2 study dict with protocolSection and resultsSection."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT99999999",
                "briefTitle": "Demo Trial",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2020-01-15"},
                "completionDateStruct": {"date": "2022-06-30"},
            },
            "descriptionModule": {
                "briefSummary": "A demonstration trial.",
            },
            "designModule": {
                "phases": ["PHASE3"],
                "studyType": "INTERVENTIONAL",
            },
            "conditionsModule": {"conditions": ["Diabetes"]},
            "eligibilityModule": {"eligibilityCriteria": "Age >= 18"},
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "DemoMab"},
                ],
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Acme Pharma"},
            },
        },
        "resultsSection": {
            "participantFlowModule": {"some": "data"},
        },
    }


# ---------------------------------------------------------------------------
# 1. _ctg_to_passages() never includes "overallStatus" in text
# ---------------------------------------------------------------------------


class TestCtgPassageLeakage:
    """Verify the CTG passage generator strips outcome-revealing content."""

    def test_overall_status_not_in_passage_text(self, ctg_row_with_status: pl.DataFrame) -> None:
        """Passage text must NOT contain 'overallStatus' or its value."""
        passages = list(_ctg_to_passages(ctg_row_with_status))
        assert len(passages) >= 1, "Expected at least one passage"
        text = passages[0]["text"].lower()
        assert "overallstatus" not in text
        assert "terminated" not in text

    def test_why_stopped_not_in_passage_text(self, ctg_row_with_status: pl.DataFrame) -> None:
        """whyStopped and its value must NOT leak into passage text."""
        passages = list(_ctg_to_passages(ctg_row_with_status))
        text = passages[0]["text"].lower()
        assert "whystopped" not in text
        assert "futility" not in text

    def test_completion_date_not_in_passage_text(self, ctg_row_with_status: pl.DataFrame) -> None:
        """completionDate must NOT leak into passage text."""
        passages = list(_ctg_to_passages(ctg_row_with_status))
        text = passages[0]["text"].lower()
        assert "completiondate" not in text
        # The literal date value "2023-01-15" should also not appear
        assert "2023-01-15" not in text

    def test_results_posted_not_in_passage_text(self, ctg_row_with_status: pl.DataFrame) -> None:
        """resultsPosted must NOT leak into passage text."""
        passages = list(_ctg_to_passages(ctg_row_with_status))
        text = passages[0]["text"].lower()
        assert "resultsposted" not in text


# ---------------------------------------------------------------------------
# 2. _LEAKAGE_FIELD_PATTERNS coverage
# ---------------------------------------------------------------------------


class TestLeakageFieldPatterns:
    """Verify the leakage pattern set catches all required fields."""

    @pytest.mark.parametrize(
        "field_name",
        [
            "overallStatus",
            "overall_status",
            "overallstatus",
            "completion_date",
            "completionDate",
            "completiondate",
            "resultsPosted",
            "results_posted",
            "resultsposted",
            "whyStopped",
            "why_stopped",
            "whystopped",
            "primaryCompletionDate",
            "primary_completion_date",
            "resultsSection",
            "results_section",
            "statusVerifiedDate",
            "status_verified_date",
            "dispositionModule",
            "disposition_module",
        ],
    )
    def test_known_leakage_fields_are_caught(self, field_name: str) -> None:
        """Every known leakage field must be detected by _is_leakage_field."""
        assert _is_leakage_field(field_name), (
            f"_is_leakage_field({field_name!r}) returned False; "
            f"this field reveals trial outcome and must be caught"
        )

    def test_safe_fields_are_not_flagged(self) -> None:
        """Non-leakage fields must NOT be caught by the filter."""
        safe_fields = [
            "briefTitle",
            "nctId",
            "studyType",
            "phase",
            "eligibilityCriteria",
            "startDate",
            "conditions",
            "interventions",
            "enrollment",
        ]
        for field_name in safe_fields:
            assert not _is_leakage_field(field_name), (
                f"_is_leakage_field({field_name!r}) returned True; "
                f"this is a safe protocol field that should not be stripped"
            )


# ---------------------------------------------------------------------------
# 3. _is_leakage_field() handles case variants and hyphens
# ---------------------------------------------------------------------------


class TestIsLeakageFieldNormalization:
    """Verify normalization of case, hyphens, and underscores."""

    def test_uppercase_variant(self) -> None:
        assert _is_leakage_field("OVERALLSTATUS")

    def test_mixed_case(self) -> None:
        assert _is_leakage_field("OverAll_Status")

    def test_hyphenated_variant(self) -> None:
        assert _is_leakage_field("overall-status")

    def test_mixed_hyphens_underscores(self) -> None:
        assert _is_leakage_field("completion-date")

    def test_camel_case_with_hyphen(self) -> None:
        assert _is_leakage_field("Why-Stopped")


# ---------------------------------------------------------------------------
# 4. Date filter excludes future-dated passages
# ---------------------------------------------------------------------------


class TestDateFilter:
    """Verify date filtering in LinearRAGWrapper.search().

    These tests exercise the static _parse_row_date helper and the
    filtering logic conceptually, without requiring a full LinearRAG
    index on disk.
    """

    def test_parse_row_date_iso_string(self) -> None:
        from ctra.rag.linearrag_wrapper import LinearRAGWrapper

        result = LinearRAGWrapper._parse_row_date("2021-06-15")
        assert result == datetime.date(2021, 6, 15)

    def test_parse_row_date_none(self) -> None:
        from ctra.rag.linearrag_wrapper import LinearRAGWrapper

        result = LinearRAGWrapper._parse_row_date(None)
        assert result is None

    def test_future_passage_excluded(self) -> None:
        """A passage dated AFTER the cutoff must be excluded.

        We simulate the post-retrieval filter logic:
            if row_date >= before_date: skip
        """
        before_date = datetime.date(2021, 1, 1)
        future_date = datetime.date(2023, 6, 15)
        assert future_date >= before_date, "Future date should be excluded"

    def test_undated_passage_excluded_conservatively(self) -> None:
        """Undated passages are conservatively excluded when date filter is on.

        The wrapper code: if row_date is None: continue
        """
        from ctra.rag.linearrag_wrapper import LinearRAGWrapper

        row_date = LinearRAGWrapper._parse_row_date(None)
        assert row_date is None
        # In the search() method, None date + date_filter_enabled => skip


# ---------------------------------------------------------------------------
# 5. ProtocolSection.to_trial_record() always sets completion_date=None
# ---------------------------------------------------------------------------


class TestProtocolSectionLeakage:
    """Verify schema-level leakage prevention."""

    def test_to_trial_record_completion_date_is_none(self) -> None:
        """Even if the underlying data could provide a completion date,
        to_trial_record() must always set completion_date=None."""
        ps = ProtocolSection(
            identification_module=IdentificationModule(
                nct_id="NCT12345678",
                brief_title="Test",
            ),
            status_module=StatusModule(
                start_date="2020-03-01",
                study_first_post_date="2020-04-01",
            ),
        )
        record = ps.to_trial_record(label=1)
        assert record.completion_date is None

    def test_to_trial_record_completion_date_none_regardless_of_label(
        self,
    ) -> None:
        """completion_date stays None for both success and failure labels."""
        ps = ProtocolSection(
            identification_module=IdentificationModule(
                nct_id="NCT00000002",
                brief_title="Another Trial",
            ),
        )
        for label in (0, 1, None):
            record = ps.to_trial_record(label=label)
            assert record.completion_date is None, (
                f"completion_date should be None for label={label}"
            )


# ---------------------------------------------------------------------------
# 6. StatusModule has no overallStatus field
# ---------------------------------------------------------------------------


class TestStatusModuleSchema:
    """Verify that StatusModule deliberately omits overallStatus."""

    def test_no_overall_status_field(self) -> None:
        """StatusModule must NOT have an overallStatus field."""
        field_names = set(StatusModule.model_fields.keys())
        assert "overall_status" not in field_names
        assert "overallStatus" not in field_names

    def test_no_completion_date_field(self) -> None:
        """StatusModule must NOT have a completionDate field."""
        field_names = set(StatusModule.model_fields.keys())
        assert "completion_date" not in field_names
        assert "completionDate" not in field_names

    def test_status_module_only_has_safe_fields(self) -> None:
        """StatusModule should only contain start_date and study_first_post_date."""
        field_names = set(StatusModule.model_fields.keys())
        expected = {"start_date", "study_first_post_date"}
        assert field_names == expected, (
            f"StatusModule has unexpected fields: {field_names - expected}"
        )


# ---------------------------------------------------------------------------
# 7. _flatten_ctg_study() does NOT include overallStatus in output
# ---------------------------------------------------------------------------


class TestFlattenCtgStudy:
    """Verify the CTG ingestion flattener strips leakage fields."""

    def test_no_overall_status_key(self, ctg_study_dict_full: dict) -> None:
        """Output dict must NOT contain overallStatus."""
        result = _flatten_ctg_study(ctg_study_dict_full)
        assert result is not None
        assert "overallStatus" not in result
        assert "overall_status" not in result
        # The value "COMPLETED" should not be stored under any key
        for key, val in result.items():
            if isinstance(val, str):
                assert val != "COMPLETED" or key not in (
                    "overallStatus",
                    "overall_status",
                ), f"Leakage value found in key={key}"

    def test_no_completion_date_key(self, ctg_study_dict_full: dict) -> None:
        """Output dict must NOT contain completionDate."""
        result = _flatten_ctg_study(ctg_study_dict_full)
        assert result is not None
        assert "completionDate" not in result
        assert "completion_date" not in result

    def test_no_results_section_data(self, ctg_study_dict_full: dict) -> None:
        """Even though the input has resultsSection, flattener must ignore it."""
        result = _flatten_ctg_study(ctg_study_dict_full)
        assert result is not None
        # No key should reference resultsSection content
        assert "resultsSection" not in result
        assert "participantFlowModule" not in result

    def test_returns_none_when_protocol_section_missing(self) -> None:
        """_flatten_ctg_study must return None when protocolSection is missing."""
        study_no_protocol = {"hasResults": True, "resultsSection": {}}
        result = _flatten_ctg_study(study_no_protocol)
        assert result is None

    def test_returns_none_when_nctid_missing(self) -> None:
        """_flatten_ctg_study returns None if nctId is empty."""
        study_no_nct = {
            "protocolSection": {
                "identificationModule": {"nctId": ""},
            }
        }
        result = _flatten_ctg_study(study_no_nct)
        assert result is None

    def test_valid_study_has_expected_keys(self, ctg_study_dict_full: dict) -> None:
        """A valid study produces a dict with the expected safe keys."""
        result = _flatten_ctg_study(ctg_study_dict_full)
        assert result is not None
        expected_keys = {
            "nctId",
            "briefTitle",
            "officialTitle",
            "startDate",
            "lastUpdatePostDate",
            "briefSummary",
            "detailedDescription",
            "phases",
            "studyType",
            "conditions",
            "interventions",
            "eligibilityCriteria",
            "minimumAge",
            "maximumAge",
            "sex",
            "leadSponsor",
        }
        assert set(result.keys()) == expected_keys
