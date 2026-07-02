"""Tests for ClinicalTrials.gov loader and protocol flattening.

Pure data-transformation tests using synthetic dicts — no Parquet files,
no network requests.
"""

from __future__ import annotations

import pytest

from ctra.data.ctg_loader import (
    _split_text,
    flatten_protocol_to_sentences,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_protocol() -> dict:
    """A synthetic CTG protocolSection dict with key modules populated."""
    return {
        "identificationModule": {
            "nctId": "NCT99999999",
            "briefTitle": "Phase 2 Study of Alpha in Asthma",
            "officialTitle": "Randomised Trial of Alpha vs Placebo",
        },
        "descriptionModule": {
            "briefSummary": "This study evaluates Alpha. The primary endpoint is FEV1.",
        },
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE2"],
        },
        "conditionsModule": {
            "conditions": ["Asthma"],
        },
        "armsInterventionsModule": {
            "interventions": [
                {"type": "DRUG", "name": "Alpha", "description": "200mg inhaled QD"},
            ],
        },
        "eligibilityModule": {
            "eligibilityCriteria": "Age >= 12 years. No COPD.",
            "minimumAge": "12 Years",
            "maximumAge": "70 Years",
            "sex": "ALL",
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {"measure": "Change in FEV1"},
            ],
        },
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"name": "BioPharma Inc"},
        },
    }


# ---------------------------------------------------------------------------
# flatten_protocol_to_sentences() tests
# ---------------------------------------------------------------------------


class TestFlattenProtocol:
    def test_extracts_title(self, sample_protocol: dict) -> None:
        sentences = flatten_protocol_to_sentences(sample_protocol)
        assert any("Phase 2 Study of Alpha in Asthma" in s for s in sentences)

    def test_extracts_conditions(self, sample_protocol: dict) -> None:
        sentences = flatten_protocol_to_sentences(sample_protocol)
        assert any("Asthma" in s for s in sentences)

    def test_extracts_interventions(self, sample_protocol: dict) -> None:
        sentences = flatten_protocol_to_sentences(sample_protocol)
        assert any("Alpha" in s for s in sentences)

    def test_empty_dict_returns_empty_list(self) -> None:
        assert flatten_protocol_to_sentences({}) == []

    def test_no_results_section_processed(self, sample_protocol: dict) -> None:
        """resultsSection keys should never appear in the output."""
        sample_protocol["resultsSection"] = {
            "outcomeModule": {"measures": [{"title": "LEAKED RESULT"}]}
        }
        sentences = flatten_protocol_to_sentences(sample_protocol)
        assert not any("LEAKED RESULT" in s for s in sentences)


# ---------------------------------------------------------------------------
# _split_text() tests
# ---------------------------------------------------------------------------


class TestSplitText:
    def test_splits_on_sentence_boundary(self) -> None:
        text = "First sentence. Second sentence. Third one."
        parts = _split_text(text)
        assert len(parts) >= 2
        assert parts[0] == "First sentence."

    def test_min_length_filters_short_fragments(self) -> None:
        text = "OK. This sentence is long enough to pass."
        parts = _split_text(text, min_length=10)
        # "OK." is only 3 chars — should be filtered
        assert all(len(p) >= 10 for p in parts)

    def test_splits_on_newlines(self) -> None:
        text = "Inclusion criteria:\nAge >= 18 years\nNo prior surgery"
        parts = _split_text(text, min_length=5)
        assert len(parts) >= 2
