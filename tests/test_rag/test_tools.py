"""Tests for ctra.rag.tools — date parsing, YAML formatting, leakage check, factories."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

from ctra.rag.linearrag_wrapper import RetrievedChunk
from ctra.rag.tools import (
    _chunks_to_yaml_blocks,
    _is_leakage_field,
    _parse_start_date,
    make_pubmed_search,
)

# ---------------------------------------------------------------------------
# _parse_start_date
# ---------------------------------------------------------------------------


class TestParseStartDate:
    def test_full_date(self) -> None:
        nct = {"startDate": "2021-06-15"}
        assert _parse_start_date(nct) == "2021-06-15"

    def test_partial_yyyy_mm(self) -> None:
        nct = {"startDate": "2021-06"}
        assert _parse_start_date(nct) == "2021-06-01"

    def test_fallback_to_start_date_key(self) -> None:
        nct = {"start_date": "2020-01-10"}
        assert _parse_start_date(nct) == "2020-01-10"

    def test_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain"):
            _parse_start_date({})

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain"):
            _parse_start_date({"startDate": ""})


# ---------------------------------------------------------------------------
# _chunks_to_yaml_blocks
# ---------------------------------------------------------------------------


class TestChunksToYamlBlocks:
    def test_empty_list(self) -> None:
        result = _chunks_to_yaml_blocks([])
        assert result == "No relevant documents found."

    def test_single_chunk(self) -> None:
        chunk = RetrievedChunk(
            text="Drug X inhibits target Y.",
            source="pubmed",
            doc_id="PMID999",
            date=datetime.date(2020, 5, 1),
            score=0.75,
        )
        result = _chunks_to_yaml_blocks([chunk])
        assert "Drug X inhibits target Y." in result
        assert "pubmed" in result
        assert "PMID999" in result
        assert "0.75" in result

    def test_multiple_chunks_have_separator(self) -> None:
        chunks = [
            RetrievedChunk(text="Passage A.", source="ctg", doc_id="NCT1", score=0.9),
            RetrievedChunk(text="Passage B.", source="pubmed", doc_id="PMID2", score=0.8),
        ]
        result = _chunks_to_yaml_blocks(chunks)
        assert "\n\n---------\n\n" in result

    def test_metadata_included(self) -> None:
        chunk = RetrievedChunk(
            text="text",
            source="chembl",
            doc_id="DB01",
            metadata={"target": "COX-2"},
        )
        result = _chunks_to_yaml_blocks([chunk])
        assert "COX-2" in result

    def test_extra_fields_callback(self) -> None:
        chunk = RetrievedChunk(text="text", source="s", doc_id="d")
        result = _chunks_to_yaml_blocks(
            [chunk],
            extra_fields=lambda c: {"custom_key": "custom_value"},
        )
        assert "custom_value" in result


# ---------------------------------------------------------------------------
# _is_leakage_field
# ---------------------------------------------------------------------------


class TestIsLeakageField:
    def test_overallstatus_is_leakage(self) -> None:
        assert _is_leakage_field("overallStatus") is True

    def test_completion_date_is_leakage(self) -> None:
        assert _is_leakage_field("completionDate") is True

    def test_primary_completion_date(self) -> None:
        assert _is_leakage_field("primary_completion_date") is True

    def test_results_posted(self) -> None:
        assert _is_leakage_field("resultsPosted") is True

    def test_why_stopped(self) -> None:
        assert _is_leakage_field("whyStopped") is True

    def test_brief_title_not_leakage(self) -> None:
        assert _is_leakage_field("briefTitle") is False

    def test_start_date_not_leakage(self) -> None:
        assert _is_leakage_field("startDate") is False

    def test_sponsor_not_leakage(self) -> None:
        assert _is_leakage_field("leadSponsorName") is False

    def test_case_insensitive(self) -> None:
        assert _is_leakage_field("OVERALLSTATUS") is True


# ---------------------------------------------------------------------------
# make_pubmed_search (factory)
# ---------------------------------------------------------------------------


class TestMakePubmedSearch:
    @patch("ctra.rag.tools._get_rag")
    @patch("ctra.rag.tools._get_cache", return_value=None)
    def test_returns_callable(self, _mock_cache: MagicMock, mock_get_rag: MagicMock) -> None:
        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_get_rag.return_value = mock_rag

        nct_info = {"startDate": "2021-06-15"}
        tool = make_pubmed_search(nct_info)
        assert callable(tool)
        assert tool.__name__ == "pubmed_search"

    @patch("ctra.rag.tools._get_rag")
    @patch("ctra.rag.tools._get_cache", return_value=None)
    def test_binds_before_date(self, _mock_cache: MagicMock, mock_get_rag: MagicMock) -> None:
        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_get_rag.return_value = mock_rag

        nct_info = {"startDate": "2021-06-15"}
        tool = make_pubmed_search(nct_info)
        tool("aspirin mechanism")

        # Verify that search was called with the correct before_date
        mock_rag.search.assert_called_once()
        call_kwargs = mock_rag.search.call_args
        assert call_kwargs.kwargs["before_date"] == datetime.date(2021, 6, 15)

    @patch("ctra.rag.tools._get_rag")
    @patch("ctra.rag.tools._get_cache", return_value=None)
    def test_returns_yaml_string(self, _mock_cache: MagicMock, mock_get_rag: MagicMock) -> None:
        mock_rag = MagicMock()
        mock_rag.search.return_value = [
            RetrievedChunk(
                text="Aspirin inhibits COX-1.",
                source="pubmed",
                doc_id="PMID123",
                score=0.9,
            ),
        ]
        mock_get_rag.return_value = mock_rag

        nct_info = {"startDate": "2021-06-15"}
        tool = make_pubmed_search(nct_info)
        result = tool("aspirin")

        assert isinstance(result, str)
        assert "Aspirin inhibits COX-1." in result
