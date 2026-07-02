"""Tests for ctra.rag.linearrag_wrapper — static/pure methods and search filtering."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from ctra.config.settings import DataSource, RAGConfig
from ctra.rag.linearrag_wrapper import LinearRAGWrapper, RetrievedChunk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def passages_df() -> pl.DataFrame:
    """A small polars DataFrame simulating the passage store."""
    return pl.DataFrame(
        {
            "text": [
                "Aspirin inhibits COX-1.",
                "Phase II trial of Drug X.",
                "Adverse events reported in Q1 2023.",
                "Drug Y targets EGFR.",
                "Undated safety signal.",
            ],
            "source": ["pubmed", "ctg", "faers", "chembl", "pubmed"],
            "doc_id": ["PMID1", "NCT001", "FAERS01", "CHEMBL001", "PMID2"],
            "date": [
                "2020-01-15",
                "2021-06-01",
                "2023-03-31",
                "2019-07-20",
                None,
            ],
        }
    )


@pytest.fixture()
def wrapper(passages_df: pl.DataFrame) -> LinearRAGWrapper:
    """A LinearRAGWrapper with a mocked LinearRAG backend and known passages."""
    config = RAGConfig(
        top_k=5,
        date_filter_enabled=True,
        sources=[DataSource.CTG, DataSource.PUBMED, DataSource.CHEMBL, DataSource.FAERS],
    )
    w = LinearRAGWrapper(config=config, index_dir=None)
    # Bypass lazy-loading; inject test data directly
    w._passages_df = passages_df
    w._linearrag = MagicMock()  # placeholder so _ensure_loaded() is a no-op
    return w


# ---------------------------------------------------------------------------
# _parse_row_date
# ---------------------------------------------------------------------------


class TestParseRowDate:
    """Tests for the static _parse_row_date helper."""

    def test_iso_string(self) -> None:
        result = LinearRAGWrapper._parse_row_date("2023-05-10")
        assert result == datetime.date(2023, 5, 10)

    def test_datetime_object(self) -> None:
        dt = datetime.datetime(2022, 3, 14, 12, 30, 0)
        result = LinearRAGWrapper._parse_row_date(dt)
        assert result == datetime.date(2022, 3, 14)

    def test_date_object(self) -> None:
        d = datetime.date(2021, 1, 1)
        result = LinearRAGWrapper._parse_row_date(d)
        assert result == datetime.date(2021, 1, 1)

    def test_none_returns_none(self) -> None:
        assert LinearRAGWrapper._parse_row_date(None) is None

    def test_nan_returns_none(self) -> None:
        assert LinearRAGWrapper._parse_row_date(float("nan")) is None

    def test_numpy_nan_returns_none(self) -> None:
        assert LinearRAGWrapper._parse_row_date(np.nan) is None

    def test_partial_date_yyyy_mm(self) -> None:
        # "2020-06" -> fromisoformat parses the first 10 chars "2020-06" which
        # is only 7 chars; fromisoformat may or may not handle it — the code
        # truncates to [:10].  If fromisoformat fails the except branch returns None.
        result = LinearRAGWrapper._parse_row_date("2020-06")
        # Python 3.11+ handles "2020-06" via fromisoformat.  Earlier versions
        # return None (caught by except).  Both are acceptable.
        assert result is None or result == datetime.date(2020, 6, 1)

    def test_garbage_returns_none(self) -> None:
        assert LinearRAGWrapper._parse_row_date("not-a-date") is None


# ---------------------------------------------------------------------------
# _resolve_passage_row
# ---------------------------------------------------------------------------


class TestResolvePassageRow:
    """Tests for mapping passage text back to a DataFrame row."""

    def test_index_prefix(self, wrapper: LinearRAGWrapper) -> None:
        """'0:Aspirin inhibits COX-1.' should resolve to row 0."""
        assert wrapper._resolve_passage_row("0:Aspirin inhibits COX-1.") == 0

    def test_index_prefix_row_2(self, wrapper: LinearRAGWrapper) -> None:
        assert wrapper._resolve_passage_row("2:Adverse events reported in Q1 2023.") == 2

    def test_text_fallback(self, wrapper: LinearRAGWrapper) -> None:
        """When no index prefix is present, fall back to text search."""
        assert wrapper._resolve_passage_row("Drug Y targets EGFR.") == 3

    def test_no_match_returns_none(self, wrapper: LinearRAGWrapper) -> None:
        assert wrapper._resolve_passage_row("This text does not exist.") is None

    def test_out_of_bounds_index_falls_back(self, wrapper: LinearRAGWrapper) -> None:
        # Index 999 is out of range; should fall through to text search.
        # The text won't match either, so expect None.
        assert wrapper._resolve_passage_row("999:nonexistent") is None


# ---------------------------------------------------------------------------
# search() with mocked _retrieve_raw
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for the main search() method with mocked retrieval."""

    def _mock_retrieve(
        self,
        wrapper: LinearRAGWrapper,
        passages: list[tuple[str, float]],
    ) -> None:
        """Replace _retrieve_raw with a function that returns fixed passages."""
        wrapper._retrieve_raw = MagicMock(return_value=passages)  # type: ignore[assignment]

    def test_basic_search_returns_chunks(
        self, wrapper: LinearRAGWrapper, passages_df: pl.DataFrame
    ) -> None:
        self._mock_retrieve(
            wrapper,
            [
                ("0:Aspirin inhibits COX-1.", 0.9),
                ("1:Phase II trial of Drug X.", 0.8),
            ],
        )
        results = wrapper.search("aspirin trial")
        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert results[0].score == 0.9

    def test_date_filter_excludes_future(self, wrapper: LinearRAGWrapper) -> None:
        """Passages with date >= before_date should be excluded."""
        self._mock_retrieve(
            wrapper,
            [
                ("0:Aspirin inhibits COX-1.", 0.9),  # date 2020-01-15
                ("2:Adverse events reported in Q1 2023.", 0.8),  # date 2023-03-31
            ],
        )
        results = wrapper.search(
            "adverse events",
            before_date=datetime.date(2022, 1, 1),
            sources=[DataSource.PUBMED, DataSource.FAERS],
        )
        # Only the 2020 passage should pass the date filter
        assert len(results) == 1
        assert results[0].doc_id == "PMID1"

    def test_source_filter(self, wrapper: LinearRAGWrapper) -> None:
        """Only passages matching the source filter should be returned."""
        self._mock_retrieve(
            wrapper,
            [
                ("0:Aspirin inhibits COX-1.", 0.9),  # pubmed
                ("1:Phase II trial of Drug X.", 0.8),  # ctg
            ],
        )
        results = wrapper.search("trial", sources=[DataSource.CTG])
        assert len(results) == 1
        assert results[0].source == "ctg"

    def test_undated_excluded_when_date_filter_active(self, wrapper: LinearRAGWrapper) -> None:
        """Passages with no date are conservatively excluded when before_date is set."""
        self._mock_retrieve(
            wrapper,
            [
                ("4:Undated safety signal.", 0.7),  # date is None
            ],
        )
        results = wrapper.search("safety", before_date=datetime.date(2025, 1, 1))
        assert len(results) == 0

    def test_top_k_limits_results(self, wrapper: LinearRAGWrapper) -> None:
        self._mock_retrieve(
            wrapper,
            [
                ("0:Aspirin inhibits COX-1.", 0.9),
                ("1:Phase II trial of Drug X.", 0.8),
                ("3:Drug Y targets EGFR.", 0.7),
            ],
        )
        results = wrapper.search("query", top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# RetrievedChunk dataclass
# ---------------------------------------------------------------------------


class TestRetrievedChunk:
    def test_construction(self) -> None:
        chunk = RetrievedChunk(
            text="test passage",
            source="pubmed",
            doc_id="PMID123",
            date=datetime.date(2020, 1, 1),
            score=0.85,
            metadata={"title": "A paper"},
        )
        assert chunk.text == "test passage"
        assert chunk.score == 0.85
        assert chunk.metadata["title"] == "A paper"

    def test_frozen(self) -> None:
        chunk = RetrievedChunk(text="t", source="s", doc_id="d")
        with pytest.raises(AttributeError):
            chunk.text = "modified"  # type: ignore[misc]
