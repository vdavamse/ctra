"""Tests for RAG indexer passage generators.

Each test exercises the pure data-transformation logic in the passage
generators — no external services, no disk I/O, no LinearRAG dependency.
Synthetic polars DataFrames stand in for real Parquet files.
"""

from __future__ import annotations

import polars as pl
import pytest

from ctra.rag.indexer import (
    _ctg_to_passages,
    _faers_to_passages,
    _pubmed_to_passages,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture()
def ctg_df() -> pl.DataFrame:
    """Minimal ClinicalTrials.gov DataFrame with one complete row."""
    return pl.DataFrame(
        {
            "nct_id": ["NCT00000001"],
            "start_date": ["2023-01-15"],
            "brief_title": ["A Study of Drug X in Diabetes"],
            "official_title": ["Phase 2 Randomized Study of Drug X"],
            "brief_summary": ["This study evaluates Drug X for type 2 diabetes."],
            "detailed_description": ["Detailed protocol description here."],
            "study_type": ["INTERVENTIONAL"],
            "phases": ["PHASE2"],
            "eligibility_criteria": ["Inclusion: age >= 18. Exclusion: pregnant."],
            "conditions": ["Type 2 Diabetes"],
        }
    )


@pytest.fixture()
def ctg_empty_row_df() -> pl.DataFrame:
    """CTG DataFrame where one row has no text fields — should be skipped."""
    return pl.DataFrame(
        {
            "nct_id": ["NCT00000002"],
            "start_date": [None],
            "brief_title": [""],
            "brief_summary": [""],
            "conditions": [""],
        }
    )


@pytest.fixture()
def pubmed_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pmid": ["12345678"],
            "publication_date": ["2022-06-01"],
            "title": ["Efficacy of Drug X in Randomized Trial"],
            "abstract": ["Background: Drug X was studied. Results: positive."],
        }
    )


@pytest.fixture()
def faers_df() -> pl.DataFrame:
    """Four adverse-event rows for two drugs => 2 passages."""
    return pl.DataFrame(
        {
            "drug_name": ["DrugA", "DrugA", "DrugB", "DrugB"],
            "event": ["Nausea", "Headache", "Rash", "Dizziness"],
            "count": [100, 50, 30, 20],
            "seriousness": ["", "serious", "", "serious"],
            "report_quarter": ["2023Q1", "2023Q1", "2023Q2", "2023Q2"],
        }
    )


# ---------------------------------------------------------------------------
# _ctg_to_passages tests
# ---------------------------------------------------------------------------


class TestCTGToPassages:
    def test_one_passage_per_trial(self, ctg_df: pl.DataFrame) -> None:
        passages = list(_ctg_to_passages(ctg_df))
        assert len(passages) == 1

    def test_passage_contains_title_and_summary(self, ctg_df: pl.DataFrame) -> None:
        passage = next(iter(_ctg_to_passages(ctg_df)))
        assert "A Study of Drug X in Diabetes" in passage["text"]
        assert "This study evaluates Drug X" in passage["text"]

    def test_passage_contains_criteria(self, ctg_df: pl.DataFrame) -> None:
        passage = next(iter(_ctg_to_passages(ctg_df)))
        assert "Inclusion: age >= 18" in passage["text"]

    def test_passage_keys(self, ctg_df: pl.DataFrame) -> None:
        passage = next(iter(_ctg_to_passages(ctg_df)))
        assert set(passage.keys()) == {"text", "source", "doc_id", "date"}

    def test_source_is_ctg(self, ctg_df: pl.DataFrame) -> None:
        passage = next(iter(_ctg_to_passages(ctg_df)))
        assert passage["source"] == "ctg"

    def test_doc_id_is_nct_id(self, ctg_df: pl.DataFrame) -> None:
        passage = next(iter(_ctg_to_passages(ctg_df)))
        assert passage["doc_id"] == "NCT00000001"

    def test_date_populated(self, ctg_df: pl.DataFrame) -> None:
        passage = next(iter(_ctg_to_passages(ctg_df)))
        assert passage["date"] == "2023-01-15"

    def test_empty_row_skipped(self, ctg_empty_row_df: pl.DataFrame) -> None:
        passages = list(_ctg_to_passages(ctg_empty_row_df))
        assert len(passages) == 0


# ---------------------------------------------------------------------------
# _pubmed_to_passages tests
# ---------------------------------------------------------------------------


class TestPubmedToPassages:
    def test_one_passage_per_article(self, pubmed_df: pl.DataFrame) -> None:
        passages = list(_pubmed_to_passages(pubmed_df))
        assert len(passages) == 1

    def test_contains_title_and_abstract(self, pubmed_df: pl.DataFrame) -> None:
        passage = next(iter(_pubmed_to_passages(pubmed_df)))
        assert "Efficacy of Drug X" in passage["text"]
        assert "Results: positive" in passage["text"]

    def test_standard_keys(self, pubmed_df: pl.DataFrame) -> None:
        passage = next(iter(_pubmed_to_passages(pubmed_df)))
        assert set(passage.keys()) == {"text", "source", "doc_id", "date"}
        assert passage["source"] == "pubmed"
        assert passage["doc_id"] == "12345678"
        assert passage["date"] == "2022-06-01"

    def test_empty_article_skipped(self) -> None:
        df = pl.DataFrame({"pmid": ["99"], "title": [""], "abstract": [""]})
        assert list(_pubmed_to_passages(df)) == []


# ---------------------------------------------------------------------------
# _faers_to_passages tests
# ---------------------------------------------------------------------------


class TestFAERSToPassages:
    def test_grouped_by_drug(self, faers_df: pl.DataFrame) -> None:
        passages = list(_faers_to_passages(faers_df))
        assert len(passages) == 2

    def test_events_listed_in_passage(self, faers_df: pl.DataFrame) -> None:
        passages = list(_faers_to_passages(faers_df))
        drug_a = next(p for p in passages if "DrugA" in p["text"])
        assert "Nausea" in drug_a["text"]
        assert "Headache" in drug_a["text"]

    def test_standard_keys(self, faers_df: pl.DataFrame) -> None:
        passages = list(_faers_to_passages(faers_df))
        for p in passages:
            assert set(p.keys()) == {"text", "source", "doc_id", "date"}
            assert p["source"] == "faers"

    def test_doc_id_format(self, faers_df: pl.DataFrame) -> None:
        passages = list(_faers_to_passages(faers_df))
        doc_ids = {p["doc_id"] for p in passages}
        assert "faers-druga" in doc_ids
        assert "faers-drugb" in doc_ids

    def test_empty_drug_name_skipped(self) -> None:
        df = pl.DataFrame(
            {
                "drug_name": [""],
                "event": ["Nausea"],
                "count": [10],
                "seriousness": [""],
                "report_quarter": ["2023Q1"],
            }
        )
        assert list(_faers_to_passages(df)) == []
