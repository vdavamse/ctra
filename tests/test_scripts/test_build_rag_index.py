"""Tests for scripts/build_rag_index.py — CLI dispatch and argument parsing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_build_command_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["build_rag_index.py", "build"])
        from build_rag_index import parse_args

        args = parse_args()
        assert args.command == "build"
        assert args.sources == "ctg,pubmed,chembl,faers,aact,primekg,drugsfda"
        assert args.output is None
        assert args.input is None

    def test_build_with_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_rag_index.py", "build", "--sources", "ctg,pubmed"],
        )
        from build_rag_index import parse_args

        args = parse_args()
        assert args.sources == "ctg,pubmed"

    def test_build_ctg_with_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_rag_index.py", "build-ctg", "--input", "data/ctg.parquet"],
        )
        from build_rag_index import parse_args

        args = parse_args()
        assert args.command == "build-ctg"
        assert args.input == "data/ctg.parquet"

    def test_build_with_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_rag_index.py", "build", "--output", "/tmp/my-index"],
        )
        from build_rag_index import parse_args

        args = parse_args()
        assert args.output == "/tmp/my-index"

    def test_invalid_command_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["build_rag_index.py", "invalid"])
        from build_rag_index import parse_args

        with pytest.raises(SystemExit):
            parse_args()


# ---------------------------------------------------------------------------
# main — dispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    def test_build_calls_build_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_rag_index.py", "build", "--sources", "ctg,pubmed"],
        )

        mock_builder = MagicMock()
        mock_builder.build_all.return_value = Path("/tmp/index")
        mock_builder_cls = MagicMock(return_value=mock_builder)

        monkeypatch.setattr("ctra.rag.indexer.IndexBuilder", mock_builder_cls)

        from build_rag_index import main

        main()

        mock_builder.build_all.assert_called_once()
        call_kwargs = mock_builder.build_all.call_args
        sources = call_kwargs.kwargs.get("sources") or call_kwargs[1].get("sources")
        assert len(sources) == 2

    def test_build_ctg_calls_build_ctg_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_rag_index.py", "build-ctg", "--input", "data/ctg.parquet"],
        )

        mock_builder = MagicMock()
        mock_builder_cls = MagicMock(return_value=mock_builder)
        monkeypatch.setattr("ctra.rag.indexer.IndexBuilder", mock_builder_cls)

        from build_rag_index import main

        main()

        mock_builder.build_ctg_index.assert_called_once()

    def test_build_chembl_calls_build_chembl_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_rag_index.py", "build-chembl"],
        )

        mock_builder = MagicMock()
        mock_builder_cls = MagicMock(return_value=mock_builder)
        monkeypatch.setattr("ctra.rag.indexer.IndexBuilder", mock_builder_cls)

        from build_rag_index import main

        main()

        mock_builder.build_chembl_index.assert_called_once()

    def test_build_all_sources_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default --sources includes all 7 data sources."""
        monkeypatch.setattr(sys, "argv", ["build_rag_index.py", "build"])

        mock_builder = MagicMock()
        mock_builder.build_all.return_value = Path("/tmp/index")
        mock_builder_cls = MagicMock(return_value=mock_builder)
        monkeypatch.setattr("ctra.rag.indexer.IndexBuilder", mock_builder_cls)

        from build_rag_index import main

        main()

        call_kwargs = mock_builder.build_all.call_args
        sources = call_kwargs.kwargs.get("sources") or call_kwargs[1].get("sources")
        assert len(sources) == 7

    def test_build_with_output_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_rag_index.py", "build", "--output", "/tmp/my-index"],
        )

        mock_builder = MagicMock()
        mock_builder.build_all.return_value = Path("/tmp/my-index")
        mock_builder_cls = MagicMock(return_value=mock_builder)
        monkeypatch.setattr("ctra.rag.indexer.IndexBuilder", mock_builder_cls)

        from build_rag_index import main

        main()

        call_kwargs = mock_builder.build_all.call_args
        output_dir = call_kwargs.kwargs.get("output_dir") or call_kwargs[1].get("output_dir")
        assert output_dir == Path("/tmp/my-index")
