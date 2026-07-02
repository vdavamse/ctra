#!/usr/bin/env python
"""Build the LinearRAG index from CTRA's 7 data sources.

Reads cleaned Parquet files, converts records into natural-language passages
with date metadata (for leakage prevention), and feeds them into LinearRAG's
NER + entity co-occurrence graph pipeline.

Usage::

    # Build full index from all 7 sources
    python scripts/build_rag_index.py build

    # Build from specific sources only
    python scripts/build_rag_index.py build --sources ctg,pubmed,chembl

    # Build a single source from a custom parquet file
    python scripts/build_rag_index.py build-ctg --input datasets/ctg_studies.parquet

    # Custom output directory
    python scripts/build_rag_index.py build --output datasets/my-index
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ctra.build_rag_index")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RAG index building.

    Returns:
        Namespace with command, sources, output, and input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build LinearRAG index for CTRA from 7 biomedical data sources.",
    )
    parser.add_argument(
        "command",
        choices=[
            "build",
            "build-ctg",
            "build-pubmed",
            "build-faers",
            "build-aact",
            "build-chembl",
            "build-primekg",
            "build-drugsfda",
        ],
        help="Index to build (build = all sources, build-<source> = single source)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="ctg,pubmed,chembl,faers,aact,primekg,drugsfda",
        help="Comma-separated list of data sources to index (only for 'build' command)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the index (default: from settings)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input parquet file (for single-source commands)",
    )
    return parser.parse_args()


def main() -> None:
    """Build LinearRAG index from one or more biomedical data sources.

    Reads cleaned Parquet files, converts records to natural-language passages
    with date metadata, and indexes them into LinearRAG (NER + entity co-occurrence
    graph). Supports full multi-source indexing or single-source builds.
    """
    args = parse_args()

    from ctra.config.settings import DataSource
    from ctra.rag.indexer import IndexBuilder

    builder = IndexBuilder()

    if args.command == "build":
        src_list = [DataSource(s.strip()) for s in args.sources.split(",")]
        output_path = Path(args.output) if args.output else None
        logger.info("Building full index from %d sources...", len(src_list))
        result_dir = builder.build_all(sources=src_list, output_dir=output_path)
        logger.info("Index built at %s", result_dir)
    else:
        source = args.command.removeprefix("build-")
        logger.info("Building %s index...", source)
        method = getattr(builder, f"build_{source}_index")
        kwargs: dict[str, str | None] = {"output_dir": args.output}
        if args.input:
            # Each build method uses a different kwarg name for the input path
            param_name = f"{source}_parquet_path"
            kwargs[param_name] = args.input
        method(**kwargs)
        logger.info("%s index complete.", source)


if __name__ == "__main__":
    main()
