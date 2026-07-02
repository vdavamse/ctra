"""RAG (Retrieval-Augmented Generation) layer built on LinearRAG.

Provides date-filtered, multi-source retrieval over ClinicalTrials.gov,
PubMed, ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA.

AutoCT-style factories (each bound to a trial's start date)::

    from ctra.rag.tools import make_pubmed_search, make_chembl_search

    nct_info = get_trial_info_dict("NCT00110279")
    tools = [make_pubmed_search(nct_info), make_chembl_search(nct_info)]
"""

from ctra.rag.indexer import IndexBuilder, build_index
from ctra.rag.linearrag_wrapper import LinearRAGWrapper, RetrievedChunk
from ctra.rag.tools import (
    get_detailed_nct_info,
    get_trial_info_dict,
    make_aact_search,
    make_chembl_search,
    make_drugsfda_search,
    make_faers_search,
    make_nct_search,
    make_primekg_search,
    make_pubmed_search,
)

__all__ = [
    "IndexBuilder",
    "LinearRAGWrapper",
    "RetrievedChunk",
    "build_index",
    "get_detailed_nct_info",
    "get_trial_info_dict",
    "make_aact_search",
    "make_chembl_search",
    "make_drugsfda_search",
    "make_faers_search",
    "make_nct_search",
    "make_primekg_search",
    "make_pubmed_search",
]
