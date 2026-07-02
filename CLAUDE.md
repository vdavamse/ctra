# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CTRA** (Clinical Trial Risk Assessment) is an implementation project for clinical trial outcome prediction (success/failure). Rather than inventing new architectures, CTRA assembles proven models from published academic research into a production-grade pipeline for predicting whether a clinical trial will succeed or fail across phases I, II, and III.

## Current State

This repository is in **active implementation**. The core AutoCT pipeline is implemented with per-phase isolation (matching AutoCT's architecture). It contains:

- `src/ctra/` — Full implementation: agents, data loaders (7 sources), RAG (LinearRAG), MCTS search (Pareto), models (XGBoost + TabPFN), API, dashboard, MLOps
- `scripts/` — Entry points: `build_rag_index.py`, `train_mcts.py`, `run_agent.py`, `predict.py`
- `tests/` — 633 tests covering agents, search, API, scripts, models, data, MLOps
- `README.md` — Implementation plan, model comparison, and roadmap
- `research/` — Literature reviews and research notes
- `docs/` — Reference PDFs with corresponding `.md` summaries for each paper
- `repositories/` — Cloned reference implementations from published papers

## Per-Phase Pipeline Isolation

Each clinical trial phase (I, II, III) runs a **completely isolated pipeline** — separate MCTS tree, separate features, separate trained model. This matches AutoCT's architecture where each phase is an independent `Task`.

- **`Task` enum** (`src/ctra/agents/data_models.py`) — `TRIAL_OUTCOME_PHASE_1`, `_PHASE_2`, `_PHASE_3`, each carrying a phase-specific LLM prompt description
- **Training:** `python scripts/train_mcts.py --task phase2 --rollouts 20` → outputs to `.output/phase2/`
- **Prediction:** `python scripts/predict.py --model-dir .output/phase2/ --nctid NCT00110279`
- **API:** POST `/api/v1/predict` with `{"trial_id": "NCT001", "phase": 2}`
- **MCTS objectives:** 2 per phase tree — accuracy (ROC-AUC) + parsimony (feature efficiency)
- **Feature cache:** Shared across phases at `output/feature_cache/` (feature values are trial-specific, not phase-specific)

## Key Models (Implementation Priority)

1. **AutoCT** (PRIMARY) — LLM agents autonomously engineer tabular features, classical ML (XGBoost, TabPFN) predicts with SHAP explanations. LLM backbone: Claude Opus 4.6. RAG: LinearRAG over 7 sources (ClinicalTrials.gov, PubMed, ChEMBL, FAERS, AACT, PrimeKG, Drugs@FDA). NER: GLiNER-BioMed with 16 zero-shot entity types. MCTS: Pareto multi-objective search (from PMMG) + adaptive branching (from AB-MCTS)
2. **HINT** (baseline) — Hierarchical attention GCN over drug/disease/criteria modalities. Includes the TOP benchmark dataset
3. **MEXA-CTP** (validation baseline) — Lightweight pairwise cross-attention with statement-level eligibility encoding. TOP benchmark SOTA
4. **LIFTED** (reference) — LLM-based multimodal fusion with Sparse MoE

Note: Reference implementations are not included in this repo. Only `repositories/ML2ClinicalTrials/Trialbench/` (benchmark dataset) is checked in. See README.md for external repo links.

## Input Modalities

AutoCT dynamically engineers features from 7 data sources via LLM agents with LinearRAG multi-hop retrieval — it is not restricted to fixed encoding pipelines or small-molecule drugs with SMILES. Data sources: ClinicalTrials.gov, PubMed, ChEMBL (drug mechanisms/targets, replaces DrugBank), FAERS/OpenFDA (adverse events), AACT (population-level trial statistics), PrimeKG (biological knowledge graph), Drugs@FDA (FDA approval history). For baseline comparison, other models encode three fixed modalities:
- **Drug:** SMILES molecular fingerprints (small molecules only)
- **Disease:** ICD-10 code embeddings
- **Eligibility criteria:** Free-text encoded at statement level (BioBERT/BioGPT)

## Implementation Roadmap

- **Phase 1** (4 weeks) — Set up LinearRAG + ChEMBL/FAERS/AACT/PrimeKG/Drugs@FDA indexes, integrate TabPFN, GLiNER-BioMed NER, reproduce AutoCT on TrialBench + HINT baseline on TOP
- **Phase 2** (5 weeks) — Upgrade LLM to Opus 4.6, align internal Merck trial data with ClinicalTrials.gov `protocolSection` format, implement Pareto multi-objective MCTS (from PMMG) + AB-MCTS adaptive branching, benchmark XGBoost + TabPFN on TOP
- **Phase 3** (5.5 weeks) — Production pipeline: data ingestion, REST API (NCT ID or internal trial ID), dashboard, MLOps
- **Phase 4** (ongoing) — Expanded retrieval sources, cross-disease transfer, failure reason classification

## Development Environment

- **Virtual environment:** `~/clinical-trial-risk-assesment-venv` (shared across all worktrees/branches)
- WSL2's `/mnt/c/` filesystem breaks uv bin entry points — the venv **must** live on the native Linux filesystem
- Run tools with: `VIRTUAL_ENV=~/clinical-trial-risk-assesment-venv uv run --active <command>`
- Install/sync with: `VIRTUAL_ENV=~/clinical-trial-risk-assesment-venv uv sync --active`

## Tech Stack

- **Python 3.10+**, uv for package management
- **LLM backbone:** Claude Opus 4.6 via DSPy framework
- **Classical ML:** XGBoost, TabPFN, shapiq (SHAP interaction values)
- **RAG:** LinearRAG (multi-hop retrieval via Personalized PageRank over entity co-occurrence graph)
- **NER:** GLiNER-BioMed (`Ihor/gliner-biomed-large-v1.0`) — zero-shot biomedical NER with 16 entity types
- **Data sources:** ClinicalTrials.gov, PubMed, ChEMBL, FAERS/OpenFDA, AACT, PrimeKG, Drugs@FDA
- **Entity resolution:** ChEMBL molecule_synonyms + PubChem PUG-REST (drug synonym expansion at query time)
- **Data:** polars, Parquet
- **Experiment tracking:** MLflow
- **Baseline models (for comparison):** PyTorch, HuggingFace Transformers, DeepChem, icdcodex, BioBERT
- **Infrastructure:** GPU recommended for TabPFN and GLiNER-BioMed; GPU needed for baseline reproduction
