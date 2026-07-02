# CTRA Implementation Project Plan

## Executive Summary

Implementation of a clinical trial outcome prediction system based on the AutoCT framework, enhanced with Claude Opus 4.6 (LLM backbone), LinearRAG (multi-hop retrieval), TabPFN (tabular foundation model), and expanded data sources (DrugBank, FAERS/OpenFDA). The system predicts trial success/failure with full per-prediction SHAP interpretability. Supports both public ClinicalTrials.gov trials and internal Merck trials via a data alignment layer.

**Total estimated effort:** 72.5 person-days (~14.5 weeks for 1 developer)
**Total estimated infrastructure cost:** ~$10,040 (first year)
**LLM API budget:** $5,000 (covers ~500 train/500 val/500 test trials with Opus 4.6 on-demand)

---

## Phase 1 — Infrastructure and Baseline Reproduction (4-6 weeks)

### 1.1 LinearRAG Setup and Indexing

Replace pgvector + txtai with LinearRAG for all retrieval.

| Task | Effort | Details |
|------|--------|---------|
| Install LinearRAG dependencies | 0.5 days | `sentence-transformers`, `spacy`, `python-igraph`, scispaCy biomedical NER model (`en_core_sci_scibert`) |
| Adapt notebook 0 (ClinicalTrials.gov) | 2 days | Convert CTG JSON → LinearRAG chunks format (`"idx:passage_text"`), index into LinearRAG with entity extraction |
| Adapt notebooks 3-5 (PubMed) | 2 days | Convert PubMed abstracts → LinearRAG chunks, index with date metadata for leakage prevention |
| Build DrugBank index | 1.5 days | Download DrugBank CSV (free academic license), parse drug properties, index into LinearRAG |
| Build FAERS index | 1 day | Query OpenFDA API for adverse event data, cache locally, index into LinearRAG |
| Implement date filtering | 1 day | Custom logic in LinearRAG wrapper — filter retrieved passages by publication/event date vs trial start date |
| Implement new tool functions | 1.5 days | Create `make_drugbank_search()`, `make_faers_search()` factory functions following existing pattern in `tools.py` |
| Replace txtai tools | 1 day | Modify `make_pubmed_search()`, `make_nct_search()` to use LinearRAG `retrieve()` instead of txtai `search()` |
| Tune LinearRAG hyperparameters | 1 day | Test `max_iterations`, `iteration_threshold`, `top_k_sentence` on clinical domain data |
| **Subtotal** | **11.5 days** | |

**Key risk:** LinearRAG has no built-in metadata filtering (date, source type). Custom wrapper logic needed for label leakage prevention. NER is handled by scispaCy's `en_core_sci_scibert` (transformer-based, trained on GENIA Treebank + PubMed Central) — already specified in LinearRAG's own install instructions and purpose-built for biomedical text. For more specialized entity recognition, supplementary scispaCy NER models are available: `en_ner_bc5cdr_md` (DISEASE + CHEMICAL, F1: 84.28%) and `en_ner_bionlp13cg_md` (15 entity types including CANCER, GENE, TISSUE, F1: 77.84%).

### 1.2 TabPFN Classifier Integration

Add TabPFN as fourth classifier alongside LR, RF, XGBoost.

| Task | Effort | Details |
|------|--------|---------|
| Install TabPFN | 0.5 days | `pip install tabpfn "tabpfn-extensions[interpretability]"` + verify GPU compatibility |
| Modify `train_simple_model_v2()` | 1 day | Add `"tabpfn"` model type branch in `agent.py:2365-2482`. TabPFN needs no `ColumnTransformer` — pass raw DataFrame directly |
| Update `OutputV2` NamedTuple | 0.5 days | Add `tabpfn_eval_output` and `test_tabpfn_eval_output` fields, update `get_best_eval_output()` |
| Update `AgentV2.forward()` | 0.5 days | Add TabPFN training + evaluation calls (lines 2111-2155) |
| Update `predict.py` | 0.5 days | Ensure best model selection works with TabPFN pipeline |
| SHAP validation | 0.5 days | Verify `tabpfn-extensions[interpretability]` SHAP output format matches existing SHAP display |
| **Subtotal** | **3.5 days** | |

**Key risk:** TabPFN-2.5 has non-commercial license. For production use, TabPFN v2 commercial license needed (contact sales@priorlabs.ai). GPU recommended — CPU feasible only for ≤1K samples.

### 1.3 Reproduce AutoCT Baseline

Validate the pipeline works end-to-end with original settings.

| Task | Effort | Details |
|------|--------|---------|
| Environment setup | 0.5 days | Create conda env, install all dependencies, verify API keys |
| Run data preparation | 1 day | Execute adapted notebooks (LinearRAG indexing instead of txtai) |
| Reproduce on TrialBench | 1.5 days | MCTS training (10 rollouts, depth 10) with gpt-4o-mini. Target: ROC-AUC 0.753/0.639/0.702 |
| Validate TabPFN improvement | 0.5 days | Compare 4-model results (LR, RF, XGBoost, TabPFN) vs published 3-model results |
| Run HINT baseline on TOP | 1 day | Reproduce HINT F1: 0.665/0.620/0.847 for comparison |
| Document results | 0.5 days | Internal benchmark report |
| **Subtotal** | **5.5 days** | |

### Phase 1 Cost Estimate

| Item | Cost | Notes |
|------|------|-------|
| gpt-4o-mini API (reproduction) | ~$150 | 10 rollouts, 100 train/100 val |
| DrugBank license | $0 | Free for academic/non-profit |
| OpenFDA API | $0 | Public, no auth needed |
| GPU instance (HINT baseline + TabPFN) | ~$50 | g4dn.xlarge ($0.526/hr) × ~100 hrs |
| Sentence-transformer embedding | ~$20 | One-time GPU cost for LinearRAG indexing |
| **Phase 1 Total** | **~$220** | |

### Phase 1 Effort Summary: **20.5 days** (~4 weeks)

---

## Phase 2 — LLM Backbone Upgrade, Internal Data Alignment, and Optimization (5 weeks)

### 2.1 Claude Opus 4.6 Integration

| Task | Effort | Details |
|------|--------|---------|
| Swap DSPy LM configuration | 0.5 days | Modify `agent.py:46-51` from `openai/gpt-4o-mini` to `anthropic/claude-opus-4-6`, swap API key env var |
| Update tokenizer reference | 0.5 days | Replace `tiktoken.encoding_for_model("gpt-4o-mini")` in `tools.py:12` |
| Test all DSPy signatures | 2 days | Validate all 15+ DSPy Signatures produce correctly structured output with Opus 4.6 (feature proposers, planners, builders, evaluators) |
| Fix output format issues | 1 day | Buffer for DSPy assertion failures / output parsing differences between models |
| **Subtotal** | **4 days** | |

### 2.2 Internal Trial Data Alignment

AutoCT's LLM agents query trial data via `get_clinical_trial_info_from_clinical_trials_gov_dict()` in `tools.py:24-40`, which reads from `datasets/ctg-studies-with-nctid.parquet` and expects the ClinicalTrials.gov `protocolSection` nested structure. Internal Merck trials must be mapped to this same schema so the feature building agents can process them.

TrialBench provides only the task splits — `(nctid, label)` pairs. It is NOT the format used for feature building. The LLM agents work against ClinicalTrials.gov data.

The pipeline has two data paths that need alignment:
1. **Task splits** — `(trial_id, label)` Parquet files read by `Task.get_train_val_test()` in `agent.py`. Simple 2-column format.
2. **Trial detail lookup** — `get_clinical_trial_info_from_clinical_trials_gov_dict()` in `tools.py:24-40` queries DuckDB for the full ClinicalTrials.gov `protocolSection.*` nested structure. This is what the LLM agents use to research each trial during feature building.

| Task | Effort | Details |
|------|--------|---------|
| Audit internal trial data schema | 1.5 days | Map internal trial management system fields to ClinicalTrials.gov `protocolSection` fields (see mapping table below) |
| Build data alignment ETL | 3 days | Transform internal trial records into ClinicalTrials.gov-compatible Parquet format. Must produce: `nctId` (or internal trial ID), `startDate`, `protocolSection.*` nested structure matching the DuckDB query in `tools.py:32` |
| Extend trial ID system | 1 day | Support internal trial IDs (not just NCT IDs) throughout: `Task.get_train_val_test()`, `tools.py` lookups, `predict.py`, LinearRAG indexing |
| Create internal task splits | 1.5 days | Generate `tasks/internal_trial_approval/{train,val,test}_data.parquet` with `(trial_id, label)` from internal trial outcomes. Stratified sampling across phases and therapeutic areas |
| Index internal trials into LinearRAG | 1 day | Add internal trial protocols to the LinearRAG index with date metadata |
| Validate end-to-end on internal data | 1.5 days | Run MCTS on internal trial splits, compare predictions against known outcomes, identify failure modes by therapeutic area |
| Document generalization gap | 0.5 days | Report comparing TrialBench performance vs internal data performance |
| **Subtotal** | **10 days** | |

**Key data mapping (ClinicalTrials.gov `protocolSection` → internal system):**

| AutoCT expects (ClinicalTrials.gov) | Description | Internal system equivalent |
|------|---------|---------|
| `protocolSection.identificationModule.nctId` | Trial identifier | Internal trial ID |
| `protocolSection.statusModule.startDateStruct.date` | Trial start date (critical for label leakage filtering) | Trial start date |
| `protocolSection.designModule.phases` | Trial phase | Trial phase |
| `protocolSection.designModule.studyType` | Study type (interventional, etc.) | Study type |
| `protocolSection.designModule.designInfo` | Allocation, masking, purpose | Trial design metadata |
| `protocolSection.eligibilityModule.eligibilityCriteria` | Eligibility criteria free text | Protocol eligibility section |
| `protocolSection.eligibilityModule.sex` | Gender eligibility | Gender criteria |
| `protocolSection.eligibilityModule.minimumAge` / `maximumAge` | Age range | Age criteria |
| `protocolSection.armsInterventionsModule.armGroups` | Treatment arms | Treatment arms |
| `protocolSection.armsInterventionsModule.interventions` | Drug/intervention details | Intervention descriptions |
| `protocolSection.outcomesModule.primaryOutcomes` | Primary endpoints | Primary endpoints |
| `protocolSection.contactsLocationsModule.locations` | Site locations | Site locations |
| `protocolSection.descriptionModule.briefSummary` | Protocol summary | Protocol synopsis |

**Key risk:** Internal trial data may have different completeness levels than ClinicalTrials.gov public data. Missing fields (e.g., no eligibility criteria text, no structured endpoints) will cause LLM feature extraction failures. The alignment layer must handle these gracefully with sensible defaults. The `FeatureBuilderV3` in `agent.py` already returns `None` for features it cannot extract, so partial data is tolerable but will reduce prediction quality.

### 2.3 Benchmarking and Optimization

| Task | Effort | Details |
|------|--------|---------|
| Benchmark Opus 4.6 vs gpt-4o-mini | 2 days | Run identical MCTS config, compare feature quality, ROC-AUC, cost per run |
| Increase MCTS rollouts | 1 day | Test 20+ rollouts with Opus 4.6 |
| Expand training sample size | 2 days | Scale to 200-500 train/val samples, measure accuracy improvement |
| Evaluate all 8 task types | 2 days | Trial approval (all phases), adverse events, dropout, mortality |
| Evaluate on TOP benchmark | 1.5 days | Direct comparison with HINT/MEXA-CTP results |
| Cost optimization (batch API + caching) | 1.5 days | Implement prompt caching for system prompts, evaluate Batch API feasibility |
| Document results | 0.5 days | Comparative benchmark report |
| **Subtotal** | **10.5 days** | |

### Phase 2 Cost Estimate

| Item | Cost | Notes |
|------|------|-------|
| Claude Opus 4.6 — benchmarking runs | ~$2,000 | Multiple MCTS runs at different sample sizes |
| Claude Opus 4.6 — full training run | ~$2,500 | 500 train/500 val, 10-20 rollouts, on-demand pricing |
| Claude Opus 4.6 — internal data run | ~$500 | MCTS run on internal trial data |
| gpt-4o-mini comparison run | ~$150 | Baseline comparison |
| GPU instance (TabPFN inference) | ~$50 | g4dn.xlarge for TabPFN during training |
| **Phase 2 Total** | **~$5,200** | |

### Phase 2 Effort Summary: **24.5 days** (~5 weeks)

---

## Phase 3 — Production Pipeline (6-8 weeks)

### 3.1 Data Ingestion Service

| Task | Effort | Details |
|------|--------|---------|
| ClinicalTrials.gov API v2 connector | 2 days | Automated download of new trials, incremental updates |
| PubMed Entrez connector | 1 day | Automated fetching of new publications with date tracking |
| DrugBank update pipeline | 1 day | Periodic re-download and re-indexing (DrugBank releases quarterly) |
| FAERS/OpenFDA connector | 1 day | Automated querying of new adverse event reports |
| LinearRAG incremental indexing | 2 days | Support adding new documents to existing index without full rebuild |
| **Subtotal** | **7 days** | |

### 3.2 Prediction Service

| Task | Effort | Details |
|------|--------|---------|
| REST API design | 1 day | Endpoints: `POST /predict` (NCT ID → prediction + SHAP), `GET /health`, `GET /models` |
| API implementation (FastAPI) | 3 days | Load best MCTS node, build features for new trial, run 4 models, return best prediction + SHAP |
| SHAP explanation serialization | 1 day | JSON format for per-feature SHAP values, feature importance ranking |
| Model versioning | 1 day | Track which MCTS checkpoint, feature plans, and model weights are in use |
| Authentication + rate limiting | 1 day | API key auth, request throttling |
| Containerization (Docker) | 1.5 days | Dockerfile for API + LinearRAG + models, docker-compose for full stack |
| **Subtotal** | **8.5 days** | |

### 3.3 Dashboard UI

| Task | Effort | Details |
|------|--------|---------|
| UI framework setup | 1 day | Streamlit or React dashboard |
| Trial prediction view | 2 days | Input NCT ID, display prediction probability, SHAP waterfall plot |
| Feature importance view | 1 day | Global feature importance across training set, per-trial feature contributions |
| Trial portfolio view | 2 days | Batch prediction across multiple trials, risk heatmap by phase/therapeutic area |
| Historical performance | 1 day | Model accuracy tracking over time, benchmark comparison charts |
| **Subtotal** | **7 days** | |

### 3.4 MLOps and Monitoring

| Task | Effort | Details |
|------|--------|---------|
| MLflow integration | 1 day | Track MCTS runs, feature plans, model metrics, SHAP distributions |
| Retraining pipeline | 2 days | Automated MCTS re-run when new data arrives, model comparison, promotion |
| Monitoring and alerting | 1 day | API latency, prediction drift, feature extraction failures |
| Documentation | 1 day | API docs, deployment guide, runbook |
| **Subtotal** | **5 days** | |

### Phase 3 Cost Estimate

| Item | Cost | Notes |
|------|------|-------|
| Claude Opus 4.6 — inference (monthly) | ~$200/mo | ~100 predictions/month at ~$2 per prediction |
| AWS EC2 — API server | ~$75/mo | t3.xlarge for API + LinearRAG |
| AWS EC2 — GPU (TabPFN) | ~$100/mo | g4dn.xlarge on-demand for batch inference |
| S3 storage | ~$10/mo | Model artifacts, LinearRAG indexes |
| **Phase 3 Annual Infra** | **~$4,620/yr** | |

### Phase 3 Effort Summary: **27.5 days** (~5.5 weeks)

---

## Phase 4 — Enhancements (Ongoing)

| Enhancement | Effort | Priority |
|-------------|--------|----------|
| Expanded retrieval sources (FDA docs, Cochrane, Europe PMC, ChEMBL) | 5-8 days | Medium |
| MCTS hyperparameter optimization (rollouts, depth, UCT alpha) | 3-5 days | High |
| Cross-disease feature transfer analysis | 3-5 days | Medium |
| Failure reason classification (4-class) | 5-7 days | Medium |
| Label noise handling (cleanlab) | 2-3 days | Low |
| Trial complexity scoring (Markey features) | 2-3 days | Low |
| Full-text paper retrieval (Europe PMC) | 3-5 days | Medium |
| Endpoint extraction NER | 3-5 days | Low |

---

## Total Project Summary

### Effort

| Phase | Duration | Effort (person-days) |
|-------|----------|---------------------|
| Phase 1 — Infrastructure + Baselines | 4 weeks | 20.5 days |
| Phase 2 — LLM Upgrade + Internal Data + Optimization | 5 weeks | 24.5 days |
| Phase 3 — Production Pipeline | 5.5 weeks | 27.5 days |
| **Total (Phases 1-3)** | **~14.5 weeks** | **72.5 days** |
| Phase 4 — Enhancements (selective) | Ongoing | 26-41 days |

### Cost

| Category | Phase 1 | Phase 2 | Phase 3 (Year 1) | Total Year 1 |
|----------|---------|---------|-------------------|--------------|
| LLM API (Claude Opus 4.6) | $150 | $5,150 | $2,400 | **$7,700** |
| GPU compute | $70 | $50 | $1,200 | **$1,320** |
| Other infrastructure | $0 | $0 | $1,020 | **$1,020** |
| Licenses/data | $0 | $0 | $0* | **$0*** |
| **Total** | **$220** | **$5,200** | **$4,620** | **$10,040** |

\*TabPFN-2.5 is free for non-commercial use. Commercial license pricing from Prior Labs TBD if needed for production.

### LLM API Cost Breakdown (Opus 4.6 on AWS Bedrock)

| Pricing | Input | Output |
|---------|-------|--------|
| On-demand | $5.00 / MTok | $25.00 / MTok |
| Batch API (50% off) | $2.50 / MTok | $12.50 / MTok |
| Cache hits (90% off input) | $0.50 / MTok | N/A |

| Scenario | On-demand | Batch API | With caching |
|----------|-----------|-----------|-------------|
| 100 train / 100 val / 100 test | ~$1,029 | ~$514 | ~$350 |
| 200 / 200 / 200 | ~$2,025 | ~$1,012 | ~$700 |
| 500 / 500 / 500 | ~$5,014 | ~$2,507 | ~$1,750 |

### Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| DSPy signatures fail with Opus 4.6 output format | High | Medium | Phase 2 includes 2 days for format debugging + assertion fixes |
| LinearRAG biomedical NER quality | Medium | Low | `en_core_sci_scibert` (already in LinearRAG install instructions) is purpose-built for biomedical text. Can supplement with `en_ner_bc5cdr_md` (disease/chemical) or `en_ner_bionlp13cg_md` (15 biomedical entity types) if needed |
| LinearRAG date filtering not native | Medium | High | Custom wrapper already planned in Phase 1 (1 day allocated) |
| TabPFN non-commercial license blocks production | High | Low | Fallback to XGBoost (already integrated); contact Prior Labs for commercial license |
| Opus 4.6 cost exceeds budget | Medium | Medium | Use Batch API + prompt caching; fall back to Sonnet 4.6 ($3/$15 per MTok) for feature building |
| DrugBank academic license restrictions | Low | Low | Free for non-profit; commercial license available if needed |
| Internal trial data incompleteness | High | High | Internal trials may lack fields AutoCT expects (eligibility text, structured endpoints). Alignment layer must handle missing fields with sensible defaults; feature builder already returns `None` for missing data |
| Internal trial data schema drift | Medium | Medium | Internal trial management system schema may change over time. ETL pipeline should validate schema on each run and flag mismatches |
| Generalization gap (TrialBench → internal) | High | Medium | Models trained on public ClinicalTrials.gov data may not transfer to internal data distribution. Phase 2 includes 1.5 days for end-to-end validation on internal splits + gap analysis |

### Key Dependencies

```
Phase 1 ──────────────────► Phase 2 ──────────────────────────► Phase 3
                                │                                    │
LinearRAG indexes ──► Opus 4.6 swap ──────────► Production API      │
DrugBank/FAERS ──────► Internal data ETL ──────► Data ingestion     │
TabPFN integration ──► Internal trial splits ──► Dashboard          │
HINT baseline ───────► TOP comparison                                │
                       Internal validation ────► Monitoring          │
                                                                     ▼
                                                                 Phase 4
                                                     (Enhancements)
```
