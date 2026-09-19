# CTRA Implementation Plan

> Comprehensive implementation plan for Clinical Trial Risk Assessment. Produced 2026-03-26 from codebase analysis of AutoCT, LinearRAG, PMMG, TreeQuest, LLMFE, AIDE, and TrialBench repositories, plus TabPFN v2.5 documentation review.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Workstream 1 — Notebook Migration to Production Python](#2-workstream-1--notebook-migration-to-production-python)
3. [Workstream 2 — XGBoost + TabPFN v2.5 (Models)](#3-workstream-2--xgboost--tabpfn-v25-models)
4. [Workstream 3 — LinearRAG with NER + Temporal Filtering](#4-workstream-3--linearrag-with-ner--temporal-filtering)
5. [Workstream 4 — Claude Opus 4.6 DSPy Integration with Extended Thinking](#5-workstream-4--claude-opus-46-dspy-integration-with-extended-thinking)
6. [Workstream 5 — MCTS Improvements](#6-workstream-5--mcts-improvements)
7. [Workstream 6 — Additional Datasets + Extended Training](#7-workstream-6--additional-datasets--extended-training)
8. [Workstream 7 — Extended MCTS Rollouts (≥20)](#8-workstream-7--extended-mcts-rollouts-20)
9. [Workstream 8 — Feature Extraction Optimization](#9-workstream-8--feature-extraction-optimization)
10. [Dependency Graph](#10-dependency-graph)
11. [Risk Register](#11-risk-register)
12. [Cost Estimate](#12-cost-estimate)

---

## 1. Executive Summary

This plan transforms the AutoCT research prototype into a production-grade CTRA pipeline with 8 workstreams. Key decisions:

- **Models:** XGBoost + TabPFN v2.5 only (remove Logistic Regression and Random Forest)
- **TabPFN:** v2.5 with fine-tuning support, pickle-compatible saving via `save_fitted_tabpfn_model()`
- **Notebooks → Scripts:** 6 Jupyter notebooks converted to idempotent CLI scripts
- **RAG:** LinearRAG replaces txtai, with `en_core_sci_scibert` NER and date-gated temporal filtering
- **LLM:** Claude Opus 4.6 via DSPy with extended thinking (budget tokens), following the DIFW agent pattern
- **MCTS:** 6 targeted fixes + Pareto multi-objective search (2 objectives per phase) + AB-MCTS adaptive branching. See `research/mcts-implementation-design.md` § "As Implemented" for current design vs. planned design.
- **Data:** TrialBench + TOP + CTOD benchmarks; training expanded to 200-500 samples
- **Rollouts:** Minimum 20 MCTS rollouts (up from default 10)
- **Feature Caching:** Per-phase feature value store (namespaced by `Task.output_subdir`, so reuse is cross-branch within one phase's tree, not global) — projected 49-66% reduction in LLM calls

---

## 2. Workstream 1 — Notebook Migration to Production Python

### Current State

AutoCT has 6 sequential data preparation notebooks in `notebooks/`:

| Notebook | Purpose | Key Dependencies |
|----------|---------|-----------------|
| `0_generate_ctg.ipynb` | Download clinical trials from ClinicalTrials.gov API | `requests`, ClinicalTrials.gov API |
| `1_generate_trialbench.ipynb` | Create benchmark splits from CTG data | `pandas`, `trialbench` |
| `2_clinical_trial_retrieval.ipynb` | Build txtai embedding index for trials (`datasets/nct-index`) | `txtai`, `sentence-transformers` |
| `3_generate_pubmed.ipynb` | Download PubMed articles from fixed PMID list (`pmids.parquet`) | `requests`, PubMed E-utilities API |
| `4_clean_pubmed.ipynb` | Clean/preprocess PubMed data | `pandas` |
| `5_pubmed_retrieval.ipynb` | Build txtai embedding index for PubMed (`datasets/pubmed-index`) | `txtai`, `sentence-transformers` |

### Target Architecture

Replace all 6 notebooks with CLI scripts under `scripts/data/`:

```
scripts/data/
├── fetch_ctg.py              ← notebook 0
├── build_trialbench_splits.py ← notebook 1
├── build_linearrag_index.py   ← notebooks 2+5 (merged, now LinearRAG)
├── fetch_pubmed.py            ← notebook 3
├── clean_pubmed.py            ← notebook 4
├── fetch_drugbank.py          ← NEW: DrugBank data acquisition
├── fetch_faers.py             ← NEW: FAERS/OpenFDA data acquisition
└── build_all.py               ← Orchestrator: runs all above in order
```

### Implementation Details

#### 2.1 `fetch_ctg.py` (from notebook 0)

```
Purpose: Download clinical trial records from ClinicalTrials.gov API v2
Input: None (or optional NCT ID list)
Output: datasets/ctg-studies.parquet
CLI: python scripts/data/fetch_ctg.py [--nctids nctids.txt] [--output datasets/ctg-studies.parquet]
```

- Extract all API calls and data transformations from notebook cells
- Add `--resume` flag for interrupted downloads (checkpoint to partial parquet)
- Add `--date-range` filter for incremental updates
- Remove all `tqdm.notebook` → use `tqdm.auto`
- Remove `IPython.display` imports
- Add structured logging (`logging` module, not `print`)

#### 2.2 `build_trialbench_splits.py` (from notebook 1)

```
Purpose: Generate train/val/test Parquet splits per task and phase
Input: datasets/ctg-studies.parquet
Output: tasks/{task_name}/[phase{1-4}_]train|val|test_data.parquet
CLI: python scripts/data/build_trialbench_splits.py --input datasets/ctg-studies.parquet --output-dir tasks/
```

- Preserve existing stratified splitting logic
- Add `--tasks` filter (e.g., `--tasks trial_approval,adverse_event`)
- Add `--sample-size` parameter (for scaling to 200-500 train/val — see Workstream 6)
- Validate output: check label distribution, no empty splits

#### 2.3 `build_linearrag_index.py` (merged notebooks 2+5, replacing txtai)

```
Purpose: Build LinearRAG graph indexes for all data sources
Input: datasets/ctg-studies.parquet, datasets/pubmed-cleaned.parquet, [drugbank.parquet], [faers.parquet]
Output: datasets/linearrag-index/ (passage/entity/sentence parquet + graph.graphml)
CLI: python scripts/data/build_linearrag_index.py --sources ctg,pubmed,drugbank,faers --output datasets/linearrag-index/
```

- **This is the biggest migration**: replaces txtai with LinearRAG (see Workstream 3)
- Merges CTG trial index + PubMed index into a single multi-source LinearRAG graph
- Adds date metadata per document for temporal filtering
- Uses `en_core_sci_scibert` NER model for biomedical entity extraction
- Outputs: `passage_embedding.parquet`, `entity_embedding.parquet`, `sentence_embedding.parquet`, `graph.graphml`, `ner_results.json`

#### 2.4 `fetch_pubmed.py` (from notebook 3)

```
Purpose: Download PubMed articles by PMID list
Input: datasets/pmids.parquet (existing list from AutoCT)
Output: datasets/pubmed-raw.parquet
CLI: python scripts/data/fetch_pubmed.py --pmids datasets/pmids.parquet --output datasets/pubmed-raw.parquet
```

- Use PubMed E-utilities API with rate limiting (3 req/sec without API key, 10 with)
- Add `--api-key` parameter for NCBI API key
- Store: PMID, title, abstract, authors, journal, publication date, MeSH terms
- Add `--resume` for interrupted downloads

#### 2.5 `clean_pubmed.py` (from notebook 4)

```
Purpose: Clean and preprocess PubMed data
Input: datasets/pubmed-raw.parquet
Output: datasets/pubmed-cleaned.parquet
CLI: python scripts/data/clean_pubmed.py --input datasets/pubmed-raw.parquet --output datasets/pubmed-cleaned.parquet
```

- Standardize date formats (ISO 8601)
- Remove duplicate entries
- Clean HTML entities from abstracts
- Validate: no null PMIDs, no empty abstracts

#### 2.6 `fetch_drugbank.py` (NEW)

```
Purpose: Download and process DrugBank data for drug mechanism, targets, interactions
Input: DrugBank XML dump (requires academic license)
Output: datasets/drugbank.parquet
CLI: python scripts/data/fetch_drugbank.py --xml drugbank_all_full_database.xml --output datasets/drugbank.parquet
```

- Parse DrugBank XML → extract: drug name, targets, mechanisms, interactions, approval status
- Map drug names to ClinicalTrials.gov intervention names
- Store publication/approval dates for temporal filtering

#### 2.7 `fetch_faers.py` (NEW)

```
Purpose: Download FAERS/OpenFDA adverse event data
Input: None (queries OpenFDA API)
Output: datasets/faers.parquet
CLI: python scripts/data/fetch_faers.py --output datasets/faers.parquet [--drug-list drugs.txt]
```

- Query OpenFDA drug/event API
- Extract: drug name, adverse event type, count, seriousness, report date
- Rate limiting (240 req/min with API key)

#### 2.8 `build_all.py` (Orchestrator)

```
Purpose: Run full data pipeline end-to-end
CLI: python scripts/data/build_all.py [--skip-fetch] [--sources ctg,pubmed,drugbank,faers]
```

- Runs scripts 2.1-2.7 in dependency order
- `--skip-fetch` skips download steps (uses cached data)
- Validates all outputs exist before proceeding to next step
- Logs total runtime and data statistics

### Effort Estimate

| Task | Days |
|------|------|
| Migrate notebooks 0-1 (CTG + splits) | 1.5 |
| Migrate notebooks 3-4 (PubMed fetch + clean) | 1 |
| New: DrugBank + FAERS fetch scripts | 2 |
| LinearRAG index builder (replaces notebooks 2+5) | 3 (covered in Workstream 3) |
| Orchestrator + testing | 1 |
| **Total** | **5.5 days** (excluding LinearRAG index builder) |

---

## 3. Workstream 2 — XGBoost + TabPFN v2.5 (Models)

### Current State

AutoCT's `train_simple_model_v2()` in `agent.py:2365` trains three models: XGBoost, Logistic Regression, Random Forest. The `OutputV2` NamedTuple at `agent.py:266` stores results for all three. No TabPFN integration exists.

### Changes Required

#### 3.1 Remove LR and RF, Add TabPFN

**File: `src/lfe/impl/agent.py`**

Modify `train_simple_model_v2()`:
- Remove `model_type` options `"rf"` and `"logistic"`
- Add `"tabpfn"` as new model type
- TabPFN **does not need** `ColumnTransformer` — it natively handles missing values, categoricals, and booleans

```python
# New model_type: Literal["xgb", "tabpfn"]
elif model_type == "tabpfn":
    from tabpfn import TabPFNClassifier
    model = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    model.fit(input_X_df, y_train)  # raw DataFrame, no ColumnTransformer
```

Modify `OutputV2` NamedTuple at `agent.py:266`:
- Remove `lr_eval_output` and `rf_eval_output` fields
- Add `tabpfn_eval_output: EvalOutput` field
- Add `test_tabpfn_eval_output: ModelEvalResult` field

Modify `get_best_eval_output()` at `agent.py:283`:
- Update to compare only XGBoost and TabPFN results

Modify `AgentV2.forward()` at `agent.py:2111`:
- Remove LR and RF training calls
- Add TabPFN training + evaluation call

#### 3.2 TabPFN v2.5 Fine-Tuning

**When to use fine-tuning vs standard TabPFN:**
- **Standard TabPFN (default):** In-context learning, no gradient updates. Best for datasets ≤10K rows where the distribution is close to TabPFN's pre-training distribution. This is the default for CTRA.
- **Fine-tuned TabPFN:** When standard TabPFN underperforms XGBoost, or when the clinical trial data distribution significantly diverges from TabPFN's priors. Requires GPU with ≥80GB VRAM (A100/H100).

**Implementation:**

```python
# Standard (default path)
from tabpfn import TabPFNClassifier
model = TabPFNClassifier(device="cuda")
model.fit(X_train, y_train)

# Fine-tuned (optional, when standard underperforms)
from tabpfn.finetuning import FinetunedTabPFNClassifier
model = FinetunedTabPFNClassifier(
    device="cuda",                     # CUDA required
    epochs=30,                         # Fine-tuning iterations
    learning_rate=2e-5,                # Gradient descent step size
    n_estimators_finetune=2,           # Ensemble members during fine-tuning
    n_estimators_final_inference=4,    # Ensemble for predictions
    random_state=42,
)
model.fit(X_train, y_train)
```

**Configuration:** Add `--tabpfn-finetune` flag to `scripts/train_mcts.py` to toggle between standard and fine-tuned TabPFN. Default: standard (no fine-tuning).

#### 3.3 Model Saving (Pickle-Compatible)

Use TabPFN's dedicated save/load API (not raw pickle) for production-grade persistence:

```python
from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model

# Save after training
save_fitted_tabpfn_model(model, "output/tabpfn_model.tabpfn_fit")

# Load later (cross-device: train on GPU, load on CPU)
model = load_fitted_tabpfn_model("output/tabpfn_model.tabpfn_fit", device="cpu")
```

**Why not raw pickle:**
- `save_fitted_tabpfn_model()` saves the model **independent of the full checkpoint** (smaller file size)
- Cross-device loading (train on GPU, deploy on CPU) works out of the box
- Not dependent on pickle serialization format changes between versions

**For XGBoost:** Continue using existing pickle serialization (already works).

#### 3.4 SHAP Interpretability

Both models support SHAP:
- **XGBoost:** TreeSHAP (fast, exact for tree models) — already integrated
- **TabPFN:** Permutation SHAP via `tabpfn-extensions[interpretability]`

```python
from tabpfn_extensions import interpretability

shap_values = interpretability.shap.get_shap_values(
    estimator=tabpfn_model,
    test_x=X_test[:n_samples],
    attribute_names=feature_names,
    algorithm="permutation",
)
fig = interpretability.shap.plot_shap(shap_values)
```

**Note:** Permutation SHAP is slower than TreeSHAP. For large test sets, subsample to ~100-200 instances for SHAP computation.

#### 3.5 Dependencies

Add to `environment.yml`:
```yaml
- tabpfn>=2.5
- tabpfn-extensions[interpretability]
```

### Effort Estimate

| Task | Days |
|------|------|
| Remove LR/RF, refactor OutputV2 and agent pipeline | 1.5 |
| Integrate TabPFN (standard + fine-tuning toggle) | 2 |
| Model saving (TabPFN dedicated API + XGBoost pickle) | 0.5 |
| SHAP integration for TabPFN | 1 |
| Testing: TabPFN vs XGBoost on TrialBench Phase II | 1 |
| **Total** | **6 days** |

---

## 4. Workstream 3 — LinearRAG with NER + Temporal Filtering

### Current State

AutoCT uses **txtai** for semantic search with two separate embedding indexes:
- `datasets/nct-index` — ClinicalTrials.gov trials
- `datasets/pubmed-index` — PubMed articles

Temporal filtering is done at query time via txtai SQL: `WHERE start_date < :s AND similar(:q)`.

### Target Architecture

Replace txtai with **LinearRAG** — a graph-based RAG system using entity-sentence-passage graphs with Personalized PageRank for multi-hop retrieval. Zero LLM cost during indexing.

#### 4.1 NER Model Selection

**Recommended: `en_core_sci_scibert`** (SciSpaCy with SciBERT backbone)

- Purpose-built for biomedical/scientific text
- Recognizes: diseases, chemicals, genes/proteins, cell types, organisms
- Installation: `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz`
- Fallback: `en_core_web_trf` (general-purpose, lower recall on biomedical entities)

**Why not the default `en_core_web_trf`:**
- Misses drug names, disease subtypes, gene symbols
- Clinical trial text has dense biomedical terminology that general NER struggles with
- `en_core_sci_scibert` achieves ~85% F1 on biomedical NER benchmarks vs ~60% for general models

**Modification in LinearRAG:**

```python
# In src/ner.py — change model initialization
class SpacyNER:
    def __init__(self, model_name="en_core_sci_scibert"):  # was "en_core_web_trf"
        self.nlp = spacy.load(model_name)
```

#### 4.2 Temporal Filtering (Data Leakage Prevention)

**Problem:** When building features for a trial that started on 2020-01-15, the RAG system must NOT return documents published after that date. Otherwise, the model learns from future information (label leakage).

**Solution: Four-layer temporal gating**

##### Layer 1: Date Metadata in Embedding Store

Extend `EmbeddingStore` to track document dates:

```python
# Modified embedding_store.py
class EmbeddingStore:
    def insert_text(self, text_list, metadata_list=None):
        """metadata_list: [{"date": "2020-01-15", "source": "pubmed"}, ...]"""
        for idx, text in enumerate(text_list):
            hash_id = compute_mdhash_id(text)
            self.hash_id_to_date[hash_id] = metadata_list[idx]["date"] if metadata_list else None
```

Parquet schema adds a `date` column alongside `hash_id`, `text`, `embedding`.

##### Layer 2: Temporal Seed Entity Filtering

When retrieving seed entities, filter by date:

```python
def get_seed_entities(self, question, query_date=None):
    question_entities = self.spacy_ner.question_ner(question)
    if query_date:
        # Only match entities from documents published before query_date
        valid_indices = [i for i, hid in enumerate(self.entity_hash_ids)
                        if self.entity_dates[hid] is None or self.entity_dates[hid] <= query_date]
        valid_embeddings = self.entity_embeddings[valid_indices]
    # ... similarity matching on valid_embeddings only
```

##### Layer 3: Temporal BFS Propagation

During entity propagation, skip sentences from future documents:

```python
def calculate_entity_scores(self, question_embedding, seed_entities, query_date=None):
    for entity_hash_id, (score, tier) in current_entities.items():
        sentence_hash_ids = self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]
        for sid in sentence_hash_ids:
            # Skip sentences from documents published after query_date
            if query_date and self.sentence_dates[sid] > query_date:
                continue
            # ... normal propagation
```

##### Layer 4: Temporal Retrieve Interface

Modify the main `retrieve()` and `qa()` methods:

```python
def retrieve(self, question, query_date=None):
    seed_entities = self.get_seed_entities(question, query_date)
    entity_scores = self.calculate_entity_scores(question_embedding, seed_entities, query_date)
    passages = self.run_ppr(entity_scores)  # PPR operates on filtered subgraph
    # Final filter: only return passages with date <= query_date
    return [p for p in passages if self.passage_dates[p.hash_id] is None or self.passage_dates[p.hash_id] <= query_date]
```

##### Date Sources Per Data Type

| Data Source | Date Field | Extraction |
|-------------|-----------|------------|
| ClinicalTrials.gov trials | `protocolSection.statusModule.startDateStruct.date` | Parse from CTG JSON |
| PubMed articles | `DateAvail` or `PubDate` | Parse from PubMed XML |
| DrugBank entries | First approval date or publication date | Parse from DrugBank XML |
| FAERS reports | `receiptdate` or `receivedate` | Parse from OpenFDA JSON |

#### 4.3 Integration with AutoCT Tools

Replace `tools.py` search functions to use LinearRAG instead of txtai:

**Current (txtai):**
```python
def make_pubmed_search(nct_info):
    start_date = nct_info["startDate"]
    pubmed_embeddings = Embeddings()
    pubmed_embeddings.load("./datasets/pubmed-index")
    def pubmed_search(query):
        return pubmed_embeddings.search("...similar(:q)...", parameters={"q": query, "s": start_date})
    return pubmed_search
```

**New (LinearRAG):**
```python
def make_pubmed_search(nct_info):
    start_date = nct_info["startDate"]
    rag = LinearRAG(LinearRAGConfig(
        dataset_name="ctra-index",
        spacy_model="en_core_sci_scibert",
        enable_temporal_filtering=True,
    ))
    rag.load_index("./datasets/linearrag-index/")

    def pubmed_search(query):
        results = rag.retrieve(query, query_date=start_date)
        return format_as_pubmed_results(results)  # Convert to existing output format
    return pubmed_search
```

**Key: Unified multi-source index.** Unlike txtai (separate indexes for CTG and PubMed), LinearRAG builds a single graph spanning ALL data sources. Entities from PubMed papers link to entities in ClinicalTrials.gov trials link to entities in DrugBank — enabling multi-hop retrieval across sources (e.g., drug → target → disease → prior trials → outcomes).

#### 4.4 LinearRAG Configuration for CTRA

```python
LinearRAGConfig(
    # Model selection
    embedding_model="all-mpnet-base-v2",          # Same as default
    spacy_model="en_core_sci_scibert",            # Biomedical NER

    # Chunking
    chunk_token_size=1000,                        # Passage length (default)
    chunk_overlap_token_size=100,                 # Overlap (default)

    # Retrieval parameters
    retrieval_top_k=10,                           # Top-10 passages (match AutoCT's current limit)
    max_iterations=3,                             # BFS depth (default)
    top_k_sentence=1,                             # Sentences per entity per BFS iteration
    iteration_threshold=0.5,                      # Min score to continue propagation

    # Graph parameters
    damping=0.5,                                  # PPR damping (default)

    # Temporal filtering
    enable_temporal_filtering=True,               # NEW: enable date gating
    temporal_field_name="date",                   # Metadata field name
    date_format="%Y-%m-%d",                       # ISO 8601

    # Performance
    use_vectorized_retrieval=False,               # GPU acceleration (enable if available)
    max_workers=16,                               # Parallel NER processing
)
```

### Effort Estimate

| Task | Days |
|------|------|
| NER model selection + validation on sample clinical trial text | 1 |
| Temporal filtering: metadata storage, seed filtering, BFS filtering, retrieve interface | 3 |
| Integration: replace tools.py search functions | 2 |
| Index builder script (`build_linearrag_index.py`) for all 4 data sources | 3 |
| Testing: verify no future data leakage, retrieval quality comparison vs txtai | 2 |
| **Total** | **11 days** |

---

## 5. Workstream 4 — Claude Opus 4.6 DSPy Integration with Extended Thinking

### Current State

AutoCT uses `gpt-4o-mini` via DSPy:

```python
# agent.py:46-51
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'), max_tokens=16000)
dspy.configure(lm=lm)
```

All LLM calls use DSPy Signatures with `ChainOfThought` modules.

### Reference Implementation

The DIFW agent at `dspy_optimize_v1.py` demonstrates the target pattern:

```python
def make_lm(model_name="claude-opus-4-6-v1", temperature=0.0, **kwargs):
    lm_kwargs = {
        "model": f"anthropic/{model_name}",
        "api_base": api_base,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": 16384,
    }
    return dspy.LM(**lm_kwargs)
```

### Implementation

#### 5.1 LLM Configuration

**File: `src/lfe/impl/llm.py`** (currently 38 lines, minimal)

Replace with Opus 4.6 configuration:

```python
import dspy
import os

def make_ctra_lm(
    model_name: str = "claude-opus-4-6",
    temperature: float = 0.0,
    max_tokens: int = 16384,
    thinking_budget: int = 10000,  # Extended thinking budget tokens
) -> dspy.LM:
    """Create DSPy LM configured for Claude Opus 4.6 with extended thinking.

    Extended thinking allows Claude to reason through complex clinical trial
    features before generating structured output. Budget tokens control how
    much internal reasoning is allowed.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    lm_kwargs = {
        "model": f"anthropic/{model_name}",
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Extended thinking configuration
    # DSPy passes extra kwargs to the underlying litellm/anthropic client
    if thinking_budget > 0:
        lm_kwargs["extra_body"] = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        }

    return dspy.LM(**lm_kwargs)


def configure_ctra_lm():
    """Configure the global DSPy LM for CTRA."""
    lm = make_ctra_lm()
    dspy.configure(lm=lm)
    return lm
```

#### 5.2 Extended Thinking Strategy

Extended thinking is critical for three AutoCT agent phases where reasoning quality directly impacts output quality:

| Agent Phase | Thinking Budget | Rationale |
|-------------|----------------|-----------|
| **FeatureProposerV2** | 10,000 tokens | Proposes Add/Remove/Refine operations — needs to reason about feature interactions and clinical trial domain knowledge |
| **FeaturePlannerV2** | 8,000 tokens | Creates structured feature schemas — needs to plan multi-step extraction from heterogeneous data sources |
| **EvaluatorV2** | 12,000 tokens | Analyzes model errors and suggests improvements — needs deep reasoning about misclassified trials |
| **FeatureBuilderV3** | 5,000 tokens | Executes plans via RAG tools — primarily tool-calling, less reasoning needed |
| **Initializer** | 5,000 tokens | Bootstrap features — one-time, moderate reasoning |

**Per-module thinking budget** (if DSPy supports per-call overrides):

```python
# If DSPy supports per-module LM override:
class FeatureProposerV2(dspy.Module):
    def __init__(self, task):
        super().__init__()
        self.proposer_lm = make_ctra_lm(thinking_budget=10000)
        self.proposer = dspy.ChainOfThought(FeatureProposerV2Signature)

    def forward(self, previous_output):
        with dspy.context(lm=self.proposer_lm):
            return self.proposer(previous_output=previous_output)
```

**If DSPy does not support per-call thinking budget:** Use a single global budget of 10,000 tokens (balanced for all phases).

#### 5.3 DSPy Signatures Preservation

All existing DSPy Signatures (`FeatureProposerV2Signature`, `FeaturePlannerV2Signature`, `EvaluatorSignature`, etc.) remain unchanged. The migration is LLM-backend only — swap `openai/gpt-4o-mini` → `anthropic/claude-opus-4-6`.

**Key compatibility notes:**
- DSPy's `ChainOfThought` works with Anthropic models via litellm
- `dspy.configure(lm=...)` is global; use `dspy.context(lm=...)` for per-call overrides
- Assertions and backtracking (`.activate_assertions(max_backtracks=5)`) work with Anthropic
- Structured output (JSON mode) works via DSPy's response parsing, not model-native JSON mode

#### 5.4 Environment Variables

```bash
# Replace OpenAI key with Anthropic key
export ANTHROPIC_API_KEY="sk-ant-..."

# Or via AWS Bedrock (if using enterprise proxy):
export AWS_BEDROCK_REGION="us-east-1"
export ANTHROPIC_API_KEY="bedrock-access-key"
```

#### 5.5 Cost Impact

| Model | Input $/MTok | Output $/MTok | Est. per Eval |
|-------|-------------|---------------|---------------|
| gpt-4o-mini (current) | $0.15 | $0.60 | ~$15-20 |
| Claude Opus 4.6 | $5.00 | $25.00 | ~$50-80 |
| Claude Opus 4.6 (Bedrock) | $5.00 | $25.00 | ~$50-80 |

**Mitigation:** Extended thinking tokens count as output tokens. The thinking budget directly affects cost. Start with 10,000 thinking tokens and tune down if cost is too high without quality loss.

**Total run cost estimate (20 rollouts, 200 train samples):**
- gpt-4o-mini: ~$300-400/run
- Opus 4.6 (10K thinking): ~$1,000-1,600/run
- Opus 4.6 (5K thinking): ~$700-1,100/run

### Effort Estimate

| Task | Days |
|------|------|
| Modify `llm.py` for Opus 4.6 + extended thinking | 0.5 |
| Test DSPy compatibility with Anthropic backend | 1 |
| Tune thinking budgets per agent phase | 1.5 |
| Update environment config and documentation | 0.5 |
| Cost benchmarking: Opus 4.6 vs gpt-4o-mini on 3 trials | 1 |
| **Total** | **4.5 days** |

---

## 6. Workstream 5 — MCTS Improvements

### Overview

Three layers of MCTS improvements, ordered by priority:

1. **Fixes** (low effort, high impact) — Backpropagation, α tuning, error penalty
2. **Pareto Multi-Objective** (medium effort, high impact) — 3-objective vector UCB
3. **AB-MCTS Adaptive Branching** (medium effort, medium impact) — Thompson Sampling GEN/CONT

### 6.1 Fixes to Published AutoCT MCTS

**File: `src/lfe/impl/treesearch.py` (397 lines)**

#### Fix 1: Backpropagate Simulation Nodes

**Problem:** `_backpropagate()` only updates nodes on the selection path. Nodes explored during `_simulate()` get `total_reward=0` despite being evaluated (~60-70 wasted evaluations per 10 rollouts).

**Fix:** After simulation, backpropagate the reward to ALL nodes on the simulation path, not just the selection path.

```python
def _simulate(self, node):
    """Simulate: run agent on random child, return best reward on full path."""
    path = [node]
    current = node
    while not current.is_terminal():
        # Select child (currently rng.choice — see Fix 2)
        child = self._select_simulation_child(current)
        path.append(child)
        current = child

    best_reward = max(n.output_node.reward for n in path if n.output_node)

    # FIX: Backpropagate to ALL nodes on simulation path
    for sim_node in path:
        self._backpropagate(sim_node, best_reward)

    return best_reward
```

#### Fix 2: Informed Simulation Selection

**Problem:** `_simulate()` uses `rng.choice()` (uniform random) to select children during simulation. Each agent run costs $15-20 — random selection wastes expensive evaluations.

**Fix:** Weight selection by evaluator's top suggestions or UCB scores.

```python
def _select_simulation_child(self, node):
    """Select simulation child weighted by evaluator suggestions."""
    if not node.children:
        return node
    # Weight by evaluator score if available, else uniform
    weights = []
    for child in node.children:
        if child.output_node and child.output_node.evaluator_score:
            weights.append(child.output_node.evaluator_score)
        else:
            weights.append(1.0)
    weights = np.array(weights) / sum(weights)
    return self.rng.choice(node.children, p=weights)
```

#### Fix 3: Reduce α (Exploration Weight) from 1.0 to ~0.15

**Problem:** `exploration_weight=1.0` makes the UCT exploration term 30-100x larger than ROC-AUC differences between nodes. Selection is effectively random.

**Fix:** Reduce to 0.15 (validated by Schmocker et al. 2025 — 40% performance loss from mismatched C).

```python
# In train_mcts.py or MCTS.__init__()
mcts = MCTS(exploration_weight=0.15)  # was 1.0
```

#### Fix 4: Error Penalty (R = -1)

**Problem:** Failed nodes receive reward 0, same as unexplored nodes. MCTS doesn't learn to avoid error-prone feature combinations.

**Fix:** Assign R = -1 to failed nodes.

```python
def _evaluate_node(self, node):
    try:
        result = run_agent_pipeline(node.state)
        return result.reward  # positive value
    except Exception:
        return -1.0  # Actively penalize failures
```

#### Fix 5: Introspective Sibling Analysis

**Problem:** The LLM doesn't see what worked/failed in sibling nodes, leading to redundant exploration.

**Fix:** Include sibling context in evaluator prompt (from I-MCTS).

```python
def _build_evaluator_context(self, node):
    """Include sibling results in evaluator prompt."""
    sibling_info = []
    if node.parent:
        for sibling in node.parent.children:
            if sibling.id != node.id and sibling.output_node:
                sibling_info.append({
                    "features": sibling.output_node.features,
                    "reward": sibling.output_node.reward,
                    "error": sibling.error,
                })
    return f"Sibling results:\n{json.dumps(sibling_info, indent=2)}"
```

### 6.2 Pareto Multi-Objective MCTS

> **Note:** This section (all of § 6.2) describes the planned design. The current implementation uses 2 objectives per phase and no multi-fidelity evaluation in the search layer. See `research/mcts-implementation-design.md` § "As Implemented" for what was actually built.

Based on detailed design in `research/mcts-implementation-design.md`.

#### Three Objectives (Not Five)

The convergence research is definitive: 5 objectives at ~140-400 evaluations is not viable (60-80% of solutions become non-dominated, making selection random). Three objectives maintain selection pressure:

| # | Objective | Metric | Direction | Why |
|---|-----------|--------|-----------|-----|
| 1 | **Weighted AUC** | `0.2×AUC_P1 + 0.5×AUC_P2 + 0.3×AUC_P3` | Maximize | Phase II weighted highest (hardest, most valuable) |
| 2 | **Calibration** | `1 - ECE` (Expected Calibration Error, 10 bins) | Maximize | Clinical decisions need trustworthy probabilities |
| 3 | **Parsimony** | `1 - (n_features / MAX_FEATURES)` | Maximize | Fewer features = lower cost, less overfitting, better SHAP |

#### Phase 2a: MVP (Pareto Ranking on Existing Scalar MCTS)

Minimal change: keep scalar UCT for tree selection, but evaluate all 3 objectives and rank final output by hypervolume contribution.

**New files:**
- `src/lfe/impl/pareto.py` (~60 lines) — `dominates()`, `pareto_filter()`, `hypervolume_contribution()`
- `src/lfe/impl/evaluator.py` (~80 lines) — `evaluate_full()` returning 3-objective vector
- `src/lfe/impl/multi_fidelity.py` (~100 lines) — 20% data proxy with threshold filtering

**Modified files:**
- `src/lfe/impl/treesearch.py` — `MCTTreeNode.total_reward: float → np.ndarray(3,)`, vector backpropagation

#### Phase 2b: Full Pareto MCTS (Vector UCB + Pareto Selection)

Replace scalar UCT with Pareto-filtered vector UCB:

```python
def pareto_select(node, exploration_weight=0.15):
    """Vector UCB + Pareto dominance filtering."""
    ucb_vectors = []
    for child in node.children:
        if child.visit_count == 0:
            ucb_vectors.append(np.full(3, np.inf))
        else:
            mean_reward = child.total_reward / child.visit_count
            exploration = exploration_weight * np.sqrt(2 * np.log(node.visit_count) / child.visit_count)
            ucb_vectors.append(mean_reward + exploration)

    non_dominated = pareto_filter(ucb_vectors)
    return node.children[random.choice(non_dominated)]
```

**Output:** Pareto front of feature sets — decision-makers choose from the trade-off surface:
- **Clinical dashboard:** 5-8 features, ECE < 0.08, AUC ~0.65
- **High-stakes model:** 20-30 features, ECE < 0.12, AUC ~0.73
- **Balanced model:** 12-15 features, ECE < 0.10, AUC ~0.70

#### Multi-Fidelity Evaluation

**Critical for Pareto MCTS viability.** Without it, ~140 raw evaluations are insufficient for 3-objective convergence.

```python
def evaluate_with_proxy(feature_state, pareto_front):
    """Quick 20% data eval → filter → full eval only for promising candidates."""
    proxy_reward = evaluate_on_subset(feature_state, fraction=0.2)  # ~1-2 min, ~$3-4
    if is_dominated_by_front(proxy_reward, pareto_front):
        return proxy_reward * PROXY_SCALING  # Skip full eval
    return evaluate_full(feature_state)  # ~5-15 min, ~$15-20
```

Expected: 3-5x effective evaluation budget increase (400-700 effective evals from 140 raw).

### 6.3 AB-MCTS Adaptive Branching

Complementary to Pareto MCTS — decides *how* to expand, while Pareto decides *which* child to visit.

**Core idea:** At each node, dynamically choose:
- **GEN (go wider):** Generate new feature proposals — when scores plateau
- **CONT (go deeper):** Refine existing promising feature sets — when a direction looks good

**Implementation:** Adapt TreeQuest's Thompson Sampling from `treequest/algos/ab_mcts_a/prob_state.py`:

```python
class AdaptiveBranching:
    """Thompson Sampling for GEN/CONT decision per node."""
    def __init__(self):
        self.gen_dist = BetaDistribution(alpha=1, beta=1)   # Prior: uniform
        self.cont_dist = BetaDistribution(alpha=1, beta=1)

    def should_generate_new(self):
        """Sample from posterior: if GEN sample > CONT sample, go wider."""
        gen_sample = self.gen_dist.sample()
        cont_sample = self.cont_dist.sample()
        return gen_sample > cont_sample

    def update(self, action, reward):
        """Update posterior with observed reward."""
        if action == "gen":
            self.gen_dist.update(reward)
        else:
            self.cont_dist.update(reward)
```

**Integration with AutoCT:** In `_expand()`, before generating all ~5 children:

```python
def _expand(self, node):
    if self.adaptive_branching.should_generate_new():
        # Generate new feature proposals (wider)
        children = self._generate_new_proposals(node)
    else:
        # Refine best existing child (deeper)
        best_child = max(node.children, key=lambda c: c.total_reward.mean())
        children = self._refine_existing(best_child)
```

### Effort Estimate (Total Workstream 5)

| Task | Days |
|------|------|
| **Fixes (6.1):** Backprop, α, error penalty, informed selection, sibling analysis | 3 |
| **Phase 2a MVP (6.2):** pareto.py, evaluator.py, multi_fidelity.py, vector rewards | 5 |
| **Phase 2b Full Pareto (6.2):** Vector UCB, Pareto selection, testing | 5 |
| **AB-MCTS (6.3):** Thompson Sampling integration | 4 |
| **Testing + validation on TrialBench Phase II** | 3 |
| **Total** | **20 days** |

---

## 7. Workstream 6 — Additional Datasets + Extended Training

### 7.1 Additional Datasets

The README identifies four benchmark datasets. Currently only TrialBench is used. Add support for:

| Dataset | Size | Status | Integration Effort |
|---------|------|--------|-------------------|
| **TrialBench** | 23 datasets, 8 tasks, 143K+ trials | ✅ Already integrated | — |
| **TOP** | 17,538 trials (Phase I/II/III) | Available in repos | 2 days |
| **CTOD** | 12,477 trials | Not in repos | 1.5 days |
| **SCT** | 4,289 drugs, 3,326 diseases | Not in repos | 1 day (low priority) |

#### TOP Dataset Integration

**Source:** `/mnt/c/Users/sanjor11/projects/crra-repos/clinical-trial-outcome-prediction/data/`

**Format:** CSV with columns: `nctid, status, why_stop, label, phase, diseases, icdcodes, drugs, smiless, criteria`

**Integration steps:**
1. Write loader in `scripts/data/load_top.py` that converts TOP CSV to AutoCT's expected format: `(nctid, label)` Parquet splits per phase
2. Map TOP's `nctid` values to ClinicalTrials.gov records in the LinearRAG index (most should already exist)
3. Create task definitions under `tasks/top_trial_approval/` with phase-specific splits
4. Temporal split: train/val before Jan 1, 2014; test after (already defined in TOP)

**Purpose:** Direct comparison with HINT, MEXA-CTP, CLaDMoP, LIFTED baselines on the standard benchmark.

#### CTOD Dataset Integration

**Source:** Need to download from LIFTED repository or request from authors.

**Integration steps:**
1. Download CTOD dataset (12,477 trials)
2. Convert to AutoCT format (same as TOP)
3. Create task definitions under `tasks/ctod_trial_approval/`
4. Cross-dataset evaluation: train on TrialBench, evaluate on CTOD (and vice versa)

**Purpose:** Cross-dataset generalization validation.

#### SCT Dataset (Lower Priority)

Useful only for Phase 4 (self-supervised pre-training experiments). Skip for now.

### 7.2 Extended Training Dataset

**Current:** AutoCT uses ~100 train + 100 val samples (limited by LLM cost with gpt-4o-mini).

**Target:** 200-500 train + 200-500 val samples.

**Implementation:**

```python
# In build_trialbench_splits.py
parser.add_argument("--train-size", type=int, default=200, help="Training samples per phase")
parser.add_argument("--val-size", type=int, default=200, help="Validation samples per phase")
```

**Sampling strategy:**
- Stratified by phase (I/II/III) and outcome (success/failure)
- Stratified by therapeutic area (to ensure diversity)
- TrialBench trial approval task has 43,202 total trials — ample headroom

**Cost impact:**
- With Opus 4.6: each trial requires ~700-1000 API calls for feature building
- 200 train + 200 val: ~$800-1,200/run (vs ~$300-400 with 100+100)
- 500 train + 500 val: ~$2,000-3,000/run

**Recommendation:** Start with 200+200, increase to 500+500 only if accuracy improves significantly.

### Effort Estimate

| Task | Days |
|------|------|
| TOP dataset loader + task definitions | 2 |
| CTOD dataset acquisition + loader | 1.5 |
| Extended training splits (200-500) + stratification | 1 |
| Cross-dataset evaluation scripts | 1.5 |
| **Total** | **6 days** |

---

## 8. Workstream 7 — Extended MCTS Rollouts (≥20)

### Current State

`scripts/train_mcts.py` default: `--rollouts 10` (configurable via CLI arg).

### Changes

#### 8.1 Configuration

```python
# train_mcts.py
parser.add_argument("--rollouts", type=int, default=20, help="MCTS rollouts (min 20)")
```

**Minimum 20 rollouts** (user requirement). Recommended: 20-30 for standard runs, 50 for final benchmarks.

#### 8.2 Impact Analysis

| Rollouts | Est. Nodes Explored | Est. Cost (Opus 4.6) | Est. Wall Time |
|----------|--------------------|-----------------------|----------------|
| 10 (current) | ~70-80 | ~$500-800 | ~6-8 hours |
| 20 (minimum) | ~140-160 | ~$1,000-1,600 | ~12-16 hours |
| 30 (recommended) | ~210-240 | ~$1,500-2,400 | ~18-24 hours |
| 50 (final benchmark) | ~350-400 | ~$2,500-4,000 | ~30-40 hours |

**With multi-fidelity evaluation (Workstream 5):** Cost reduced by ~40-60% (proxy filtering avoids full evaluation on dominated candidates).

#### 8.3 Checkpointing

With 20+ rollout runs lasting 12-40 hours, checkpointing is essential:

```python
# Already supported via --resume in train_mcts.py
python scripts/train_mcts.py --rollouts 20 --resume checkpoint.pkl
```

Verify that the existing `dill`-based checkpoint/resume works correctly with:
- Vector rewards (Pareto MCTS)
- Adaptive branching state (Thompson Sampling posteriors)
- Multi-fidelity evaluation cache

#### 8.4 Parallel Evaluation (Optional)

AutoCT currently evaluates nodes sequentially. For 20+ rollouts, parallelism saves wall-clock time:

```python
# Option A: ProcessPoolExecutor (already used for feature building)
with ProcessPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(evaluate_node, node) for node in nodes_to_eval[:2]]
    results = [f.result() for f in futures]

# Option B: Async evaluation (saves wall time, not cost)
# Each eval still costs $15-20, but 2 run in parallel
```

**Recommendation:** Start with sequential (simpler, no state management issues). Add parallelism in Phase 3 if wall-clock time is a bottleneck.

### Effort Estimate

| Task | Days |
|------|------|
| Update default rollouts, validate checkpoint/resume with new MCTS features | 1 |
| Wall-clock benchmarking at 20/30/50 rollouts | 1 |
| **Total** | **2 days** |

---

## 9. Workstream 8 — Feature Extraction Optimization

> Detailed analysis: [research/feature-extraction-optimization.md](./feature-extraction-optimization.md)

### Problem

Feature building is **99.7% of per-node LLM cost**. For each ADD operation with 500 trials (200 train + 200 val + 100 test), the pipeline makes ~3,000 LLM calls (ReAct research + construction per trial). Over a 20-rollout MCTS run, this totals ~370,000 LLM calls (~$740 with Opus 4.6).

While AutoCT correctly reuses parent features within a tree path (ADD only builds the new feature, REMOVE is free), three categories of waste remain:

| Category | Wasted LLM Calls | % of Total |
|----------|-----------------|------------|
| **Duplicate features across branches** — same feature proposed on different branches rebuilds from scratch because the feature builder cache uses a group hash, not per-feature hash | ~60,000-100,000 | 16-27% |
| **Simulation node exploration** — `_simulate()` creates temporary nodes running full agent pipelines that are never reused | ~120,000-240,000 | 32-65% |
| **Redundant RAG research** — same trial researched independently for different features | ~10,000-30,000 | 3-8% |

### Solution: Global Feature Value Store

Replace the group-hash-based feature builder cache with a **per-feature cache** keyed by `(nctid, feature_name, plan_content_hash)`:

```
Current cache key:  (task, SHA-256(ALL plans in group), nctid)  → group changes = ALL features miss
Proposed cache key: (task, feature_name, SHA-256(individual plan), nctid)  → only changed features miss
```

**Hash safety for REFINE operations:** The `plan_content_hash` includes `feature_idea` — the field that carries the refinement chain. On REFINE, AutoCT appends `"\n---\n{refinement}"` to this string, guaranteeing the hash differs from the original. The store is append-only: a refined feature creates a NEW entry, never overwrites the original. Other branches using the unrefineed version are unaffected because they inherit feature values from their parent's `raw_features`, not from the global store.

**Modified `WrappedFeatureBuilderV3`:**
1. Check global store for each feature individually → return cached values for hits
2. Build only uncached features via `FeatureBuilderV3`
3. Store each built feature individually in the global store
4. Merge cached + freshly built values and return

### Additional Optimizations

| Optimization | Mechanism | Savings |
|-------------|-----------|---------|
| **Research results caching** | Separate ReAct research from construction; cache research by `(nctid, feature_set)`. Same trial researched once, construction runs per feature. | ~20% of feature builder cost |
| **Lightweight simulation** | Check feature store before full pipeline in `_simulate()`. If all features cached, skip research entirely. | Eliminates ~70% of simulation LLM calls |
| **Warm-start feature store** | Pre-compute common features (trial_phase, enrollment_count, etc.) before MCTS starts. | First rollout 30-40% faster |
| **Batch store queries** | Query store for ALL trials at once before spawning ProcessPoolExecutor workers. Only dispatch uncached (nctid, feature) pairs to workers. | Reduces worker spawn overhead |

### Projected Savings

| Configuration | LLM Calls | Est. Cost (Opus 4.6) | Savings |
|--------------|-----------|----------------------|---------|
| Current (no optimization) | ~370,000 | ~$740 | — |
| With global feature store | ~190,000 | ~$380 | 49% |
| + research caching + simulation optimization | ~124,000 | ~$248 | 66% |

Over 15 validation runs: **$5,400-$7,380 saved.**

### Effort Estimate

| Task | Days |
|------|------|
| Phase 1: `feature_store.py` + modified `WrappedFeatureBuilderV3` + batch queries | 4 |
| Phase 2: Separate research/construction caching in `FeatureBuilderV3` | 3 |
| Phase 3: Feature-store-aware simulation in `_simulate()` | 2 |
| Phase 4: Semantic plan matching (optional, for non-refined features only) | 2 |
| **Total** | **9-11 days** |

### New Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/lfe/impl/feature_store.py` | Global feature value store: get/put/batch_get, plan hashing, stats | ~120 |

### Modified Files

| File | Change |
|------|--------|
| `src/lfe/impl/agent.py` (`WrappedFeatureBuilderV3`) | Check/write global store per-feature instead of per-group |
| `src/lfe/impl/agent.py` (`FeatureBuilderV3`) | Separate research from construction; cache research results |
| `src/lfe/impl/treesearch.py` (`_simulate`) | Check feature store before full pipeline execution |

---

## 10. Dependency Graph

```
Workstream 1 (Notebooks → Scripts)
├── fetch_ctg.py, fetch_pubmed.py, clean_pubmed.py     ← independent
├── fetch_drugbank.py, fetch_faers.py                   ← independent
├── build_trialbench_splits.py                          ← depends on fetch_ctg
└── build_linearrag_index.py                            ← depends on ALL fetch + Workstream 3

Workstream 2 (XGBoost + TabPFN)                         ← independent
├── Remove LR/RF
├── Add TabPFN integration
└── Model saving + SHAP

Workstream 3 (LinearRAG + NER + Temporal Filtering)      ← blocks Workstream 1 index builder
├── NER model selection
├── Temporal filtering implementation
├── tools.py integration
└── Index builder script

Workstream 4 (Opus 4.6 DSPy + Extended Thinking)        ← independent
├── llm.py modification
├── DSPy compatibility testing
└── Thinking budget tuning

Workstream 5 (MCTS Improvements)                         ← depends on Workstreams 2, 4
├── Phase 1: Fixes (backprop, α, error penalty)          ← independent
├── Phase 2a: Pareto MVP                                 ← depends on Phase 1
├── Phase 2b: Full Pareto MCTS                           ← depends on Phase 2a
└── AB-MCTS Adaptive Branching                           ← depends on Phase 2a

Workstream 6 (Additional Datasets + Extended Training)   ← depends on Workstream 3
├── TOP loader
├── CTOD loader
└── Extended splits

Workstream 7 (Extended Rollouts ≥20)                     ← depends on Workstream 5
├── Config update
└── Checkpoint validation

Workstream 8 (Feature Extraction Optimization)            ← independent (can start immediately)
├── Phase 1: Global feature store                         ← independent
├── Phase 2: Research/construction separation              ← depends on Phase 1
├── Phase 3: Feature-store-aware simulation                ← depends on Phase 1 + WS5
└── Phase 4: Semantic plan matching (optional)             ← depends on Phase 1
```

### Recommended Execution Order

**Parallel Track A (Infrastructure):** Workstreams 1, 3, 4 — can start immediately
**Parallel Track B (Models):** Workstream 2 — can start immediately
**Parallel Track C (Cost optimization):** Workstream 8 Phases 1-2 — can start immediately (modifies agent.py, independent of other workstreams)
**Sequential:** Workstream 5 → 7 (after Tracks A and B complete)
**Parallel with Track A:** Workstream 6 (after index builder is ready)
**After WS5 + WS8:** Workstream 8 Phase 3 (simulation optimization, needs both MCTS changes and feature store)

### Critical Path

```
Workstream 3 (LinearRAG, 11 days)
  → Workstream 1 index builder (3 days, included in WS3)
  → Workstream 6 (6 days)
  → Workstream 5 Phase 2a (5 days)
  → Workstream 5 Phase 2b (5 days)
  → Workstream 7 (2 days)

Total critical path: ~29 days (~6 weeks)
```

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **TabPFN v2.5 fine-tuning requires 80GB VRAM** | HIGH | MEDIUM | Use standard TabPFN (no fine-tuning) as default. Fine-tuning is optional — only activate if standard underperforms XGBoost. Budget A100 instance ($3.50/hr) for fine-tuning experiments only. |
| **Opus 4.6 cost 3-4x higher than gpt-4o-mini** | HIGH | MEDIUM | Extended thinking budget is configurable. Start at 10K, tune down to 5K if quality doesn't degrade. Multi-fidelity evaluation reduces total evaluations by 40-60%. |
| **LinearRAG temporal filtering adds latency** | MEDIUM | LOW | Date filtering is O(n) scan on metadata — negligible vs LLM call latency. Pre-filter at index time to create date-partitioned subindexes if needed. |
| **Multi-fidelity proxy unreliable** | MEDIUM | HIGH | Validate proxy correlation (Spearman ρ > 0.7) on first 30 full evaluations. If proxy fails, fall back to full evaluation only (accept higher cost). |
| **3 Pareto objectives still too many at 140 evals** | LOW-MEDIUM | MEDIUM | Monitor non-dominated fraction. If >50% at eval 100, drop parsimony (use as constraint instead), reduce to 2 objectives. |
| **`en_core_sci_scibert` NER misses novel drug names** | MEDIUM | LOW | Supplement with regex patterns for known drug name formats. Add custom entity rules for CTRA-specific terminology. |
| **DSPy + Anthropic compatibility issues** | LOW-MEDIUM | MEDIUM | DSPy uses litellm under the hood, which supports Anthropic. Test all 5 Signature types before full run. Fallback: direct Anthropic SDK with manual prompt construction. |
| **Checkpoint/resume fails with new vector rewards** | LOW | MEDIUM | Test checkpoint round-trip with Pareto MCTS before long runs. Use `dill` (already in AutoCT) which handles numpy arrays. |
| **Feature store hash collision on REFINE** — refined plan hashes to same key as original | VERY LOW | HIGH | Hash includes `feature_idea` which physically differs on REFINE (appends `\n---\n{refinement}`). Append-only store — refined features are new entries, never overwrite originals. See [feature-extraction-optimization.md §6.5](./feature-extraction-optimization.md#65-refine-safety-why-the-store-never-corrupts-other-branches). |
| **Feature store disk growth** | LOW | LOW | ~500 trials × 20 features = ~10K files, ~1-5KB each = 10-50MB total. Add periodic cleanup of unreferenced entries. |

---

## 11. Cost Estimate

### Per-Run Costs

| Configuration | Train/Val | Rollouts | Est. Cost | With Feature Store |
|---------------|-----------|----------|-----------|-------------------|
| Current (gpt-4o-mini, 100+100, 10 rollouts) | 100+100 | 10 | ~$150-200 | — |
| Phase 1 baseline (Opus 4.6, 100+100, 10 rollouts) | 100+100 | 10 | ~$500-800 | ~$250-400 |
| Phase 2 standard (Opus 4.6, 200+200, 20 rollouts) | 200+200 | 20 | ~$1,000-1,600 | ~$500-800 |
| Phase 2 with multi-fidelity (Opus 4.6, 200+200, 20 rollouts) | 200+200 | 20 | ~$600-1,000 | ~$300-500 |
| Final benchmark (Opus 4.6, 500+500, 30 rollouts) | 500+500 | 30 | ~$2,500-4,000 | ~$1,250-2,000 |

### Development Costs (Compute)

| Item | Hours | Rate | Cost |
|------|-------|------|------|
| GPU for TabPFN (standard, g4dn.xlarge) | 50 | $0.53/hr | $27 |
| GPU for TabPFN fine-tuning (A100, p4d.24xlarge) | 10 | $32.77/hr | $328 |
| LinearRAG indexing (CPU) | 20 | $0.10/hr | $2 |
| **Total compute** | | | **~$357** |

### Effort Summary

| Workstream | Days | Can Parallelize With |
|------------|------|---------------------|
| WS1: Notebook migration | 5.5 | WS2, WS4 |
| WS2: XGBoost + TabPFN | 6 | WS1, WS3, WS4 |
| WS3: LinearRAG + NER + Temporal | 11 | WS2, WS4 |
| WS4: Opus 4.6 DSPy | 4.5 | WS1, WS2, WS3 |
| WS5: MCTS Improvements | 20 | Partially (fixes only) |
| WS6: Additional Datasets | 6 | After WS3 |
| WS7: Extended Rollouts | 2 | After WS5 |
| WS8: Feature Extraction Optimization | 9-11 | WS1, WS2, WS3, WS4 (Phases 1-2 independent) |
| **Total (sequential)** | **64-66 days** | |
| **Total (with parallelism)** | **~37 days (~7.5 weeks)** | |

### Run Budget for Validation

| Run | Purpose | Cost |
|-----|---------|------|
| 3× baseline reproduction (TrialBench, Opus 4.6, 10 rollouts) | Validate Opus 4.6 migration | ~$1,500-2,400 |
| 3× Pareto MVP (TrialBench Phase II, 20 rollouts) | Validate multi-objective framework | ~$1,800-3,000 |
| 3× Full Pareto MCTS (TrialBench, 20 rollouts) | Benchmark Pareto MCTS | ~$1,800-3,000 |
| 1× TOP benchmark (20 rollouts) | Compare with HINT/MEXA-CTP baselines | ~$600-1,000 |
| 1× CTOD cross-dataset (20 rollouts) | Generalization test | ~$600-1,000 |
| 1× Final benchmark (500+500, 30 rollouts) | Best configuration | ~$2,500-4,000 |
| **Total validation runs** | | **~$8,800-14,400** |
| **Total with feature store (WS8)** | | **~$4,400-7,200** |

---

## Appendix A: File Change Summary

| File | Workstream | Change Type |
|------|-----------|-------------|
| `src/lfe/impl/llm.py` | WS4 | **Rewrite** — Opus 4.6 + extended thinking |
| `src/lfe/impl/agent.py` | WS2, WS4, WS5, WS8 | **Modify** — Remove LR/RF, add TabPFN, update OutputV2, sibling analysis, per-feature caching in WrappedFeatureBuilderV3, research/construction separation in FeatureBuilderV3 |
| `src/lfe/impl/feature_store.py` | WS8 | **New** — Global feature value store: get/put/batch_get, plan content hashing, store stats |
| `src/lfe/impl/treesearch.py` | WS5, WS7 | **Major modify** — Vector rewards, Pareto selection, backprop fix, α tuning, adaptive branching |
| `src/lfe/impl/tools.py` | WS3 | **Rewrite** — Replace txtai with LinearRAG, temporal filtering |
| `src/lfe/impl/pareto.py` | WS5 | **New** — Pareto dominance, filtering, hypervolume |
| `src/lfe/impl/evaluator.py` | WS5 | **New** — Multi-objective evaluation wrapper |
| `src/lfe/impl/multi_fidelity.py` | WS5 | **New** — Proxy evaluation, threshold filtering |
| `src/lfe/impl/adaptive_branching.py` | WS5 | **New** — Thompson Sampling GEN/CONT |
| `scripts/train_mcts.py` | WS7 | **Modify** — Default rollouts=20, new CLI flags |
| `scripts/predict.py` | WS2 | **Modify** — TabPFN model loading |
| `scripts/data/*.py` | WS1 | **New** — 8 data pipeline scripts |
| `environment.yml` | WS2, WS3 | **Modify** — Add tabpfn, scispacy, LinearRAG deps |

## Appendix B: Key Repository Paths

| Component | Path |
|-----------|------|
| AutoCT source | `/mnt/c/Users/sanjor11/projects/crra-repos/autoct/src/lfe/impl/` |
| AutoCT scripts | `/mnt/c/Users/sanjor11/projects/crra-repos/autoct/scripts/` |
| AutoCT notebooks | `/mnt/c/Users/sanjor11/projects/crra-repos/autoct/notebooks/` |
| LinearRAG source | `/mnt/c/Users/sanjor11/projects/crra-repos/LinearRAG/src/` |
| TreeQuest (AB-MCTS) | `/mnt/c/Users/sanjor11/projects/crra-repos/treequest/src/treequest/algos/` |
| PMMG (Pareto MCTS) | `/mnt/c/Users/sanjor11/projects/crra-repos/PMMG/chemtsv2/` |
| LLMFE | `/mnt/c/Users/sanjor11/projects/crra-repos/LLMFE/llmfe/` |
| AIDE | `/mnt/c/Users/sanjor11/projects/crra-repos/aideml/aide/` |
| TrialBench data | `/mnt/c/Users/sanjor11/projects/crra-repos/ML2ClinicalTrials/Trialbench/data/` |
| TOP data | `/mnt/c/Users/sanjor11/projects/crra-repos/clinical-trial-outcome-prediction/data/` |
| DSPy reference (DIFW) | `/mnt/c/Users/sanjor11/projects/difw-agent/worktrees/.../dspy_optimize_v1.py` |
