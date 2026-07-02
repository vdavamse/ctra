# LinearRAG Integration Research

> RAG-ENGINEER research findings for replacing txtai with LinearRAG in the CTRA/AutoCT pipeline. Produced 2026-03-27. Research only -- no code.

---

## Table of Contents

1. [LinearRAG API and Installation](#1-linearrag-api-and-installation)
2. [AutoCT Tool Interface](#2-autoct-tool-interface)
3. [DrugBank Data Format](#3-drugbank-data-format)
4. [OpenFDA FAERS API](#4-openfda-faers-api)
5. [Date Filtering for Label Leakage Prevention](#5-date-filtering-for-label-leakage-prevention)
6. [scispaCy NER Model Recommendation](#6-scispacy-ner-model-recommendation)

---

## 1. LinearRAG API and Installation

### Repository

- **GitHub**: https://github.com/DEEP-PolyU/LinearRAG (ICLR 2026)
- **Paper**: arXiv 2510.10114v4
- **License**: Not a pip package -- source code only, cloned or vendored
- **Python**: 3.9 recommended

### Installation

```bash
# Step 1: Python dependencies
pip install -r requirements.txt

# Step 2: spaCy model (default general-purpose)
python -m spacy download en_core_web_trf

# Step 3: For biomedical/clinical text, install scispaCy model instead
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz

# Step 4: Embedding model -- download all-mpnet-base-v2 locally
# Place at: model/all-mpnet-base-v2/
```

### Dependencies (from requirements.txt)

```
httpx==0.25.2
numpy==1.21.0
openai==1.54.5
pandas==1.3.0
python-igraph==0.11.8
scikit-learn==1.3.2
scipy>=1.7.0
sentence-transformers==2.2.2
spacy==3.6.1
tqdm==4.67.1
transformers==4.30.2
huggingface-hub==0.16.4
pyarrow==12.0.1
```

Key dependencies: `python-igraph` (graph construction + PPR), `sentence-transformers` (embeddings), `spacy` (NER), `pyarrow` (Parquet storage).

### Source Code Structure

```
src/
  LinearRAG.py        # Core class: index(), retrieve(), qa(), run_ppr()
  config.py           # LinearRAGConfig dataclass
  embedding_store.py  # EmbeddingStore: Parquet-backed embedding persistence
  evaluate.py         # Evaluation utilities
  ner.py              # SpacyNER: entity extraction for graph construction
  utils.py            # LLM_Model wrapper, normalize_answer(), compute_mdhash_id()
```

### LinearRAGConfig -- All Parameters

```python
@dataclass
class LinearRAGConfig:
    dataset_name: str                          # Name for working directory
    embedding_model: str = "all-mpnet-base-v2" # Sentence embedding model
    llm_model: LLM_Model = None                # For QA generation (not retrieval)
    chunk_token_size: int = 1000               # Passage chunking size
    chunk_overlap_token_size: int = 100        # Chunk overlap
    spacy_model: str = "en_core_web_trf"       # NER model name
    working_dir: str = "./import"              # Storage directory for index files
    batch_size: int = 128                      # Embedding batch size
    max_workers: int = 16                      # Parallel NER workers
    retrieval_top_k: int = 5                   # Number of passages to retrieve
    max_iterations: int = 3                    # BFS propagation depth (n-hop)
    top_k_sentence: int = 1                    # Sentences per entity per BFS iteration
    passage_ratio: float = 1.5                 # Passage selection ratio
    passage_node_weight: float = 0.05          # Weight for passage nodes in PPR
    damping: float = 0.5                       # PPR damping factor
    iteration_threshold: float = 0.5           # Min score to continue propagation
    use_vectorized_retrieval: bool = False      # GPU-accelerated sparse tensor mode
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    attribute_query_keywords: list[str] = field(default_factory=lambda: [...])
```

### Core API -- How to Use LinearRAG

#### 1. Index Documents

```python
from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig

config = LinearRAGConfig(
    dataset_name="ctra-index",
    spacy_model="en_core_sci_scibert",
    embedding_model="model/all-mpnet-base-v2",
    retrieval_top_k=10,
    max_iterations=3,
)
rag = LinearRAG(global_config=config)

# passages: list of strings, formatted as "idx:text"
# e.g., ["0:This is the first passage about drug X...", "1:A clinical trial for..."]
passages = [f"{i}:{text}" for i, text in enumerate(corpus_texts)]
rag.index(passages)
```

**Document Format**: Passages are a `list[str]` where each string is `"index:content"`. The index prefix is a numeric chunk ID separated by a colon. The content is the raw text of the passage. LinearRAG internally:
1. Segments each passage into sentences (by punctuation)
2. Runs NER on each sentence to extract entities
3. Builds Tri-Graph: entity nodes, sentence nodes, passage nodes
4. Computes embeddings for all three node types
5. Stores everything as Parquet files + GraphML

**Index Output Files** (in `working_dir`):
- `passage_embedding.parquet` (hash_id, text, embedding)
- `entity_embedding.parquet` (hash_id, text, embedding)
- `sentence_embedding.parquet` (hash_id, text, embedding)
- `graph.graphml` (iGraph graph with entity-passage and entity-sentence edges)
- `ner_results.json` (cached NER output per passage)

#### 2. Retrieve Passages

```python
# questions: list of dicts with at minimum an "id" and "question" field
questions = [{"id": "q1", "question": "What are the adverse effects of drug X?"}]
results = rag.retrieve(questions)
# Returns: list of retrieved passage sets, one per question
```

The `retrieve()` method internally:
1. Extracts entities from query via NER
2. Finds matching seed entities in graph via embedding similarity
3. Runs BFS propagation through sentence-entity bipartite graph (Stage 1: Semantic Bridging)
4. Runs Personalized PageRank on entity-passage subgraph (Stage 2: Global Importance)
5. Returns top-k passages ranked by PPR score

#### 3. QA (Retrieve + Generate)

```python
# Full QA pipeline (retrieve + LLM generation)
predictions = rag.qa(questions)
# Returns: list of generated answers using retrieved context
```

The `qa()` method calls `retrieve()` then sends passages + question to the LLM for answer generation. For CTRA, we only need `retrieve()` -- we do NOT need `qa()` since the AutoCT agents handle LLM reasoning separately.

### Tri-Graph Architecture

Three node types connected by two sparse adjacency matrices:

```
Entity Nodes (V_e) <--- Mention Matrix M ---> Sentence Nodes (V_s)
      |
      |--- Contain Matrix C ---> Passage Nodes (V_p)
```

- **Contain matrix C** (|V_p| x |V_e|): C_ij = 1 if passage i contains entity j
- **Mention matrix M** (|V_s| x |V_e|): M_ij = 1 if sentence i mentions entity j
- No explicit relation edges -- relations are implicit via shared entities

### Two-Stage Retrieval

**Stage 1 -- Local Semantic Bridging (Entity Activation)**:
1. Extract query entities via NER
2. Match to graph entities by embedding similarity
3. Iteratively propagate through sentence-entity bipartite graph:
   `a^t = MAX(M^T(sigma * (M * a^{t-1})), a^{t-1})`
4. Dynamic pruning: discard entities below threshold delta

**Stage 2 -- Global Importance Aggregation (PPR)**:
1. Initialize passage scores from DPR similarity + entity activation scores
2. Run Personalized PageRank on entity-passage subgraph
3. Return top-k passages by PPR score

### Key Performance Characteristics

- **Zero LLM token cost** for indexing and retrieval
- **Indexing**: 249.78s on 2WikiMultiHopQA (vs. 4,933s for LightRAG)
- **Retrieval**: 0.093s per query
- **Linear scalability**: O(|P| * T) for both construction and retrieval
- **Medical dataset accuracy**: 63.72% GPT-Acc (best among all GraphRAG methods)

### Critical Gap: No Metadata Filtering

LinearRAG has **no native support** for metadata filtering (dates, source types, etc.). The `retrieve()` method does not accept any filter parameters. This is the primary engineering challenge for CTRA integration -- see Section 5.

---

## 2. AutoCT Tool Interface

### Repository

- **GitHub**: https://github.com/CogComp/autoct (EMNLP 2025)
- **Note**: The URL `linyongver/AutoCT` from CLAUDE.md returns 404. The correct repo is `CogComp/autoct`.

### Key Files

```
src/lfe/impl/
  tools.py       # Search tool factory functions
  agent.py       # DSPy ReAct agent definitions
  globals.py     # Cache (FanoutCache), DuckDB connection, process pool
  pubmed.py      # PubMed data retrieval via Entrez API
  treesearch.py  # MCTS implementation
  utils.py       # Utility functions
```

### tools.py -- Complete Interface (What We Must Match)

The file uses **txtai** for vector search with two pre-built embedding indexes:

```python
from txtai.embeddings import Embeddings

pubmed_embeddings = Embeddings()
pubmed_embeddings.load("./datasets/pubmed-index")

nct_embeddings = Embeddings()
nct_embeddings.load("./datasets/nct-index")
```

#### Factory Function: `make_pubmed_search(nct_info)`

**Input**: `nct_info` dict containing at minimum `{"startDate": "YYYY-MM-DD"}`

**Returns**: A callable `pubmed_search(query: str) -> str`

**Internal search call (txtai SQL)**:
```python
@cache.memoize(tag="pmemb-v0")
def _search_pubmed_embeddings(query, start_date):
    return pubmed_embeddings.search(
        "select text, data from txtai where DateAvail < :s and similar(:q)",
        limit=10,
        parameters={"q": query, "s": start_date},
    )
```

**Output format**: Concatenated YAML strings separated by `"\n\n---------\n\n"`. Each result is a YAML-serialized dict with fields:
- `AbstractText` (populated from search result `text`)
- `AuthorList` (first 5 entries)
- Other PubMed article metadata
- (Excludes: `ArticlePubmedDataReferenceList`, `ArticleAuthorList`, `ArticleInvestigatorList`)

#### Factory Function: `make_nct_search(nct_info)`

**Input**: `nct_info` dict with `{"startDate": "YYYY-MM-DD"}`

**Returns**: A callable `related_trials_nct_search(query: str) -> str`

**Internal search call (txtai SQL)**:
```python
@cache.memoize(tag="nctemb-v0")
def _search_nct_embeddings(query, start_date):
    return nct_embeddings.search(
        "select text, data from txtai where start_date < :s and similar(:q)",
        limit=10,
        parameters={"q": query, "s": start_date},
    )
```

**Output format**: Concatenated YAML strings separated by `"\n\n---------\n\n"`. Each result is a YAML-serialized dict with:
- Trial metadata fields
- `text` field (search text)
- `outcome` normalized to `"Success"` or `"Failure"` (from binary 1/0)
- (Excludes: `brief_summary/textblock`, `brief_summary`, `detailed_description`)

#### Direct Function: `get_detailed_nct_info(nctid: str) -> str`

Returns YAML-serialized clinical trial info from DuckDB query on `ctg-studies-with-nctid.parquet`. Includes all `protocolSection` fields (excluding `statusModule`).

### How Tools Are Used in agent.py

DSPy ReAct agents receive tools as a list:

```python
builderstep = dspy.ReAct(
    FeatureBuilderStepSignature,
    tools=[
        make_pubmed_search(nct_info),    # Returns callable
        make_nct_search(nct_info),        # Returns callable
        get_detailed_nct_info,            # Direct function
    ],
    max_iters=5,
)
```

**Tool requirements for DSPy ReAct**:
1. Must be a callable (function) with a docstring
2. First argument must be a string (the query)
3. Must return a string (the search results)
4. DSPy uses the function name and docstring to tell the LLM what the tool does

### Key Design Pattern: Trial-Specific Date Binding

Each tool factory is called **once per trial** with that trial's `nct_info`, which binds the `startDate` into the closure. This means every search automatically filters by that trial's start date -- preventing label leakage. The LinearRAG replacement must preserve this pattern.

### Caching

- `FanoutCache` at `.cache/dc/` with tag-based memoization
- Both `_search_nct_embeddings` and `_search_pubmed_embeddings` are memoized by `(query, start_date)` tuple
- Cache tags: `"nctemb-v0"`, `"pmemb-v0"`, `"cti-v0"`

### Interface Contract Summary (What LinearRAG Must Match)

| Property | Current (txtai) | LinearRAG Replacement |
|----------|----------------|-----------------------|
| Input | `query: str` | `query: str` |
| Output | `str` (YAML blocks separated by `\n\n---------\n\n`) | Same format |
| Date filtering | txtai SQL `WHERE date < :s` | Must implement externally |
| Results count | `limit=10` | `retrieval_top_k=10` |
| Memoization | `FanoutCache` per (query, date) | Keep same caching |
| Tool signature | `def search(query: str) -> str` with docstring | Same |

---

## 3. DrugBank Data Format

### Data Access Tiers

| Tier | License | Content | Relevant for CTRA |
|------|---------|---------|-------------------|
| **CC0 Open Data** | Public domain | DrugBank Vocabulary (IDs, names, CAS, UNII), DrugBank Structures (SMILES, InChI, InChIKey) | Yes -- drug identification and linking |
| **CC BY-NC 4.0 Academic** | Non-commercial research | Full database (XML/CSV/JSON): descriptions, indications, pharmacology, targets, interactions | Yes -- primary data source |
| **Commercial** | Paid license | Same as academic + commercial use rights | Not needed for research |

### CC0 Open Data: DrugBank Vocabulary (Free)

Available at https://go.drugbank.com/releases/latest under CC0.

Columns:
- `DrugBank ID` (e.g., DB00001)
- `Accession Numbers` (semicolon-delimited alternate IDs)
- `Common name` (generic drug name)
- `CAS` (CAS registry number)
- `UNII` (FDA unique ingredient identifier)
- `Synonyms` (semicolon-delimited)
- `Standard InChI Key`

### CC0 Open Data: DrugBank Structures (Free)

Columns:
- `DrugBank ID`
- `Name`
- `CAS Number`
- `Drug Groups` (approved, investigational, experimental, etc.)
- `InChI`
- `InChI Key`
- `SMILES` (as `moldb_smiles`)
- `Molecular Formula`
- `Molecular Weight`
- PubChem CID, PubChem SID, ChEBI, ChEMBL IDs

### Academic License: Full CSV Tables

#### `drugs` table (primary)
Key columns:
- `id`, `drugbank_id`, `name`, `type` (small molecule / biotech)
- `description` (free text)
- `state` (solid, liquid, gas)
- `cas_number`
- `moldb_smiles`, `moldb_inchi`, `moldb_inchikey`
- `moldb_average_mass`, `moldb_mono_mass`, `moldb_formula`
- Group flags: `investigational`, `approved`, `vet_approved`, `experimental`, `nutraceutical`, `illicit`, `withdrawn` (boolean 0/1)

#### `structured_pharmacology` table
Columns:
- `id`, `drug_id`
- `indication` (free text: what the drug treats)
- `pharmacodynamics` (free text: how the drug works)
- `mechanism_of_action` (free text)
- `absorption`, `toxicity`, `protein_binding`
- `metabolism`, `half_life`
- `route_of_elimination`, `volume_of_distribution`, `clearance`

#### `drug_interactions` table
Columns:
- `subject_drug_id`, `affected_drug_id`
- `subject_drug_accession`, `affected_drug_accession`
- `description` (free text, e.g., "The metabolism of Lepirudin can be increased when combined with St. John's Wort")

#### `drug_targets` table
Columns (inferred from XML schema and parsers):
- `drug_id`, `target_id`
- `target_name`, `gene_name`
- `actions` (e.g., inhibitor, agonist, antagonist)
- `organism`
- UniProt ID linkage

#### `drug_mappings` table
- ATC codes, ATCVet codes, MeSH vocabulary codes
- `direct` column (1 = direct mapping, 0 = parent concept)

### Recommended Approach for CTRA

1. **For drug identification/linking**: Use CC0 open data (DrugBank Vocabulary + Structures). No license required.
2. **For RAG corpus enrichment**: Parse the full XML dump (`drugbank_all_full_database.xml`) under academic license. Extract per-drug text passages from `description`, `indication`, `mechanism_of_action`, `pharmacodynamics`, and `toxicity` fields.
3. **Python parser**: Use https://github.com/dhimmel/drugbank or https://github.com/Zhangs996/DrugBankParser for XML-to-DataFrame conversion.

### Passage Construction for LinearRAG

For each drug, construct a text passage:
```
Drug: {name} (DrugBank ID: {drugbank_id})
Type: {type} | Groups: {groups}
Indication: {indication}
Mechanism of Action: {mechanism_of_action}
Pharmacodynamics: {pharmacodynamics}
Toxicity: {toxicity}
Known Targets: {target_1 (gene_1, action_1)}, {target_2}, ...
```

These passages become input to `rag.index(passages)`. Entities like drug names, target genes, and disease names will be automatically extracted by scispaCy NER and linked in the Tri-Graph.

---

## 4. OpenFDA FAERS API

### Endpoint

```
https://api.fda.gov/drug/event.json
```

### Data Source

FDA Adverse Event Reporting System (FAERS). Contains adverse event and medication error reports submitted to FDA. Data available from 2004 to present, updated quarterly.

### Query Syntax

```
https://api.fda.gov/drug/event.json?search=FIELD:VALUE+AND+FIELD:VALUE&limit=N&skip=M
```

**Operators**:
- `+AND+` : Both conditions must match
- `+OR+` : Either condition matches (space between terms)
- `[DATE_START+TO+DATE_END]` : Date range (inclusive)
- `.exact` suffix: Match exact phrase (required for counting)

**Date format**: `YYYYMMDD` (e.g., `20200115`)

### Key Searchable Fields

#### Report-Level Fields
| Field | Type | Description |
|-------|------|-------------|
| `safetyreportid` | string | Unique report ID (8-digit, last digit is checksum) |
| `receivedate` | date | When FDA received the report (YYYYMMDD) |
| `receiptdate` | date | Date of most recent info in report |
| `serious` | string | "1" = serious event, "2" = not serious |
| `seriousnessdeath` | string | "1" = resulted in death |
| `seriousnesshospitalization` | string | "1" = required hospitalization |
| `seriousnesslifethreatening` | string | "1" = life-threatening |
| `seriousnessdisabling` | string | "1" = resulted in disability |
| `occurcountry` | string | Two-letter country code (FAERS) or full name (AERS) |

#### Patient Fields
| Field | Type | Description |
|-------|------|-------------|
| `patient.patientonsetage` | string | Age at onset |
| `patient.patientonsetageunit` | string | Age unit (800=decade, 801=year, 802=month, etc.) |
| `patient.patientsex` | string | "0"=unknown, "1"=male, "2"=female |
| `patient.patientweight` | string | Weight in kg |

#### Drug Fields
| Field | Type | Description |
|-------|------|-------------|
| `patient.drug.medicinalproduct` | string | Drug name as reported |
| `patient.drug.drugindication` | string | Why drug was prescribed |
| `patient.drug.drugdosagetext` | string | Dosage as reported |
| `patient.drug.drugstartdate` | date | When drug use began |
| `patient.drug.drugenddate` | date | When drug use ended |
| `patient.drug.drugcharacterization` | string | "1"=suspect, "2"=concomitant, "3"=interacting |
| `patient.drug.openfda.brand_name` | string[] | Harmonized brand name(s) |
| `patient.drug.openfda.generic_name` | string[] | Harmonized generic name(s) |
| `patient.drug.openfda.substance_name` | string[] | Active ingredient(s) |
| `patient.drug.openfda.product_type` | string[] | Product type |

#### Reaction Fields
| Field | Type | Description |
|-------|------|-------------|
| `patient.reaction.reactionmeddrapt` | string | Adverse reaction (MedDRA preferred term) |
| `patient.reaction.reactionoutcome` | string | "1"=recovered, "2"=recovering, "3"=not recovered, "4"=recovered with sequelae, "5"=fatal, "6"=unknown |

### Example API Calls

```bash
# 1. Search for adverse events mentioning aspirin
https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:"aspirin"&limit=10

# 2. Count top reactions for a specific drug
https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name.exact:"metformin"&count=patient.reaction.reactionmeddrapt.exact&limit=20

# 3. Search by date range
https://api.fda.gov/drug/event.json?search=receivedate:[20200101+TO+20201231]+AND+patient.drug.medicinalproduct:"pembrolizumab"&limit=100

# 4. Serious events only
https://api.fda.gov/drug/event.json?search=serious:1+AND+patient.drug.openfda.generic_name.exact:"nivolumab"&limit=100

# 5. Count adverse events by drug
https://api.fda.gov/drug/event.json?search=receivedate:[20200101+TO+20251231]&count=patient.drug.openfda.generic_name.exact&limit=100
```

### Rate Limits

| Authentication | Rate Limit |
|----------------|-----------|
| No API key | 40 requests per minute, 1000 per day |
| With API key | 240 requests per minute |

API key: register at https://open.fda.gov/apis/authentication/

### Response JSON Structure

```json
{
  "meta": {
    "disclaimer": "...",
    "terms": "...",
    "license": "...",
    "last_updated": "2025-12-15",
    "results": {
      "skip": 0,
      "limit": 10,
      "total": 123456
    }
  },
  "results": [
    {
      "safetyreportid": "12345678",
      "receivedate": "20200315",
      "serious": "1",
      "seriousnessdeath": "0",
      "seriousnesshospitalization": "1",
      "patient": {
        "patientsex": "2",
        "patientonsetage": "65",
        "patientweight": "70",
        "drug": [
          {
            "medicinalproduct": "PEMBROLIZUMAB",
            "drugindication": "NON-SMALL CELL LUNG CANCER",
            "drugcharacterization": "1",
            "openfda": {
              "brand_name": ["KEYTRUDA"],
              "generic_name": ["PEMBROLIZUMAB"],
              "substance_name": ["PEMBROLIZUMAB"]
            }
          }
        ],
        "reaction": [
          {
            "reactionmeddrapt": "Pneumonitis",
            "reactionoutcome": "1"
          }
        ]
      }
    }
  ]
}
```

### Important Caveats for CTRA

1. **No causal relationship**: A report does NOT establish that a drug caused the adverse event. Multiple drugs may be listed per report with no drug-reaction linkage.
2. **Duplicate reports**: The same event may be reported multiple times.
3. **Drug name inconsistency**: `medicinalproduct` is free-text as reported; use `openfda.generic_name` for harmonized names. ~86% of records have `openfda` annotations.
4. **AERS vs FAERS differences**: Pre-2012 data (AERS) uses uppercase reaction names and full country names; post-2012 (FAERS) uses mixed case and ISO country codes.

### Recommended Approach for CTRA

For each drug in the trial being assessed:
1. Query `patient.drug.openfda.generic_name.exact:"{drug_name}"` with date range filter
2. Count top adverse reactions: `&count=patient.reaction.reactionmeddrapt.exact`
3. Get seriousness distribution: count queries on `serious`, `seriousnessdeath`, etc.
4. Construct a text passage per drug summarizing adverse event profile:

```
Drug: {generic_name}
FAERS Reports (before {trial_start_date}): {total_count}
Top Adverse Events: {reaction_1} ({count_1}), {reaction_2} ({count_2}), ...
Serious Event Rate: {serious_count}/{total_count} ({percentage}%)
Death Reports: {death_count}
Hospitalization Reports: {hosp_count}
```

---

## 5. Date Filtering for Label Leakage Prevention

### The Problem

LinearRAG has **no native metadata filtering**. The `retrieve()` method takes a question and returns passages based purely on graph structure and semantic similarity. There is no `WHERE date < X` equivalent like txtai SQL provides.

When predicting whether a clinical trial will succeed, we must NOT retrieve:
- PubMed articles published after the trial's start date
- ClinicalTrials.gov records from trials that started after the current trial
- FAERS reports received after the trial's start date
- DrugBank information added after the trial's start date

This is **label leakage** -- the model would learn from future information that would not have been available at prediction time.

### How txtai Currently Handles This

txtai uses SQL-like filtering at query time:
```python
pubmed_embeddings.search(
    "select text, data from txtai where DateAvail < :s and similar(:q)",
    parameters={"q": query, "s": start_date},
)
```

The `WHERE DateAvail < :s` clause filters results BEFORE similarity ranking. This is efficient because txtai indexes metadata alongside embeddings.

### Proposed Solution: Four-Layer Temporal Gating

Since LinearRAG cannot filter internally, we implement filtering as a wrapper at four layers.

#### Layer 1: Date Metadata Storage

Extend `EmbeddingStore` to store a `date` column alongside each passage:

```
Parquet schema: hash_id | text | embedding | date | source
```

When indexing, each passage carries metadata:
```python
metadata = {"date": "2019-06-15", "source": "pubmed"}
```

Implementation: Add a `hash_id_to_date` dict and a `date` column in the Parquet file. This requires modifying `embedding_store.py`.

#### Layer 2: Seed Entity Filtering

When `retrieve()` finds seed entities from the query, filter out entities that only appear in future documents:

```python
# Only match entities from documents published before query_date
valid_entity_indices = [
    i for i, hid in enumerate(entity_hash_ids)
    if entity_dates[hid] is None or entity_dates[hid] <= query_date
]
```

This prevents future-knowledge entities from entering the BFS propagation.

#### Layer 3: BFS Propagation Filtering

During the iterative entity activation (Stage 1), skip sentences from future documents:

```python
for sentence_hash_id in connected_sentences:
    if sentence_dates[sentence_hash_id] > query_date:
        continue  # Do not propagate through future sentences
```

This prevents multi-hop paths that traverse future documents.

#### Layer 4: Post-Retrieval Filtering

After PPR ranking, filter the final passage list:

```python
filtered_passages = [
    p for p in retrieved_passages
    if passage_dates[p.hash_id] is None or passage_dates[p.hash_id] <= query_date
]
```

This is the safety net -- even if some future information leaked through earlier layers, it gets caught here.

### Alternative Approaches Considered

| Approach | Pros | Cons |
|----------|------|------|
| **Post-retrieval filter only (Layer 4)** | Simple, minimal code changes | Wastes retrieval slots on future docs; may return fewer than top_k |
| **Separate indexes per date window** | Native isolation | Massive storage overhead; loses cross-temporal entity links |
| **Pre-filter corpus per query** | Clean separation | Re-indexing per query is prohibitively expensive |
| **Four-layer gating (recommended)** | Comprehensive, efficient | Requires modifying LinearRAG source code |

### Date Sources

| Data Source | Date Field | Format | Extraction |
|-------------|-----------|--------|------------|
| ClinicalTrials.gov | `protocolSection.statusModule.startDateStruct.date` | YYYY-MM or YYYY-MM-DD | Parse from CTG JSON/Parquet |
| PubMed | `DateAvail` or `PubDate` | YYYY-MM-DD | Parse from PubMed XML response |
| DrugBank | First approval date or initial publication date | YYYY-MM-DD | Parse from DrugBank XML |
| FAERS/OpenFDA | `receivedate` | YYYYMMDD | Parse from OpenFDA JSON, convert to YYYY-MM-DD |

### Implementation Effort

The four-layer approach requires modifying three LinearRAG source files:
1. `embedding_store.py` -- Add date column to Parquet schema and in-memory dict
2. `LinearRAG.py` -- Add `query_date` parameter to `retrieve()`, `calculate_entity_scores()`, `graph_search_with_seed_entities()`
3. `ner.py` -- No changes needed (NER is date-independent)

The implementation plan already specifies this at 3 person-days (see `research/implementation-plan.md`, Workstream 3, Section 4.2).

---

## 6. scispaCy NER Model Recommendation

### Available Models

#### Core Pipeline Models (full spaCy pipeline: tokenizer, tagger, parser, NER)

| Model | Vocab | Vectors | Backbone | GPU Needed |
|-------|-------|---------|----------|-----------|
| `en_core_sci_sm` | ~100k | None | CNN | No |
| `en_core_sci_md` | ~360k | 50k word vectors | CNN | No |
| `en_core_sci_lg` | ~785k | 600k word vectors | CNN | No |
| `en_core_sci_scibert` | ~785k | SciBERT transformer | `allenai/scibert-base` | Recommended |

#### Specialized NER Models (NER-only, no parser)

| Model | Training Corpus | Entity Types |
|-------|----------------|-------------|
| `en_ner_craft_md` | CRAFT | GGP, SO, TAXON, CHEBI, GO, CL |
| `en_ner_jnlpba_md` | JNLPBA | DNA, RNA, cell_line, cell_type, protein |
| `en_ner_bc5cdr_md` | BC5CDR (1500 PubMed articles) | **CHEMICAL, DISEASE** (4409 chemicals, 5818 diseases, 3116 interactions) |
| `en_ner_bionlp13cg_md` | BIONLP13CG | Cancer, Organ, Tissue, Organism, Cell, Amino_acid, Gene_or_gene_product, Anatomical |

### Installation

```bash
# Core model (recommended for LinearRAG)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz

# Specialized NER model (for drug/disease extraction)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

### Performance Benchmarks

**BC5CDR NER (drugs + diseases)**:
- SciBERT backbone: **90.01 F1** (SOTA on this benchmark)
- BioBERT: 88.85 F1
- `en_core_sci_md`: 78.79 recall
- `en_core_sci_sm`: 75.62 recall
- General `en_core_web_trf`: ~60% F1 on biomedical text

### Recommendation for CTRA: `en_core_sci_scibert`

**Primary model: `en_core_sci_scibert`** for LinearRAG graph construction.

Rationale:
1. **Best biomedical entity coverage**: SciBERT backbone pre-trained on 1.14M papers from Semantic Scholar (82% biomedical, 18% CS). Recognizes drug names, disease subtypes, gene symbols, protein names, cell types that general NER misses.
2. **Full pipeline**: Includes tokenizer, POS tagger, dependency parser, and NER -- LinearRAG uses the full pipeline for sentence segmentation and entity extraction.
3. **LinearRAG already supports it**: The repo's README explicitly lists this as the model for medical datasets:
   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz
   ```
4. **Performance**: ~85% F1 on biomedical NER vs ~60% for the default `en_core_web_trf`.

**Why not `en_ner_bc5cdr_md`?**
While BC5CDR specifically targets chemicals and diseases (the most directly relevant entity types), it is a NER-only model without the full spaCy pipeline. LinearRAG's `SpacyNER` class uses `spacy.load(model)` and relies on sentence segmentation and entity extraction from the full pipeline. `en_core_sci_scibert` provides both.

**Optional: Combined approach for richer entity extraction**:
```python
# Use en_core_sci_scibert as the primary pipeline
nlp = spacy.load("en_core_sci_scibert")

# Add BC5CDR NER as a secondary component for drug/disease labels
# (requires custom pipeline merging -- adds complexity)
```

This combined approach is more complex and may not be worth the engineering effort for Phase 1. Start with `en_core_sci_scibert` alone and evaluate entity coverage on a sample of clinical trial texts before adding complexity.

### Entity Type Mapping for Clinical Trial Text

| Entity in Trial Text | Captured by `en_core_sci_scibert` | Example |
|---------------------|-----------------------------------|---------|
| Drug names | Yes | pembrolizumab, metformin, nivolumab |
| Disease names | Yes | non-small cell lung cancer, type 2 diabetes |
| Gene/protein names | Yes | PD-L1, EGFR, BRCA1 |
| Biomarkers | Yes | HbA1c, PSA, CA-125 |
| Anatomical terms | Partial | liver, kidney (recognized), but not always typed correctly |
| Dosage/measurements | No (filtered out as CARDINAL/ORDINAL) | 200mg, 10mg/kg |

LinearRAG's `SpacyNER.extract_entities_sentences()` already filters out ORDINAL and CARDINAL entities, which is the correct behavior -- quantities are not useful as graph nodes for multi-hop retrieval.

---

## Summary: Key Findings and Next Steps

| Topic | Key Finding | Implication for CTRA |
|-------|------------|---------------------|
| LinearRAG API | Source-only (no pip). Core API: `LinearRAG(config).index(passages)` then `.retrieve(questions)`. Passages are `"idx:text"` strings. | Vendor/fork the repo; only use `index()` and `retrieve()`, skip `qa()` |
| AutoCT tools | Factory functions return `(query: str) -> str` callables. Output is YAML blocks. Date filtering via txtai SQL closure. | LinearRAG wrapper must match this exact interface |
| DrugBank | Academic XML dump has richest data. Free CC0 data has identifiers + structures only. Key tables: drugs, structured_pharmacology, drug_interactions, drug_targets. | Parse XML for RAG passages; CC0 data for drug linking |
| OpenFDA FAERS | REST API at `api.fda.gov/drug/event.json`. Key fields: `patient.drug.openfda.generic_name`, `patient.reaction.reactionmeddrapt`, `receivedate`. 240 req/min with key. | Query per drug, construct summary passages for indexing |
| Date filtering | LinearRAG has zero metadata filtering. Four-layer temporal gating needed: metadata storage, seed entity filter, BFS filter, post-retrieval filter. | 3 days of LinearRAG source modifications |
| scispaCy NER | `en_core_sci_scibert` is the recommended model. ~85% F1 on biomedical NER vs ~60% for general models. LinearRAG already documents this model for medical use. | Drop-in config change: `spacy_model="en_core_sci_scibert"` |

---

Sources:
- [DEEP-PolyU/LinearRAG GitHub](https://github.com/DEEP-PolyU/LinearRAG)
- [LinearRAG Paper (arXiv 2510.10114)](https://arxiv.org/abs/2510.10114)
- [CogComp/autoct GitHub](https://github.com/CogComp/autoct)
- [AutoCT Paper (arXiv 2506.04293)](https://arxiv.org/abs/2506.04293)
- [DrugBank CSV Format Reference](https://docs.drugbank.com/csv/)
- [DrugBank Data Packages](https://go.drugbank.com/data_packages)
- [DrugBank Releases](https://go.drugbank.com/releases/latest)
- [OpenFDA Drug Adverse Event API](https://open.fda.gov/apis/drug/event/)
- [OpenFDA Searchable Fields](https://open.fda.gov/apis/drug/event/searchable-fields/)
- [OpenFDA Example Queries](https://open.fda.gov/apis/drug/event/example-api-queries/)
- [OpenFDA FAERS Field Mapping (GitHub)](https://github.com/FDA/openfda/blob/master/schemas/faers_mapping.md)
- [scispaCy Documentation](https://allenai.github.io/scispacy/)
- [scispaCy GitHub](https://github.com/allenai/scispacy)
- [SciBERT Paper](https://aclanthology.org/D19-1371.pdf)
- [DrugBank XML Parser (Python)](https://github.com/dhimmel/drugbank)
