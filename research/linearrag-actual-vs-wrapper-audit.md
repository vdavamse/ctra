# LinearRAG Actual Source vs CTRA Wrapper Audit

**Date:** 2026-03-27
**Scope:** Comparison of the real LinearRAG library (`DEEP-PolyU/LinearRAG`, ICLR'26) against our CTRA wrapper at `src/ctra/rag/linearrag_wrapper.py`.

---

## 1. API Differences

### 1.1 Constructor: `LinearRAG.__init__(global_config)`

**Real signature:**
```python
class LinearRAG:
    def __init__(self, global_config):
```

`global_config` is a `LinearRAGConfig` dataclass instance.

**What we do (correct):**
```python
rag = LinearRAG(global_config=config)
```
This matches. No issue here.

### 1.2 `LinearRAGConfig.embedding_model` -- TYPE MISMATCH (CRITICAL)

**Real type:** `LinearRAGConfig.embedding_model` is a **`SentenceTransformer` instance**, not a string path.

Evidence from `run.py`:
```python
embedding_model = SentenceTransformer(args.embedding_model, device="cuda")
config = LinearRAGConfig(
    dataset_name=args.dataset_name,
    embedding_model=embedding_model,  # <-- SentenceTransformer object
    ...
)
```

The `EmbeddingStore` class calls `self.embedding_model.encode(texts, ...)` directly on this object (line 45 of `embedding_store.py`). The config field's default `"all-mpnet-base-v2"` is misleading -- it is only a default string value in the dataclass definition, but the real code in `run.py` always passes a pre-loaded `SentenceTransformer` object.

**What we pass:**
```python
config = LinearRAGConfig(
    ...
    embedding_model=self._config.embedding_model,  # This is a STRING like "sentence-transformers/all-MiniLM-L6-v2"
    ...
)
```

**Impact: WILL CRASH.** `EmbeddingStore.__init__` immediately calls `self.embedding_model.encode(...)`. Passing a string will fail with `AttributeError: 'str' object has no attribute 'encode'`.

### 1.3 `LinearRAGConfig.llm_model` -- TYPE MISMATCH (NON-CRITICAL for retrieval)

**Real type:** `LLM_Model` instance (wrapper around OpenAI client). Required for `qa()` method only.

**What we pass:** Nothing (defaults to `None`). This is fine because we only use `index()` and `retrieve()`, never `qa()`. But if we ever called `qa()`, it would crash.

### 1.4 `index()` -- Input Format

**Real signature:**
```python
def index(self, passages):
```

`passages` is a **list of strings**. The expected format is `"idx:text"` where `idx` is a sequential integer and `text` is the passage content.

Evidence from `run.py`:
```python
passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
rag_model.index(passages)
```

And from `add_adjacent_passage_edges()` (line 586):
```python
index_pattern = re.compile(r'^(\d+):')
```
This regex extracts the `idx:` prefix to build adjacency edges between sequential passages. If the prefix is missing, **no adjacent-passage edges are created**, which degrades retrieval quality (multi-hop reasoning relies on these edges for document-order continuity).

**What we do:**
```python
passages_for_index = [
    f"{i}:{row['text']}" for i, row in self._passages_df.iterrows()
]
rag.index(passages_for_index)
```

This is **correct in format**. However, `self._passages_df.iterrows()` yields `(index, row)` tuples where `i` is the DataFrame index. If the DataFrame has been filtered or reindexed, `i` may not be sequential from 0, which would produce non-sequential prefixes like `"3:text"`, `"7:text"`, etc. The `add_adjacent_passage_edges()` method sorts by the integer prefix, so non-sequential integers still produce correct adjacency. But the `_resolve_passage_row()` method in our wrapper assumes the prefix IS the DataFrame row index, which would only be true if the DataFrame index is the default `RangeIndex`. This is fragile.

### 1.5 `retrieve()` -- Input Format (CRITICAL MISMATCH)

**Real signature:**
```python
def retrieve(self, questions):
```

`questions` is a **list of dicts**, each with at least:
- `"question"` (str): The query text
- `"answer"` (str): The gold answer (used to populate result; referenced on line 127)

It returns a **list of dicts**, each containing:
```python
{
    "question": str,
    "sorted_passage": list[str],        # passage texts, ranked
    "sorted_passage_scores": list[float], # corresponding scores
    "gold_answer": str                    # copied from input
}
```

**What we do:** We **DO NOT call `retrieve()` directly**. Instead, our `_retrieve_raw()` method manually replicates the retrieve logic:
```python
# Lines 302-348 of our wrapper
rag.entity_hash_ids = list(rag.entity_embedding_store.hash_id_to_text.keys())
# ... (manually sets up all the data structures)
question_embedding = rag.config.embedding_model.encode(...)
seed_entity_indices, seed_entities, ... = rag.get_seed_entities(query)
# ... calls graph_search_with_seed_entities or dense_passage_retrieval
```

This is an acceptable approach -- we manually call the internal methods rather than using the high-level `retrieve()` API. However, it means:
1. We must replicate the setup code that `retrieve()` does at lines 85-93 (setting `self.entity_hash_ids`, `self.entity_embeddings`, etc.).
2. We are tightly coupled to LinearRAG's internal implementation. Any refactoring upstream will silently break us.
3. We skip the vectorized retrieval precomputation (`_precompute_sparse_matrices()`) that `retrieve()` calls when `use_vectorized_retrieval=True`.

**Risk:** Medium. The current approach works if internals don't change, but bypasses the vectorized retrieval path.

### 1.6 `retrieve()` Return Format vs Our Assumptions

**Real return:** List of dicts with `sorted_passage` (list of passage texts) and `sorted_passage_scores`.

**Our assumption:** We expect `(passage_text, score)` tuples. Since we bypass `retrieve()` entirely and call internal methods directly, this is not a direct mismatch, but it means we can never switch to calling `retrieve()` without adapting the return format handling.

### 1.7 `config.embedding_model.encode()` Call in `_retrieve_raw()`

**Our code (line 320-322):**
```python
question_embedding = rag.config.embedding_model.encode(
    query, normalize_embeddings=True, show_progress_bar=False,
    batch_size=rag.config.batch_size,
)
```

**Real code (line 110):**
```python
question_embedding = self.config.embedding_model.encode(
    question, normalize_embeddings=True, show_progress_bar=False,
    batch_size=self.config.batch_size,
)
```

These match in structure. But as noted in 1.2, `config.embedding_model` must be a `SentenceTransformer` object, not a string.

---

## 2. Config Differences

### 2.1 Real `LinearRAGConfig` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_name` | `str` | (required) | Name used for working directory subdirectory |
| `embedding_model` | `str` (default) / `SentenceTransformer` (actual) | `"all-mpnet-base-v2"` | **Must be a SentenceTransformer object in practice** |
| `llm_model` | `LLM_Model` | `None` | Only needed for `qa()` |
| `chunk_token_size` | `int` | `1000` | Not used in retrieval; may be for pre-chunking |
| `chunk_overlap_token_size` | `int` | `100` | Not used in retrieval |
| `spacy_model` | `str` | `"en_core_web_trf"` | spaCy model name for NER |
| `working_dir` | `str` | `"./import"` | Root directory for all index artifacts |
| `batch_size` | `int` | `128` | Embedding batch size |
| `max_workers` | `int` | `16` | Parallelism for NER and QA |
| `retrieval_top_k` | `int` | `5` | Number of passages to return |
| `max_iterations` | `int` | `3` | BFS propagation depth |
| `top_k_sentence` | `int` | `1` | Sentences per entity per iteration |
| `passage_ratio` | `float` | `1.5` | Weight for DPR score in passage scoring |
| `passage_node_weight` | `float` | `0.05` | Scale factor for passage node weights in PPR |
| `damping` | `float` | `0.5` | PPR damping factor |
| `iteration_threshold` | `float` | `0.5` | Minimum score to continue BFS propagation |
| `use_vectorized_retrieval` | `bool` | `False` | GPU-accelerated retrieval |
| `enable_hybrid_attribute_fallback` | `bool` | `False` | Keyword boost for attribute queries |
| `attribute_keyword_boost` | `float` | `0.25` | Boost magnitude |
| `attribute_query_keywords` | `list[str]` | `[born, birth, where, ...]` | Keywords triggering attribute boost |

### 2.2 What We Set vs What Exists

| Our wrapper param | Maps to real field | Value | Issue |
|---|---|---|---|
| `embedding_model=self._config.embedding_model` | `embedding_model` | String `"sentence-transformers/all-MiniLM-L6-v2"` | **BREAKING: Must be SentenceTransformer object** |
| `spacy_model=self._config.ner_model` | `spacy_model` | `"en_core_sci_scibert"` | OK (valid spaCy model name) |
| `working_dir=str(index_path.parent)` | `working_dir` | Parent of index dir | OK |
| `damping=self._config.ppr_alpha` | `damping` | `0.15` | **SEMANTIC MISMATCH**: We call it `ppr_alpha` and set `0.15` (standard PPR alpha). In LinearRAG, `damping=0.5` is the default and is passed directly to `igraph.personalized_pagerank(damping=...)`. In igraph, `damping` means the probability of following a link (1-restart_prob). So LinearRAG's `damping=0.5` means restart_prob=0.5, while our `0.15` means restart_prob=0.85. These are very different behaviors: our setting causes much weaker personalization. |
| `max_iterations=self._config.ppr_iterations` | `max_iterations` | `50` | **SEMANTIC MISMATCH**: `max_iterations` in LinearRAG is the BFS entity propagation depth (default=3), NOT the PPR iteration count. Setting this to 50 means the BFS will propagate entities 50 hops deep, which is computationally wasteful and semantically wrong. The `ppr_iterations` concept from our config does not map to any LinearRAG parameter -- igraph's PPR uses PRPACK which converges internally. |
| (not set) | `passage_ratio` | defaults to `1.5` | Should be tuned for clinical text; `run.py` uses `2` |
| (not set) | `passage_node_weight` | defaults to `0.05` | Not configurable from our wrapper |
| (not set) | `top_k_sentence` | defaults to `1` | `run.py` uses `3`; should be configurable |
| (not set) | `iteration_threshold` | defaults to `0.5` | `run.py` uses `0.4`; controls BFS pruning |
| (not set) | `batch_size` | defaults to `128` | Not configurable from our wrapper |
| (not set) | `max_workers` | defaults to `16` | Not configurable from our wrapper |
| (not set) | `use_vectorized_retrieval` | defaults to `False` | Our `_retrieve_raw()` never calls `_precompute_sparse_matrices()` |

### 2.3 Embedding Model Mismatch

Our CTRA config uses `"sentence-transformers/all-MiniLM-L6-v2"` (384-dim), while the LinearRAG default and examples use `"all-mpnet-base-v2"` (768-dim). This means our embeddings will be in a different vector space than what LinearRAG was tested with. Not a bug per se, but a quality concern.

---

## 3. Index Format Differences

### 3.1 Directory Structure

**Real LinearRAG** writes artifacts under `{working_dir}/{dataset_name}/`:
- `passage_embedding.parquet` -- hash_id, text, embedding columns
- `entity_embedding.parquet` -- hash_id, text, embedding columns
- `sentence_embedding.parquet` -- hash_id, text, embedding columns
- `ner_results.json` -- `{"passage_hash_id_to_entities": {...}, "sentence_to_entities": {...}}`
- `LinearRAG.graphml` -- igraph graph with entity and passage nodes

**What we assume:**
- `{index_path}/passages.parquet` -- our custom metadata file (text, source, doc_id, date)
- `{index_path}/{index_path.name}/LinearRAG.graphml` -- we look for graph at `index_path / index_path.name`
- `{index_path}/{index_path.name}/ner_results.json` -- NER results at same nested path

**Issue:** Our `passages.parquet` is **a different file** from LinearRAG's `passage_embedding.parquet`. We use it for metadata (source, doc_id, date); LinearRAG's passage store contains (hash_id, text, embedding). These are complementary, not conflicting, but our wrapper must maintain both files.

The nested directory structure `index_path / index_path.name` is correct because LinearRAG creates `{working_dir}/{dataset_name}/` and we set `working_dir=str(index_path.parent)`, `dataset_name=index_path.name`, so the artifacts end up at `index_path/` (which is `working_dir/dataset_name`). Wait -- that means the graph is at `index_path / "LinearRAG.graphml"`, NOT `index_path / index_path.name / "LinearRAG.graphml"`.

**Let me verify:** If `index_path = /data/myindex`, then:
- `working_dir = str(index_path.parent)` = `/data`
- `dataset_name = index_path.name` = `myindex`
- LinearRAG writes to `os.path.join(working_dir, dataset_name, ...)` = `/data/myindex/...`

So `passage_embedding.parquet` goes to `/data/myindex/passage_embedding.parquet` and `LinearRAG.graphml` goes to `/data/myindex/LinearRAG.graphml`.

But our wrapper checks:
```python
graph_path = index_path / index_path.name / "LinearRAG.graphml"  # = /data/myindex/myindex/LinearRAG.graphml
```

**BREAKING: Wrong path.** The graph is at `index_path / "LinearRAG.graphml"`, but we look for it at `index_path / index_path.name / "LinearRAG.graphml"` (one level too deep). This means `graph_path.exists()` will always be `False`, forcing a full re-index every time instead of loading the pre-built graph.

Similarly for NER:
```python
ner_path = index_path / index_path.name / "ner_results.json"  # WRONG: one level too deep
```

### 3.2 Passage Text Storage

**Real LinearRAG:** Passages are stored as-is in the embedding store. If you pass `"0:some text"`, the stored text IS `"0:some text"` (including the prefix). The `hash_id_to_text` mapping stores the full string with prefix.

**Our `_resolve_passage_row()`** strips the `"idx:"` prefix from returned passage texts and tries to match against `_passages_df["text"]` which does NOT have the prefix. This is correct in principle, but it assumes the prefix format is always `"digits:"`. If a passage text itself starts with digits followed by a colon (e.g., `"42: The result was..."` as actual clinical text), the regex would incorrectly try to parse that as an index.

### 3.3 Hash ID Format

LinearRAG generates hash IDs using `compute_mdhash_id(content, prefix=namespace+"-")`:
```python
md5(content.encode()).hexdigest()  # with prefix like "passage-"
```

This means the hash includes the `"idx:text"` prefix. Two passages with identical text but different indices will have different hash IDs. This is important: our wrapper's passage ordering affects the hash IDs and therefore the graph structure.

---

## 4. Retrieval Differences

### 4.1 The Real Retrieval Pipeline

1. **Setup phase** (done once per `retrieve()` call, lines 85-106):
   - Cache `entity_hash_ids`, `entity_embeddings`, `passage_hash_ids`, `passage_embeddings`, `sentence_hash_ids`, `sentence_embeddings` as numpy arrays
   - Build `node_name_to_vertex_idx` and `vertex_idx_to_node_name` from graph
   - If vectorized: precompute sparse adjacency matrices

2. **Per-question loop** (lines 108-128):
   - Encode question with embedding model
   - `get_seed_entities()`: Run spaCy NER on query, encode entity mentions, find closest entities in entity embedding store by cosine similarity
   - If seed entities found: `graph_search_with_seed_entities()`:
     - BFS entity propagation through entity-sentence-entity chains (up to `max_iterations` hops)
     - Calculate passage scores (DPR similarity + entity overlap bonus)
     - Run Personalized PageRank on graph with combined entity+passage weights
     - Return sorted passage hash_ids and scores
   - If no seed entities: fall back to `dense_passage_retrieval()` (pure cosine similarity)

3. **Return:** List of result dicts with `sorted_passage` (texts) and `sorted_passage_scores`

### 4.2 What Our `_retrieve_raw()` Does

Our wrapper replicates steps 1-2 manually but has these differences:

1. **Missing vectorized precomputation:** We never call `_precompute_sparse_matrices()`, so vectorized retrieval is silently broken.
2. **Missing `passage_node_indices` setup:** Our code (lines 314-318) computes `passage_node_indices` but the real `retrieve()` does NOT set this attribute -- it is set during `index()` in `add_nodes()` (line 615). If we loaded a pre-built graph, we do set it in lines 314-318. But if we just called `index()`, it is already set. This is redundant but not harmful.
3. **Correct internal method calls:** The calls to `get_seed_entities()`, `graph_search_with_seed_entities()`, and `dense_passage_retrieval()` match the real API.

### 4.3 PPR Behavior Difference

The real LinearRAG uses igraph's `personalized_pagerank()` with `damping=0.5` (default). Our wrapper passes `damping=0.15` via `ppr_alpha`. In igraph, `damping` is the probability of following a link (not restarting). So:
- LinearRAG default: 50% chance of following links, 50% chance of restart (strong personalization)
- Our setting: 15% chance of following links, 85% chance of restart (extremely strong personalization, almost no graph exploration)

This will produce very different ranking behavior. The real LinearRAG's `0.5` was presumably tuned for their benchmarks.

---

## 5. Breaking Issues

### 5.1 CRITICAL: `embedding_model` is a string, not a `SentenceTransformer` (WILL CRASH)

**File:** `linearrag_wrapper.py`, lines 126-134 and 457-465
**Impact:** `EmbeddingStore.__init__()` calls `self.embedding_model.encode()` on the config's `embedding_model`. Passing a string crashes immediately with `AttributeError`.
**Fix:** Load the `SentenceTransformer` before creating `LinearRAGConfig`:
```python
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer(self._config.embedding_model)
config = LinearRAGConfig(
    ...
    embedding_model=st_model,
    ...
)
```

### 5.2 CRITICAL: `max_iterations` set to `ppr_iterations=50` (WRONG SEMANTIC)

**File:** `linearrag_wrapper.py`, line 133 and line 464
**Impact:** `max_iterations` controls BFS entity propagation hops (default=3, `run.py` uses 3). Setting it to 50 causes the BFS to attempt 50 propagation rounds, which is extremely slow and semantically wrong. Entity propagation beyond 3-5 hops produces diminishing returns and noise.
**Fix:** Use a dedicated config field (e.g., `entity_propagation_depth: int = 3`) or hardcode 3.

### 5.3 CRITICAL: Graph path is one directory level too deep (ALWAYS RE-INDEXES)

**File:** `linearrag_wrapper.py`, lines 144-145
```python
graph_path = index_path / index_path.name / "LinearRAG.graphml"
ner_path = index_path / index_path.name / "ner_results.json"
```
**Impact:** These paths will never exist because LinearRAG writes to `index_path/LinearRAG.graphml`. The condition on line 147 will always be False, causing a full re-index on every load. With large corpora this means minutes of unnecessary NER + embedding computation.
**Fix:**
```python
graph_path = index_path / "LinearRAG.graphml"
ner_path = index_path / "ner_results.json"
```

### 5.4 HIGH: `damping=0.15` produces wrong PPR behavior

**File:** `linearrag_wrapper.py`, line 131 and line 463
**Impact:** With damping=0.15, PPR barely explores the graph (85% restart probability). The graph-based multi-hop reasoning that makes LinearRAG superior to dense retrieval is almost entirely disabled.
**Fix:** Use LinearRAG's default (`0.5`) or make it a separate config field with a sensible default:
```python
damping=getattr(self._config, 'ppr_damping', 0.5)
```

### 5.5 MEDIUM: Missing `top_k_sentence`, `passage_ratio`, `iteration_threshold` configuration

**Impact:** These default to `1`, `1.5`, and `0.5` respectively, while the `run.py` example uses `3`, `2`, and `0.4`. The defaults are probably fine for initial usage but may underperform vs the tuned values.

### 5.6 LOW: Vectorized retrieval path silently broken

**Impact:** If `use_vectorized_retrieval=True` is set, our `_retrieve_raw()` never calls `_precompute_sparse_matrices()`. The sparse matrices would not exist, causing an `AttributeError` when `graph_search_with_seed_entities` tries to access them.
**Mitigation:** We never set `use_vectorized_retrieval=True`, so this does not currently crash.

### 5.7 LOW: `ner_results_path` not set before `save_ner_results()` in edge case

When calling `index()`, the code calls `self.load_existing_data(hash_id_to_passage.keys())` which sets `self.ner_results_path`. But if `len(new_passage_hash_ids) == 0` (all passages already indexed), it still calls `self.save_ner_results()` which uses `self.ner_results_path`. This works because `load_existing_data` always sets the attribute, so no issue.

---

## 6. Recommendations

### 6.1 Immediate Fixes (Required Before First Use)

1. **Load SentenceTransformer before passing to config:**
   ```python
   from sentence_transformers import SentenceTransformer
   st_model = SentenceTransformer(self._config.embedding_model)
   ```
   Pass `st_model` as `embedding_model` in `LinearRAGConfig`. Do this in both `_ensure_loaded()` and `build_index()`.

2. **Fix `max_iterations`:** Do NOT map `ppr_iterations` to `max_iterations`. Add a new field `entity_hops: int = 3` to `RAGConfig` or hardcode `max_iterations=3`.

3. **Fix graph/NER paths:** Remove the extra `/ index_path.name` nesting:
   ```python
   graph_path = index_path / "LinearRAG.graphml"
   ner_path = index_path / "ner_results.json"
   ```

4. **Fix damping:** Change `ppr_alpha=0.15` to use LinearRAG's default of `0.5`, or add a separate `ppr_damping` field with default `0.5`. Note: in igraph's PPR, `damping` is NOT the restart probability -- it is `1 - restart_probability`. LinearRAG's `0.5` default means 50% follow / 50% restart.

### 6.2 Short-Term Improvements

5. **Expose missing config fields** in `RAGConfig`:
   - `top_k_sentence: int = 3`
   - `passage_ratio: float = 2.0`
   - `iteration_threshold: float = 0.4`
   - `passage_node_weight: float = 0.05`

6. **Use `retrieve()` API directly** instead of manually replicating internal logic. This reduces coupling to LinearRAG internals. Adapt the input/output format:
   ```python
   questions = [{"question": query, "answer": ""}]
   results = rag.retrieve(questions)
   passages = results[0]["sorted_passage"]
   scores = results[0]["sorted_passage_scores"]
   ```
   The `"answer"` field is only used for passthrough; an empty string works.

7. **Consider embedding model alignment:** Our default `all-MiniLM-L6-v2` (384-dim) differs from LinearRAG's tested `all-mpnet-base-v2` (768-dim). For clinical text, consider `all-mpnet-base-v2` or a biomedical model like `pritamdeka/S-PubMedBert-MS-MARCO`.

### 6.3 Longer-Term Improvements

8. **Add integration test** that indexes a small set of passages and retrieves against known queries, using the real LinearRAG library. This would have caught issues 5.1-5.4 immediately.

9. **Pin LinearRAG version:** Since we depend on internal APIs (embedding store attributes, graph structure, NER result format), any upstream change can break us silently. Fork or pin a specific commit.

10. **Bio-NER model:** LinearRAG's README notes that the `medical` dataset uses `en_core_sci_scibert`. Our config already defaults to this (`ner_model: str = "en_core_sci_scibert"`), which is correct for clinical text. Good.

---

## 7. Summary of Issues by Severity

| # | Severity | Issue | File:Line |
|---|----------|-------|-----------|
| 5.1 | CRITICAL | `embedding_model` is string, not SentenceTransformer | wrapper:127, wrapper:459 |
| 5.2 | CRITICAL | `max_iterations=50` (should be 3) | wrapper:133, wrapper:464 |
| 5.3 | CRITICAL | Graph path nested one level too deep | wrapper:144-145 |
| 5.4 | HIGH | `damping=0.15` disables graph exploration | wrapper:131, wrapper:463 |
| 5.5 | MEDIUM | Missing tunable config fields | wrapper:126-134 |
| 5.6 | LOW | Vectorized retrieval silently broken | wrapper:_retrieve_raw |
| 2.3 | LOW | Embedding model dimensionality mismatch | settings.py:225 |

Three of these (5.1, 5.2, 5.3) are show-stoppers that will cause crashes or silent degradation on first real use. They should be fixed before any indexing or retrieval is attempted.
