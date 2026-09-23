# Improving CTRA's LinearRAG in place — ideas from SPRIG and schemagraph

**Date:** 2026-09-23
**Scope:** Keep LinearRAG's architecture and keep GLiNER as the NER backend. Fix the implementation.
**Companions:** `docs/2602.23372v1.md` (SPRIG) · `research/rag-alternatives-hybrid-retrieval.md` (the replace-it case + hardware measurements) · `research/rag-synthesis-sprig-schemagraph.md` (the rebuild-it case) · `notes/bench_rag_hardware/` (measured throughput)

---

## 0. The finding that motivates this document

The provenance audit (`research/rag-alternatives-hybrid-retrieval.md` § 5) established that CTRA's vendored LinearRAG is upstream code, essentially unmodified — so its limits are the published architecture's, not CTRA's vendoring. What that framing obscured is this:

**Most of those limits live in LinearRAG's *data structures*, not its *algorithm*.** The scoring maths (entity activation → passage weighting → PPR) is sound. What breaks at scale is that passages are scanned as Python strings per query, embeddings are held as in-RAM Python lists, and the PPR system is re-solved from scratch on every call. All three are replaceable underneath an unchanged algorithm.

This document lists those replacements, sourced from SPRIG and `schemagraph`, ordered by value per line changed.

**Shared-substrate note:** items 1, 3, 6 and 9 build a BM25 index, an ANN index, and a precomputed entity→passage matrix. That is the same infrastructure a hybrid retriever needs. These two paths are not a fork — build the substrate, keep PPR on top, and the hybrid comparison later becomes a config flag rather than a rewrite.

---

## 1. Precompute the entity→passage mention matrix

**Where:** `src/ctra/rag/linearrag/core.py:728` `calculate_passage_scores`

**Problem.** The method iterates over *every* passage, lowercases its full text, then runs `str.count()` for *every* activated entity:

```python
for i, dpr_passage_index in enumerate(dpr_passage_indices):      # ALL passages
    passage_text_lower = ...hash_id_to_text[passage_hash_id].lower()
    for entity_hash_id, (_id, entity_score, tier) in actived_entities.items():
        entity_occurrences = passage_text_lower.count(entity_lower)
```

At CTRA scale (2M+ passages, ~50 activated entities, 5.5 KB mean passage) that is ~100M Python substring scans over ~11 GB of transient lowercased strings — **per query**. It is the dominant query cost, well above the embedding matmul.

**Fix.** SPRIG's entity–document bipartite graph is precisely this, precomputed at index time with TF–IDF weights:

> w(e,d) = tf(e,d) · log((N+1)/(df(e)+1)) + 1

**CTRA already has the data.** `passage_hash_id_to_entities` is persisted in `ner_results.json` and loaded by `extract_nodes_and_edges` (`core.py:1025`); `add_entity_to_passage_edges` (`core.py:1000`) already walks it. It is simply never materialised as a matrix. Build a `scipy.sparse` CSR of shape (|entities|, |passages|) at index time — mirroring `_precompute_sparse_matrices` (`core.py:269`), which already does this for entity↔sentence — and replace the double loop with a single `entity_scores @ entity_to_passage`.

**Semantics.** Near-identical, and arguably more correct: `str.count()` matches substrings inside unrelated words (`"il2"` inside `"il20"`), whereas an index-time token-aware count does not. Preserve the existing `log(1 + bonus)` and tier division.

**Effort:** ~1 day · **Risk:** low · **Quality impact:** neutral-to-positive

---

## 2. Cache the PPR matrix; add a push-based mode

**Where:** `core.py:385` `run_ppr`

**Problem.** `igraph.personalized_pagerank(..., implementation="prpack")` is called over the full vertex set on every query. PRPACK is a direct linear solve — exact, but it re-derives the system each call, and at millions of entity+passage nodes that is not a per-query operation.

**Fix — from `schemagraph/graph/ppr.py`:**

- `PPRMatrix`: row-normalised sparse CSR **built once per graph** and reused. Its docstring records the motivation — converting a NetworkX view per query cost 300–400 ms against a few ms for the iteration itself.
- **Teleport-vector initialisation**: start `x = p` so mass never enters components the seeds do not touch; those nodes stay exactly zero. This replaces any need to restrict the graph to touched components.
- **Explicit dangling-node handling** (`self.dangling`), absent from LinearRAG.
- **Documented convergence**: `tol=1e-12`, because at `1e-6` a 7,000-node schema stopped with L1 error near 1e-2 — *"enough to reorder near-tied tables at rank 1."*

**Fix — from SPRIG § 3.3:** a push-based approximation with residual threshold ε, confining work to the seeds' neighbourhood. SPRIG reports push-based helps on HotpotQA while power iteration is competitive on 2Wiki, so make it a config option, not a default.

⚠️ **Do not copy SPRIG's `max_iter=5`.** At 0.85 damping, ~44% of the initial error survives five iterations — that is a decayed neighbourhood expansion, not converged PPR. Use schemagraph's converged settings. (This is also why SPRIG's weak graph-only numbers should be read as a pessimistic bound.)

**Effort:** ~2–3 days · **Risk:** medium (prpack is exact; these approximate) · **Mitigation:** `tol=1e-12` is effectively exact; validate against prpack on a small index

---

## 3. ANN + memory-mapping for dense passage retrieval

**Where:** `core.py:777` `dense_passage_retrieval`; `src/ctra/rag/linearrag/embedding_store.py:189`

**Problem.** Two compounding issues:

```python
question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]   # FULL argsort
```

— a brute-force matmul over all passages plus a full argsort, returning *every* passage sorted, per query. And `EmbeddingStore` holds all embeddings as a Python list, with `get_embeddings` (line 206) doing `np.array(self.embeddings)[indices]` — materialising the whole store on every call.

**Fix.** SPRIG uses HNSW (M=32, efConstruction=200, efSearch=64). Swap the in-RAM list for a memory-mapped `numpy` array plus a FAISS index. **LinearRAG's algorithm does not care where the vectors live** — this is contained entirely within `EmbeddingStore` and `dense_passage_retrieval`.

This is what actually dissolves the RAM wall documented in `research/rag-alternatives-hybrid-retrieval.md` § 2, without changing the retrieval maths.

⚠️ **One real semantic change.** `calculate_passage_scores` currently applies `min_max_normalize` over *all* DPR scores. Restricting to top-K changes that normalisation's range. SPRIG does exactly this (top-k seeds) so it is defensible, but it must be measured, not assumed.

SPRIG also found ANN-vs-exact differences visible on 2k-query subsets washed out at full validation (Table 3: tuned and default HNSW gave *identical* full-validation numbers), which is mildly reassuring.

**Effort:** ~3–4 days · **Risk:** medium (the normalisation change) · **Impact:** removes the RAM wall

---

## 4. Fix the order-dependence bug

**Where:** `core.py:460`–`519`, the BFS loop in `calculate_entity_scores`

**Problem.** `used_sentence_hash_ids` is a **global** set: each sentence is consumed once across the entire traversal, and entities are visited in `dict` iteration order. Which entity claims a shared sentence therefore depends on insertion order, making retrieval results non-deterministic across runs with different hash seeds.

**Fix.** `schemagraph/graph/ppr.py` handles the same hazard explicitly — `for fqn in sorted(...)` carries the comment *"stable order: ties must not depend on the hash seed."* Iterate `current_entities` sorted by `(-entity_score, entity_hash_id)`.

**Effort:** ~1 hour · **Risk:** none · **Note:** this is a correctness fix worth making regardless of any other decision, and it should land before the benchmark harness, or run-to-run variance will be misread as signal.

---

## 5. Date/source as a pre-filter mask

**Where:** `src/ctra/rag/linearrag_wrapper.py:295` `search`, `:44` `_OVERSAMPLE_FACTOR`

**Problem.** Retrieve `top_k × 3`, then discard by source and date, **conservatively dropping every undated document**. With aggressive source filters this silently under-returns — the logged `"No passages match filters"` warning is the symptom. This is the primary label-leakage defence (`config/settings.py` § 2a) implemented as a best-effort post-hoc trim.

**Fix.** Carry `date` and `source` as arrays aligned to passage index and mask the PPR score vector (and the ANN candidate set from item 3) **before** top-k. Exact, cheap, and removes the oversample heuristic entirely.

Neither SPRIG nor schemagraph needs this — it is CTRA-specific, and it is the item with the clearest correctness argument.

**Effort:** ~1 day · **Risk:** low · **Impact:** turns a best-effort leakage guard into a hard guarantee

---

## 6. Seed PPR from BM25 hits, not only query entities

**Where:** `core.py:728` `calculate_passage_scores` (builds `passage_weights`), `core.py:167` `retrieve`

**This is SPRIG's strongest result, and it is a small change here.** LinearRAG already blends two signals into passage node weights:

```python
passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)
```

Adding BM25 mass to that vector is SPRIG's **SPRIG-MIX**, and seeding PPR from a fused candidate list is **GraphRRF**:

| SPRIG variant | Hotpot R@10 | 2Wiki R@10 |
|---|---|---|
| `Graph` — query-entity seeds only (**what CTRA does today**) | 0.464 | 0.357 |
| `GraphHybrid` — BM25 seeds | 0.775 | 0.743 |
| `GraphDense` — dense seeds | 0.844 | 0.747 |
| **`GraphRRF`** — fused seeds | **0.867** | **0.794** |
| `RRF+PPR` — **score-side** fusion | 0.782 | 0.602 |

Two readings matter. First, query-entity-only PPR scores **below plain BM25** (0.742 / 0.643) — the graph is a re-scorer, not a retriever. Second, **score-side fusion is worse than not fusing at all**; the gain comes from richer seeds feeding traversal, not from ensembling rankings afterwards. Fuse at the seed.

SPRIG's mixing rule: `s = norm(α_mix·s_e + (1−α_mix)·s_d)`, with an adaptive option `α_mix = (n_e+1)/(n_e+n_d+2)`. Rank-based seed weighting beat raw and softmax in their ablations. Seed sizes k=5–10.

The BM25 index this needs is the same one item 9 needs, and the same one a hybrid retriever would need.

**Effort:** ~3–4 days (including the BM25F index) · **Risk:** medium — a genuine quality change · **Requires the benchmark harness**

---

## 7. Multi-seed activation instead of `argmax`

**Where:** `core.py:799` `get_seed_entities`

```python
best_entity_idx = np.argmax(entity_scores)
```

Exactly **one** graph entity is kept per query entity. One ambiguous or misspelled match and the seed is silently wrong with no recovery path — and the BFS then amplifies that error.

`schemagraph/linking/lexical.py` instead activates many nodes with graded weights (exact token 1.0, n-gram 1.6, abbreviation/lemma 0.8, fuzzy ≥0.86 → 0.6, description-only 0.35). Take top-*m* above a similarity floor, weighted by score.

**Effort:** ~half a day · **Risk:** low-medium · **Requires the harness**

---

## 8. Node specificity on the reset vector

**Where:** `core.py:424` / `:385`, the `entity_weights` vector fed to `run_ppr`

LinearRAG applies **no** specificity weighting. Both comparison systems do, arrived at independently:

- `schemagraph/graph/ppr.py` — `specificity_weights`: `1/log(1 + n_nodes_sharing_name)`, described as "the analogue of inverse document frequency for schema objects."
- SPRIG — hub downweighting `df(e)⁻ᵖ` on document→entity edges (p=0.5) plus seed-entity downweighting `df(e)⁻ᑫ` (q=0.5 HotpotQA, 1.0 2Wiki).

Biomedical corpora have worse hubs than either benchmark: *patients*, *treatment*, *study*, *safety*, *efficacy* will appear in nearly every passage and currently seed PPR with the same authority as *pembrolizumab*.

SPRIG's related lever, **hub pruning** (remove top 1% of entities by `df`), cut query time 16–28% with negligible Recall@10 change, and Appendix A.10 shows pruned entities rarely overlap gold mentions — it removes generic hubs, not rare bridge entities.

**Effort:** ~1 day · **Risk:** low-medium · **Requires the harness**

---

## 9. BM25 fallback when NER finds no entities

**Where:** `core.py:799` `get_seed_entities` returns empty → `retrieve` falls back to `dense_passage_retrieval`

The current fallback is the O(N) brute-force dense scan — the most expensive path in the system, taken exactly when the system has the least information. SPRIG evaluates a BM25 top-1 fallback for entity-less queries (Appendix Table A.8) and finds it an improvement.

**Effort:** ~2 hours (once a BM25 index exists) · **Risk:** low

---

## 10. GLiNER: switch to the bi-encoder

**Where:** `RAGConfig.ner_model` / `ner_labels` (`src/ctra/config/settings.py:230`), `src/ctra/rag/ner_config.py`, `src/ctra/rag/gliner_batched.py`

CTRA runs `Ihor/gliner-biomed-large-v1.0` — the **uni-encoder** at **large** scale. From the GLiNER-BioMed paper (`docs/2504.00676v2.md` § "Inference Efficiency"):

- Bi-encoder throughput: **+39–63%** with dataset-specific label counts; **+92–568%** at 127 labels. The cause is structural — uni-encoder cost is **quadratic**, O((|SE|+|T|)²), while bi-encoder is linear, O(|T|²), because label embeddings are precomputed and cached.
- At small and base scale the **bi-encoder outperforms the uni-encoder** (+4.40 and +3.94 F1). Only at large does uni-encoder win (+4.87).
- **GLiNER-BioMed-small, with 7x fewer parameters, matches GLiNER-v2.5-large** with no significant difference (p>0.05).

`docs/ner-evaluation-guide.md:143` already lists *"Test a different GLiNER variant (e.g., the bi-encoder model)"* as a supported operation — and it appears never to have been run. The NER evaluation compared GLiNER against scibert and scispaCy, never against its own smaller and bi-encoder siblings.

### MEASURED 2026-09-23 — the bi-encoder is **not** the win; downscaling is

All six variants, on 120 real CTG chunks (mean 1,147 chars), GTX 1650 Ti fp16 batch 8, production call path, threshold 0.4. Scripts and raw JSON: `notes/bench_rag_hardware/bench_gliner_variants.py`, `results_gliner_variants.json`.

| Model | Labels | doc/s | vs prod | VRAM | CTG est | Agreement vs prod |
|---|---|---|---|---|---|---|
| **uni-large** (production) | 16 | 1.13 | 1.00x | 1,616 M | 22.8 d | — |
| uni-large | 6 | 1.22 | 1.08x | 1,616 M | 21.1 d | — |
| bi-large | 16 | 1.43 | **1.27x** | 1,618 M | 18.0 d | 0.46 |
| **uni-base** | 16 | 3.95 | **3.50x** | 744 M | **6.5 d** | **0.56** |
| bi-base | 16 | 3.40 | 3.02x | 1,013 M | 7.6 d | 0.45 |
| **uni-small** | 16 | 6.35 | **5.63x** | 655 M | **4.1 d** | 0.50 |
| bi-small | 16 | 4.84 | 4.29x | 903 M | 5.3 d | 0.44 |

**Three findings, two of which contradict the prediction above:**

1. **The bi-encoder only helps at `large`, and less than advertised.** `bi-large` gives 1.27x — below the paper's 39–63% band. At base and small it is *slower* than the uni-encoder (`bi-base` 0.86x of `uni-base`; `bi-small` 0.76x of `uni-small`) and uses **more** VRAM, consistent with carrying a second encoder. The paper's gains come from amortising cached label embeddings across many labels; at 16 labels and ~1,147-char documents, text encoding dominates and the extra encoder is pure overhead. Their measurements were RTX3090 / fp32 / "dataset-specific labels" — a different regime.

2. **Model scale is the real lever.** `uni-small` is **5.63x** production (22.8 d → 4.1 d), `uni-base` **3.50x** (→ 6.5 d). That is the 3–6x hoped for, from a different knob than expected.

3. **⚠️ The speedup is not free — agreement with production is only 0.44–0.56 Jaccard.** Even the closest variant (`uni-base`, 0.56) shares barely half its extracted entity set with the current model. These are materially different extractions, not cheaper approximations of the same output. Note also that bi-encoder variants cluster low (0.44–0.46) regardless of scale, suggesting the architecture extracts systematically differently.

**Recommendation:** drop the bi-encoder idea. Evaluate **`uni-base`** (best speed/agreement trade) and **`uni-small`** on F1 — and unlike the retrieval question, **the harness for this already exists**: `src/ctra/rag/eval/` (`NERExperiment`, CHIA/BioNLP benchmarks, threshold sweeps) is precisely the right tool. This is the one item on the list that is decidable today.

**Effort:** config change + an existing-harness run · **Risk:** low to trial, medium to adopt (0.5 agreement is a real quality change)

---

## 11. Cut the label count

### ⚠️ MEASURED 2026-09-23 — the cost rationale below is wrong

Cutting 16 labels → 6 gave only **1.08x** at large (1.13 → 1.22 doc/s), and similarly small deltas at every scale. The quadratic term in `(|labels| + |text|)` is real but irrelevant here: CTRA's chunks are ~1,147 characters, so `|text|` dominates `|labels|` completely. The paper's 92–568% figures were measured at **127 labels**, where the ratio inverts.

**Label count is not a cost lever for CTRA.** The *precision* rationale below stands on its own and is unaffected — a curated gazetteer is exact for in-vocabulary terms where GLiNER is probabilistic — but this item should be justified on quality, not speed, and it drops well down the priority list.

---

Of CTRA's 16 labels, three have authoritative vocabularies:

| Label | Vocabulary | Status |
|---|---|---|
| `Drug` | ChEMBL `molecule_synonyms` + PubChem | **Already loaded** by `EntityResolver` |
| `Disease` | MeSH, CTG `conditionsModule` | Available |
| `Gene or protein` | HGNC | Available |

Aho-Corasick over 100k+ phrases tags documents in milliseconds, single-pass. Run the gazetteer for those three; reserve GLiNER for the types with no vocabulary — `Mechanism of action`, `Clinical endpoint`, `Adverse event`, `Patient population`, `Biomarker`. GLiNER then stays where it genuinely generalises, and drug/disease/gene mentions become exact rather than probabilistic. ~~16 → ~6 labels is a superlinear saving under the quadratic cost~~ — measured at 1.08x; see above.

⚠️ `docs/ner-evaluation-guide.md` found GLiNER's *typed* extraction acts as an implicit precision filter and beat scispaCy on downstream retrieval. That compared two statistical taggers; a curated gazetteer is a third option neither was measured against. It wins on precision by construction for in-vocabulary terms and loses on out-of-vocabulary recall. **Measure the in-vocabulary hit rate on real CTG interventions before committing.**

**Effort:** ~2–3 days · **Risk:** medium · **Requires the harness**

---

## 12. Field-selective NER

CTG passages average 5,544 chars (measured over 3,000 real trials), but entity density is concentrated in `briefTitle`, `conditions` and `interventions`. The `eligibilityCriteria` block is long and comparatively entity-poor.

`_ctg_to_passages` (`src/ctra/rag/indexer.py:52`) concatenates all fields into one blob. Emitting fields separately would allow skipping low-yield ones for NER — plausibly 50%+ of tokens — and would simultaneously enable **BM25F** field weighting for item 6.

**Effort:** ~1–2 days · **Risk:** low-medium

---

## Summary

| # | Change | Source | Effort | Risk | Needs harness |
|---|---|---|---|---|---|
| 1 | Precompute entity→passage matrix | SPRIG | 1 d | low | no |
| 2 | Cached PPR matrix + push-based mode | schemagraph + SPRIG | 2–3 d | med | no |
| 3 | ANN + memmap for dense retrieval | SPRIG | 3–4 d | med | partly |
| 4 | Fix order-dependence bug | schemagraph | 1 h | none | no |
| 5 | Date/source pre-filter mask | CTRA-specific | 1 d | low | no |
| 6 | Seed PPR from BM25 (GraphRRF) | SPRIG | 3–4 d | med | **yes** |
| 7 | Multi-seed instead of argmax | schemagraph | 0.5 d | low-med | **yes** |
| 8 | Node specificity + hub pruning | both | 1 d | low-med | **yes** |
| 9 | BM25 fallback for entity-less queries | SPRIG | 2 h | low | no |
| 10 | ~~GLiNER bi-encoder~~ → **downscale to `uni-base`/`uni-small`** | measured | config | low-med | NER harness (**exists**) |
| 11 | Gazetteer for Drug/Disease/Gene (precision, **not** speed) | CTRA-specific | 2–3 d | med | **yes** |
| 12 | Field-selective NER + BM25F fields | schemagraph | 1–2 d | low-med | no |

### What this fixes

| Blocker | After |
|---|---|
| Query-time O(passages × entities) `str.count()` | **Gone** (1) |
| Full-corpus matmul + argsort per query | **Gone** (3) |
| RAM wall — embeddings as in-RAM Python lists | **Largely gone** (3) |
| PPR system re-solved per query | **Gone** (2) |
| Non-deterministic results | **Gone** (4) |
| Best-effort leakage filter | **Gone** (5) |
| Single-seed activation fragility | **Gone** (6–8) |
| GLiNER index cost — 22 days for CTG | **Measured 4.1 d (`uni-small`) / 6.5 d (`uni-base`)** — but at 0.50–0.56 entity agreement, so quality must be adjudicated first (10) |
| Sentence embedding store, 13–100 GB | **Still hard** — see below |

### The residual

LinearRAG's **sentence tier** is its most expensive structure and the one item on that list with no in-place fix. Worth noting: **SPRIG has no sentence layer at all** — an entity–document bipartite graph, and PPR works. Dropping the sentence tier would be a genuine architecture change rather than an implementation fix, so it is flagged here, not recommended.

### Suggested order

1. **Item 4** (1 hour, no risk) — otherwise run-to-run variance pollutes every later measurement.
2. **Item 10's F1 run** — `uni-base` and `uni-small` on the **existing** `src/ctra/rag/eval/` NER harness. Throughput is now measured (3.50x / 5.63x, 22.8 d → 6.5 d / 4.1 d); only quality is open, and the tool for it already exists. **This is the one decidable item on the list today**, and a 3–6x faster index makes every later experiment cheaper.
3. **The retrieval benchmark harness** — still the blocking prerequisite for everything else (`research/rag-alternatives-hybrid-retrieval.md` § 8). Items 6, 7, 8 and 11 are undecidable without it.
4. **Items 1, 2, 3, 5** — pure implementation wins, no quality risk, and they make iteration fast enough to evaluate the rest.
5. **Items 6–9, 11–12** — quality changes, one variable per run, scored on the harness.
