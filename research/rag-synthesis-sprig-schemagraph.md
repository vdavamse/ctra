# Incorporating SPRIG and schemagraph into CTRA's RAG

**Date:** 2026-09-23
**Question:** How should SPRIG (`docs/2602.23372v1.md`) and the `schemagraph` implementation inform CTRA's retrieval architecture?
**Companion:** `research/rag-alternatives-hybrid-retrieval.md` (the hybrid-vs-LinearRAG case and hardware measurements), `notes/bench_rag_hardware/` (measured throughput).

---

## 1. The reframing: CTRA is a schemagraph problem, not a SPRIG problem

Both SPRIG and LinearRAG solve *"build a knowledge graph from unstructured text, cheaply, without an LLM."* They extract entities with lightweight NER and infer structure from **co-occurrence**.

CTRA does not have that problem in the same form, because it already owns the structure:

| Asset | What it is | Current use |
|---|---|---|
| **PrimeKG** | **4M+ curated, typed biological relations** across drugs, targets, pathways, diseases, phenotypes (MIT licence) | **Flattened into prose** by `_primekg_to_passages`, capped at 30 relations per entity, then fed to GLiNER so co-occurrence can re-infer the relations that were just discarded |
| **EntityResolver** | ChEMBL + PubChem **synonym→canonical gazetteer** (`src/ctra/data/entity_resolver.py`) | Query-time drug expansion only |
| **CTG `protocolSection`** | Explicitly fielded records (title, summary, conditions, interventions, eligibility, outcomes, sponsor) | Concatenated into one flat text blob by `_ctg_to_passages` |

`_primekg_to_passages` is the sharpest example. It takes `x_name`, `display_relation`, `y_name` — a typed edge list — renders it as *"Pembrolizumab targets PDCD1, CD274, ..."*, and then spends GPU-days running a 460M-parameter NER model to recover "Pembrolizumab" and "PDCD1" and guess that they are related because they co-occur. **The graph is destroyed in order to be rebuilt worse.**

This makes CTRA structurally much closer to **schemagraph** than to SPRIG or LinearRAG:

| schemagraph | CTRA equivalent |
|---|---|
| `t:<table>` / `c:<column>` nodes | trial / chunk |
| `relation` edges (FK, lineage) with `kind` + provenance | **PrimeKG typed relations** (`display_relation`, x_type/y_type) |
| `k:<term>` business-glossary nodes | **ChEMBL / MeSH / HGNC entities** |
| `w:<token>` lexical anchors + `mention` edges | gazetteer-matched entity mentions in passages |
| `contains` (table ↔ column) | trial ↔ chunk |
| column→table `top3` aggregation | **chunk→trial aggregation** (needed: 4.94 chunks/trial, measured) |
| rank-tiered column cap | rank-tiered chunk budget |
| BM25F over fielded documents | BM25F over `protocolSection` fields |

schemagraph's architecture is the better template precisely because it was built for a corpus that *already had* curated relations and a glossary. LinearRAG's design has nowhere to put PrimeKG.

## 2. The two cost levers, validated

CTRA's measured blockers (`notes/bench_rag_hardware/`, 2026-09-23) were: GLiNER-BioMed-large at 1.16 chunk/s (22 days for CTG alone, 84–320 days full corpus) and MiniLM-L6 embedding at 71.2 chunk/s GPU (33–125 h). Both have an answer in these two systems.

### 2.1 Static embeddings (from schemagraph) — kills the embedding cost

`schemagraph/linking/embed.py` uses `minishlab/potion-base-8M` — **model2vec**: static token embeddings, mean-pooled, **numpy only, no torch, no transformer forward pass**. Published figures: `potion-base-32M` reaches **94.66%** of all-MiniLM-L6-v2's average MTEB score and **~70x faster on CPU**; `potion-retrieval-32M` reaches **81.69%** of MiniLM specifically on *retrieval*, "orders of magnitude faster."

Derived for CTRA (not measured — see § 6): at ~70x CTRA's measured CPU rate of 42.7 chunk/s, the full 8.4–32.1M-chunk corpus embeds in roughly **1–3 CPU-hours, with no GPU at all**, versus 33–125 GPU-hours for MiniLM.

**The upgrade path is the interesting part.** model2vec's core operation is *distillation*: `distill(model_name=...)` passes a vocabulary through any sentence-transformer, PCA-reduces, and applies Zipf weighting — **no training data required**, and it accepts a **custom vocabulary**. So a biomedical static embedder is a cheap experiment: distil **MedCPT** or **PubMedBERT** against a vocabulary built from ChEMBL synonyms + MeSH + CTG condition terms. That is the one plausible route to biomedical embedding quality on this hardware — MedCPT itself was priced at 1–3.5 weeks corpus-wide and ruled out.

### 2.2 Gazetteer entity extraction — kills the NER cost

SPRIG's finding is that **lightweight NER suffices** (spaCy `en_core_web_sm` at 9–10 ms/doc; regex at 0.02 ms/doc), and that the **entity** graph beats the TF–IDF **term** graph decisively (R@10 0.464/0.357 vs 0.419/0.367 — and both far below BM25, but the entity topology is the better one). The author's conclusion: *"explicit entity co-occurrence provides a more effective topology for multi-hop navigation than term-level edges... This supports the use of lightweight NER even without full entity linking."*

CTRA can go further than "lightweight NER" because it has **authoritative vocabularies**. Aho-Corasick matching over 100k+ phrases tags documents in **milliseconds**, single-pass; QuickUMLS reports similar precision/recall to MetaMap/cTAKES at **up to 135x** the speed. For CTRA's entity types that actually matter — drugs, diseases, genes — ChEMBL/MeSH/HGNC *are* the ground truth, and `EntityResolver` already holds the drug half.

**Zero-shot NER was solving a problem CTRA doesn't have.** GLiNER earns its keep where no vocabulary exists; for `Drug`, `Disease`, `Gene or protein` it is an expensive way to approximate a lookup table. Estimated cost: **~1–3 CPU-hours** for the full corpus against a measured 84–320 GPU-days.

Note this does *not* invalidate `docs/ner-evaluation-guide.md`'s finding that GLiNER beats scispaCy for retrieval — that compared two *statistical* taggers. A curated gazetteer is a third option neither was measured against, and it wins on precision by construction for in-vocabulary terms while losing on recall for out-of-vocabulary ones (§ 6).

## 3. Proposed architecture

```
query + before_date + sources
  │
  ├─[1]─ query normalization: gazetteer match + ChEMBL synonym expansion
  │         ("Keytruda" → {Keytruda, pembrolizumab, MK-3475, ...})
  │
  ├─[2]─ BM25F over protocolSection fields        ─┐
  ├─[3]─ dense: static (model2vec) over chunks    ─┤→ RRF (k=60) → candidates
  │                                                │
  ├─[4]─ metadata PRE-filter (date, source) ───────┘   exact, before top-k
  │
  ├─[5]─ (optional) graph leg — SEED-side:
  │         candidates' matched entities + query entities
  │              → PPR over the PrimeKG-backed entity graph
  │              → chunk scores via one sparse matmul
  │              → re-fuse into the candidate ranking
  │
  ├─[6]─ chunk → trial aggregation (top3 rule)
  ├─[7]─ TinyBERT-L2 cross-encoder rerank, depth 30–50
  └─[8]─ rank-tiered chunk budget → RetrievedChunk[]
```

### Layer notes

**[1] Query normalization.** `EntityResolver` already does this; it moves from patching LinearRAG's NER to expanding BM25F query terms directly — a native lexical operation rather than a graph-seeding trick. **This is the single highest-confidence component**: biomedical retrieval's dominant failure is surface-form fragmentation (Keytruda / pembrolizumab / MK-3475 share zero tokens), and this is exactly what a synonym table fixes.

**[2] BM25F, not flat BM25.** Direct transfer from `schemagraph/linking/bm25.py` (110 lines, per-field weights and length normalisation). `_ctg_to_passages` currently concatenates fields into one blob; a trial's `briefTitle` should not be scored identically to a paragraph of `detailedDescription`. schemagraph's weights (`name 1.0, columns 1.0, business 0.9, tags 0.5, desc 0.35`) map onto (`title 1.0, conditions/interventions 1.0, outcomes 0.9, keywords 0.5, summary/description 0.35`).

**[3] Dense leg = static embeddings.** Per § 2.1. Keep MiniLM-on-GPU as the quality ceiling to measure against, but static is the default so the pipeline has **no GPU dependency at all**.

**[4] Metadata pre-filter.** CTRA-specific; neither source system needs it. Today `search()` oversamples 3x and discards by date/source afterwards, conservatively dropping every undated document — a best-effort trim on the primary **label-leakage defence** (`config/settings.py` § 2a). BM25 and FAISS both support exact pre-filtering via candidate masks. This must become a hard guarantee.

**[5] Graph leg — deferred, and seed-side.** Two design decisions, both evidence-driven:

- **Entity-only graph, chunks as a sparse matrix.** PrimeKG is ~129k nodes / 4M edges — PPR over it is milliseconds and ~50–80 MB. Chunks are *not* graph nodes; they are reached by one sparse matmul `entity_scores @ entity_to_chunk`. This is what LinearRAG's `calculate_entity_scores_vectorized` gestures at, done correctly — and it structurally eliminates the `O(passages × entities)` Python `str.count()` scan that makes LinearRAG's `calculate_passage_scores` unusable at scale, because mentions are precomputed at index time.
- **Seeds come from the fused candidate list, not just query NER.** SPRIG measured both: `GraphRRF` (fusion → seeds → PPR) is the **best R@10 on both datasets** (0.867 / 0.794), while `RRF+PPR` score-fusion is **worse than not fusing at all** (0.782 / 0.602 vs RRF's 0.851 / 0.697). And query-entity-only `Graph` scores 0.464 / 0.357 — *below plain BM25*. The graph is not a retriever; it is a re-scorer fed by one.

**[6] Chunk→trial aggregation.** Forced by measurement: trials are 5,544 chars mean and must be split into 4.94 chunks. schemagraph solved the identical column→table problem — `own + best + 0.5·second + 0.25·third + 0.02·rest` — after finding that a plain sum let wide objects with many weak matches swamp one strong match. A trial with forty mediocre chunks should not outrank a trial with one exact hit.

**[7] Reranker.** Measured: TinyBERT-L2 at **222 ms** (CPU, depth 50) vs MiniLM-L6 CE at **2,139 ms** (unusable). SPRIG independently used TinyBERT and still got the study's best MRR (0.887). Both sources agree the CE buys **precision, not recall** — SPRIG's RRF+CE slightly *lowers* R@10 versus RRF alone. That is the right trade for CTRA, since the feature-builder agent reads the top chunks, but it should be a stated choice. `tools.py`'s existing diskcache memoisation amortises the cost across MCTS rollouts.

**[8] Rank-tiered budget.** schemagraph's most surprising measured win: uncapping columns for the **top-1** table lifted column strict recall **81.2 → 91.2 at −0.5% tokens**, because *"table rank, not column evidence, predicts gold columns."* CTRA analogue: return every chunk of the top-ranked trial, cap lower-ranked trials. Same token budget, better coverage.

## 4. Attribution — what comes from where

| Component | Source | Confidence |
|---|---|---|
| BM25F multi-field over fielded records | schemagraph `linking/bm25.py` | High — direct structural match |
| Static embeddings (model2vec), distillation for biomedical | schemagraph `linking/embed.py` | High on cost, **unvalidated on biomedical quality** |
| RRF fusion of sparse + dense | both | High |
| `top3` chunk→trial aggregation | schemagraph `graph/ppr.py` | High — same problem shape |
| Rank-tiered chunk budget | schemagraph `linking/linker.py` | Medium-high |
| Cached sparse PPR matrix, teleport init, dangling handling, `tol=1e-12` | schemagraph `graph/ppr.py` | High (if a graph leg is built) |
| Specificity / IDF hub suppression | both, independently | High |
| Cost vs. affinity edge separation | schemagraph `graph/build.py` | High — PrimeKG relation kinds differ in trust exactly as FK vs. inferred do |
| **Seed-side fusion (GraphRRF), not score-side** | SPRIG Table 5 | High — directly measured, both directions |
| Push-based PPR (residual threshold) | SPRIG § 3.3 | Medium — needed only at scale |
| Hub pruning (top 1% by `df`) | SPRIG Table 4 | Medium-high — 16–28% query time, no recall loss; biomedical hubs are severe |
| Rank-based seed weighting > raw/softmax | SPRIG § 6 | Medium |
| BM25 fallback when no entities match | SPRIG A.8 | High — cheap safety net |
| Gazetteer entity extraction over ChEMBL/MeSH/HGNC | **neither** — CTRA-specific | Medium — see § 6 |
| PrimeKG as graph backbone instead of prose | **neither** — CTRA-specific | Medium-high |
| Metadata pre-filter for leakage | **neither** — CTRA-specific | High, mandatory |

## 5. What NOT to take

- **schemagraph's path-finding and PathRAG pruning** (`pathfinding.py`, `pruning.py`). These exist because schemagraph's output contract is a *join-complete subgraph* — bridge tables are structurally mandatory. CTRA returns ranked passages; there is no join to complete. Skip entirely.
- **schemagraph's score-side RRF for the graph leg.** Its own gain is small (+0.19 strict) and SPRIG measured this exact shape as actively harmful. Use seed-side.
- **SPRIG's `max_iter=5` PPR.** At 0.85 damping, ~44% of initial error survives 5 iterations — that is a decayed neighbourhood expansion, not converged PPR. Use schemagraph's converged solve. This also means **SPRIG's weak graph-only numbers may be a pessimistic bound**, since schemagraph documents that under-convergence reorders the head of the ranking.
- **SPRIG's regex NER** (`\b[A-Z][a-z]+...`). Capitalisation-based extraction is meaningless for biomedical text — drug and gene names have no such convention.
- **GLiNER for corpus-wide indexing.** Keep it for query-side extraction (small volume, CPU-fine) if wanted; it is the wrong tool for 8–32M chunks.

## 6. Risks, and what would falsify this

1. **Static embeddings may degrade much worse than 81.69% on biomedical text.** model2vec is *uncontextualised* — a mean of static token vectors — so it cannot disambiguate polysemy or bind multiword terms. Published figures are general-domain (MTEB). Biomedical text is dense with multiword entities and abbreviation collisions. **Mitigation:** use it as the recall leg beside BM25F (which handles exact identifiers) with the CE fixing ordering — schemagraph uses it this way, as *additive* seeds worth +0.4 strict, never as a replacement. **Falsifier:** if static-only NDCG@10 on TREC-CT falls below BM25F alone, the leg isn't paying and MiniLM-on-GPU comes back.

2. **Gazetteer NER has a recall ceiling by construction.** It finds only what's listed — no novel compounds, no unlisted synonyms, no typos. **Mitigation:** SPRIG's own finding that *"GraphHybrid is more robust because BM25 seeding compensates for NER errors"* applies directly; the gazetteer feeds a boost layer, not the retriever. **Falsifier:** measure in-vocabulary hit rate on a CTG sample — if a large fraction of interventions don't resolve to ChEMBL, the vocabulary is too narrow.

3. **SPRIG's BM25-2step (entity query expansion) did *not* beat BM25** (0.729/0.653 vs 0.742/0.643). That is evidence against naive graph-driven query expansion. **However**, it does not transfer cleanly to layer [1]: SPRIG expanded with *co-occurrence* entities from a noisy graph (semantic expansion); CTRA expands with *curated synonyms* (lexical normalisation of the same referent). Different mechanism, much higher prior. But it does argue for keeping PrimeKG-relation expansion (pembrolizumab → PD-1 → nivolumab) **measured and optional**, not on by default.

4. **Memory at the graph leg needs sizing.** Entity→chunk mentions at ~10 entities/chunk × 8.4M chunks ≈ 84M nnz ≈ 340–670 MB depending on dtype — on a 7 GB machine that is real but survivable, and SPRIG's hub pruning plus a per-entity edge cap (`top-L`) reduce it directly. Restricting the graph leg to high-value sources (CTG + ChEMBL + PrimeKG, excluding PubMed) cuts it further.

5. **Everything quality-related here is unmeasured for CTRA.** The cost arguments rest on measurements (`notes/bench_rag_hardware/`) plus published throughput figures. The *quality* arguments rest on two out-of-domain benchmarks — Wikipedia multi-hop QA (SPRIG) and text-to-SQL schema linking (schemagraph). Neither is biomedical retrieval. **This is why § 7 step 0 is non-negotiable.**

## 7. Phased plan

**Step 0 — the retrieval harness (blocking, ~1 day).** Rebuild the TREC-CT 2021 benchmark as reusable code (`src/ctra/rag/eval/retrieval.py`: NDCG@10, MRR, Recall@k) against a pluggable retriever interface. It exists today only as prose in `docs/ner-evaluation-guide.md`; `grep -ri trec` finds no code. Adopt schemagraph's harness discipline: one variable per run, per-variant CSV+JSON, a dated iteration log, and rejected options kept as flags with the reason. **Nothing below is decidable without this.**

**Step 1 — the CPU core (~1.5–2 weeks).** Layers [1]–[4], [6]–[8] behind `RAGConfig.backend = "hybrid"`, reusing `indexer.py` verbatim. No graph. No GPU. Score against the step-0 harness with the recorded LinearRAG baseline.

**Step 2 — embedder bake-off (~2 days).** On the harness: `potion-base-8M` vs `potion-retrieval-32M` vs a **MedCPT-distilled** static model vs MiniLM-L6-on-GPU. Also test a **512-token** embedder (e5-small, bge-small) to halve the 4.94x chunking multiplier — fewer, larger chunks may be net-cheaper *and* better.

**Step 3 — restructure the wasted assets (~1 week).** Stop flattening PrimeKG; load it as a typed edge list with `weight`/`affinity` separation per relation kind. Build the gazetteer mention index (Aho-Corasick over ChEMBL + MeSH + HGNC + CTG conditions). Field-split `_ctg_to_passages` for BM25F. **These are valuable independent of whether the graph leg ships** — the gazetteer improves query normalisation and BM25F needs the fields.

**Step 4 — graph leg, only if step 1 shows a multi-hop deficit (~1 week).** The signal to look for is *not* "lower NDCG" but a **recall deficit concentrated on bridge-requiring queries** — where the answer needs chaining and the bridging passage shares no vocabulary with the query ("trials of drugs targeting the same pathway as X"). A broad, uniform loss is an embedder problem, not a multi-hop problem. If the deficit is real: entity-only PPR over PrimeKG, seed-side fusion, schemagraph's `PPRMatrix`, SPRIG's push-based mode and hub pruning.

## 8. Bottom line

Neither system ports wholesale. What they jointly supply is:

- **schemagraph** — the architecture template (curated relations + glossary + lexical anchors + typed edges), four directly transferable components (BM25F, `top3` aggregation, rank-tiered budget, the PPR engine), and the static-embedding idea that **removes CTRA's GPU dependency entirely**.
- **SPRIG** — the empirical guardrails: graph-only retrieval is weak, seeds must come from retrieval, fuse at the seed not the score, the CE buys MRR not recall, hub pruning is free latency, and all of it fits a 4 GB CPU budget.
- **CTRA's own assets** — PrimeKG and the ChEMBL gazetteer, which make the expensive parts of both systems unnecessary rather than merely cheaper.

The combined effect on the measured blockers: corpus indexing goes from **84–320 GPU-days** (GLiNER + MiniLM) to roughly **2–6 CPU-hours** (gazetteer + static embeddings), with no GPU required and peak RAM in the 1–2 GB range. Quality remains unproven for CTRA until step 0 exists.
