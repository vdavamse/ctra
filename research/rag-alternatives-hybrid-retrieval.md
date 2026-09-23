# RAG Alternatives: Hybrid Retrieval (BM25 + Dense + Reranker) vs LinearRAG

**Date:** 2026-09-22 (hardware measurements added 2026-09-23)
**Question:** Can CTRA replace LinearRAG with BM25 + vector search + cross-encoder reranking, and does that make the pipeline viable without a datacenter GPU?
**Follow-up:** `research/rag-synthesis-sprig-schemagraph.md` — how SPRIG and the `schemagraph` implementation combine into a concrete CTRA architecture, and why PrimeKG + the ChEMBL gazetteer make the expensive parts of both unnecessary.

**Short answer:** Yes — confirmed by measurement on the target machine. The GPU is not even the strongest reason. LinearRAG as currently wired cannot run on this machine at CTRA's corpus scale *regardless* of GPU, because of an in-RAM embedding-store design. Hybrid retrieval removes both the GPU wall and the RAM wall, and moves CTRA *closer* to AutoCT's published retrieval design rather than further from it.

**Measured bottom line (2026-09-23, real CTG data, this hardware):** indexing ClinicalTrials.gov alone with GLiNER-BioMed-large takes **22 days** of continuous GPU; embedding the same chunks takes **8.7 hours**. The NER pass costs **61x** the embedding pass and is the whole GPU dependency. Two secondary findings change the component choices: **ONNX int8 is unusable on this CPU** (no AVX512-VNNI — it runs slower than fp32), and the **MiniLM-L6 cross-encoder is too slow** at 2.1 s/query, so the reranker must be TinyBERT-L2 (222 ms).

---

## 1. Where the GPU cost actually sits

Measured numbers already in this repo (`docs/ner-evaluation-guide.md` § "NER Throughput Optimization", 2026-04-10):

| Stage | Model | Measured throughput | Cost over 1.7M–6.5M passages |
|---|---|---|---|
| **NER (index)** | GLiNER-BioMed **large** (~460M params) | 35 docs/s on T4 + fp16 + batching | **14–57 h** |
| Embeddings (index) | all-MiniLM-L6-v2 (22M params) | not separately measured | small by comparison |
| Graph build | igraph, CPU | — | minutes–hours |
| Retrieval (query) | dense matmul + PPR | 0.093 s (paper, 2Wiki scale) | scales linearly with corpus |

The GPU requirement is **~95% one component**: a 460M-parameter span-enumeration NER model run over every passage in the corpus. That pass exists *only* to build the entity co-occurrence graph. Everything else in LinearRAG is either a 22M-param embedder or CPU graph math.

Both entry points hard-fail without CUDA — `linearrag_wrapper.py:139` (`_ensure_loaded`) and `:553` (`build_index`) raise `RuntimeError` if `torch.cuda.is_available()` is false. There is no CPU path at all today.

### This machine vs. the requirement — **measured 2026-09-23**

Hardware: **Intel i7-10750H** (6 cores / 12 threads, AVX2, **no AVX512, no VNNI**), **GTX 1650 Ti 4 GB** (Turing TU117, compute 7.5, **no Tensor Cores**), **7 GB RAM** (~4.9 GB available), torch 2.14.0+cu130.

Benchmarked on **3,000 real Phase 2/3 trials** pulled from the ClinicalTrials.gov v2 API and converted with the repo's own `_ctg_to_passages`, using the same `model.inference(..., flat_ner=True, batch_size=...)` call path as `gliner_batched.py:249`. Scripts and raw JSON: scratchpad `bench/`.

**GLiNER-BioMed-large (`Ihor/gliner-biomed-large-v1.0`), 16 CTRA labels, threshold 0.4:**

| Device | Doc length | batch | docs/s | VRAM |
|---|---|---|---|---|
| GPU fp16 | 535 ch (CHIA-comparable) | 8 | **2.30** | 1,167 MiB |
| GPU fp16 | 1,136 ch (real CTG chunk) | 8 | **1.16** | 1,616 MiB |
| CPU fp32 | 535 ch | 4 | 1.17 | — |
| CPU fp32 | 1,148 ch | 4 | 0.64 | — |

**The headline: 2.30 docs/s here versus the repo's recorded 35 docs/s on a T4 at the same document length — 15.2x slower.** The GPU buys only ~1.8x over the CPU (2.30 vs 1.17), because without Tensor Cores the fp16 path has little left to exploit. VRAM was never the binding constraint — 1.6 GB peak of 4 GB.

**Consequence.** At the measured 1.16 chunks/s on real CTG text, and the measured 4.94 chunks per trial (below):

| Scope | GLiNER-large, GPU fp16 |
|---|---|
| ClinicalTrials.gov alone (450K trials, 2.22M chunks) | **532 h — 22 days** |
| Full 7-source corpus (8.4–32.1M chunks) | **2,011–7,689 h — 84–320 days** |

Twenty-two days of continuous GPU for *one* of seven sources. The earlier "~1–3 weeks" extrapolation was optimistic by roughly 4x. This is not a slow plan; it is not a plan.

## 2. The RAM wall — the argument that does not depend on the GPU

This matters more than the GPU, and it is easy to miss.

`src/ctra/rag/linearrag/embedding_store.py` keeps **all** embeddings as a Python list in memory (`self.embeddings: list[Any]`), persisted as a single parquet, loaded whole on init. `get_embeddings()` (line 206) does:

```python
embeddings = np.array(self.embeddings)[indices]
```

— materializing the entire store into a dense array on every call. Retrieval then runs a brute-force dense matmul over *every* sentence (`core.py:557`):

```python
sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()
```

There is no ANN index and no memory mapping anywhere in the design.

LinearRAG builds **three** such stores — passages, sentences, entities. At CTRA's own estimated scale, with the paper's stated density (~4 entities/sentence, ~10/passage):

| Store | Count at 1.7M passages | Count at 6.5M passages | fp32 @384-d |
|---|---|---|---|
| Passages | 1.7M | 6.5M | 2.6–10 GB |
| Sentences (~5–10 per passage) | 8.5–17M | 32–65M | **13–100 GB** |
| Entities (deduplicated) | ~2–5M | ~5–15M | 3–23 GB |

The sentence store alone exceeds this machine's total RAM by 2–15x, and that is the *optimistic* fp32-contiguous accounting — as a Python list of per-row arrays it is substantially worse. An A100 would not change this: it is a host-memory and algorithmic-design problem.

**Conclusion:** LinearRAG at CTRA scale needs a machine with ~128 GB+ RAM and a datacenter GPU, or it needs the embedding stores rewritten to be memory-mapped and ANN-backed. If you are rewriting that layer anyway, you have already done most of the work of a hybrid retriever.

## 3. Proposed architecture

```
query
  ├─ BM25 (bm25s, memory-mapped sparse index)      → top 100
  ├─ Dense  (MiniLM/MedCPT via ONNX int8 + FAISS)  → top 100
  ├─ fuse: Reciprocal Rank Fusion                  → top ~100
  ├─ metadata PRE-filter (source, before_date)     ← cheap, exact
  └─ cross-encoder rerank (ONNX int8, CPU)         → top_k
```

### Per-component sizing — **measured 2026-09-23**

First, a measurement that reshapes the arithmetic. `_ctg_to_passages` emits **one passage per trial**, and real trials are long: mean **5,544 chars (~1,386 tokens)**, p50 4,751, p90 9,941, max 36,100. all-MiniLM-L6-v2 truncates at **256 tokens**, so each trial must be split into **4.94 chunks** (measured over 608 trials) or ~82% of every trial is silently discarded. The corpus to embed is therefore **8.4–32.1M chunks**, not 1.7–6.5M passages.

**Embedding throughput** (256-token chunks of real CTG text, 12 threads):

| Config | chunk/s | trials/s | CTG only (2.22M chunks) | Full corpus |
|---|---|---|---|---|
| GPU fp16 1650 Ti | **71.2** | 14.4 | **8.7 h** | 33–125 h |
| CPU fp32 torch | 42.7 | 8.6 | 14.5 h | 55–209 h |
| CPU ONNX int8 (VNNI) | 37.2 | 7.5 | — | — |

**ONNX int8 does not work on this CPU.** The widely-cited ~3.2x speedup needs AVX512-VNNI; the i7-10750H has AVX2 only, so the quantized kernels fall back and run *slower* than fp32 (37.2 vs 42.7). Drop int8 from the plan for this machine — it is a datacenter-CPU optimization.

**The GPU figure is a genuine ceiling, not framework overhead.** Re-running the loop in raw `transformers`/torch gives 69.2 chunk/s against sentence-transformers' 71.2, at a sustained 804 GFLOP/s (~14% of the card's theoretical fp16 peak — memory-bound on a 22M-param model). Tokenization is 2.3% of runtime. There is nothing left to tune.

**Cross-encoder rerank latency** (per query, real 180-word candidates):

| Model | depth 30 | depth 50 | depth 100 |
|---|---|---|---|
| ms-marco-MiniLM-L6 CPU | 1,347 ms | 2,139 ms | 4,249 ms |
| ms-marco-MiniLM-L6 GPU | — | 1,298 ms | 2,467 ms |
| **ms-marco-TinyBERT-L2 CPU** | **131 ms** | **222 ms** | 390 ms |
| **ms-marco-TinyBERT-L2 GPU** | — | **108 ms** | 151 ms |

**The reranker choice flips.** MiniLM-L6 at 2.1 s/query is unusable when agents issue many queries per trial. **TinyBERT-L2 (4M params) is the only viable reranker here** — and notably it is exactly what SPRIG used for its CE baseline, where it still produced that study's best MRR (0.881 on HotpotQA). Rerank depth 30–50 on CPU, or 50–100 on GPU, fits a sane budget.

| Component | Choice | License | Index cost | Query cost | RAM |
|---|---|---|---|---|---|
| Lexical | `bm25s` (scipy sparse, mmap) | MIT | minutes | <10 ms | low (mmap) |
| Embedder | all-MiniLM-L6-v2 fp16 **on GPU** | Apache-2.0 | **8.7 h CTG / 33–125 h full** | ~14 ms | 206 MiB VRAM |
| Embedder (upgrade) | MedCPT / BMRetriever-410M | varies | 5–20x MiniLM — **likely unaffordable here** | 25–100 ms | 0.4–1.6 GB |
| ANN | FAISS IVF + SQ8, or flat mmap | MIT | minutes | ~10 ms | ~3 GB fp32 → 0.75 GB int8 @ 8.4M chunks |
| Reranker | **ms-marco-TinyBERT-L2** | Apache-2.0 | none | **222 ms @ depth 50 CPU** | 17 MB |

**Net, measured:** the GLiNER pass costs **61x** what embedding the same chunks costs on GPU (71.2 vs 1.16 chunk/s), 67x on CPU. Dropping the graph turns **22 days into 8.7 hours** for ClinicalTrials.gov, and **84–320 days into 1.4–5.2 days** for the full corpus. Peak RAM goes from >13 GB (§2) to ~1–2 GB.

Two honest caveats on that headline. First, **1.4–5.2 days of GPU is still a real one-time cost** — resumable and overnight-able, but not the "few hours" I first estimated; that claim was wrong by ~10x, from compounding the 4.94x chunking factor with an over-optimistic throughput figure. Second, **MedCPT is probably out of reach**: at 5–20x MiniLM's cost it lands at 1–3.5 weeks for the full corpus, so the biomedical-embedder upgrade in § 9 step 4 is likely a CTG-only option, not a corpus-wide one.

### Why the reranker matters more here than usual

Because MiniLM-L6 is a weak 256-token embedder being fed 1,386-token trials in 5 pieces, first-stage dense recall will be mediocre. BM25 compensates on exact identifiers (NCT IDs, gene symbols, drug codes), and the cross-encoder — which sees query and passage jointly at 512 tokens — compensates on ordering. `src/ctra/rag/tools.py` already memoizes every tool call through diskcache (`_cache_memoize`), so the 222 ms amortizes across MCTS rollouts rather than recurring per rollout.

## 4. What you give up, and whether it matters

**Lost:** multi-hop retrieval via Personalized PageRank over the entity graph — LinearRAG's actual contribution.

Three reasons this is a smaller loss than it looks for CTRA specifically:

1. **AutoCT's published implementation did not use graph RAG.** Per README § "RAG backend", the reference implementation is pgvector + txtai with *single-hop* semantic search. LinearRAG is a CTRA enhancement, not a reproduction requirement. Hybrid + reranker is strictly stronger than the published baseline; the graph was the speculative part.
2. **The agent loop is the multi-hop mechanism.** ReAct feature-builder agents already issue successive, refined queries. Graph multi-hop and agentic multi-hop are substitutes, and you are paying for both.
3. **Query-side entity expansion survives the migration.** The ChEMBL/PubChem synonym resolver currently patched into query NER (`_patch_query_ner_with_synonyms`) retargets cleanly onto BM25 query expansion — and arguably works *better* there, since "Keytruda OR pembrolizumab" is a native lexical operation rather than a graph-seeding trick.

## 5. Evidence on retrieval quality

From this repo's own TREC-CT 2021 evaluation (`docs/ner-evaluation-guide.md` § 629):

| System | NDCG@10 | MRR | Corpus |
|---|---|---|---|
| GLiNER + LinearRAG (ours) | 0.328 | **0.689** | 5,000 (1.3% subset) |
| BM25 baseline | ~0.30 | ~0.29 | full ~375K |
| BM25 + BERT rerank | 0.36 | — | full ~375K |
| **BM25 + BioBERT rerank** | **0.46+** | — | full ~375K |
| CSIROmed (best automatic) | ~0.53 | — | full ~375K |

Read this carefully — it is suggestive, not conclusive:

- The comparison is **not apples-to-apples**. A 5K subset containing all judged documents inflates MRR (fewer distractors) and deflates NDCG@10 (unretrievable judged docs are missing). The repo's own analysis says exactly this.
- Still: neural reranking on the *full* corpus reaches 0.46+ NDCG@10 against our 0.328 on an easier subset. That is the single most relevant data point available, and it favours the reranker.

External evidence is more nuanced than "hybrid + reranker wins." The most relevant study is **SPRIG** ([arXiv:2602.23372](https://arxiv.org/abs/2602.23372), Qizhi Wang, PingCAP) — read in full 2026-09-23. It is the closest prior work to this decision: a CPU-only, linear-time, token-free GraphRAG pipeline (lightweight NER → entity–document co-occurrence graph → Personalized PageRank), benchmarked against exactly the alternatives CTRA is choosing between, **under a strict 4 GB RAM budget on CPU-only hardware**.

Main results, R@10 / MRR (Tables 1, 2, 5):

| Method | HotpotQA R@10 | MRR | 2Wiki R@10 | MRR |
|---|---|---|---|---|
| BM25 | 0.742 | 0.784 | 0.643 | 0.819 |
| Dense (bge-small-en-v1.5 + HNSW) | 0.811 | 0.878 | 0.609 | 0.885 |
| **RRF (BM25+dense)** | 0.851 | 0.865 | 0.697 | 0.914 |
| BM25 + cross-encoder (TinyBERT, top-100) | 0.810 | **0.881** | 0.676 | 0.891 |
| **RRF + cross-encoder** | 0.846 | **0.887** | 0.701 | 0.901 |
| Graph only (query-entity PPR) | **0.464** | 0.448 | **0.357** | 0.430 |
| GraphHybrid (BM25-seeded PPR) | 0.775 | 0.778 | 0.743 | 0.839 |
| GraphDense (dense-seeded PPR) | 0.844 | 0.875 | 0.747 | 0.901 |
| **GraphRRF (RRF-seeded PPR)** | **0.867** | 0.852 | **0.794** | 0.912 |

Four findings that bear directly on CTRA:

1. **Pure entity-graph PPR is weak on its own** — R@10 of 0.464 / 0.357, far *below* plain BM25. This is the closest published analogue to LinearRAG's retrieval mechanism, and it says the graph is not a standalone retriever. It only becomes competitive when seeded from dense or lexical retrieval. That is a caution about the mechanism CTRA currently depends on, not just about its cost.
2. **The cross-encoder buys precision, not recall.** RRF+CE *slightly lowers* R@10 versus RRF alone on HotpotQA (0.846 vs 0.851) while giving the best MRR in the study (0.887). So the reranker's value is ordering the top of the list, not finding more evidence. For CTRA this is probably the right trade — the feature-builder agent reads the top chunks and extracts a value — but it should be an explicit choice, and CE should sit on top of **RRF**, not on top of BM25 alone.
3. **Well-seeded graphs do add recall.** GraphRRF is the best R@10 on both datasets. If CTRA turns out to have a genuine multi-hop deficit, the fix is PPR seeded by hybrid retrieval — which requires the hybrid retriever to exist first. The migration in § 7 is therefore a prerequisite for the graph option, not an alternative to it.
4. **The whole design fits a 4 GB budget on CPU.** SPRIG's graph construction with `en_core_web_sm` costs **9–10 ms/doc** (0.02 ms/doc with a regex extractor) — so even the graph variant would index 2M CTRA passages in ~5.5 CPU-hours, versus ~1 s/doc for GLiNER-BioMed-large. The gap is the NER model, exactly as § 1 argues.

**Caveat on transferring these numbers:** HotpotQA and 2WikiMultiHopQA are Wikipedia multi-hop QA with gold supporting-passage labels. They are not biomedical, not clinical-trial retrieval, and their entity distribution is nothing like drug/gene/condition vocabulary. Note also that SPRIG's graph uses generic `en_core_web_sm` NER — which this repo's own evaluation (`docs/ner-evaluation-guide.md`) found markedly inferior to GLiNER for biomedical retrieval. A biomedical graph with good NER might well fare better than SPRIG's graph does here. These numbers justify *ordering the work* hybrid-first; they do not settle the question for CTRA. Only the TREC-CT harness in § 8 can.

## 6. Side benefits of the migration

1. **Date filtering becomes a pre-filter instead of a post-filter.** Today `search()` retrieves `top_k * 3` (`_OVERSAMPLE_FACTOR = 3`) and then discards by source/date, conservatively dropping every undated document. With aggressive source filters this silently under-returns — the logged "No passages match filters" warning is the symptom. BM25 and FAISS both support exact metadata pre-filtering, so `before_date` (the primary label-leakage defence, per `config/settings.py` § 2a) becomes a hard guarantee rather than a best-effort post-hoc trim.
2. **Removes GPL-3 code from a proprietary-licensed repo.** `src/ctra/rag/linearrag/` is ~1,500 lines vendored from LinearRAG under GPL-3, inside a project declaring `license = {text = "Proprietary"}`. bm25s (MIT), FAISS (MIT) and sentence-transformers (Apache-2.0) carry no such tension.
3. **Drops the GLiNER-BioMed indexing dependency entirely** — along with the `gliner-spacy==0.0.10` pin, the custom `gliner_batched.py` component, and the fp16/batching workarounds. GLiNER stays useful for query-side entity extraction (small volume, CPU-fine) if wanted.
4. **Kills the 12–57 h index rebuild cycle**, which is what currently makes retrieval iteration impractical.

## 7. Migration cost — the seam is already clean

The codebase is better positioned for this than expected. The entire coupling is two methods:

- `LinearRAGWrapper.search(query, before_date, sources, top_k) -> list[RetrievedChunk]`
- `LinearRAGWrapper.build_index(passages, output_dir, rag_config) -> Path`

Everything downstream — all seven `make_*_search` tool factories in `tools.py`, the agents, the API — consumes only `RetrievedChunk`. `IndexBuilder` in `indexer.py` (906 lines of source→passage conversion, the genuinely valuable part) calls `build_index` three times and is otherwise backend-agnostic.

So: introduce a `HybridRAGWrapper` with the same two-method surface, add a `RAGConfig.backend` switch, and nothing above the RAG layer changes. `indexer.py` is reused verbatim.

**Also relevant:** the RAG stack has never actually been run. `torch`, `sentence-transformers`, `gliner`, `spacy` and `igraph` are absent from the venv (the `rag` extra was never installed), `datasets/` does not exist, and no index artifacts are present. `tests/test_rag/` mocks the backend throughout. There is **no working LinearRAG deployment to regress** — the switching cost is close to its theoretical minimum, and it will only grow.

Estimated effort: ~1.5–2 weeks for indexer + retriever + reranker + tests, on top of the eval harness below.

## 8. The blocking gap: there is no retrieval benchmark

`src/ctra/rag/eval/` is **NER-only** (GLiNER threshold sweeps, CHIA/BioNLP label ablations). The TREC-CT 2021 retrieval run quoted in § 5 exists only as prose in `docs/ner-evaluation-guide.md` — `grep -ri trec` finds no code anywhere in `src/`, `scripts/` or `tests/`. That benchmark is not reproducible today.

This was already flagged as recommendation #3 in that same guide ("Add TREC-CT retrieval benchmark to regression test suite", 1 day) and never done.

**Do this first.** Without it, "hybrid vs LinearRAG" is an argument about priors rather than a measurement, and § 5's numbers stay un-comparable.

## 9. Recommended sequence

1. **Rebuild the TREC-CT 2021 retrieval harness** as reusable code under `src/ctra/rag/eval/retrieval.py` (NDCG@10, MRR, Recall@k) against a pluggable retriever interface. ~1 day. This is the prerequisite for every decision below.
2. ~~**Measurement spike**~~ — **DONE 2026-09-23.** Results in §§ 1 and 3; scripts and raw JSON in `notes/bench_rag_hardware/`. Headline: GLiNER-large runs at 2.30 docs/s (15.2x slower than the T4 figure the plan was built on), embedding at 71.2 chunk/s on GPU, TinyBERT-L2 rerank at 222 ms @ depth 50. ONNX int8 is unavailable on this CPU.
3. **Build the hybrid retriever** behind `RAGConfig.backend = "hybrid"`, reusing `indexer.py` unchanged. Score it on the harness from step 1 against the recorded LinearRAG baseline.
4. **Decide on the embedder** empirically: MiniLM-L6 (fast, generic) vs MedCPT (biomedical, 5–20x cost — measured sizing puts a corpus-wide MedCPT index at 1–3.5 weeks on this GPU, so treat it as CTG-only at best). The reranker may recover most of the gap; measure on the step-1 harness rather than assume. Also worth testing: a **512-token embedder** (e5-small, bge-small) to halve the 4.94x chunking multiplier — that trades per-chunk cost against chunk count and may be net-cheaper as well as better.
5. **Defer the graph — but keep the door open.** If step 3 shows a real multi-hop deficit on TREC-CT, add PPR seeded from the hybrid candidate list (SPRIG's GraphRRF), not a restoration of GPU LinearRAG. SPRIG's evidence is that RRF-seeded PPR is the strongest recall configuration tested, and it costs one CPU NER pass with a small model — but it is only reachable *after* the hybrid retriever exists.

## 10. Caveats

- §§ 1 and 3 throughput figures are now **measured on the target hardware** (2026-09-23, 3,000 real CTG trials). They are single-run timings with warm-up, not averaged across repeated runs, so treat them as ±10% rather than precise. They also cover CTG-shaped text only; PubMed abstracts are shorter and will embed faster per document.
- The corpus-hour extrapolations assume every source has CTG's 4.94-chunks-per-document density. PubMed abstracts (~250 words) are closer to 1.4 chunks, so the full-corpus figures are conservative — likely overstated by roughly 2x at the high end.
- The § 5 TREC comparison mixes corpus sizes and is directional only.
- Passage-count estimates (1.7M–6.5M) are the repo's own, carried forward unverified; PubMed subset size is the dominant uncertainty and is a policy choice, not a fact.
- SPRIG's numbers in § 5 are read from the paper's Tables 1, 2 and 5 (full text, 2026-09-23). Its appendix tables (A.x), including the dense-model sensitivity sweep that covers all-MiniLM-L6-v2, were not examined and may affect embedder choice in step 4.

---

## Sources

- [SPRIG: Democratizing GraphRAG — Linear, CPU-Only Graph Retrieval for Multi-Hop QA (arXiv:2602.23372)](https://arxiv.org/abs/2602.23372)
- [BM25S: fast BM25 in Python via scipy sparse](https://bm25s.github.io/)
- [Sentence Transformers — Speeding up Inference (ONNX / int8 benchmarks)](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
- [Hybrid Search: BM25, Vector & Reranking Reference 2026](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)
- [BMRetriever: Tuning LLMs as Better Biomedical Text Retrievers](https://pmc.ncbi.nlm.nih.gov/articles/PMC12949633/)
- [Best Rerankers for RAG in 2026 — model/latency comparison](https://mixpeek.com/curated-lists/best-rerankers)
- [GLiNER-BioMed (arXiv:2504.00676)](https://arxiv.org/pdf/2504.00676) — also `docs/2504.00676v2.md`
- In-repo: `docs/2510.10114v4.md` (LinearRAG paper), `docs/ner-evaluation-guide.md` (TREC-CT results, throughput measurements), `research/linearrag-actual-vs-wrapper-audit.md`
