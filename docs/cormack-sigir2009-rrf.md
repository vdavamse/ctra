# Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods

## Authors and Affiliations

- **Gordon V. Cormack** — University of Waterloo, Waterloo, Ontario, Canada
- **Charles L. A. Clarke** — University of Waterloo, Waterloo, Ontario, Canada
- **Stefan Büttcher** — Google, Redmond, WA, USA

## Venue / Publication Info

- **Venue**: SIGIR '09 — 32nd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, Boston, Massachusetts, USA, July 19–23, 2009
- **Pages**: 758–759 (two-page short paper)
- **DOI**: 10.1145/1571941.1572114
- **PDF**: https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- **Why it is in this repo**: this is the origin of **RRF**, the fusion step inside SPRIG's `GraphRRF` variant ([2602.23372v1](2602.23372v1.md)) and inside the hybrid retriever proposed in `research/rag-alternatives-hybrid-retrieval.md`. SPRIG uses RRF with *k* = 60 but **does not cite this paper**; this document supplies the missing provenance.

## Abstract

Reciprocal Rank Fusion (RRF), a simple method for combining the document rankings from multiple IR systems, consistently yields better results than any individual system, and better results than the standard method Condorcet Fuse. This result is demonstrated by using RRF to combine the results of several TREC experiments, and to build a meta-learner that ranks the LETOR 3 dataset better than any previously reported method.

## Problem Statement

Supervised learning-to-rank was the dominant research direction at the time, but it requires training examples — labelled relevance judgments that most deployments do not have. The authors set out to build an *unsupervised* fusion baseline against which to measure learning-to-rank methods, and found the baseline beat the methods it was meant to calibrate.

The practical problem RRF solves is combining ranked lists produced by systems whose **scores are not comparable**. A BM25 score and a cosine similarity live on different scales with different distributions; normalizing them requires either calibration data or an arbitrary choice. RRF sidesteps this by discarding scores and using ranks only.

## Key Contributions

1. **The RRF formula** — a one-line, parameter-light, unsupervised rank-fusion rule.
2. **Empirical demonstration** that RRF beats Condorcet Fuse, CombMNZ, and the best individual input system across four TREC collections and the LETOR 3 benchmark.
3. **A meta-learner** built by fusing LETOR 3's supplied learning-to-rank baselines that, at time of publication, exceeded every previously reported result on that dataset.
4. **An argument from mechanism** for *why* rank-only fusion beats vote-based fusion (below).

## Methodology / Architecture

### The formula

Given a set of documents *D* and a set of rankings *R*, each a permutation on 1..|D|:

```
RRFscore(d ∈ D) = Σ_{r ∈ R}  1 / (k + r(d))
```

where `r(d)` is the rank of document *d* in ranking *r*, and **k = 60**.

### Design rationale (stated by the authors)

- The reciprocal shape encodes that "while highly-ranked documents are more important, the importance of lower-ranked documents does not vanish as it would were, say, an exponential function used."
- **The constant *k* "mitigates the impact of high rankings by outlier systems"** — it damps the influence of a single system that ranks something #1 idiosyncratically.

### Provenance of k = 60

Explicitly stated: *"k = 60 was fixed during a pilot investigation and not altered during subsequent validation."* It is a single pilot-tuned constant, carried forward unchanged — **not** a value optimized per dataset. Table 1 sweeps it:

| k | 0 | 10 | 20 | 30 | 40 | 50 | **60** | 70 | 80 | 90 | 100 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAP | .2072 | .2123 | .2134 | .2139 | .2138 | .2144 | **.2145** | .2146 | .2147 | .2145 | .2142 | .2098 |

The peak is at k = 80 (.2147), not 60 — but the plateau from k ≈ 20 to k ≈ 100 spans only 0.0013 MAP. **k is genuinely non-critical in that range**; only k = 0 (.2072) and k = 500 (.2098) are meaningfully worse.

### Two properties that matter for implementation

1. **No score calibration.** RRF "combines ranks without regard to the arbitrary scores returned by particular ranking methods." Adding a new retriever requires no renormalization of the existing ones.
2. **Streaming / low memory.** *"RRF requires no special voting algorithm or global information; ranks may be computed and summed one system at a time, avoiding the necessity of keeping all rankings in memory."* Fusion is an accumulate-into-a-dict pass per retriever.

### Baselines compared

- **Condorcet Fuse** — sorts by the pairwise relation `r(d1) < r(d2)`, decided for each pair by majority vote among input rankings.
- **CombMNZ** — `|{r : r(d) ≤ c}| · Σ_{r: r(d) ≤ c} s_r(d)`; requires per-system scoring functions `s_r` and a cutoff rank *c*, so unlike RRF it *is* exposed to score scale.

## Datasets

- **Pilot**: four TREC collections, each fusing 30 configurations of Wumpus Search. Table 1 reports TREC topics 351–400.
- **TREC submissions**: actual participant runs (not the authors' own configurations) for TREC 3, TREC 5, TREC 9 ad hoc, and the TREC 2004 Robust track — sets chosen because prior metaranking evaluations used them.
- **LETOR 3**: 583,850 document–query pairs across seven sub-collections, combined into one pooled MAP figure.

## Results

### Fusion of submitted TREC runs (Table 2, MAP)

| Collection | **RRF** | Best individual | Condorcet | CombMNZ |
|---|---|---|---|---|
| TREC Robust | **.3686** | .3586 | .3652 | .3575 |
| TREC 3 | .4350 | .4226 | .4256 | **.4381** |
| TREC 5 | **.3394** | .3165 | .3213 | .3237 |
| TREC 9 | .2830 | .3519 (manual) / (.2801) | .2750 | .2671 |

The TREC 9 "best individual" used a **human in the loop**; against the best automated run (.2801) RRF still wins.

### LETOR 3 meta-learner (Table 3, pooled MAP over 583,850 pairs)

| Method | MAP (95% CI) | RRF − method | p |
|---|---|---|---|
| **RRF** | **0.6051** (0.58–0.63) | — | — |
| CombMNZ | 0.6107 (0.58–0.64) | −0.0056 | .2 |
| Condorcet | 0.5917 (0.56–0.62) | 0.0134 | .004 |
| ListNet | 0.5846 (0.56–0.61) | 0.0205 | .001 |
| LGD | 0.5837 (0.56–0.61) | 0.0214 | .003 |
| AdaRank-MAP | 0.5778 (0.55–0.61) | 0.0273 | .000 |
| RankSVM | 0.5737 (0.55–0.60) | 0.0314 | .000 |
| RankBoost | 0.5622 (0.53–0.59) | 0.0429 | .000 |

**Read this honestly: CombMNZ edges RRF here (0.6107 vs 0.6051), and the difference is not significant (p ≈ .2).** RRF's claim is not that it is uniformly best, but that it is uniformly *robust* — CombMNZ's results range "from insubstantially better than RRF to substantially worse than Condorcet," which the authors attribute to its dependence on uncalibrated scores.

### Significance

Aggregated by sign test over the pilot and TREC experiments: RRF beat Condorcet 7/7 (p ≈ .008), CombMNZ 6/7 (p ≈ .04), and the best individual system 6–7 times (.008 ≤ p ≤ .04). Average margin over Condorcet, CombMNZ and the best system: **4–5%**.

MAP is reported for brevity; the authors state P@k, R-precision and NDCG "yield comparable results."

## Limitations

- **Two pages, 2009.** No dense retrieval, no neural rankers — the input systems are lexical IR runs and classical learning-to-rank models. Transfer to BM25+dense fusion is by analogy, not by measurement in this paper.
- **k = 60 is pilot-tuned on TREC ad hoc data**, not on any modern or biomedical collection. Table 1 shows the choice is forgiving, but the plateau was measured on one collection type.
- **RRF discards score magnitude by construction.** When one retriever's scores carry genuine confidence information (a well-calibrated cross-encoder, say), RRF throws it away. CombMNZ's occasional wins are the visible edge of this.
- **No analysis of correlated inputs.** All experiments fuse either 30 configurations of one engine or independent TREC participant runs. How RRF degrades when two retrievers are near-duplicates is not studied.
- **Equal weighting only.** Every input ranking contributes identically; there is no weighted-RRF variant here.

## Relevance to CTRA

1. **It supplies the missing citation for `GraphRRF`.** SPRIG's best Recall@10 configuration on both HotpotQA (0.867) and 2Wiki (0.794) is PPR seeded from an RRF-fused candidate list, and SPRIG's RRF is exactly this formula with this *k*. `research/linearrag-improvements.md` item 6 recommends that mechanism; this is its primary source.

2. **The no-calibration property is why seed-side fusion is cheap to try.** CTRA would fuse BM25 over trial text with dense scores from an embedding model. Those scales are unrelated and CTRA has no labelled data to calibrate them with — RRF requires neither. This is the specific reason RRF, rather than a weighted score sum, is the right first fusion to implement.

3. **The streaming property matters here specifically.** Per the paper, ranks "may be computed and summed one system at a time, avoiding the necessity of keeping all rankings in memory." Given the RAM wall documented in `research/rag-alternatives-hybrid-retrieval.md` § 2 (7 GB total, vs a 13–100 GB in-RAM sentence store), a fusion step that never materializes both full ranking lists is a real constraint satisfied, not a footnote.

4. **Set k = 60 and stop thinking about it.** Table 1 is the evidence that tuning *k* is not worth a sweep — the 20–100 plateau is 0.0013 MAP wide. On a machine where every experiment costs GPU-days, knowing which knobs *not* to turn has direct value.

5. **Caveat to carry forward.** RRF's wins here are over *lexical* input systems on newswire. CTRA's inputs would be BM25 and a dense biomedical encoder over clinical trial text — more correlated than TREC participant runs, and in a domain where the two retrievers fail differently. The TREC-CT harness (still the outstanding prerequisite in `research/rag-alternatives-hybrid-retrieval.md` § 8) is what would confirm the gain locally; this paper establishes the prior, not the result.

## Code / Data Availability

No code release accompanies the paper — the method is four lines to implement. The LETOR 3 dataset referenced in the paper was hosted by Microsoft Research; it is now distributed via the [LETOR project pages](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/). RRF is implemented in most modern search stacks (Elasticsearch, OpenSearch, Vespa, LangChain) with `k = 60` as the default, inherited from this paper.

## Citation

```bibtex
@inproceedings{cormack2009reciprocal,
  title     = {Reciprocal rank fusion outperforms condorcet and individual rank learning methods},
  author    = {Cormack, Gordon V. and Clarke, Charles L. A. and B{\"u}ttcher, Stefan},
  booktitle = {Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '09)},
  pages     = {758--759},
  year      = {2009},
  publisher = {ACM},
  doi       = {10.1145/1571941.1572114}
}
```
