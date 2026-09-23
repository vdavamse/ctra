# Okapi at TREC-3 (the paper that introduced BM25)

## Authors and Affiliations

- **S. E. Robertson**, **S. Walker**, **S. Jones**, **M. M. Hancock-Beaulieu**, **M. Gatford**
- Centre for Interactive Systems Research, Department of Information Science, City University, Northampton Square, London EC1V 0HB, UK
- Advisers: E. Michael Keen (University of Wales, Aberystwyth), Karen Spärck Jones (Cambridge University), Peter Willett (University of Sheffield)

## Venue / Publication Info

- **Venue**: TREC-3 — Third Text REtrieval Conference, Gaithersburg, Maryland, USA, 2–4 November 1994
- **Proceedings**: NIST Special Publication 500-225, pages 109–126
- **PDF as vendored**: Microsoft Research copy (18 pages)
- **Why it is in this repo**: this is the first appearance of **BM25** by name. It is the seeding retriever in SPRIG's `GraphHybrid` variant ([2602.23372v1](2602.23372v1.md)), one of the two inputs to `GraphRRF`, and the lexical half of the hybrid retriever proposed in `research/rag-alternatives-hybrid-retrieval.md`.

> **Note on the PDF**: this is a 1994 document typeset with a character encoding that modern text extractors mangle (digits and punctuation appear as `/1`, `/#28`, and so on). All formulas and numbers below were decoded and verified against the rendered pages.

## Abstract

*(The paper carries no formal abstract; the following summarizes §1 and §9.)* The paper reports City University's TREC-3 participation with the Okapi system. The emphasis is on four threads: further refinement of term-weighting functions, run-time passage determination and searching, expansion of ad hoc queries by terms extracted from the top documents of a trial search, and new methods for choosing query-expansion terms after relevance feedback. The term-weighting work introduces BM25, which unifies the previously separate BM11 and BM15 functions under a single tunable document-length parameter.

## Problem Statement

Before TREC-1, Okapi ranked documents with the "classical" Robertson–Spärck Jones probabilistic model, **taking no account of document length or of within-document term frequency**. The results were, in the authors' words, "undistinguished."

The paper contains a striking piece of self-correction on this point. The authors note that their TREC-1 and probably TREC-2 results "would have been considerably worse had it not been that the system at that time could not handle documents longer than 64K, and so the longest few hundred documents in the database were truncated." Re-running TREC-1 ad hoc on the untruncated database drops 11-point average precision from 0.12 to 0.10 and P@5 from 0.50 to 0.37 — **a bug had been masking the length bias**. The diagnosis: "the simple weighting scheme tends to favour long documents, particularly FR, few of which are relevant."

BM25 is the response to that diagnosis: a principled, tunable document-length normalization.

## Key Contributions

1. **BM25** — a term-weighting function that subsumes BM11 and BM15 as the endpoints of a single parameter *b*, making the degree of length normalization a continuous choice rather than a binary one.
2. **Run-time passage determination** — scoring best-matching sub-documents rather than whole documents, with a threshold trick that makes it tractable.
3. **Blind (pseudo-relevance) query expansion** — expanding ad hoc queries from the top-ranked documents of a trial search *without* relevance information, which the authors report "somewhat to our surprise" proved beneficial.
4. **Term ordering and stepwise selection** procedures for choosing expansion terms after relevance feedback.

## Methodology / Architecture

### Notation

| Symbol | Meaning |
|---|---|
| *N* | number of documents in the collection |
| *n* | collection frequency — number of documents containing a specific term |
| *R* | number of documents known to be relevant to a topic |
| *r* | number of those containing the term |
| *tf* | frequency of the term within a specific document |
| *qtf* | frequency of the term within the query |
| *dl* | document length (arbitrary units) |
| *avdl* | average document length |
| *k_i*, *b* | constants used in the BM functions |

### The base relevance weight

All BM functions build on the Robertson–Spärck Jones weight:

```
w⁽¹⁾ = log [ (r + 0.5) / (R − r + 0.5) ] / [ (n − r + 0.5) / (N − n − R + r + 0.5) ]     (1)
```

With no relevance information (*R* = *r* = 0) this reduces to an inverse collection frequency weight — **this is BM1, the function used in TREC-1**.

### BM15 and BM11 (TREC-2)

Both add within-document frequency, within-query frequency, and a length component, but normalize length differently:

```
BM15:  w = s₁ s₃ · tf/(k₁ + tf) · w⁽¹⁾ · qtf/(k₃ + qtf)      and  k₂ · nq · (avdl − dl)/(avdl + dl)

BM11:  w = s₁ s₃ · tf/(k₁·dl/avdl + tf) · w⁽¹⁾ · qtf/(k₃ + qtf)  and  k₂ · nq · (avdl − dl)/(avdl + dl)
```

*nq* is the number of query terms. The "and" marks a **global** correction: it is added once at the end, after the per-term weights are summed, and is independent of which terms matched.

### BM25 — the unification

> "In the course of investigating variant functions for TREC-3, we in effect combined BM11 and BM15 into a single function BM25."

The term-frequency component becomes:

```
tf^c / (K^c + tf)        with    K = k₁ · ((1 − b) + b · dl/avdl)        (2)
```

- With *c* = 1: **b = 1 gives BM11**, **b = 0 gives BM15**, and intermediate *b* gives a mix.
- The rationale for mixing: BM11 rests on the "verbosity hypothesis," one of two possible models of document length, "which might be expected to **exaggerate** the document length effect."
- *c* > 1 gives the function an s-shape, motivated by conditions under which the 2-Poisson model generates such a curve. The authors parameterize *c* = 1 + *mK*, *m* ≥ 0 — and then report plainly: **"Non-zero m was not helpful."** In the event *m* was largely ignored.
- Scaling factors: *s₁* = *k₁* + 1 and *s₃* = *k₃* + 1. Where *k₃* is given as ∞, the factor *s₃·qtf/(k₃ + qtf)* is implemented as *qtf* on its own.

### Parameter naming convention

BM25 is written **BM25(k₁, k₂, k₃, b)**, with *m* = 0 assumed unless stated. The operational settings in this paper are **BM25(2.0, 0, 8.0, 0.75)** and **BM25(2.0, 0, 1, 0.75)** — note *k₂* = 0 throughout, i.e. **the global length correction is switched off**, and *b* = 0.75, the value that has been the field's default ever since.

On *b*: "b < 1 can give some improvement. Values around 0.75 were usually used, sometimes with a higher k₁ than for BM11."

### Passage retrieval

Passages are determined at run time, with a minimum of four paragraphs. Two efficiency devices:

- The first "passage" considered is the whole document; if it fails to reach a threshold weight, no further processing is done. Setting that threshold at the weight of the 10,000th whole document "reduced the number of documents considered by a factor of ten or more."
- *avdl* is **reduced for sub-document weighting only** — the true *avdl* (about 2600) "is far too high for true weighting of short passages."

Final document weight = max(best proper sub-document weight, whole-document weight).

## Datasets

TREC disks 1 & 2, using TREC ad hoc topics 101–150 (training/development) and 151–200 (test), plus the TREC-3 routing task. The Federal Register (FR) sub-collection is called out repeatedly as the source of the long-document bias.

## Results

### Passage retrieval (Table 1; topics 151–200, TND fields, disks 1&2, all BM25(2.0, 0.0, 1, 0.75))

| min | max | step | avdl | AveP | P@5 | P@30 | P@100 | R-Prec | Rcl |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2 | 12 | 1800 | **0.345** | 0.720 | 0.585 | 0.440 | 0.392 | 0.692 |
| 4 | 2 | 20 | 1800 | 0.345 | 0.716 | 0.585 | 0.440 | 0.392 | 0.692 |
| 4 | 2 | 24 | 1800 | 0.344 | 0.716 | 0.584 | 0.440 | 0.392 | 0.692 |
| 8 | 4 | 24 | 1800 | 0.342 | 0.728 | 0.589 | 0.434 | 0.387 | 0.687 |
| — | — | — | — | 0.337 | 0.732 | 0.590 | 0.431 | 0.382 | 0.681 |

The last row is whole-document retrieval. **Passage retrieval alone buys +0.008 AveP and actually loses P@5 and P@30** — the authors describe it as "difficult to obtain more than a small improvement over whole-document searching."

### Ad hoc query expansion (Table 3; topics 101–150)

| FB docs | terms | Conditions | AveP | P@5 | P@30 | P@100 | R-Prec | Rcl |
|---|---|---|---|---|---|---|---|---|
| 20 | 20 | | 0.328 | 0.580 | 0.553 | 0.473 | 0.371 | 0.700 |
| 30 | 40 | | 0.333 | 0.600 | 0.569 | 0.478 | 0.370 | 0.704 |
| 100 | 50 | | 0.321 | 0.556 | 0.546 | 0.466 | 0.365 | 0.702 |
| 30 | 40 | **passages** | **0.345** | 0.604 | 0.577 | 0.483 | 0.375 | **0.719** |
| — | — | unexpanded BM25(2.0,0,1,0.75) | 0.302 | 0.592 | 0.532 | 0.451 | 0.361 | 0.674 |

Blind expansion from 30 documents / 40 terms lifts AveP from 0.302 to 0.333 (**+10%**), and combining it with passage retrieval reaches 0.345 (**+14%**). The authors flag the interaction as unexplained: passages help far more *in conjunction with* expansion than alone, and "it is not at all obvious why this should be so."

### Stated conclusions (§9.2)

- **Term weighting**: the "rough model" methods developed for TREC-2 "have now been shown to be effective under the full rigour of the official TREC procedures." But — "attempts at somewhat less rough models have shown only small benefit."
- **Passages**: "feasible, if computationally expensive… some benefits for document retrieval, though not very large ones."
- **Blind expansion**: beneficial with short queries, and "combined effectively with passage retrieval."
- **Term ordering**: "We have not managed to improve on the term-ordering measures used in previous experiments."

## Limitations

- **BM25's parameters are empirical, not derived.** *b* = 0.75 and *k₁* = 2.0 are reported as values that "were usually used." The paper explicitly declines to show the sweep: "Evaluation results for BM25 with various parameter values are not explicitly given in this paper."
- **The *c* / *m* s-shape extension failed.** It is in the formula but was abandoned in practice — an honest negative result, and a reminder that the surviving form of BM25 is simpler than the one proposed here.
- **Newswire and government documents only.** No biomedical or scientific text; no short structured records of the kind clinical trial registry entries are.
- **Passage retrieval is "computationally expensive"** and its standalone benefit is within noise on most measures.
- **The 64K truncation bug** means the TREC-1/TREC-2 comparisons in the literature that cite those Okapi results are comparing against an artificially favourable baseline — documented here by the authors themselves.

## Relevance to CTRA

1. **It is the definition of record for CTRA's lexical retriever.** The migration in `research/rag-alternatives-hybrid-retrieval.md` § 7 puts BM25 (or BM25F over trial fields) at the base of the stack. `BM25(k₁=2.0, k₂=0, k₃=8.0, b=0.75)` is the configuration this paper actually ran, and is the right starting point — not a value to sweep before there is a benchmark to sweep against.

2. **The document-length story transfers directly to clinical trial records.** Okapi's failure mode was that unnormalized weighting "tends to favour long documents, particularly FR, few of which are relevant." CTRA's corpus has exactly this shape: `protocolSection` eligibility criteria blocks run to thousands of words while intervention and condition fields are a line or two. *b* is the knob that governs whether long criteria blocks swamp short, precise matches — and `_primekg_to_passages` (see `research/rag-synthesis-sprig-schemagraph.md`) produces passages of wildly varying length from the same source.

3. **Blind query expansion is an unclaimed, CPU-only +10%.** Table 3's best unexpanded-to-expanded jump is 0.302 → 0.333 AveP with no relevance judgments and no GPU. SPRIG benchmarks this as `BM25+RM3`. Given that CTRA's GLiNER indexing pass is measured at **1.13 doc/s (22.8 days for ClinicalTrials.gov alone)**, a pure-CPU query-side technique with a published double-digit gain deserves evaluation before any further graph work.

4. **Passage retrieval is a caution, not a recommendation.** The measured standalone gain is +0.008 AveP at substantial compute cost, and P@5 *dropped*. CTRA already chunks passages for GLiNER (a 4.94× chunking factor, measured); this paper is evidence not to expect retrieval gains from finer chunking on its own.

5. **Provenance for `GraphHybrid`.** SPRIG seeds PPR from top-*k* BM25 passages (k=10 HotpotQA, k=5 2Wiki) to reach R@10 0.775 / 0.743. That seeding retriever is this function.

## Code / Data Availability

No code release (1994). The original Okapi BSS is not publicly distributed. Modern, faithful implementations: [Lucene](https://lucene.apache.org/) (`BM25Similarity`, defaults k₁=1.2, b=0.75), [rank_bm25](https://github.com/dorianbrown/rank_bm25), [Pyserini/Anserini](https://github.com/castorini/pyserini), and [bm25s](https://github.com/xhluca/bm25s). TREC disks 1 & 2 are available from NIST under a data agreement; proceedings are free at https://trec.nist.gov/pubs/trec3/.

> **Implementation note**: most modern libraries default to *k₁* ≈ 1.2, not the *k₁* = 2.0 used here, and implement the simplified `(k₁+1)·tf / (K + tf)` form with *c* = 1 — i.e. the version *without* the s-shape extension the authors found unhelpful. The `b` = 0.75 default is inherited directly from this paper.

## Citation

```bibtex
@inproceedings{robertson1994okapi,
  title     = {Okapi at {TREC}-3},
  author    = {Robertson, Stephen E. and Walker, Steve and Jones, Susan and
               Hancock-Beaulieu, Micheline M. and Gatford, Mike},
  booktitle = {Proceedings of the Third Text REtrieval Conference (TREC-3)},
  series    = {NIST Special Publication 500-225},
  pages     = {109--126},
  year      = {1994},
  publisher = {National Institute of Standards and Technology}
}
```
