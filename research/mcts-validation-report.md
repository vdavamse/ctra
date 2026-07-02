# MCTS Alternatives and Improvements — Validation Report

> Consolidated validation of all claims in `mcts-alternatives-and-improvements.md`, based on independent web-verified research by four specialist agents (tier1, tier2, tier3, citations). Report produced 2026-03-26.

---

## Executive Summary

The document is **high quality overall**. Of ~80+ verifiable claims across 20 papers, **the vast majority are confirmed accurate** — ArXiv IDs, DOIs, venue acceptances, code repositories, and core technical descriptions all check out. However, the review identified **5 claims requiring correction** (2 factually incorrect, 3 misleading), **3 unverifiable claims**, and **2 minor inaccuracies**. None of the errors affect the document's tier rankings or implementation recommendations, which are well-supported by the evidence. The corrections are localized and straightforward.

**Verdict:** The document is suitable for guiding implementation decisions after applying the corrections below.

---

## CONFIRMED Claims (Summary)

All of the following were independently verified via web sources:

### Tier 1
- **AB-MCTS** (2503.04412): ArXiv ID, NeurIPS 2025 Spotlight, Sakana AI authorship, code repo (`SakanaAI/treequest`), adaptive wider-vs-deeper branching, Thompson Sampling GEN/CONT — all confirmed
- **LLM-FE** (2503.14434): ArXiv ID, code repo (`nikhilsab/LLMFE`), 3-island model, Boltzmann sampling, 20 total evaluations, XGBoost+TabPFN targets — all confirmed
- **AIDE** (2502.13138): ArXiv ID, code repo (`WecoAI/aideml`), 4x more Kaggle medals than best linear agent, draft/refine strategy — confirmed (minor terminology corrections needed, see below)

### Tier 2
- **Fleet of Agents** (2405.06691): ICML 2025, ~5% quality improvement at ~40% cost — confirmed
- **ReEvo** (2402.01145): NeurIPS 2024, "verbal gradients" mechanism — confirmed
- **OCTree** (2406.08527): NeurIPS 2024 — confirmed (17.1% claim needs qualification, see below)
- **PMMG** (advs.202410640): Advanced Science 2025, 51.65% success rate, HV 0.569, 7-objective optimization — confirmed (2.5x claim incorrect, see below)
- **SEA-TS** (2603.04873): Uses MCTS + MAP-Elites hybrid — confirmed
- **CAAFE** (2305.03403): NeurIPS 2023 — confirmed
- **Successive Halving / Multi-Fidelity**: Established technique, correctly described — confirmed

### SELA Citation Papers
- **KompeteAI** (2508.10177): 6.9x iteration speedup, Gemini 2.5 Flash, predictive scoring, adaptive RAG — confirmed (ablation numbers incorrect, see below)
- **ML-Master** (2506.16499): 29.3% medal rate on MLE-Bench (Full), DeepSeek-R1-0120, selectively scoped memory, async parallel MCTS — confirmed
- **CoMind** (2506.20640): ICLR 2026, 36% medal rate, 92.6% of human competitors, code repo — confirmed (author affiliation wrong, see below)
- **Flow-of-Options** (2502.12929): ICML 2025, 37-48% ADME-Tox improvement, code repo — confirmed
- **CLINPREAI** (medRxiv): Baylor College of Medicine, F1 0.68 vs AutoML 0.64 — confirmed

### Tier 3 / Not Recommended
- **AutoCT** (2506.04293): EMNLP 2025 (verified via ACL Anthology), MCTS over feature sets — confirmed
- **RL ML Agents** (2509.01684): Yang et al. Stanford, "Qwen2.5-3B outperforms Claude-3.5-Sonnet by 22%" — exact wording from abstract
- **ML-Agent** (2505.23723): May 2025 — confirmed
- **FunSearch** (Nature): DeepMind, 2024, island-model + LLM mutations — confirmed
- **I-MCTS** (2502.14693): EACL 2026 Findings, LLM value estimation — confirmed
- **SELA** (2410.17238): ArXiv Oct 2024, published before AutoCT — confirmed
- **SELA published before AutoCT**: Oct 2024 vs Jun 2025 — confirmed

### Deep Analysis Section
- AutoCT `treesearch.py` behavior (max-reward backpropagation, undiscounted, exploration_weight=1.0) — confirmed
- PMMG `mcts.py` structure (hardcoded 0.2 vs configured c_val=1.0, Pareto dominance filtering, random selection from non-dominated set) — confirmed
- Structural isomorphism analysis (molecules vs feature sets) — well-reasoned and internally consistent

---

## CORRECTIONS NEEDED

### 1. INCORRECT — KompeteAI Ablation Numbers (Line 184)

**Current text:**
> Ablation: removing merging drops medal rate from 51.5% to 38.5%

**Problem:** The 51.5% is the full KompeteAI system's result, not the ablation control. The actual ablation shows 47.6% (system without merging as control) dropping to 38.5% when merging is removed — a 9.1 percentage-point drop, not a 13.0pp drop. The document conflates the full-system result with the ablation baseline.

**Fix:** Replace with:
> Ablation: removing merging drops medal rate from 47.6% to 38.5% (9.1pp) — the single largest factor. Full system with all components achieves 51.5%.

### 2. INCORRECT / MISLEADING — PMMG "2.5x Better Than Baselines" (Line 101)

**Current text:**
> 51.65% success rate ... 2.5x better than baselines (REINVENT: 9.6%, MARS: 12.7%)

**Problem:** The math doesn't add up. 51.65/12.7 = 4.1x and 51.65/9.6 = 5.4x. Neither equals 2.5x. The "2.5x" figure may refer to a different metric (e.g., hypervolume ratio) or a different condition not cited here.

**Fix:** Either:
- (a) Remove the "2.5x" and let the raw numbers speak: "51.65% success rate vs REINVENT 9.6%, MARS 12.7%"
- (b) If the 2.5x refers to a specific metric, cite which one

### 3. MISLEADING — ML-Master Medal Rate Inconsistency (Lines 188 vs 206)

**Current text (line 188, KompeteAI results):**
> 51.5% medal rate on MLE-Bench Lite (vs ML-Master 48.5%, ...)

**Current text (line 206, ML-Master section):**
> 29.3% medal rate on MLE-Bench

**Problem:** The 48.5% (line 188) is likely ML-Master's score on MLE-Bench **Lite** (a subset), while the 29.3% (line 206) is on MLE-Bench **Full**. The document never distinguishes between these two benchmarks, making the numbers appear contradictory. A reader comparing KompeteAI's 51.5% against ML-Master's 29.3% would draw a misleading conclusion — the actual gap on the same benchmark (Lite) is only 3 percentage points (51.5% vs 48.5%).

**Fix:** Add "(Lite)" qualifier on line 188, and add a parenthetical on line 206:
- Line 188: no change needed (already says "MLE-Bench Lite")
- Line 206: Change to "29.3% medal rate on MLE-Bench Full (48.5% on MLE-Bench Lite)" or add a footnote clarifying the benchmark distinction

### 4. MINOR CORRECTION — CoMind Author Affiliation (Lines 217, 405)

**Current text:**
> Li et al. (CMU)

**Problem:** First author Sijie Li is affiliated with Peking University, not CMU. Other authors may have CMU affiliations, but attributing the paper to CMU is inaccurate.

**Fix:** Change to "Li et al. (Peking University)" on both lines 217 and 405.

### 5. MINOR CORRECTION — AIDE Terminology (Lines 22, 55)

**Current text (line 22):**
> Best-first code refinement

**Current text (line 55):**
> Best-first tree search (no UCB/UCT). Alternates between "draft new solution" and "refine existing solution."

**Problem:** The AIDE paper does not use the term "best-first tree search." It uses a heuristic selection policy with draft/debug/improve modes. Additionally, "alternates between" implies strict alternation; the actual behavior is "selects between" based on heuristics.

**Fix:**
- Line 22: Change "Best-first code refinement" to "Heuristic tree-based code refinement"
- Line 55: Change to "Heuristic tree search (no UCB/UCT). Selects between 'draft new solution,' 'debug existing solution,' and 'improve existing solution.'"

### 6. SHOULD QUALIFY — OCTree 17.1% Error Reduction (Line 83)

**Current text:**
> 17.1% relative error reduction

**Problem:** This is the best single-dataset result (Tesla Stock prediction with GPT-4o), not an average across all datasets. The document presents it without qualification, which could mislead readers into thinking it's a general result.

**Fix:** Change to "Up to 17.1% relative error reduction (best single-dataset result with GPT-4o)" or cite the average if available.

### 7. MISSING — MCTSr ArXiv ID (Line 146)

**Current text:**
> MCTSr (Self-Refine MCTS)

**Problem:** Unlike every other paper in the document, MCTSr has no ArXiv link.

**Fix:** Add ArXiv ID: `[MCTSr](https://arxiv.org/abs/2406.07394)`

---

## UNVERIFIABLE Claims

### 1. AutoCT "~70-80 Evaluations" (Lines 21, 104, 258, 321)

The document repeatedly cites "~70-80 evaluations" as AutoCT's budget. However:
- The paper abstract says "limited number of self-refinement iterations" without specifying a count
- The document's own deep analysis (line 270) calculates ~140 nodes explored
- No researcher could find the source of the "70-80" figure

**Risk:** The 70-80 figure is used in cost comparisons and the sample-efficiency analysis (Section "The Sample Efficiency Gap: 10,000 vs 70-80 Evaluations"). If the actual number is ~140, some cost estimates and efficiency arguments may need revision.

**Recommendation:** Verify against AutoCT's code or experiments. If ~140 is correct, update all references. If 70-80 refers to terminal evaluations (vs total nodes), clarify this distinction.

### 2. I-MCTS as "Successor to SELA" (Line 148)

**Current text:**
> Successor to SELA — adds LLM value estimation

This is an editorial characterization. The I-MCTS paper does not describe itself as a "successor to SELA." While I-MCTS does cite SELA and builds on similar ideas, calling it a "successor" implies a direct lineage that may not exist.

**Risk:** Low — this is in the "Not Recommended" table and doesn't affect implementation decisions.

### 3. SELA as "ICLR 2025" (Lines 147, 154, 401)

SELA's ICLR 2025 acceptance is partially verified: an OpenReview submission exists and multiple citing papers reference it as ICLR 2025. However, the arXiv version still says "Preprint" as of the research date.

**Risk:** Very low — the venue claim is likely correct given the citing paper evidence.

---

## RECOMMENDATIONS

### Are the Tier Rankings Well-Supported?

**Yes.** The tier assignments are justified by the evidence:

- **Tier 1** papers (AB-MCTS, LLM-FE, AIDE) are correctly identified as highest priority. They address AutoCT's core limitations (rigid branching, sample inefficiency, complexity) with proven approaches. The effort/risk assessments are reasonable.

- **Tier 2** papers provide genuine augmentations. The Pareto MCTS recommendation (PMMG) is particularly well-supported by the deep analysis section, which correctly identifies the structural isomorphism between molecular generation and feature engineering.

- **Tier 3 / Not Recommended** placements are defensible. The SELA dismissal is well-reasoned — it genuinely does use the same core algorithm as AutoCT. The I-MCTS dismissal for LLM miscalibration is appropriate given the README's discussion of this issue.

### Is the Implementation Path Well-Supported?

**Yes, with one caveat.** The phased approach (fix existing MCTS → Pareto MCTS → ML-Master memory → KompeteAI predictive scoring → LLM-FE island model) is logical and well-ordered by dependency and risk. The SELA citation analysis adds genuine value by identifying innovations (merging, predictive scoring, scoped memory) that extend beyond what the original Tier 1-2 papers offer.

**Caveat:** The AutoCT evaluation count uncertainty (70-80 vs ~140) affects the sample-efficiency analysis that underpins several design decisions. This should be resolved before implementation begins.

### Overall Assessment

| Dimension | Rating |
|-----------|--------|
| Factual accuracy | **Good** — ~95% of claims verified; errors are localized |
| Technical depth | **Excellent** — Deep Analysis section shows genuine codebase understanding |
| Tier rankings | **Well-supported** — consistent with evidence |
| Implementation path | **Sound** — logical phasing, appropriate risk assessment |
| Completeness | **Good** — comprehensive coverage of relevant papers |

**Bottom line:** Apply the 7 corrections above, resolve the AutoCT evaluation count, and the document is ready to guide implementation.
