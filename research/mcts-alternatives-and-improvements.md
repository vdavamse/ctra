# MCTS Alternatives and Improvements for AutoCT Feature Search

> Research notes from 2026-03-25. Covers search strategy alternatives to AutoCT's MCTS for LLM-based feature engineering in clinical trial outcome prediction.

---

## Context

AutoCT uses standard UCT-based MCTS to search over feature sets (Add/Remove/Refine operations). Each node evaluation runs the full agent pipeline (LLM feature proposal → RAG retrieval → ML training → evaluation) at ~$15-20 per evaluation. The search space is combinatorial feature combinations, not game moves. Reward signal is validation ROC-AUC from XGBoost/LR/RF.

The core insight from this research: **MCTS was designed for game tree search with cheap evaluations. AutoCT's problem is automated feature engineering with expensive evaluations.** The field has purpose-built approaches for this.

---

## Comparison Table

| Paper | Year | Venue | Approach | Search Strategy | Evaluations | Tested on |
|-------|------|-------|----------|----------------|-------------|-----------|
| [CAAFE](https://arxiv.org/abs/2305.03403) | 2023 | NeurIPS 2023 | Sequential LLM feature proposal | Greedy (no search) | ~10-15 | Various classifiers |
| [OCTree](https://arxiv.org/abs/2406.08527) | 2024 | NeurIPS 2024 | Decision tree feedback to LLM | Iterative refinement | ~5-10 | XGBoost |
| [AutoCT](https://arxiv.org/abs/2506.04293) | 2025 | EMNLP 2025 | MCTS over feature sets | UCT tree search | ~70-80 | XGBoost, LR, RF |
| [AIDE](https://arxiv.org/abs/2502.13138) | 2025 | Feb 2025 | Best-first code refinement | Simplified tree (no UCT) | ~20-40 | Kaggle tasks |
| [AB-MCTS](https://arxiv.org/abs/2503.04412) | 2025 | NeurIPS 2025 Spotlight | Adaptive wider-vs-deeper branching | Dynamic UCT with feedback | ~same as MCTS | Coding/engineering |
| [LLM-FE](https://arxiv.org/abs/2503.14434) | 2025 | Mar 2025 | Evolutionary program search, 3-island model | Boltzmann sampling + LLM mutation | **20 total** | XGBoost, TabPFN |
| [PMMG](https://doi.org/10.1002/advs.202410640) | 2025 | Advanced Science | Pareto MCTS for multi-objective molecular generation | Vector UCB + Pareto dominance pools | 10k molecules / ~16h | ZINC, EGFR/HER2 dual-target |
| [ML-Master](https://arxiv.org/abs/2506.16499) | 2025 | arXiv | Async parallel MCTS + selectively scoped memory | UCT + cross-branch memory injection | ~12h budget | MLE-Bench (29.3% medal) |
| [KompeteAI](https://arxiv.org/abs/2508.10177) | 2025 | arXiv | Solution merging + predictive scoring + RAG | Tree search + crossover merging | 12.5 iters (6.9x speedup) | MLE-Bench (51.5% medal) |

---

## Tier 1: Highest Priority for CTRA

### AB-MCTS — Adaptive Branching (Sakana AI)

- **Paper:** [arXiv:2503.04412](https://arxiv.org/abs/2503.04412), NeurIPS 2025 Spotlight
- **Code:** https://github.com/SakanaAI/treequest
- **Key idea:** At each node, dynamically decide whether to "go wider" (generate new candidate branches) or "go deeper" (refine existing candidates using feedback). Adapts based on external feedback signals.
- **Why it matters:** Least disruptive upgrade — keeps MCTS framework, replaces fixed branching with dynamic wider-vs-deeper decisions. When a feature set scores well, go deeper (refine). When scores plateau, go wider (try different directions). Maps directly to AutoCT's Add/Refine/Remove tension.
- **Effort:** Medium — modify UCT selection logic in `treesearch.py`
- **Risk:** Low — still MCTS at its core

### LLM-FE — Evolutionary Feature Engineering

- **Paper:** [arXiv:2503.14434](https://arxiv.org/abs/2503.14434), Mar 2025
- **Code:** https://github.com/nikhilsab/LLMFE
- **Key idea:** Feature engineering as program search. 3-island model with Boltzmann sampling. LLM mutates successful programs. Only **20 total evaluations** needed.
- **Why it matters:** Most directly comparable work — same problem (LLM-based feature engineering for tabular ML with XGBoost and TabPFN), dramatically more sample-efficient. Supports crossover between feature sets (MCTS can't — paths are independent). Island diversity maintains different feature philosophies.
- **Effort:** High — would replace MCTS entirely with evolutionary loop
- **Risk:** Medium — untested on clinical trial data, but proven on tabular ML

### AIDE — Simplified Tree Search for ML Engineering

- **Paper:** [arXiv:2502.13138](https://arxiv.org/abs/2502.13138), Feb 2025
- **Code:** https://github.com/WecoAI/aideml
- **Key idea:** Best-first tree search (no UCB/UCT). Alternates between "draft new solution" and "refine existing solution." 4x more Kaggle medals than best linear agent.
- **Why it matters:** Simpler than MCTS, proven on exact same problem type (iteratively improving ML code to maximize a metric). The draft/refine alternation maps to AutoCT's Add vs Refine.
- **Effort:** Medium-High — simpler architecture but requires rewrite
- **Risk:** Low — extensively validated on Kaggle

---

## Tier 2: Valuable Augmentations

### Fleet of Agents — Genetic Particle Filtering

- **Paper:** [arXiv:2405.06691](https://arxiv.org/abs/2405.06691), ICML 2025
- **Key idea:** Multiple agents explore in parallel; periodically resample (duplicate promising, kill poor). ~5% quality improvement at **~40% of the cost** of MCTS.
- **Why it matters:** Natural diversity maintenance, significant cost reduction. Particle filtering is conceptually simpler than MCTS.
- **Effort:** High — fundamentally different architecture
- **Risk:** Medium — not tested on feature engineering specifically

### ReEvo — Reflective Evolution

- **Paper:** [arXiv:2402.01145](https://arxiv.org/abs/2402.01145), NeurIPS 2024
- **Key idea:** "Verbal gradients" — LLM compares two solutions and generates insights about why one outperforms, guiding mutations. Richer signal than scalar rewards.
- **Why it matters:** Could replace AutoCT's evaluator with pairwise comparison reflections: "Feature set A (ROC-AUC 0.72) vs Feature set B (ROC-AUC 0.68): what explains the difference?"
- **Effort:** Medium — modify evaluator prompts
- **Risk:** Low — augmentation, not replacement

### OCTree — Decision Tree as LLM Feedback

- **Paper:** [arXiv:2406.08527](https://arxiv.org/abs/2406.08527), NeurIPS 2024
- **Key idea:** Convert trained decision trees into natural language so LLM gets interpretable feedback about which features helped. 17.1% relative error reduction.
- **Why it matters:** Richer feedback than raw ROC-AUC. The LLM sees "drug_target_count > 3 AND disease_severity > 0.7 → predict success" instead of just "ROC-AUC = 0.71".
- **Effort:** Low-Medium — add decision tree rendering to evaluator
- **Risk:** Low — purely additive

### Successive Halving / Multi-Fidelity Evaluation

- **Established technique** from hyperparameter optimization (ASHA, Hyperband)
- **Key idea:** Quick evaluation on 20% of data first, eliminate bad candidates, full evaluation only for promising ones.
- **Why it matters:** Directly addresses expensive evaluation bottleneck. Train quick RF on subset → filter → full XGB+LR+RF only on survivors.
- **Effort:** Low — add early stopping within existing pipeline
- **Risk:** Very low — well-proven technique

### PMMG — Pareto Multi-Objective MCTS

- **Paper:** [doi:10.1002/advs.202410640](https://doi.org/10.1002/advs.202410640), Liu et al., Advanced Science 2025
- **Code:** https://github.com/Liuyifeii/PMMG
- **Key idea:** Replaces scalar UCB with n-dimensional vector UCB for multi-objective optimization. At each node, maintains a pool of non-dominated (Pareto-optimal) children; dominated nodes are pruned. Backpropagation updates reward vectors across all objectives simultaneously. Applied to SMILES-based molecular generation with an RNN policy.
- **Results:** 51.65% success rate on 7-objective optimization (EGFR, HER2, solubility, permeability, metabolic stability, toxicity, QED) — 2.5x better than baselines (REINVENT: 9.6%, MARS: 12.7%). Hypervolume 0.569 vs 0.433 best baseline (+31.4%). Generated dual-target EGFR/HER2 inhibitors comparable to lapatinib.
- **Why it matters for CTRA:** AutoCT currently optimizes a single scalar (ROC-AUC). In practice, we care about multiple objectives: accuracy across phases (I/II/III), calibration, interpretability (SHAP stability), feature parsimony. PMMG's vector UCB + Pareto pool could replace AutoCT's scalar UCT to simultaneously optimize these. The Pareto front gives decision-makers a menu of trade-off solutions rather than a single "best" feature set.
- **Effort:** Medium — modify UCT selection and backpropagation to use reward vectors and Pareto dominance instead of scalar averaging
- **Risk:** Medium — proven on molecular generation (different domain), and 16h runtime for 10k molecules is expensive; would need to validate that vector UCB converges well with AutoCT's ~70-80 evaluations rather than 10k
- **Key limitation:** Computational cost (~16x slower than baselines). May need to combine with successive halving or surrogate models.

### MAP-Elites Archive for Feature Set Diversity

- **Based on:** SEA-TS ([arXiv:2603.04873](https://arxiv.org/abs/2603.04873)) MCTS + MAP-Elites hybrid
- **Key idea:** Maintain a grid of diverse high-performing feature sets along behavioral dimensions (data source emphasis, feature type distribution, clinical domain focus). Ensures safety-focused and efficacy-focused features both get explored.
- **Why it matters:** Phase I/II/III likely benefit from different feature sets. MAP-Elites prevents convergence to a single feature philosophy.
- **Effort:** Medium — add QD archive alongside MCTS
- **Risk:** Low — augmentation, not replacement

---

## Tier 3: Longer-Term (Phase 3-4)

### RL-Trained ML Agents

- **Paper:** Yang et al. [arXiv:2509.01684](https://arxiv.org/abs/2509.01684), Stanford, 2025
- **Also:** ML-Agent [arXiv:2505.23723](https://arxiv.org/abs/2505.23723), May 2025
- **Key idea:** Train a smaller model via RL to do feature engineering. Fine-tuned Qwen2.5-3B outperforms Claude-3.5-Sonnet by 22%.
- **When:** After Phase 2 when enough historical evaluation data exists for RL training.

### FunSearch — Evolutionary Program Discovery

- **Paper:** DeepMind, [Nature 10.1038/s41586-023-06924-6](https://www.nature.com/articles/s41586-023-06924-6), 2024
- **Key idea:** Island-model evolutionary search over programs with LLM as mutation operator. Prompt constructed from diverse high-scoring programs.
- **When:** The island model is transferable now; full FunSearch requires cheaper evaluations.

### Surrogate-Assisted MCTS

- **Key idea:** Train a cheap surrogate model predicting ROC-AUC from feature set descriptions (using historical evaluation data). Use for MCTS simulation rollouts, only evaluate true function for most promising nodes. Could reduce costs 50-80%.
- **When:** After enough historical evaluations to train the surrogate.

---

## Not Recommended

| Approach | Why not |
|----------|---------|
| Tree of Thoughts / Graph of Thoughts | Designed for single-inference reasoning, not iterative search with expensive evaluations |
| OPRO (Optimization by Prompting) | Too unstructured for combinatorial feature spaces |
| Full RL training (near-term) | Too expensive at current per-evaluation costs |
| MCTSr (Self-Refine MCTS) | Math reasoning domain; adds cost without addressing evaluation expense |
| SELA (Tree-Search Enhanced LLM Agents) | Same core algorithm as AutoCT: UCT-MCTS + LLM candidate generation + real ML execution + undiscounted scalar backpropagation. Differences are superficial — UCT-DP is a one-line tweak (`n(x)=0.8` for unvisited nodes), stage decomposition mirrors AutoCT's Add/Remove/Refine structure, stage caching is an engineering optimization. SELA was published first (ICLR 2025, Oct 2024); AutoCT (EMNLP 2025) specialized the same idea for feature engineering. Adopting SELA would not change anything we don't already have. |
| I-MCTS Hybrid Value Model | LLMs 30%+ miscalibrated on biomedical tasks (see MCTS Optimization section in README). Successor to SELA — adds LLM value estimation, which is the problematic part |

---

## SELA Citation Analysis (Forward Citations as of 2026-03-26)

> SELA (Chi et al., ICLR 2025) uses the same core algorithm as AutoCT — UCT-MCTS + LLM candidate generation + real ML execution + undiscounted scalar backpropagation — applied to general AutoML pipeline search. It was published first (Oct 2024); AutoCT (EMNLP 2025) specialized the idea for feature engineering. SELA itself adds nothing new for CTRA, but its 17 citing papers reveal the direction the field has moved since. Three are significant for CTRA.

### Citing Papers — Full Inventory

| Paper | Year | Venue | ArXiv | Relevance to CTRA |
|-------|------|-------|-------|-------------------|
| **KompeteAI** | 2025 | arXiv | 2508.10177 | **HIGH** — Solution merging + predictive scoring + RAG from Kaggle/arXiv |
| **ML-Master** | 2025 | arXiv | 2506.16499 | **HIGH** — Selectively scoped memory + async branch-parallel MCTS |
| **CoMind** | 2025 | ICLR 2026 | 2506.20640 | **MEDIUM** — Parallel exploration + community knowledge, 92.6% of human Kaggle competitors |
| **Flow-of-Options** | 2025 | ICML 2025 | 2502.12929 | **MEDIUM** — Network-of-options diversity mechanism, 37-48% improvement on therapeutic chemistry |
| **CLINPREAI** | 2025 | medRxiv | — | **MEDIUM** — Agentic AI for clinical prediction (postpartum depression, F1 0.68 vs AutoML 0.64) |
| **AgentGym-RL** | 2025 | arXiv | 2509.08755 | LOW — RL training for LLM agents (Phase 4 relevance) |
| **AgentDS** | 2026 | arXiv | 2603.19005 | LOW — Domain-specific benchmark; AI-only underperforms human-AI collab |
| **EvoAgents** | 2025 | eScience | — | LOW — Evolutionary architecture search for CNNs |
| **FALCON** | 2025 | ICCS | — | LOW — Multi-objective MCTS for industrial fault diagnosis |
| Evaluation Agent Framework | 2026 | arXiv | 2602.22442 | LOW — Meta-evaluation of AutoML agent decisions |
| LLM Data Science Survey | 2025 | arXiv | 2508.02744 | NONE — Survey |
| LLM Statistics Survey | 2024 | American Statistician | 2412.14222 | NONE — Survey |
| MLE-Dojo | 2025 | arXiv | 2505.07782 | NONE — Training environment |
| MLRC-Bench | 2025 | arXiv | 2504.09702 | NONE — Benchmark |
| CA-HIL | 2025 | DSAA | — | NONE — Human-AI competency framework |
| Multi-Modal Finance Agents | 2025 | BigData Congress | — | NONE — Finance domain |
| Evolutionary Agentic Workflows | 2025 | arXiv | 2505.04354 | NONE — Position paper |

### HIGH Priority: KompeteAI — Solution Merging + Predictive Scoring

- **Paper:** [arXiv:2508.10177](https://arxiv.org/abs/2508.10177), Kulibaba et al., 2025
- **LLM:** Gemini 2.5 Flash (primary), o1-preview and DeepSeek-R1 for comparisons
- **Key innovations over SELA/AutoCT:**

1. **Solution merging:** After exploring feature engineering (FE) and model training (MT) nodes independently, KompeteAI *recombines* the best FE and MT nodes by sampling pairs and merging their structural and statistical traits via `MergeFE()`. MCTS (including AutoCT) can't do this — tree paths are independent. This is analogous to crossover in evolutionary algorithms but within a tree structure. **Ablation: removing merging drops medal rate from 51.5% to 38.5%** — the single largest factor.
2. **Predictive scoring for early pruning:** Instead of running full training, an LLM predicts solution quality from feature descriptions using few-shot prompting with anchor examples from prior evaluations. Enables **6.9x more iterations** within the same compute budget (12.5 vs 1.8 iterations). Directly addresses CTRA's $15-20/evaluation bottleneck.
3. **Adaptive RAG:** Retrieves winning Kaggle solutions and arXiv papers via function calling, triggered only when external knowledge is expected to help. Not a static retrieval — the agent decides when to invoke RAG based on problem context.

- **Results:** 51.5% medal rate on MLE-Bench Lite (vs ML-Master 48.5%, RD-Agent 48.2%, AIDE 16.9%)
- **Why it matters for CTRA:**
  - **Solution merging** could combine the best RAG retrieval strategy from one branch with the best feature set from another — something AutoCT's independent tree paths cannot do
  - **Predictive scoring** could serve as a lightweight surrogate for CTRA's expensive evaluations: "Given these features and this RAG context, predict ROC-AUC" before committing to full pipeline execution. This overlaps with our Phase 2-3 surrogate idea but is simpler (LLM few-shot vs trained model)
  - **Adaptive RAG triggering** matches CTRA's architecture — LinearRAG retrieval is the most expensive component; triggering it selectively could cut costs substantially
- **Effort:** High — merging requires rethinking AutoCT's tree structure; predictive scoring is medium effort
- **Risk:** Medium — merging is novel and unvalidated on feature engineering specifically

### HIGH Priority: ML-Master — Selectively Scoped Memory + Async Parallel MCTS

- **Paper:** [arXiv:2506.16499](https://arxiv.org/abs/2506.16499), Liu et al. (SJTU), 2025
- **LLM:** DeepSeek-R1-0120 (reasoning model)
- **Key innovations over SELA/AutoCT:**

1. **Selectively scoped memory (ℳₜ):** At each node, the agent receives curated insights from two sources — the parent node (continuation signal) and sibling nodes at the same depth (contrastive signal). An extraction function ε(·) distills verbose reasoning traces into actionable knowledge: "key analytical insights, identified patterns, debugging strategies, improvement directions." This prevents context overflow while preserving cross-branch learning.
2. **Async branch-parallel MCTS:** Expands root nodes jointly, then selects top-k highest UCT-valued nodes as independent thread entry points. Threads explore asynchronously without interference. On completion, threads return to root and select the best unoccupied child. Sibling information creates contrastive signals to avoid redundant exploration.
3. **Steerable reasoning:** Memory is "explicitly embedded into the LLM's 'think' component" — the reasoning model's chain-of-thought is steered by insights from other branches, not just the current path's history.

- **Results:** 29.3% medal rate on MLE-Bench (vs AIDE 16.9% with o1-preview). **Medium-complexity tasks: 20.2% vs previous best 9.0%** — the biggest gain is on tasks requiring deep exploration, exactly CTRA's profile. Achieved in 12 hours vs 24-hour limit used by baselines.
- **Why it matters for CTRA:**
  - **Selectively scoped memory** directly addresses AutoCT's context problem — with 700-1000 API calls per evaluation, the LLM loses track of what's been tried. Cross-branch insights ("Phase II features from branch A failed because of criteria encoding, not drug features") would prevent redundant exploration
  - **Async parallel MCTS** could parallelize AutoCT's evaluations across workers — currently sequential. With $15-20/evaluation, parallelism doesn't save money but saves wall-clock time
  - **Steerable reasoning** is particularly relevant for clinical feature engineering — insights like "drug-target interaction features only help Phase II/III, not Phase I (safety endpoints differ)" could propagate across branches
  - The **2.2x improvement on medium-complexity tasks** suggests the memory mechanism helps most when the search space is deceptive — which is exactly CTRA's delayed-reward problem (see Deep Analysis below)
- **Effort:** Medium-High — memory mechanism is modular, but async parallelism requires infrastructure changes
- **Risk:** Low-Medium — proven on MLE-Bench, reasoning model (DeepSeek-R1) aligns with CTRA's use of Claude Opus 4.6

### MEDIUM Priority: CoMind — Community Knowledge for ML Engineering

- **Paper:** [arXiv:2506.20640](https://arxiv.org/abs/2506.20640), Li et al. (CMU), ICLR 2026
- **Key idea:** Multi-agent system where agents share knowledge from a simulated research community. Iterative parallel exploration balances breadth and depth. 36% medal rate on 75 past Kaggle competitions; 92.6% of human competitors on 8 live competitions (top 5% on 3, top 1% on 1).
- **Why it matters:** The community knowledge sharing is a form of population-based diversity maintenance — similar in spirit to LLM-FE's island model but with richer inter-agent communication. The live competition results are the strongest published for any ML agent.
- **Limitation:** No details on feature engineering specifically; ICLR 2026 paper may not be fully public yet.
- **Code:** https://github.com/comind-ml/CoMind

### MEDIUM Priority: Flow-of-Options — Diversified Reasoning for Chemistry + ML

- **Paper:** [arXiv:2502.12929](https://arxiv.org/abs/2502.12929), Nair et al. (Flagship Pioneering), ICML 2025
- **Key idea:** Models reasoning options as a network of nodes. Uses beam search with variable widths + case-based reasoning to enforce diversity. Unlike MCTS which explores a single tree, FoO explores a *network* of alternative solution strategies.
- **Results:** 38-69% improvement on data science tasks; **37-48% improvement on ADME-Tox therapeutic chemistry tasks** (from Therapeutic Data Commons). Cost < $1/task.
- **Why it matters:** The therapeutic chemistry results are directly relevant — ADME-Tox prediction is structurally similar to clinical trial prediction (drug properties → outcome). The network-of-options paradigm is fundamentally different from tree search and could offer better diversity in feature engineering strategies.
- **Limitation:** No published comparison to MCTS-based approaches. The mechanism details are only in the PDF (not fully extractable).
- **Code:** https://github.com/flagshippioneering/Flow-of-Options

### MEDIUM Priority: CLINPREAI — Agentic AI for Clinical Prediction (Validation)

- **Paper:** [medRxiv 10.1101/2025.11.14.25340265](https://doi.org/10.1101/2025.11.14.25340265), Palacios et al. (Baylor College of Medicine), 2025
- **Key idea:** First application of agentic AI to perinatal mental health prediction. Autonomous system predicts postpartum depression risk from multimodal EHR (27 structured variables + clinical notes). F1 0.68 vs traditional AutoML F1 0.64.
- **Why it matters for CTRA:** Proof-of-concept that agentic AI (SELA-style) works for clinical prediction from structured + unstructured data. The modest improvement (0.04 F1) over traditional AutoML suggests the bottleneck is not the search algorithm but the data representation — supporting CTRA's emphasis on RAG-powered feature engineering over generic AutoML.
- **Implication:** Validates our architectural choice — AutoCT + domain-specific RAG should outperform generic agentic AutoML on clinical prediction tasks.

---

## Deep Analysis: Pareto MCTS for Feature Engineering (PMMG ↔ AutoCT)

> Research from 2026-03-26. Detailed analysis of why PMMG's Pareto MCTS mechanism is applicable to AutoCT's feature engineering search, based on deep codebase analysis of both repositories.

### Structural Isomorphism: Molecules ↔ Feature Sets

Both problems share the same fundamental structure: **sequential construction of a composite object where intermediate states have no measurable value**.

| Dimension | PMMG (Molecules) | AutoCT (Features) |
|-----------|-------------------|---------------------|
| Token | SMILES symbol (C, N, O, branch markers) | Feature operation (Add/Remove/Refine) |
| Path | Partial → complete molecule | Partial → complete feature set |
| Intermediate value | None — partial SMILES meaningless | Near-zero — single features lack predictive power |
| Reward | Multi-objective vector (7-8 pharma properties) | Currently scalar (ROC-AUC) |
| Depth | ~30-80 tokens per molecule | Max 7 operations (configurable) |
| Branching factor | ~40-70 valid tokens | ~5 LLM suggestions per node |
| Evaluation cost | Milliseconds (LGB property prediction) | 5-15 min, ~$15-20 (LLM+RAG+ML pipeline, 700-1000 API calls) |
| Budget | ~10,000 molecules / 16h | ~70-80 evaluations total |

### The Delayed Reward Problem: Why Deep Exploration Matters

**The epistasis analogy.** In genetics, pure epistasis means individual SNPs show zero marginal association with phenotype, but specific multi-SNP combinations are deterministic predictors (the biological XOR problem). Clinical trial features exhibit the same structure: "number of eligibility criteria" alone is uninformative, "drug target count" alone is uninformative, but "high criteria count AND low target count" strongly predicts Phase II failure. Univariate filter methods rank both features low and discard them, missing the interaction entirely.

**Deceptive fitness landscapes.** This creates a landscape where the immediate reward gradient points away from the global optimum. Forward (greedy) feature selection fails catastrophically. Research on AutoML search spaces (Pimenta & Sa 2020) confirms these landscapes exhibit strong *neutrality* — as you approach good solutions, the landscape flattens with many equally-performing neighbors. MCTS must traverse large neutral regions before reaching discriminating terrain.

**How AutoCT handles it today** (`treesearch.py`):
- `_simulate()` returns the **max reward along the entire root-to-leaf path**, not just the leaf score
- Backpropagation is undiscounted — all ancestors receive the raw max score equally
- `exploration_weight=1.0` in UCT keeps exploration bonus substantial
- With ~140 nodes explored out of ~78M possible (branching factor 5, depth 7), coverage is **0.0002%** — exploration appears random because it is sampling a vanishing fraction of the tree

**How PMMG handles it** (`mcts.py`, `utils.py`):
- Uses the **RNN as a learned completion policy** — from any partial SMILES, the RNN greedily completes the molecule for evaluation (`chem_kn_simulation`)
- Backpropagation is also undiscounted — the full vector reward flows unchanged to all ancestors
- **Critical code finding:** `c_val=1.0` is defined in config but **never used** — a hardcoded `0.2` coefficient replaces it in `mcts.py:63`, making actual exploration more exploitative than configured

### Why Pareto MCTS Prevents Premature Convergence

**The stepping-stone mechanism.** Maintaining a Pareto front of non-dominated solutions is functionally equivalent to Quality-Diversity algorithms' stepping-stone mechanism. QD methods outperform single-objective methods by orders of magnitude on deceptive landscapes (p<10^-9 vs genetic algorithms, p<10^-7 vs TD3 RL). Traditional methods "plateau at suboptimal fitness values early" while QD methods show "sustained improvement throughout the learning process."

**How it works in PMMG's selection** (`mcts.py:49-87`):
1. Compute vector UCB for all children: `score = (total_reward / visits) + 0.2 * sqrt(2 * ln(parent_visits) / visits)`
2. Apply Pareto dominance filtering — remove children whose score vector is dominated by another child on ALL objectives
3. **Randomly select from the non-dominated set** — this is the key divergence from scalar MCTS

In scalar MCTS (AutoCT today), UCB deterministically selects the highest-scoring child → convergence to a single exploitation path. In Pareto MCTS, the non-dominated pool contains children excelling on different objectives → random selection forces diverse path exploration.

**Concrete CTRA example:** A feature set with high Phase III accuracy but poor calibration is *non-dominated* by a well-calibrated set with lower accuracy. Both are preserved and expanded. The Phase III-optimal path might, when refined deeper, discover a calibration-improving feature interaction that the calibration-optimal path would never reach. Single-objective MCTS would prune one of these.

### Why Clinical Trial Prediction Needs Multiple Objectives

Single-objective optimization (ROC-AUC alone) leads to degenerate solutions:
- **Phase-biased performance** — aggregate AUC is dominated by Phase III (more data, higher base rate), sacrificing Phase I/II
- **Miscalibrated probabilities** — discrimination preserved but probability estimates meaningless for risk-based decisions
- **Overfitting to leaky features** — trial duration, enrollment rate as proxies for success
- **No interpretability guarantee** — black-box feature sets that clinicians cannot trust

Natural objectives for Pareto MCTS:
1. **Phase-specific accuracy** — separate ROC-AUC for Phase I, II, III (3 objectives)
2. **Calibration** — Expected Calibration Error (ECE) or Brier score
3. **Feature parsimony** — fewer features = lower operational cost (each requires RAG retrieval + LLM processing), less overfitting
4. **Interpretability** — SHAP stability, decision tree depth, overlap with known clinical predictors
5. **Robustness** — performance on out-of-distribution trials (novel therapeutic areas)

The Pareto front gives decision-makers a **menu of trade-off solutions** — an interpretable 5-feature set for clinical dashboards, a 30-feature set for high-stakes predictions, a phase-specific specialist for each phase.

### Search Space Comparison

| | Molecular Space | Feature Space |
|---|---|---|
| Raw size | ~10^60 (drug-like molecules) | ~10^15 to 10^30 (feature subsets) |
| With variants | Constrained by chemistry | ~10^100 (if features have parameter variants) |
| Evaluation cost | Milliseconds (property prediction) | Minutes + dollars (LLM pipeline) |
| Reward deceptiveness | Moderate (RNN priors guide search) | **High** (no analogous prior for feature interactions) |
| Valid fraction | ~1 in 10^3-10^6 yield valid molecules | ~1 in 10^1-10^3 yield non-degenerate models |

The feature space is nominally smaller but **harder to search** — higher evaluation cost, more deceptive reward structure, and no learned prior to guide expansion (AutoCT uses LLM suggestions rather than a trained policy).

### The Sample Efficiency Gap: 10,000 vs 70-80 Evaluations

PMMG generates ~10,000 molecules. AutoCT's budget is ~70-80 evaluations. Three mitigations:

1. **Multi-fidelity evaluation**: Quick RF on 20% of data as cheap proxy → filter → full XGB+LR+RF only for survivors. Could 3-5x the effective evaluation budget.
2. **Shallower tree**: AutoCT depth 7 vs PMMG depth 80. Far fewer evaluations needed to reach terminal nodes. Each AutoCT evaluation is informationally rich (full ML pipeline, not a single property prediction).
3. **Surrogate-assisted Pareto MCTS**: After ~30 evaluations, train a cheap surrogate predicting the reward vector from feature set descriptions. Use for MCTS rollouts, only run full pipeline for nodes near the Pareto front. Could reduce cost 50-80%.

### Implementation Sketch: Pareto MCTS for AutoCT

```
Objectives: [phase1_auc, phase2_auc, phase3_auc, calibration, parsimony]

At each node:
  1. Compute vector UCB for all children (5-dimensional)
  2. Pareto-filter: remove dominated children
  3. Random select from non-dominated pool
  4. Expand: LLM proposes Add/Remove/Refine operation
  5. Simulate: run full agent pipeline → evaluate all 5 objectives
  6. Backpropagate: vector reward to all ancestors (undiscounted)

Output: Pareto front of feature sets spanning the trade-off surface
```

Key adaptation from PMMG:
- Replace SMILES vocabulary → feature operation space
- Replace RNN completion policy → LLM agent pipeline (already exists in AutoCT)
- Replace LGB property prediction → ML training + multi-metric evaluation
- Replace molecular filters → feature validity checks (no duplicate features, minimum feature count)
- Add `dominate_score()` reward computation comparing new feature sets against historical Pareto front
- Modify `select_node()` to use Pareto-filtered vector UCB instead of scalar UCT

### Key Code References

- **AutoCT MCTS core:** [`src/lfe/impl/treesearch.py`](https://github.com/linyongver/AutoCT) (397 lines)
- **AutoCT agent pipeline:** [`src/lfe/impl/agent.py`](https://github.com/linyongver/AutoCT) (2765 lines)
- **PMMG MCTS core:** [`chemtsv2/mcts.py`](https://github.com/Liuyifeii/PMMG) (564 lines) — `select_node()` at line 49, `dominate()` at line 183, reward computation at line 426
- **PMMG vector UCB:** [`policy/ucb1.py`](https://github.com/Liuyifeii/PMMG) (9 lines) — defined but unused; actual implementation inline in mcts.py:63
- **PMMG backpropagation:** [`chemtsv2/utils.py:50-56`](https://github.com/Liuyifeii/PMMG) — undiscounted vector reward to all ancestors
- **AB-MCTS Thompson Sampling:** [`src/treequest/algos/ab_mcts_a/prob_state.py`](https://github.com/SakanaAI/treequest) — GEN/CONT adaptive decision

---

## Recommended Implementation Path

**Phase 2 (current):**
1. Fix existing MCTS first (backpropagate simulation nodes, α tuning, informed selection — already in PR #11)
2. Add OCTree decision-tree feedback to evaluator (low effort, additive)
3. Add multi-fidelity evaluation / successive halving (low effort, direct cost savings)

**Phase 2-3 transition:**
4. **Implement Pareto MCTS for multi-objective feature search** — adapt PMMG's vector UCB + Pareto dominance selection into AutoCT's `treesearch.py`. Define objectives: phase-specific AUC (I/II/III), calibration (ECE), parsimony. This addresses premature convergence and enables deep exploration of feature interactions. (medium effort, high impact — see Deep Analysis section above)
5. **Add ML-Master's selectively scoped memory** — at each node, inject curated insights from parent (continuation) and sibling (contrastive) nodes into the LLM prompt. Prevents redundant exploration across branches and addresses context loss over 700-1000 API calls per evaluation. 2.2x improvement on medium-complexity tasks in MLE-Bench. (medium effort, high impact — see SELA Citation Analysis)
6. Evaluate AB-MCTS adaptive branching as complementary upgrade — Thompson Sampling GEN/CONT decision on top of Pareto selection (medium effort, low risk)
7. **Add KompeteAI's predictive scoring for early pruning** — LLM few-shot predicts evaluation quality from feature descriptions before committing to full pipeline execution. Simpler than training a surrogate model; KompeteAI achieved 6.9x more iterations within the same compute budget. (medium effort, direct cost savings — see SELA Citation Analysis)

**Phase 3:**
8. Prototype LLM-FE island model with Pareto objectives as A/B alternative (high effort, high potential)
9. **Evaluate KompeteAI's solution merging** — recombine best feature engineering and model configuration nodes across tree paths. MCTS paths are independent; merging enables crossover-like recombination. Ablation showed merging is the single largest factor (51.5% → 38.5% without it). (high effort, high potential — see SELA Citation Analysis)
10. Add MAP-Elites archive for multi-phase feature diversity
11. Evaluate AIDE as simpler replacement if Pareto MCTS still underperforms

**Phase 4:**
12. Begin collecting evaluation data for RL training
13. Investigate hybrid: AB-MCTS adaptive branching + Pareto vector UCB + ML-Master memory + KompeteAI merging

---

## Fixes to Published AutoCT MCTS

Code analysis of AutoCT's `src/lfe/impl/treesearch.py` and research from I-MCTS [[23]](../docs/2502.14693v3.pdf) and SEA-TS [[24]](../docs/2603.04873v2.pdf) identified the following implementation fixes. Status reflects what is implemented in CTRA as of 2026-04-08.

| Change | Rationale | Effort | Grounding | Status |
|--------|-----------|--------|-----------|--------|
| **Backpropagate simulation nodes** — update all in-tree nodes on the simulation path, not just the selection path | ~60-70 explored nodes per 10 rollouts have `total_reward=0` despite being evaluated | Low | Browne et al. 2012 [27]; LATS [25] | **Implemented** (`mcts.py:554-571`) |
| **Backpropagation rule** — PMMG's subtree mean of the reward vector (current) vs AutoCT's max, vectorised (`max`), vs the best realised vector by hypervolume (`max_hv`) | Whether averaging a subtree hides the best line below it | Config (`MCTSConfig.backprop`) | Issue #16 ablation, [backprop-ablation.md](backprop-ablation.md) | **Measured; default kept at `mean`** — within noise where UCB decides, `max_hv` loses two cells |
| **Informed simulation selection** — replace `rng.choice()` with weighted selection favoring evaluator's top suggestions | Random selection wastes expensive agent runs | Low | AlphaGo Zero [28]; LATS [25]; Browne [27] | Not implemented |
| **Reduce α from 1.0 to ~0.15** | α=1.0 makes UCT exploration term 30-100x larger than ROC-AUC differences, making selection effectively random | Trivial | Schmocker et al. 2025 [31]: 40% performance loss from mismatched C | Not implemented (`settings.py` has `exploration_constant: 1.414`) |
| **Increase rollouts to 30-50** | More iterations yield better results; backprop fix reduces waste | Config | AutoCT [5] Fig 3; SELA [26]; I-MCTS [23] | **Implemented** (configurable) |
| **Error penalty (R=-1)** | Failed nodes receive -1 instead of 0, actively pushing MCTS away | Trivial | SEA-TS [24] Eq. 7 | Not implemented |
| **Introspective sibling analysis** | Show LLM what worked/failed in siblings — avoids dead ends | Medium | I-MCTS [23] Section 2 | Not implemented |

---

## Key References

- [AB-MCTS](https://arxiv.org/abs/2503.04412) — Inoue et al., NeurIPS 2025 Spotlight
- [LLM-FE](https://arxiv.org/abs/2503.14434) — Abhyankar et al., Mar 2025
- [AIDE](https://arxiv.org/abs/2502.13138) — Jiang et al. (Weco AI), Feb 2025
- [Fleet of Agents](https://arxiv.org/abs/2405.06691) — Klein et al., ICML 2025
- [ReEvo](https://arxiv.org/abs/2402.01145) — NeurIPS 2024
- [OCTree](https://arxiv.org/abs/2406.08527) — NeurIPS 2024
- [CAAFE](https://arxiv.org/abs/2305.03403) — NeurIPS 2023
- [PMMG](https://doi.org/10.1002/advs.202410640) — Liu et al., Advanced Science 2025
- [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) — DeepMind, Nature 2024
- [RL for ML Agents](https://arxiv.org/abs/2509.01684) — Yang et al., Stanford, 2025
- [SEA-TS](https://arxiv.org/abs/2603.04873) — Xu et al., 2026
- [I-MCTS](https://arxiv.org/abs/2502.14693) — Liang et al., EACL 2026
- [SELA](https://arxiv.org/abs/2410.17238) — Chi et al., ICLR 2025 (same algorithm as AutoCT; see Not Recommended + Citation Analysis)
- [AutoCT](https://arxiv.org/abs/2506.04293) — Liu et al., EMNLP 2025
- [KompeteAI](https://arxiv.org/abs/2508.10177) — Kulibaba et al., 2025 (solution merging + predictive scoring; found via SELA citations)
- [ML-Master](https://arxiv.org/abs/2506.16499) — Liu et al. (SJTU), 2025 (selectively scoped memory + async parallel MCTS; found via SELA citations)
- [CoMind](https://arxiv.org/abs/2506.20640) — Li et al. (CMU), ICLR 2026 (community-driven parallel exploration; found via SELA citations)
- [Flow-of-Options](https://arxiv.org/abs/2502.12929) — Nair et al. (Flagship Pioneering), ICML 2025 (network-of-options diversity; found via SELA citations)
- [CLINPREAI](https://doi.org/10.1101/2025.11.14.25340265) — Palacios et al. (Baylor), medRxiv 2025 (agentic AI for clinical prediction; found via SELA citations)
