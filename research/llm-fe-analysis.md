# LLM-FE: Automated Feature Engineering with LLMs as Evolutionary Optimizers

**Paper:** arXiv:2503.14434 (Abhyankar, Shojaee, Reddy -- Virginia Tech)
**Code:** https://github.com/nikhilsab/LLMFE
**Date reviewed:** 2026-03-26

---

## 1. The 3-Island Evolutionary Model

### What are the three islands?

LLM-FE uses **m=3 homogeneous islands** -- they are structurally identical sub-populations, not three different island *types*. Each island is an independent `Island` object maintaining its own set of program clusters. The design is inherited from FunSearch (Romera-Paredes et al., 2024) and LLM-SR (Shojaee et al., 2024).

Each island:
- Is initialized with the **same seed program** (a trivial `modify_features_v0()` that does minimal transformation)
- Maintains its own dictionary of **clusters** keyed by program signature (the tuple of per-fold validation scores)
- Tracks its own `_num_programs` counter that drives temperature annealing independently
- Evolves completely independently -- islands never directly share programs during normal operation

### How do solutions migrate between islands?

There is **no continuous migration**. Instead, LLM-FE uses a **periodic catastrophic reset** mechanism:

1. **Reset trigger:** Every `reset_period` seconds (default: 14,400 seconds = 4 hours), `reset_islands()` fires
2. **Ranking:** Islands are sorted by their best-ever score (with tiny noise to break ties)
3. **Reset the weak half:** The bottom `num_islands // 2` islands (i.e., 1 island out of 3) are completely destroyed and reinitialized as empty
4. **Founder seeding:** Each reset island receives the **best program from a randomly chosen surviving island** as its founding member

From `buffer.py`:
```python
def reset_islands(self) -> None:
    indices_sorted_by_score = np.argsort(
        self._best_score_per_island +
        np.random.randn(len(self._best_score_per_island)) * 1e-6)
    num_islands_to_reset = self._config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    for island_id in reset_islands_ids:
        self._islands[island_id] = Island(...)  # fresh island
        founder_island_id = np.random.choice(keep_islands_ids)
        founder = self._best_program_per_island[founder_island_id]
        founder_scores = self._best_scores_per_test_per_island[founder_island_id]
        self._register_program_in_island(founder, None, None, island_id, founder_scores)
```

**Practical note for CTRA:** With only 20 LLM evaluations total and the reset period set to 4 hours, the reset mechanism will almost certainly **never trigger** in a typical run. At ~3 samples/iteration, a full run of ~7 iterations completes in minutes. The reset is a vestige of FunSearch's longer-running regime. The islands primarily serve to provide **parallel independent search trajectories** that are randomly selected at prompt time.

### Island selection at prompt time

At each iteration, one island is chosen **uniformly at random**:
```python
def get_prompt(self) -> Prompt:
    island_id = np.random.randint(len(self._islands))
    code, version_generated, data_input, data_output = self._islands[island_id].get_prompt()
    return Prompt(code, version_generated, island_id, data_input, data_output)
```

New programs are registered **only to the island that produced the prompt** (the `island_id` is carried through the entire sample-evaluate-register cycle). The exception is the initial seed program, which is registered to **all islands** (`island_id=None`).

---

## 2. Boltzmann Sampling and Selection Pressure

### How cluster selection works

Within each island, programs are grouped into **clusters** based on their **signature** -- the tuple of per-fold validation scores. Programs with identical score vectors share a cluster. Boltzmann sampling selects which cluster(s) to draw in-context examples from.

**Probability of selecting cluster i:**

```
P_i = exp(s_i / tau_c) / sum_j exp(s_j / tau_c)
```

where:
- `s_i` = the score of cluster i (the mean validation score that defines that cluster)
- `tau_c` = temperature, which anneals over time

**Temperature schedule:**

```
tau_c = T_0 * (1 - (u mod N) / N)
```

where:
- `T_0 = 0.1` (initial temperature, from `cluster_sampling_temperature_init`)
- `N = 30,000` (period, from `cluster_sampling_temperature_period`; the paper says 10,000 but the code says 30,000)
- `u` = `_num_programs` on that island (total programs ever registered)

From `buffer.py`:
```python
temperature = self._cluster_sampling_temperature_init * (
    1 - (self._num_programs % period) / period)
probabilities = _softmax(cluster_scores, temperature)
```

### How selection pressure changes

- **Early iterations** (u small): temperature is close to `T_0 = 0.1`, which is already quite low for a softmax. Even initially, higher-scoring clusters get substantially more probability mass.
- **As u grows:** temperature decreases toward 0, making selection increasingly greedy -- almost always picking the highest-scoring cluster.
- **Cyclical reset:** When `u mod N = 0`, temperature resets to `T_0`, briefly restoring exploration. With N=30,000 and only ~20 total evaluations, this reset never occurs in practice.

**Within-cluster sampling** uses a separate mechanism: programs are sampled with probability inversely proportional to their code length (shorter programs preferred), using softmax over negative normalized lengths with temperature=1.0:
```python
def sample_program(self) -> code_manipulation.Function:
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
        max(self._lengths) + 1e-6)
    probabilities = _softmax(-normalized_lengths, temperature=1.0)
    return np.random.choice(self._programs, p=probabilities)
```

This implements a parsimony pressure (Occam's razor) -- among programs with equal fitness, shorter ones are preferred as in-context examples.

### Two-prompt-style diversity mechanism

The code randomly selects between **two different prompt templates** at each generation:
```python
random_number = random.randint(0, 2)
if random_number == 0:
    # "operations" prompt: emphasizes advanced operators (log, sigmoid, groupby, etc.)
else:
    # "domain" prompt: emphasizes domain knowledge and semantic reasoning
```

This 1/3 vs 2/3 split introduces prompt-level diversity independent of the island model.

---

## 3. Why Only 20 Evaluations? Sample Efficiency

### The budget

`global_max_sample_num = 20` in `main.py`, multiplied by `splits=5` for cross-validation, giving `max_sample_nums = 100` total LLM calls across all folds. Per fold: 20 LLM calls, with `samples_per_prompt = 3` candidates per call, yielding ~7 iterations per fold.

### Why it works with so few evaluations

1. **LLM as a strong prior:** The LLM encodes vast domain knowledge about feature engineering. It doesn't search blindly -- it generates semantically meaningful transformations from the start. The ablation shows removing domain knowledge causes a 37.4% performance drop, confirming the LLM's knowledge is doing most of the heavy lifting.

2. **In-context learning as cheap refinement:** Each iteration provides the LLM with the k=2 best-so-far programs as in-context examples, plus their version history (v0, v1, v2 with docstrings like "Improved version of v1"). The LLM can reason about what worked and what to change. This is far more informative than random mutation.

3. **The search space is constrained:** Programs operate on a fixed input DataFrame and must return an augmented DataFrame. The specification template with `@equation.evolve` constrains the output format. The prompt templates further constrain the operator vocabulary (especially the "operations" prompt which lists specific allowed operators).

4. **Clustering prevents redundancy:** Programs with identical validation score signatures share a cluster. Boltzmann sampling then preferentially selects high-scoring clusters, avoiding wasted evaluations on programs similar to already-evaluated ones.

5. **Feature engineering is additive, not combinatorial:** Unlike architecture search or algorithm discovery, feature engineering builds on the original feature set. Each new program adds/removes features but the base data persists. This makes the landscape smoother -- small improvements compound rather than requiring radical structural changes.

6. **The evaluation is cheap:** Each evaluation trains an XGBoost model with 4-fold cross-validation. This takes seconds, not hours. The bottleneck is LLM calls, not evaluation compute.

### Comparison to MCTS budget requirements

MCTS-based approaches (e.g., AutoCT) typically require 50-200+ evaluations because tree expansion is breadth-first and most nodes are exploratory. LLM-FE's evolutionary approach is more exploitation-heavy -- it rapidly converges by mutating the best-so-far programs rather than systematically exploring a tree.

---

## 4. Multi-Objective vs Single-Objective

### LLM-FE is strictly single-objective

The fitness function is a single scalar: validation accuracy (classification) or negative NRMSE (regression).

```
max_T E(f*(T(X_val)), Y_val)
s.t. f* = argmin_f L(f(T(X_tr)), Y_tr)
```

The bilevel structure (train model, then evaluate) is nested single-objective, not multi-objective.

### Could Pareto fitness replace scalar fitness?

**Architecturally, yes, but several components need modification:**

1. **Cluster signatures already use multi-dimensional scores.** The `Signature` is a tuple of per-fold scores `(s_fold1, s_fold2, ..., s_fold5)`. Replacing this with `(accuracy, interpretability, calibration)` or similar multi-objective tuple is straightforward.

2. **Boltzmann sampling needs rethinking.** Currently, `cluster_scores` is a 1D array of scalar scores fed to softmax. For Pareto, you'd need:
   - Replace scalar ranking with **Pareto rank** (non-domination level)
   - Use Pareto rank as the "score" for Boltzmann sampling, OR
   - Use **hypervolume contribution** as the scalar proxy for each cluster
   - Or switch to **NSGA-II style tournament selection** instead of Boltzmann

3. **Best-program tracking needs generalization.** `_best_score_per_island` is a scalar. For Pareto, each island would maintain a **Pareto front** instead of a single best.

4. **Island reset logic needs adjustment.** Currently resets the island with the lowest scalar best. For Pareto, reset based on **hypervolume** of each island's Pareto front.

5. **Prompt construction needs multi-objective context.** The in-context examples would need to showcase programs from different regions of the Pareto front, not just the top-2 by a single metric.

**Estimated effort:** Moderate. The core loop (generate -> evaluate -> register -> select) is clean and modular. The main work is in `buffer.py` (cluster scoring, island reset) and the prompt templates (communicating multi-objective tradeoffs to the LLM).

---

## 5. Reported Results

### Datasets

**Classification (19 datasets):** adult, arrhythmia, balance-scale, bank-marketing, breast-w, blood-transfusion, car, cdc-diabetes, cmc, communities, covtype, credit-g, eucalyptus, heart, jungle_chess, myocardial, pc1, tic-tac-toe, vehicle

**Regression (10 datasets):** airfoil_self_noise, bike, cpu_small, crab, diamonds, forest-fires, housing, insurance, plasma_retinol, wine

Sources: OpenML, UCI, Kaggle. Sizes range from ~300 (balance-scale) to 581K (covtype). Dimensionality up to 279 features (arrhythmia).

### Baselines compared

| Method | Type | Notes |
|--------|------|-------|
| Base (no FE) | -- | Raw features with XGBoost |
| AutoFeat | Classical | Automated feature generation |
| OpenFE | Classical | Effective feature generation via expansion-reduction |
| CAAFE | LLM-based | Context-aware automated feature engineering |
| FeatLLM | LLM-based | Feature engineering with LLM prompting |
| OCTree | LLM-based | Optimization with code generation |
| LLM-FE | LLM+Evolutionary | This paper |

### Key results

**Classification (XGBoost):**
- LLM-FE mean rank: **1.47** (best)
- Next best: OpenFE 3.26, CAAFE 3.31
- Largest improvements: jungle_chess (0.869 -> 0.969), balance-scale (0.856 -> 0.990)

**Regression (XGBoost):**
- LLM-FE mean rank: **1.00** (won every single dataset)
- Next best: OpenFE 2.20

**Cross-model generalization (Table 4):**
- Works with XGBoost, MLP, and TabPFN
- GPT-3.5 > Llama 3.1-8B as the LLM backbone
- Even Llama 3.1-8B (open-source, local) provides meaningful improvements

### No MCTS comparison

The paper does **not** compare against any MCTS-based approach (e.g., AutoCT, MCTS-AHD, or FunSearch). The baselines are all feature-engineering-specific methods. This is a gap -- we cannot directly assess whether the island evolutionary approach outperforms MCTS for this task from this paper alone.

---

## 6. Crossover Between Feature Programs

### LLM-FE does NOT implement explicit crossover

The paper mentions "adaptive mutation and crossover operations" only in the related work section when describing prior evolutionary LLM approaches. LLM-FE itself does **not** implement crossover.

### What happens instead (implicit crossover via in-context learning)

The prompt includes k=2 programs sampled from (potentially different) clusters. The LLM is asked to produce an "Improved version" that builds on these examples. This creates an **implicit crossover** effect:

```
# In the prompt:
def modify_features_v0(df):
    '''First approach: created interaction features...'''
    # [body of program A from cluster X]

def modify_features_v1(df):
    '''Improved version of modify_features_v0...'''
    # [body of program B from cluster Y]

def modify_features_v2(df):
    '''Improved version of modify_features_v1. Think and suggest new features.'''
    # LLM generates this, potentially combining ideas from v0 and v1
```

Because v0 and v1 can come from **different clusters** (selected independently via Boltzmann sampling), the LLM may combine features from both programs -- effectively crossing them over. But this is emergent behavior, not a controlled genetic operator.

### Could explicit crossover be added?

Yes. The simplest approach:
1. Sample two parent programs from different clusters/islands
2. Prompt the LLM: "Combine the best features from Program A and Program B into a single program"
3. This would be a separate operator alongside the current mutation-style generation

The code structure in `buffer.py` already supports sampling multiple programs (`functions_per_prompt=2`), so the infrastructure exists.

---

## 7. Mutation Mechanism

### LLM-as-mutator (implicit mutation)

LLM-FE does not define explicit mutation operators (e.g., "swap feature X with feature Y" or "change operator from log to sqrt"). Instead, **the LLM itself is the mutation operator.**

The process:
1. **Select island** uniformly at random
2. **Sample k=2 programs** from the island via Boltzmann cluster selection + within-cluster length-biased sampling
3. **Construct prompt** with these programs as versioned in-context examples (v0, v1), plus dataset description, feature metadata, serialized data examples, and one of two prompt templates (operations-focused or domain-focused)
4. **LLM generates b=3 candidate programs** as the "next version" (v2)
5. **Each candidate is evaluated** independently and registered if valid

The prompt explicitly instructs:
- **Domain prompt (2/3 probability):** "Use your domain knowledge to derive features that capture meaningful patterns, trends, or relationships inherent in the data"
- **Operations prompt (1/3 probability):** "Create meaningful and insightful features using advanced operators [list of specific operators: log, sigmoid, groupby, etc.]. Avoid basic arithmetic."

The docstring of the function to generate says: `"Improved version of modify_features_v1. Think and suggest new features."` -- this implicitly asks for mutation (improve upon the previous version) rather than generation from scratch.

### Mutation intensity

LLM temperature is set to **0.8** for generation, balancing creativity and adherence. With `samples_per_prompt=3`, each prompt yields 3 diverse candidates from the same in-context examples.

### What kinds of mutations occur in practice

Based on the prompt structure and operator list:
- **Feature addition:** New derived features (e.g., `df['BP_HR_ratio'] = df['RestingBP'] / df['MaxHR']`)
- **Feature deletion:** Removing uninformative columns (`df.drop(columns=[...])`)
- **Operator substitution:** Changing `log` to `sqrt`, `mean` to `median` in groupby operations
- **Feature combination:** Creating interaction terms, polynomial features
- **Domain-driven features:** Clinically meaningful composites based on LLM's medical knowledge

---

## 8. Relevance to CTRA / AutoCT

### Advantages of LLM-FE's approach for CTRA

1. **Extreme sample efficiency (20 evals):** Clinical trial feature engineering is expensive if it involves API calls to external databases. LLM-FE's 20-evaluation budget is much cheaper than MCTS's ~140 evaluations.

2. **Feature programs are interpretable:** Each program is a Python function with docstrings. This aligns with CTRA's SHAP explainability requirement -- you get both SHAP values on features AND natural-language rationale for why features were created.

3. **Works with XGBoost and TabPFN:** Exactly the models in CTRA's stack.

4. **LLM backbone is swappable:** Code uses OpenAI API but the interface is generic. Claude Opus 4.6 could replace GPT-3.5 trivially.

### Disadvantages / gaps for CTRA

1. **No multi-objective support:** CTRA needs to balance accuracy, calibration, and interpretability. LLM-FE is single-objective.

2. **No RAG integration:** LLM-FE relies solely on the LLM's parametric knowledge plus serialized data examples. CTRA's LinearRAG retrieval over ClinicalTrials.gov/PubMed/DrugBank is not natively supported. Would need to inject retrieved context into the prompt.

3. **No MCTS comparison:** Cannot directly assess whether LLM-FE's evolutionary search outperforms the Pareto MCTS planned for CTRA.

4. **Feature engineering only, not full pipeline:** LLM-FE generates features but doesn't do trial-level reasoning, evidence retrieval, or structured prediction. It would be one component in CTRA, not a replacement.

5. **Island reset is irrelevant at 20 evals:** The 3-island model provides marginal benefit at this budget. Most of the value comes from the LLM's domain knowledge + in-context learning, not the evolutionary search structure.

### Hybrid possibility: LLM-FE's evolutionary loop + CTRA's Pareto MCTS

A compelling hybrid:
- Use LLM-FE's prompt structure and in-context demonstration approach for feature generation
- Replace the island model with CTRA's Pareto MCTS for multi-objective search
- Inject LinearRAG context into the feature generation prompt
- Keep the Boltzmann sampling idea for node selection within MCTS (as an alternative to UCB)

---

## 9. Key Hyperparameters Summary

| Parameter | Default | Location |
|-----------|---------|----------|
| `num_islands` | 3 | `config.py` |
| `functions_per_prompt` | 2 | `config.py` (k in paper) |
| `samples_per_prompt` | 3 | `config.py` (b in paper) |
| `global_max_sample_num` | 20 | `main.py` |
| `cluster_sampling_temperature_init` | 0.1 | `config.py` (T_0 in paper) |
| `cluster_sampling_temperature_period` | 30,000 | `config.py` (N; paper says 10,000) |
| `reset_period` | 14,400 sec | `config.py` |
| `evaluate_timeout_seconds` | 30 | `config.py` |
| `LLM generation temperature` | 0.8 | sampler code |
| `splits` (cross-validation folds) | 5 | `main.py` |
| `within-cluster length temperature` | 1.0 | `buffer.py` (hardcoded) |

---

## Sources

- [arXiv paper (HTML)](https://arxiv.org/html/2503.14434)
- [arXiv abstract](https://arxiv.org/abs/2503.14434)
- [GitHub repository](https://github.com/nikhilsab/LLMFE)
- [HuggingFace paper page](https://huggingface.co/papers/2503.14434)
- [OpenReview discussion](https://openreview.net/forum?id=2l4lDof7O8)
