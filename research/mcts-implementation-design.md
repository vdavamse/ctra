# AutoCT Search Strategy: Implementation Design

> **⚠ SUPERSEDED** — This document specifies a design with 3 objectives (cross-phase weighted ROC-AUC + calibration/ECE + parsimony). The current implementation uses per-phase isolation with 2 objectives (per-phase ROC-AUC + parsimony): the weighted AUC became each phase's own ROC-AUC, calibration was dropped, and parsimony was kept. See "As Implemented" section below for what was built.

> Synthesized from codebase analysis, convergence validation, approach comparison, and augmentation compatibility research (2026-03-26). This document resolves the Pareto MCTS vs LLM-FE debate and specifies a concrete implementation plan for multi-objective feature search in CTRA.

---

## 1. Recommended Approach

### Decision: Pareto Fitness MVP on Existing MCTS, Then Head-to-Head

The research doc recommends Pareto MCTS first; the approach-challenger recommends LLM-FE first. **Both are partially right, and both understate the risk of premature commitment.** The correct move is to validate the multi-objective framework cheaply before committing to either search strategy.

**The core insight:** Multi-objective fitness and search strategy are orthogonal concerns. We can decouple them:

1. **Phase 2a (MVP):** Bolt Pareto fitness (hypervolume-based ranking) onto the existing scalar MCTS. Keep UCT, keep the loop, but evaluate 3 objectives and rank by hypervolume contribution. Simultaneously implement multi-fidelity evaluation. This validates the objective definitions and multi-fidelity proxy at minimal code risk.

2. **Phase 2b (parallel tracks):** Implement both Pareto MCTS (vector UCB + Pareto selection) and LLM-FE (evolutionary program search + Pareto fitness). Run head-to-head on TrialBench Phase II (hardest phase, most deceptive landscape).

3. **Phase 2c (commit):** Pick the winner. Add augmentations (memory, predictive scoring, OCTree) to the winning strategy.

### Justification Against Alternatives

| Strategy | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Pareto MCTS (full)** | Theoretically strongest for deceptive landscapes; composes with AB-MCTS; leverages existing tree infrastructure | At 140 raw evals, degenerates to random selection unless multi-fidelity works (convergence research confirmed); PMMG code needs clean-room rewrite | Build it, but only after multi-fidelity is validated |
| **LLM-FE (full)** | Most sample-efficient (20 evals); same problem domain; implicit crossover; $300-800/run | Replaces MCTS entirely (high effort); untested on clinical trial data; no published multi-objective variant | Build it as challenger; lower risk per run but higher implementation risk |
| **AIDE + Pareto post-hoc** | Simplest to implement; proven on Kaggle | Structurally greedy — cannot handle deceptive landscapes; no multi-objective selection during search | Phase 1 baseline only, not a serious contender |
| **Pareto fitness on existing MCTS (MVP)** | Minimal code change (~100 lines); validates objectives and multi-fidelity before committing to either strategy; preserves optionality | Scalar UCT with Pareto post-ranking is weaker than true vector UCB selection | This is the starting point, not the endpoint |

### Why Not LLM-FE First (Approach-Challenger's Recommendation)

The challenger's argument is compelling on cost: 20 evals at $15-20 = $300-400/run vs $2,100-2,800 for MCTS. But it assumes:

1. **LLM-FE's sample efficiency transfers to clinical trials.** The 20-eval result is on standard tabular ML benchmarks. Clinical trial feature engineering has higher epistasis (feature interactions are non-obvious) and delayed reward (adding a feature that helps Phase II may hurt Phase I). The landscape may require deeper exploration than LLM-FE's evolutionary mutations provide.

2. **Implicit crossover substitutes for explicit multi-objective selection.** LLM-FE's island model provides diversity, but diversity along *feature type* dimensions, not *objective* dimensions. Two islands may converge to the same objective trade-off while exploring different features. Pareto selection forces diversity along the objective surface.

3. **Replacing MCTS entirely is low-risk.** AutoCT's agent pipeline (`agent.py`, 2765 lines) is deeply coupled to the tree structure. LLM-FE requires a new evolutionary loop, new population management, and new mutation operators. This is a larger rewrite than modifying `treesearch.py`.

The right answer: **build both, compare empirically, commit to the winner.**

---

## As Implemented (Current State)

The recommendation to "build both" was set aside; Pareto MCTS was built with modifications to the original design:

- **Per-phase isolation:** Each clinical trial phase (I, II, III) runs a completely independent MCTS tree with its own objectives and Pareto front. The original design expected cross-phase weighted AUC as the accuracy objective. Per-phase isolation (each phase is an independent `Task`, matching AutoCT) makes a cross-phase weighted AUC undefined within one tree; the accuracy objective is the phase's own validation ROC-AUC.

- **Two objectives, not three:** Implementation uses accuracy (validation ROC-AUC) + parsimony (feature count efficiency). The original design's three objectives were weighted AUC, calibration (ECE, expected calibration error) and parsimony; calibration was never implemented, and `MCTSConfig.objectives` is typed `list[Literal["accuracy", "parsimony"]]`, so no third objective can be enabled without a code change. No rationale for dropping it is recorded.

- **Reference point `[0.5, 0.0]`:** Each phase's MCTS ranks hypervolume contributions against `[0.5, 0.0]` — accuracy at the ROC-AUC chance baseline, parsimony at its floor (final selection and best-on-path; in-search child selection via `pareto_select` still measures from the origin, see README). This avoids inflating the rank of any one-feature root that happens to get a random accuracy bump. See issue #18 for the full rationale.

- **No multi-fidelity in search:** Multi-fidelity evaluation (25%, 50%, 75%, 100% data schedules per rollout) was not implemented in the search layer (only `mlops/retraining.py` has a fidelity notion); no decision on it is recorded.

For full implementation details, see `src/ctra/config/settings.py` (MCTSConfig, objectives list), `src/ctra/search/objectives.py` (FeatureSet, ObjectiveResult), and `docs/mcts-implementation-infographic.md` (§ "The Two Per-Phase Objectives").

---

## 2. Multi-Objective Definition

### Three Objectives (Not Five)

The convergence research is definitive: **5 objectives at 140 evaluations (or even 400-700 effective evals with multi-fidelity) is not viable.** With 5 objectives, 60-80% of solutions are non-dominated, and Pareto selection degenerates to random choice. Three objectives maintain selection pressure while capturing the key trade-offs.

### Objective Specification

| # | Objective | Metric | Direction | Measurement | Why This One |
|---|-----------|--------|-----------|-------------|-------------|
| 1 | **Aggregate performance** | Weighted mean ROC-AUC across phases | Maximize | `0.2 * auc_p1 + 0.5 * auc_p2 + 0.3 * auc_p3` | Phase II weighted highest (hardest, most valuable). Single aggregate avoids 3 correlated objectives eating the budget. |
| 2 | **Calibration** | Expected Calibration Error (ECE), 10 bins | Minimize | `ece = mean(|accuracy_bin - confidence_bin|)` | Clinical decisions require trustworthy probabilities, not just ranking. A model with AUC 0.70 and ECE 0.05 is more useful than AUC 0.75 and ECE 0.20. |
| 3 | **Parsimony** | Feature count (inverse) | Minimize | `1.0 / n_features` (or simply `-n_features` normalized) | Fewer features = lower RAG cost per prediction, less overfitting, more interpretable SHAP plots. Each feature costs ~$0.50-1.00 in RAG retrieval at inference time. |

### Why These Three

**Performance + calibration + parsimony** captures three fundamentally different desiderata that cannot be optimized simultaneously:

- High performance often requires more features (conflicts with parsimony)
- High calibration often requires simpler models (conflicts with performance)
- Few features often sacrifice discriminative power (conflicts with performance)

This creates a genuine Pareto surface with meaningful trade-offs. Decision-makers get a menu:

- **Clinical dashboard model:** 5-8 features, ECE < 0.08, AUC ~0.65 — interpretable, cheap to run
- **High-stakes model:** 20-30 features, ECE < 0.12, AUC ~0.73 — maximum accuracy
- **Balanced model:** 12-15 features, ECE < 0.10, AUC ~0.70 — practical sweet spot

### Objectives Considered and Rejected

| Objective | Why Rejected |
|-----------|-------------|
| Phase-specific AUC (3 separate) | Highly correlated (~0.6-0.8 between phases); consumes 3 of 3-5 objective slots for correlated metrics. Weighted aggregate is more efficient. |
| SHAP stability | Expensive to compute (requires multiple bootstrap runs); correlates with parsimony. Defer to Phase 3. |
| Robustness (OOD) | Requires held-out therapeutic areas; not available in TrialBench setup. Defer to Phase 3. |
| Interpretability (tree depth) | Subjective, hard to automate. Parsimony is a proxy. |

### Phase Weighting Rationale

The `0.2 / 0.5 / 0.3` weights for Phase I / II / III reflect:

- **Phase II (0.5):** Highest failure rate (65-70%), most expensive to get wrong, most deceptive landscape (efficacy endpoints are noisy). This is where CTRA adds the most value.
- **Phase III (0.3):** High cost per failure ($50M-300M+) but higher base rate of success. Moderate weight.
- **Phase I (0.2):** Safety endpoints are qualitatively different (toxicity, not efficacy). Lower weight because Phase I features are structurally different and Phase I data is smaller.

These weights are configurable and should be validated against stakeholder priorities.

---

## 3. Code-Level Design

### Architecture Decision: Modify treesearch.py, Not Rewrite

The codebase analysis confirms `treesearch.py` is 397 lines with clean separation between the MCTS loop and the agent pipeline. The core changes are:

1. `MCTTreeNode.total_reward`: `float` → `np.ndarray` of shape `(3,)`
2. New `pareto_select()` function replacing scalar UCT argmax
3. `_backpropagate()`: vector addition instead of scalar addition
4. `_simulate()`: return vector reward instead of scalar max
5. New `hypervolume_contribution()` for solution ranking

### Modified Data Structures

```python
from typing import NamedTuple
import numpy as np

# --- Objectives ---
N_OBJECTIVES = 3
OBJECTIVE_NAMES = ["weighted_auc", "calibration", "parsimony"]
# Reference point for hypervolume (worst acceptable values)
REFERENCE_POINT = np.array([0.5, 0.3, 0.0])  # AUC=0.5, ECE=0.3, parsimony=0

class MCTTreeNode(NamedTuple):
    """Modified from AutoCT's original scalar node."""
    state: Any                          # Feature set state (unchanged)
    parent: Optional['MCTTreeNode']     # (unchanged)
    children: list                      # (unchanged)
    total_reward: np.ndarray            # CHANGED: float → np.ndarray shape (3,)
    visit_count: int                    # (unchanged)

    @staticmethod
    def create(state, parent=None):
        return MCTTreeNode(
            state=state,
            parent=parent,
            children=[],
            total_reward=np.zeros(N_OBJECTIVES),  # CHANGED
            visit_count=0,
        )
```

### Modified MCTS Loop (Pseudocode)

```python
def mcts_search(root_state, budget=140, exploration_weight=1.0):
    """Multi-objective MCTS with Pareto selection.

    Phase 2a MVP: Uses Pareto ranking for solution output but
    scalar UCT (weighted scalarization) for tree selection.
    Phase 2b upgrade: Replace scalar UCT with vector UCB + Pareto filtering.
    """
    root = MCTTreeNode.create(root_state)
    pareto_front = []  # List of (node, reward_vector) on current front

    for iteration in range(budget):
        # --- SELECT ---
        node = root
        while node.children and not node.is_terminal():
            if USE_PARETO_UCB:  # Phase 2b: full Pareto MCTS
                node = pareto_select(node, exploration_weight)
            else:  # Phase 2a MVP: scalarized UCT
                node = scalar_uct_select(node, exploration_weight)

        # --- EXPAND ---
        # LLM proposes ~5 feature operations (unchanged from AutoCT)
        child_states = llm_propose_operations(node.state)
        children = [MCTTreeNode.create(s, parent=node) for s in child_states]
        node.children.extend(children)

        # --- EVALUATE ---
        # Pick one child to evaluate (e.g., random from new children)
        child = random.choice(children)

        if USE_MULTI_FIDELITY:
            # Quick eval on 20% data subset
            proxy_reward = evaluate_proxy(child.state)  # ~1-2 min, ~$3-4
            if not passes_proxy_threshold(proxy_reward, pareto_front):
                # Dominated even on proxy — skip full eval
                reward_vector = proxy_reward * PROXY_SCALING
            else:
                # Full evaluation on complete dataset
                reward_vector = evaluate_full(child.state)  # ~5-15 min, ~$15-20
        else:
            reward_vector = evaluate_full(child.state)

        # reward_vector = np.array([weighted_auc, 1-ece, parsimony_score])

        # --- UPDATE PARETO FRONT ---
        pareto_front = update_pareto_front(pareto_front, child, reward_vector)

        # --- BACKPROPAGATE ---
        backpropagate(child, reward_vector)

    return pareto_front


def evaluate_full(feature_state) -> np.ndarray:
    """Run full agent pipeline, return 3-objective reward vector."""
    # Existing AutoCT pipeline: LLM feature build → RAG → ML training
    results = run_agent_pipeline(feature_state)  # Returns dict with metrics

    # Objective 1: Weighted aggregate AUC
    weighted_auc = (
        0.2 * results['phase1_roc_auc'] +
        0.5 * results['phase2_roc_auc'] +
        0.3 * results['phase3_roc_auc']
    )

    # Objective 2: Calibration (inverted — higher is better)
    calibration = 1.0 - results['expected_calibration_error']

    # Objective 3: Parsimony (normalized — higher is better)
    n_features = len(feature_state.features)
    parsimony = 1.0 - (n_features / MAX_FEATURES)  # e.g., MAX_FEATURES=50

    return np.array([weighted_auc, calibration, parsimony])
```

### Pareto Selection (Phase 2b)

```python
def pareto_select(node, exploration_weight):
    """Vector UCB + Pareto dominance filtering.

    Adapted from PMMG mcts.py:49-87, clean-room rewrite.
    """
    if not node.children:
        return node

    parent_visits = node.visit_count
    ucb_vectors = []

    for child in node.children:
        if child.visit_count == 0:
            # Unvisited: infinite UCB (explore first)
            ucb_vectors.append(np.full(N_OBJECTIVES, np.inf))
        else:
            mean_reward = child.total_reward / child.visit_count
            exploration = exploration_weight * np.sqrt(
                2 * np.log(parent_visits) / child.visit_count
            )
            ucb_vectors.append(mean_reward + exploration)

    # Pareto filter: remove dominated children
    non_dominated_indices = pareto_filter(ucb_vectors)

    # Random selection from non-dominated pool
    selected_idx = random.choice(non_dominated_indices)
    return node.children[selected_idx]


def pareto_filter(vectors):
    """Return indices of non-dominated vectors. O(n^2) but n<=5."""
    n = len(vectors)
    non_dominated = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(vectors[j], vectors[i]):
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)
    return non_dominated


def dominates(a, b):
    """True if a dominates b (a >= b on all objectives, a > b on at least one)."""
    return np.all(a >= b) and np.any(a > b)


def backpropagate(node, reward_vector):
    """Undiscounted vector reward to all ancestors (same as PMMG)."""
    current = node
    while current is not None:
        current = current._replace(
            total_reward=current.total_reward + reward_vector,
            visit_count=current.visit_count + 1,
        )
        current = current.parent
```

### Hypervolume for Solution Ranking

```python
def update_pareto_front(front, node, reward_vector):
    """Maintain the global Pareto front of evaluated solutions."""
    # Check if new solution is dominated by any existing
    for _, existing_reward in front:
        if dominates(existing_reward, reward_vector):
            return front  # Dominated, don't add

    # Remove solutions dominated by the new one
    front = [(n, r) for n, r in front if not dominates(reward_vector, r)]
    front.append((node, reward_vector))
    return front


def hypervolume_contribution(front, reference_point):
    """Compute hypervolume contribution of each solution.

    Used for final ranking and reporting, not during search.
    With 3 objectives and ~10-30 Pareto-optimal solutions,
    exact computation is fast (< 1ms).
    """
    # Use pymoo or pygmo for exact 3D hypervolume
    from pymoo.indicators.hv import HV

    points = np.array([r for _, r in front])
    hv = HV(ref_point=reference_point)

    contributions = []
    total_hv = hv(points)
    for i in range(len(points)):
        reduced = np.delete(points, i, axis=0)
        reduced_hv = hv(reduced) if len(reduced) > 0 else 0.0
        contributions.append(total_hv - reduced_hv)

    return contributions
```

### Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/lfe/impl/treesearch.py` | Replace scalar reward with vector; add Pareto selection, vector backprop | ~150 lines modified/added |
| `src/lfe/impl/evaluator.py` (new) | Multi-objective evaluation wrapper; ECE computation; parsimony scoring | ~80 lines |
| `src/lfe/impl/pareto.py` (new) | Pareto dominance, filtering, hypervolume utilities | ~60 lines |
| `src/lfe/impl/multi_fidelity.py` (new) | Proxy evaluation on 20% data; threshold-based filtering | ~100 lines |
| `src/lfe/impl/agent.py` | Minimal changes: pass phase-specific eval config | ~10 lines modified |

**Total new/modified code: ~400 lines** (Phase 2a MVP: ~200 lines; Phase 2b full Pareto: +200 lines)

---

## 4. Augmentation Stack

### Minimum Viable Stack (9-14 days, implements alongside Phase 2a)

| Priority | Augmentation | Effort | Impact | Dependency |
|----------|-------------|--------|--------|------------|
| 1 | **Multi-fidelity evaluation** | LOW (3-4 days) | HIGH | None — prerequisite for Pareto MCTS viability |
| 2 | **OCTree decision tree feedback** | LOW (2-3 days) | MEDIUM | None — strategy-agnostic |
| 3 | **Predictive scoring** (KompeteAI-style) | MEDIUM (4-7 days) | HIGH | Requires ~15 anchor evaluations |

**Multi-fidelity** is not optional — it is a prerequisite. Without it, Pareto MCTS at 140 raw evals degenerates. With 20% data proxy + selective full evaluation, effective budget rises to 400-700 evals.

**OCTree** gives the LLM structured feedback about which features help. Instead of "ROC-AUC = 0.71", the LLM sees "drug_target_count > 3 AND phase = II AND criteria_count < 15 → predict failure (confidence: 0.83)". This costs almost nothing (fit a shallow decision tree on existing predictions) and improves LLM proposal quality.

**Predictive scoring** is the second critical cost-reduction mechanism. After ~15 full evaluations, use Opus 4.6 with few-shot examples to predict whether a proposed feature set is worth full evaluation. KompeteAI achieved 6.9x throughput improvement. Expected accuracy: 70-80% correlation with actual performance (sufficient for filtering, not for final ranking).

### Full Stack (adds 10-15 days on top of MVP)

| Priority | Augmentation | Effort | Impact | When |
|----------|-------------|--------|--------|------|
| 4 | **Selectively scoped memory** (ML-Master) | MEDIUM (5-7 days) | HIGH | After MVP validated |
| 5 | **Verbal gradients** (ReEvo-style) | MEDIUM (3-4 days) | MEDIUM | After OCTree |
| 6 | **AB-MCTS adaptive branching** | MEDIUM (4-5 days) | MEDIUM | If Pareto MCTS wins head-to-head |
| 7 | **Solution merging** (KompeteAI) | HIGH (7-10 days) | HIGH | Phase 3 only |

### Synergy Chain

```
OCTree feedback → Verbal gradients → Memory → Merging
     ↓                  ↓                ↓          ↓
 Better LLM       Richer signal     Cross-branch   Recombination
  proposals      for mutations      learning       of best parts
```

Each augmentation feeds the next:
- OCTree gives structured feature importance → verbal gradients can compare "why did features A outperform B?"
- Verbal gradients produce textual insights → memory stores and retrieves them across branches
- Memory enables cross-branch knowledge → merging recombines structurally different solutions

### Strategy-Agnostic Augmentations

These work regardless of whether Pareto MCTS or LLM-FE wins the head-to-head:

- Multi-fidelity evaluation
- OCTree decision tree feedback
- Predictive scoring
- Verbal gradients

**Tree-only augmentations** (require MCTS structure): AB-MCTS branching, selectively scoped memory (adaptable to LLM-FE with modifications), solution merging.

---

## 5. Risk Analysis

### Critical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Multi-fidelity proxy is unreliable** — 20% data subset doesn't correlate with full-dataset performance | MEDIUM | CRITICAL — without multi-fidelity, Pareto MCTS is nonviable at 140 evals | Validate proxy correlation (Spearman rho > 0.7) on first 30 full evals before switching to proxy. If proxy fails, fall back to LLM-FE (which needs only 20 evals). |
| **3 objectives still too many** — even with 400-700 effective evals, Pareto selection pressure is insufficient | LOW-MEDIUM | HIGH — Pareto MCTS adds complexity without benefit | Monitor non-dominated fraction during runs. If >50% of solutions are non-dominated at eval 100, scalarize to 2 objectives (drop parsimony, handle via constraint). |
| **Phase-weighted AUC masks phase-specific failures** — aggregate metric hides poor Phase I performance | MEDIUM | MEDIUM — Phase I model is unreliable | Track per-phase AUC as diagnostic metrics (not objectives). Alert if any phase AUC drops below 0.55. |
| **LLM-FE doesn't transfer to clinical trials** — evolutionary mutations fail on domain-specific features | MEDIUM | MEDIUM — LLM-FE track is a dead end | Head-to-head benchmark catches this in Phase 2b. If LLM-FE fails, we only lose ~2 weeks of implementation. |
| **ECE is unstable on small test sets** — clinical trial datasets have ~200-500 test samples per phase | HIGH | MEDIUM — calibration objective is noisy | Use 10-fold cross-validated ECE instead of single holdout. Smooth with Platt scaling before measurement. |
| **Predictive scoring is too inaccurate** — LLM predictions don't correlate with actual performance | MEDIUM | LOW — we just skip more to full eval | Start conservatively: only filter solutions predicted to be in bottom 25%. Tighten threshold as anchor set grows. |

### Convergence Risks Specific to Pareto MCTS

| Risk | Indicator | Response |
|------|-----------|----------|
| Non-dominated fraction > 60% at eval 50 | Pareto selection is near-random | Reduce to 2 objectives; add epsilon-dominance (relax dominance by epsilon per objective) |
| Hypervolume plateaus for 30+ consecutive evals | Search is stuck | Increase exploration_weight from 1.0 to 2.0; inject diversity via random restarts |
| One objective dominates UCB variance | All selection driven by one noisy objective | Normalize objectives to [0, 1] before UCB computation; use rank-based normalization |
| Multi-fidelity proxy Spearman rho < 0.5 | Proxy is misleading the search | Disable proxy; fall back to full evaluation only; reduce budget expectations |

### What We Won't Try to Mitigate

- **LLM backbone cost** — $15-20/eval is acceptable given the research budget (~$10K Year 1). Cost reduction mechanisms (multi-fidelity, predictive scoring) target throughput, not per-eval cost.
- **PMMG code quality** — clean-room rewrite is the mitigation. We will not patch the existing O(n^2) SMILES-coupled code.
- **AutoCT's NamedTuple immutability** — at ~140 evals, the overhead of creating new tuples on backpropagation is negligible. No need to refactor to mutable nodes.

---

## 6. Phase Plan

### Phase 2a: Multi-Objective MVP (3 weeks / 15 days)

**Goal:** Validate the 3-objective framework and multi-fidelity proxy on TrialBench.

| Week | Task | Output | Days |
|------|------|--------|------|
| 1 | Implement `pareto.py` (dominance, filtering, hypervolume) | Unit-tested Pareto utilities | 2 |
| 1 | Implement `evaluator.py` (3-objective scoring: weighted AUC, ECE, parsimony) | Objective measurement on existing AutoCT results | 2 |
| 1 | Implement OCTree feedback (shallow DT → text for LLM) | Decision tree rendering in evaluation loop | 1 |
| 2 | Modify `treesearch.py` — vector rewards, Pareto ranking of final output (keep scalar UCT for selection) | MVP MCTS with multi-objective output | 3 |
| 2 | Implement `multi_fidelity.py` — 20% data proxy with correlation validation | Proxy evaluation pipeline | 2 |
| 3 | Run on TrialBench Phase II | Baseline results: Pareto front quality, proxy correlation, per-objective scores | 3 |
| 3 | Analyze results: is proxy reliable? Are 3 objectives well-separated? | Go/no-go decision for Phase 2b | 2 |

**Milestone:** Pareto front of 5-15 solutions from existing MCTS with objective scores. Proxy correlation (Spearman rho) measured. Go/no-go for Pareto selection.

**Cost:** ~$700-1,000 (2-3 full TrialBench runs at ~$300-400 each)

### Phase 2b: Head-to-Head (4 weeks / 20 days)

**Goal:** Empirically determine whether Pareto MCTS or LLM-FE is the better search strategy.

| Week | Task | Output | Days |
|------|------|--------|------|
| 1-2 | Implement full Pareto MCTS (vector UCB, Pareto selection, vector backprop) | `treesearch.py` v2 with Pareto selection | 5 |
| 1-2 | Implement LLM-FE evolutionary loop (3-island model, Boltzmann sampling, LLM mutation) with Pareto fitness | New `evolutionary_search.py` | 7 |
| 2 | Implement predictive scoring (few-shot LLM scoring with anchor examples) | Scoring module, integrated into both strategies | 4 |
| 3 | Run head-to-head: Pareto MCTS vs LLM-FE on TrialBench Phase II (3 runs each for variance) | Comparative results: hypervolume, per-objective Pareto fronts, wall-clock time, cost | 2 |
| 3-4 | Implement selectively scoped memory (ML-Master-style) for winning strategy | Cross-branch insight injection | 5 |
| 4 | Re-run winner + memory on full TrialBench (all phases) | Production-quality results | 2 |

**Milestone:** Winner selected. Full Pareto front on TrialBench all phases with winning strategy.

**Decision criteria for head-to-head:**

| Metric | Weight | Measurement |
|--------|--------|-------------|
| Hypervolume of Pareto front | 40% | Higher = better multi-objective coverage |
| Best weighted AUC on front | 25% | Must not sacrifice peak performance |
| Cost per run | 20% | $/run including all evaluations |
| Variance across 3 runs | 15% | Lower = more reliable |

**Cost:** ~$3,000-4,500 (6 full runs at ~$300-600 each, plus development iterations)

### Phase 2c: Augmentation Integration (2 weeks / 10 days)

**Goal:** Add remaining augmentations to winning strategy.

| Week | Task | Output | Days |
|------|------|--------|------|
| 1 | Add verbal gradients (pairwise comparison prompts) | Richer LLM feedback on why features help/hurt | 3 |
| 1 | Add AB-MCTS adaptive branching (if Pareto MCTS won) OR crossover tuning (if LLM-FE won) | Strategy-specific enhancement | 3 |
| 2 | Ablation study: run with/without each augmentation | Contribution measurement per augmentation | 4 |

**Cost:** ~$1,500-2,500

### Phase 3: Production Integration (per existing roadmap)

Solution merging (KompeteAI), MAP-Elites archive, async parallelism — these are Phase 3 items that build on the winning strategy from Phase 2.

### Total Phase 2 Timeline

| Sub-phase | Duration | Cost | Cumulative |
|-----------|----------|------|------------|
| 2a: MVP | 3 weeks (15 days) | ~$850 | $850 |
| 2b: Head-to-head | 4 weeks (20 days) | ~$3,750 | $4,600 |
| 2c: Augmentations | 2 weeks (10 days) | ~$2,000 | $6,600 |
| **Total** | **9 weeks (45 days)** | **~$6,600** | |

This is 4 weeks longer than the original Phase 2 estimate (5 weeks / 24.5 days). The additional time buys empirical validation instead of betting on a single strategy.

---

## 7. Cost Model

### Per-Run Cost Breakdown

| Component | Phase 2a (MVP) | Phase 2b (Pareto MCTS) | Phase 2b (LLM-FE) |
|-----------|---------------|----------------------|-------------------|
| Raw evaluations | 140 | 140 | 20 |
| Multi-fidelity proxy evals | 0 | ~300-400 (cheap) | 0 |
| Full evals (after filtering) | 140 | ~80-100 | 20 |
| Predictive scoring calls | 0 | ~200 | 0 |
| Cost per full eval | $15-20 | $15-20 | $15-20 |
| Cost per proxy eval | — | $3-5 | — |
| Cost per predictive score | — | $0.10-0.20 | — |
| **Total per run** | **$2,100-2,800** | **$1,800-2,500** | **$300-400** |
| **Effective evaluations** | 140 | 400-700 | 20 |
| **Cost per effective eval** | $15-20 | $3-6 | $15-20 |

### Expected Improvement Over Baseline

| Metric | Current AutoCT (scalar MCTS) | Phase 2a MVP | Phase 2b Winner (projected) |
|--------|-----|------|------|
| Best ROC-AUC (Phase II) | 0.639 | 0.64-0.66 | 0.66-0.70 |
| Calibration (ECE) | Unknown (not measured) | Baseline established | < 0.12 |
| Pareto front size | 1 solution | 5-15 solutions | 10-25 solutions |
| Feature sets available | 1 (best AUC) | Multiple trade-offs | Full trade-off menu |

**Conservative estimate:** 3-5% absolute AUC improvement on Phase II, plus calibrated probabilities and parsimony trade-offs that don't exist today.

**Why conservative:** The primary value of multi-objective search is not raw AUC improvement — it's producing a *menu* of solutions with different trade-off profiles. A 2% AUC improvement with calibrated probabilities and 50% fewer features is more valuable than a 5% AUC improvement with an opaque 40-feature model.

---

## 8. Head-to-Head Benchmark Plan

### Setup

| Parameter | Value |
|-----------|-------|
| Dataset | TrialBench Phase II (hardest, most deceptive landscape) |
| Runs per strategy | 3 (for variance estimation) |
| Budget per run | 140 raw evals (Pareto MCTS) / 20 evals (LLM-FE) |
| Multi-fidelity | Enabled for Pareto MCTS; not applicable for LLM-FE |
| Predictive scoring | Enabled for both (after anchor set collected) |
| Objectives | Same 3 for both: weighted AUC, calibration, parsimony |
| LLM backbone | Claude Opus 4.6 for both |
| Random seed | Fixed 3 seeds: 42, 137, 2026 |

### Evaluation Protocol

1. **Run Phase 2a MVP first** to establish objective baselines and validate multi-fidelity proxy.
2. **Run both strategies** with identical evaluation function and objective computation.
3. **Measure after each strategy completes** (do not compare mid-run — different evaluation budgets mean different timelines).

### Metrics Collected

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Hypervolume** (3D, relative to reference point) | Quality of Pareto front | `pymoo.indicators.hv.HV` with reference `[0.5, 0.3, 0.0]` |
| **Spread** (Delta metric) | Diversity of Pareto front | Distance between extreme solutions |
| **Best single-objective scores** | Peak performance per objective | Max weighted AUC, min ECE, min features on front |
| **Cost** | Total $ spent | Sum of all eval costs |
| **Wall-clock time** | Practical runtime | End-to-end including LLM calls |
| **Convergence curve** | How fast front improves | Hypervolume vs evaluation count |
| **Non-dominated fraction** | Selection pressure health | Fraction of all evaluated solutions on front (lower is better) |

### Decision Framework

```
IF hypervolume(Pareto MCTS) > hypervolume(LLM-FE) * 1.15:
    → Pareto MCTS wins (15% margin justifies higher cost)
ELIF hypervolume(LLM-FE) >= hypervolume(Pareto MCTS) * 0.9:
    → LLM-FE wins (within 10% at fraction of cost)
ELSE:
    → Hybrid: LLM-FE for initial population, Pareto MCTS for refinement
```

### Empirical Question Being Answered

**Does the clinical trial feature landscape actually require deep exploration (favoring Pareto MCTS) or does LLM domain knowledge handle most of the complexity (favoring LLM-FE)?**

If the landscape is primarily deceptive (feature interactions dominate, greedy search fails), Pareto MCTS should show:
- Higher hypervolume (finds solutions in hard-to-reach regions)
- Better convergence at eval 100+ (late-stage discovery)
- Larger Pareto front (more diverse trade-offs)

If LLM domain knowledge is sufficient, LLM-FE should show:
- Comparable hypervolume at 1/7th the cost
- Fast convergence (most improvement in first 10 evals)
- Compact but high-quality Pareto front

---

## 9. Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Number of objectives | 3 | 5 objectives degenerates at available eval budget (convergence research) |
| Objective set | Weighted AUC, ECE, parsimony | Captures three genuinely conflicting desiderata; all cheaply measurable |
| Phase weights | 0.2 / 0.5 / 0.3 (I/II/III) | Phase II is hardest and most valuable; configurable |
| MVP approach | Pareto fitness on existing MCTS | Minimal code change; validates objectives before committing to search strategy |
| Strategy commitment | Deferred to head-to-head | Neither MCTS nor LLM-FE has clear theoretical advantage; empirical resolution needed |
| Multi-fidelity | 20% data proxy | Critical enabler; must validate correlation before trusting |
| Surrogate model | Deferred (use predictive scoring instead) | Convergence research: RF surrogate at 30 evals is premature; LLM predictive scoring viable at ~15 evals |
| PMMG code reuse | Clean-room rewrite of algorithm only | PMMG code is SMILES-coupled, poor quality, hardcoded constants; algorithm is correct |
| AB-MCTS integration | Post-head-to-head, MCTS-only | Composes well with Pareto MCTS but adds complexity; validate base strategy first |
| Solution merging | Phase 3 | Highest-impact augmentation per KompeteAI ablation, but also highest effort and most invasive |

---

## Appendix A: Phased Surrogate Strategy

From convergence research — when to introduce cost-reduction mechanisms:

| Eval Range | Strategy | Rationale |
|-----------|----------|-----------|
| 1-15 | Full evaluation only, pure exploration | Too few samples for any surrogate or scoring |
| 15-30 | Add predictive scoring (LLM few-shot) | 15 anchor examples sufficient for Opus 4.6 to estimate feature set quality |
| 30-70 | Predictive scoring as first filter, multi-fidelity as second | Reject bottom 25% via LLM prediction, then proxy eval survivors, full eval top candidates |
| 70+ | Optional: train RF/linear surrogate from evaluation history | 70+ feature-set → reward-vector pairs may enable a cheap surrogate; validate before trusting |

## Appendix B: LLM-FE Integration Sketch

If LLM-FE wins the head-to-head, the evolutionary loop replaces `treesearch.py`:

```python
def llm_fe_search(task, budget=20, n_islands=3, pop_per_island=5):
    """3-island evolutionary feature engineering with Pareto fitness."""
    islands = [initialize_island(task, pop_per_island) for _ in range(n_islands)]

    for generation in range(budget // n_islands):
        for island in islands:
            # Select parent via Boltzmann sampling on hypervolume contribution
            parent = boltzmann_select(island.population, temperature=1.0)

            # LLM mutation: propose modifications to parent's feature program
            child_program = llm_mutate(parent.program, island.theme)

            # Evaluate child
            reward_vector = evaluate_full(child_program)

            # Update island population (Pareto ranking)
            island.population = update_population(
                island.population, child_program, reward_vector
            )

        # Migration: exchange best solutions between islands every 3 generations
        if generation % 3 == 0:
            migrate(islands)

    # Merge all island Pareto fronts
    return merge_pareto_fronts([island.pareto_front for island in islands])
```

**Island themes** (diversity mechanism):
1. **Drug-focused:** Prioritize drug mechanism, target interaction, SMILES-derived features
2. **Trial-design-focused:** Prioritize eligibility criteria, enrollment, study design features
3. **Outcome-focused:** Prioritize historical outcome data, FAERS safety signals, prior phase results

## Appendix C: Cross-Reference to Existing Research

This design builds on and in some cases revises recommendations from [mcts-alternatives-and-improvements.md](./mcts-alternatives-and-improvements.md):

| Research Doc Recommendation | This Design | Rationale |
|---------------------------|-------------|-----------|
| Implement Pareto MCTS in Phase 2-3 transition | Phase 2b (parallel track) | Must validate multi-fidelity first; run head-to-head against LLM-FE |
| 5 objectives | 3 objectives | Convergence research: 5 objectives at available budget degenerates to random selection |
| Surrogate model after ~30 evals | Predictive scoring at ~15 evals; surrogate deferred to 70+ | LLM few-shot scoring is cheaper and available earlier |
| LLM-FE as Phase 3 prototype | Phase 2b (parallel track with Pareto MCTS) | Approach challenger identified LLM-FE as potentially best risk-adjusted choice; deserves equal evaluation |
| QD/MAP-Elites for diversity | Deferred to Phase 3 | Three objectives already provide diversity pressure; MAP-Elites adds complexity |
| Solution merging in Phase 2-3 | Phase 3 only | Highest effort augmentation; validate base strategy first |
| OCTree as Phase 2 item #2 | Phase 2a (MVP) | Confirmed low-effort, high-value; no reason to delay |
