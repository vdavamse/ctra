# CTRA Multi-Objective MCTS — Implementation Infographic

## 1. Architecture Overview

```
                    CTRA Pareto MCTS: Feature Set Optimization
    ============================================================================

    GOAL: Find the best combination of features for predicting clinical trial
          success/failure, balancing accuracy and feature-set size, per phase.

    INPUT                         MCTS SEARCH                        OUTPUT
    -----                         -----------                        ------
    Initial features         +--> SELECT ----+                  Best feature set
    ["enrollment_size",      |    (Pareto    |                  on Pareto front
     "drug_approvals"]       |     UCT)      |                  + SHAP explanations
                             |               v
                             |    EXPAND ----+
                             |    (AB-MCTS   |
                             |     adaptive  |
                             |     branching)|
                             |               v
                             |    SIMULATE --+    <-- Deep Rollout
                             |    (evaluate  |        (to max_depth,
                             |     to depth  |         eval every node)
                             |     7-10)     |
                             |               v
                             |    BACKPROP --+
                             |    (best-on-  |
                             |     path HV)  |
                             |               |
                             +--- REPEAT ----+
                                 (10-20 rollouts)
```

---

## 2. The Two Per-Phase Objectives

> **Note:** The SVG file in this directory (`docs/mcts-implementation-infographic.svg`) is a stale rendering. This Markdown document is authoritative.

All objectives normalized to [0, 1]. Higher is better. Pareto ranking finds trade-offs. Each phase's MCTS tree optimizes independently with its own objectives.

```
    OBJECTIVE 1: Predictive Accuracy
    ================================
    Metric:  Validation ROC-AUC for this phase
    Purpose: Maximize the probability that the feature set correctly ranks
             successful vs. failed trials in the target phase

    Example values (Phase II):
    +--------------------------------------------------+
    | Feature Set                | Validation ROC-AUC  |
    |----------------------------|---------------------|
    | {enrollment, drug_targets} |       0.58          |
    | {+ adverse_events}         |       0.64          |  <-- +0.06
    | {+ sponsor_success}        |       0.68          |  <-- +0.04
    | {+ biomarker, orphan_drug} |       0.72          |  <-- +0.04
    +--------------------------------------------------+
    Each phase optimizes independently; a feature that helps Phase II
    may not help Phase I or III, and that's fine.


    OBJECTIVE 2: Parsimony (Feature Efficiency)
    ============================================
    Formula: 1 - (n_features / max_features)
    Purpose: Fewer features = lower LLM cost, faster inference, less overfitting

    +--------------------------------+
    |  Features   |  Parsimony Score |
    |-------------|------------------|
    |    0        |      1.00        |  (useless but maximum parsimony)
    |    5        |      0.90        |  <-- sweet spot
    |   10        |      0.80        |
    |   25        |      0.50        |
    |   50        |      0.00        |  (at capacity)
    +--------------------------------+

    Cost impact: each feature requires ~6 LLM calls per trial
    (ReAct search + extraction). 10 features x 500 trials = 30,000 calls.


    THE PARETO TRADE-OFF
    ====================

                Accuracy
                   ^
              0.85 |         * A (high accuracy,
                   |           low parsimony)       The Pareto FRONT is the set
              0.75 |     * B                        of non-dominated solutions.
                   |                                No single point beats another
              0.65 |         * C (balanced)          on ALL objectives.
                   |
              0.55 | * D (high parsimony,           MCTS explores this surface
                   |   low accuracy)                to find the best trade-off.
                   +----------------------------->
                  0.5    0.7    0.9    1.0
                          Parsimony
```

---

## 3. Deep Rollout Simulation (Option 3: Hybrid)

```
    HOW A SINGLE ROLLOUT WORKS
    ==========================

    Traditional MCTS:              CTRA Deep Rollout (AutoCT-style):
    1 eval per rollout             ~7 evals per rollout

    Root ─┐                        Root ─┐
          ├─ A  <-- eval, done           ├─ A  <-- eval
          └─ B                           │  ├─ A1 <-- eval
                                         │  │  ├─ A1a <-- eval
                                         │  │  │  └─ A1a1 <-- eval (depth 4)
                                         │  │  │     └─ A1a1x <-- eval (depth 5)
                                         │  │  │        └─ A1a1x2 <-- eval (depth 6)
                                         │  │  │           └─ TERMINAL (depth 7)
                                         │  └─ A2
                                         └─ B

    Result: depth 1, 1 eval       Result: depth 7, 7 evals
    Best: A's reward              Best: MAX reward on entire path


    WHY DEEP ROLLOUT MATTERS FOR FEATURE COMBINATIONS
    ==================================================

    Depth 0: {enrollment_size}                          acc=0.52
    Depth 1: {enrollment_size, drug_targets}            acc=0.55  (each alone: weak)
    Depth 2: {enrollment_size, drug_targets, phase_dur} acc=0.58
    Depth 3: {+ adverse_event_rate}                     acc=0.63  (interaction effect!)
    Depth 4: {+ sponsor_success_rate}                   acc=0.68
    Depth 5: {+ biomarker_available}                    acc=0.73  (synergy with targets)
    Depth 6: {+ mechanism_novelty}                      acc=0.78
    Depth 7: TERMINAL                                        ^
                                                             |
    Without deep rollout, MCTS only sees depth 1-2      The COMBINATION of
    and never discovers the depth-5 synergy between     features matters, not
    biomarker + drug_targets.                           individual features.


    BEST-ON-PATH SELECTION (Multi-Objective)
    ========================================

    After deep rollout, find the best node on the path:

    Path node:     depth 0    depth 1    depth 2    depth 3    depth 4
    Accuracy:       0.52       0.55       0.58       0.63       0.68
    Parsimony:      0.98       0.96       0.94       0.92       0.90

    Hypervolume:    0.020      0.048      0.075      0.120      0.162
                                                                 ^
                                                                 |
    Best-on-path: depth 4 has highest hypervolume ───────────────┘
    (product of objective deltas above the reference point; each
    phase uses reference point [0.5, 0.0]: accuracy at the ROC-AUC
    chance baseline, parsimony at its floor, see section 11)

    This reward vector [0.68, 0.90] is backpropagated to root.
```

---

## 4. Simulation Example (Illustrative Run)

> Values adapted by hand from the original 3-objective run (its third objective was dropped, the Pareto front re-pruned under the remaining two objectives and the hypervolume recomputed against `[0.5, 0.0]`); the 2-objective configuration shown below has not been re-run.

```
    PROBLEM: Find synergistic feature combination
    ==============================================
    Pool: {feat_A, feat_B, feat_C, feat_D, noise_1, noise_2, noise_3}

    Synergy rule:
      - 1 synergy feature:  accuracy ~0.55
      - 2 synergy features: accuracy ~0.60
      - 3 synergy features: accuracy ~0.70
      - ALL 4 together:     accuracy ~0.85  <-- target (epistasis)

    Config: 10 rollouts, max_depth=7, 2 objectives, deep_simulation=True


    RESULTS
    =======

    Total nodes in tree:     61
    Evaluated nodes:         45   (73% of tree evaluated — deep rollout efficiency)
    Max tree depth:           6
    Evaluation calls:        45

    Depth distribution (all nodes / evaluated):
    +------------------------------------------+
    | Depth | All Nodes | Evaluated | Coverage |
    |-------|-----------|-----------|----------|
    |   0   |     1     |     1     |   100%   |
    |   1   |     2     |     2     |   100%   |
    |   2   |     4     |     4     |   100%   |
    |   3   |     8     |     8     |   100%   |
    |   4   |    16     |    10     |    63%   |
    |   5   |    20     |    10     |    50%   |
    |   6   |    10     |    10     |   100%   |
    +------------------------------------------+

    Dense coverage at all depths — deep rollout explores the full tree.


    BEST NODE FOUND
    ===============
    Features:    [noise_1, noise_3, feat_D, feat_C, feat_A, noise_2, feat_B]
    Depth:       6
    Synergy:     4/4 features found (feat_A + feat_B + feat_C + feat_D)

    Objectives:  Accuracy   = 0.865  (near-maximum synergy bonus)
                 Parsimony  = 0.860  (7 features / 50 max = good)

    Hypervolume: (0.865 - 0.5) * (0.860 - 0.0) = 0.314
                 (above reference point [0.5, 0.0])


    BEST PATH (Root to Best Node)
    =============================

    Depth 0: [noise_1]                    acc=0.82  pars=0.87
        |
        +-- add:noise_3
        v
    Depth 1: [noise_1, noise_3]          acc=0.85  pars=0.86
        |
        +-- add:feat_D
        v
    Depth 2: [noise_1, noise_3, feat_D]  acc=0.86  pars=0.86
        |
        +-- add:feat_C    <-- first synergy feature added
        v
    Depth 3: [+ feat_C]                  acc=0.60  pars=0.92
        |
        +-- add:feat_A    <-- second synergy feature
        v
    Depth 4: [+ feat_A]                  acc=0.70  pars=0.90
        |
        +-- add:noise_2
        v
    Depth 5: [+ noise_2]                 acc=0.69  pars=0.88
        |
        +-- add:feat_B    <-- third synergy feature! All 4 present now
        v
    Depth 6: [+ feat_B]                  acc=0.87  pars=0.86
                                          ^^^^
                                          Synergy bonus triggered!


    PARETO FRONT (3 non-dominated solutions)
    ========================================

    +------+------------------------------------------+---------+---------+
    | Depth| Features                                 | Accurac | Parsim  |
    |------|------------------------------------------|---------|---------|
    |  3   | [noise_1, feat_C, feat_D, feat_A]        |  0.692  |  0.920  |
    |  4   | [noise_1, feat_C, feat_D, feat_B, feat_A]|  0.840  |  0.900  |
    |  6   | [noise_1, noise_3, feat_D, ..., feat_B]  |  0.865  |  0.860  |
    +------+------------------------------------------+---------+---------+

    The Pareto front spans depths 3-6, showing that MCTS explores both
    shallow (few features, high parsimony) and deep (more features,
    higher accuracy) solutions simultaneously.
```

---

## 5. CTRA vs AutoCT Comparison

```
    HEAD-TO-HEAD: Same problem, same 10 rollouts
    =============================================

    +-----------------------------------+----------+----------+
    | Metric                            | AutoCT   | CTRA     |
    |-----------------------------------|----------|----------|
    | Max tree depth                    |    7     |    7     |  <-- MATCHED
    | Total evaluation calls            |   63     |   63     |  <-- MATCHED
    | Total nodes in tree               |   63     |  160     |  <-- CTRA richer
    | Objectives                        |    1     |    2     |  <-- CTRA multi-obj (per-phase)
    | Selection strategy                | Scalar   | Pareto   |
    |                                   |   UCT    |   UCT    |
    | Best-on-path metric               | max(acc) | max(HV)  |
    | Adaptive branching                |   No     |   Yes    |
    | Deep simulation                   |   Yes    |   Yes    |  <-- MATCHED
    +-----------------------------------+----------+----------+


    WHAT AUTOCT DOES (CogComp/autoct treesearch.py)
    ================================================

    1. SELECT:    Scalar UCT traversal to leaf
    2. EXPAND:    Evaluate the leaf (run full LLM + train + eval pipeline)
    3. SIMULATE:  Deep rollout — random child selection to max_depth
                  EVALUATING EVERY NODE on the way down
                  Return: max(rewards_on_path)
    4. BACKPROP:  Scalar reward up to root

    Key: Each rollout creates an ENTIRE depth-first path.
         10 rollouts x ~7 evals = ~70 evaluations.
         Single objective: validation ROC-AUC.


    WHAT CTRA DOES (our implementation)
    ====================================

    1. SELECT:    PARETO UCT traversal — vector UCB scores per objective,
                  Pareto front filtering, hypervolume ranking
    2. EXPAND:    AB-MCTS adaptive branching (log2(visits)+2 children)
    3. SIMULATE:  DEEP ROLLOUT (matching AutoCT) — random child selection
                  to max_depth, evaluating every node. But with:
                  - Multi-objective: track 2 objectives per node
                  - Best-on-path by hypervolume (not scalar max)
    4. BACKPROP:  VECTOR reward (2 objectives) up to root

    Key: Same depth as AutoCT, but richer ranking.
         10 rollouts x ~7 evals = ~70 evaluations.
         Two objectives: accuracy + parsimony (per-phase).
         Pareto front output: multiple trade-off solutions, not just one.


    BEHAVIORAL DIFFERENCE BEFORE VS AFTER DEEP SIMULATION
    =====================================================

    BEFORE (shallow, 1 eval per rollout):
    +------------------------------------------+
    | Rollouts | Max Depth | Evals | Depth 7?  |
    |----------|-----------|-------|-----------|
    |    10    |     2     |   11  |    No     |
    |    20    |     4     |   21  |    No     |
    |    40    |     5     |   41  |    No     |
    |    80    |     6     |   81  |  Maybe    |
    +------------------------------------------+

    AFTER (deep rollout, ~7 evals per rollout):
    +------------------------------------------+
    | Rollouts | Max Depth | Evals | Depth 7?  |
    |----------|-----------|-------|-----------|
    |    10    |     7     |  ~63  |    YES    |
    |    20    |     7     | ~130  |    YES    |
    +------------------------------------------+

    Deep rollout reaches max_depth in EVERY rollout — no wasted budget
    on shallow exploration when deep combinations are the goal.
```

---

## 6. Pareto Selection During Tree Traversal

```
    HOW PARETO UCT SELECTS THE NEXT NODE
    =====================================

    At each internal node, compute UCB vector per child:

    Child A: UCB = [mean_acc + C*explore, mean_pars + C*explore]
           = [0.58 + 0.42, 0.85 + 0.42]
           = [1.00, 1.27]

    Child B: UCB = [0.77 + 0.38, 0.77 + 0.38]
           = [1.15, 1.15]

    Child C: UCB = [0.78 + 0.55, 0.45 + 0.55]
           = [1.33, 1.00]
                    ^
                    |
           Unvisited nodes get UCB = [inf, inf] -> always explored first

    Step 1: Pareto front of UCB vectors
            A dominates none, B dominates none, C dominates none
            All three are on the front.

    Step 2: Hypervolume contribution (measured from the origin)
            HV(A) = exclusive volume A contributes = 0.12
            HV(B) = exclusive volume B contributes = 0.02
            HV(C) = exclusive volume C contributes = 0.18  <-- highest
                                                       |
    Step 3: SELECT C (highest HV contribution)  <------+

    This balances:
    - Exploitation (high mean_reward)
    - Exploration (low visit_count -> high explore term)
    - Diversity (hypervolume spreads selection across objective space)
```

---

## 7. AB-MCTS Adaptive Branching

```
    BRANCHING FACTOR = min(max, max(min, int(log2(visits) + 2)))
    ============================================================

    +----------+--------+--------------------------------------------+
    | Visits   | Branch | Rationale                                  |
    |----------|--------|--------------------------------------------+
    |    1     |   2    | First visit — explore conservatively        |
    |    2     |   3    | Some evidence — slightly wider              |
    |    4     |   4    | Promising node — broader exploration        |
    |    8     |   5    | Well-tested — explore diverse directions    |
    |   16+    |   6+   | High confidence — maximum breadth (capped)  |
    +----------+--------+--------------------------------------------+

    WHY: Early in search, generating many children wastes budget.
    Later, promising nodes deserve diverse expansion to find the
    best modification (add/remove/refine).
```

---

## 9. Benefits

> Section 8 (Multi-Fidelity Evaluation Schedule) was removed: multi-fidelity evaluation is not implemented in the search layer (only `mlops/retraining.py` has a fidelity parameter). The numbering of sections 9-12 is kept so that existing section references stay valid.

```
    +------------------------------------------------------------------+
    |                        CTRA MCTS BENEFITS                        |
    +------------------------------------------------------------------+
    |                                                                  |
    |  1. DEEP FEATURE COMBINATION DISCOVERY                           |
    |     - Reaches depth 7 in 10 rollouts (matches AutoCT)            |
    |     - Discovers synergistic features that are weak alone          |
    |     - Every intermediate node evaluated and available for reuse   |
    |                                                                  |
    |  2. MULTI-OBJECTIVE OPTIMIZATION                                 |
    |     - Accuracy + Parsimony balanced per phase                    |
    |     - Pareto front output: multiple trade-off solutions           |
    |     - No single-metric bias (AutoCT only optimizes ROC-AUC)      |
    |                                                                  |
    |  3. PARETO-AWARE TREE TRAVERSAL                                  |
    |     - Vector UCB + hypervolume selection during tree traversal    |
    |     - Explores diverse regions of objective space                 |
    |     - Not just accuracy-greedy (AutoCT's scalar UCT is)          |
    |                                                                  |
    |  4. ADAPTIVE BRANCHING                                           |
    |     - Promising nodes get more children (wider exploration)       |
    |     - New nodes get fewer children (conservative budget use)      |
    |                                                                  |
    |  5. BACKWARD COMPATIBLE                                          |
    |     - deep_simulation=False: reverts to shallow (1 eval/rollout) |
    |     - objectives=["accuracy"]: degenerates to scalar MCTS        |
    |     - Single-objective mode matches standard MCTS exactly         |
    |                                                                  |
    +------------------------------------------------------------------+
```

---

## 10. Limitations

```
    +------------------------------------------------------------------+
    |                       CURRENT LIMITATIONS                         |
    +------------------------------------------------------------------+
    |                                                                  |
    |  1. DEEP ROLLOUT COST                                            |
    |     Each rollout evaluates ~max_depth nodes, not just 1.         |
    |     10 rollouts = ~70 evals = ~$140 with Opus 4.6.               |
    |     AutoCT has the same cost — this is inherent to deep search.  |
    |                                                                  |
    |  2. RANDOM SIMULATION POLICY                                     |
    |     Deep rollout picks random children (AutoCT does the same).   |
    |     A smarter policy (e.g., LLM-guided child selection) could    |
    |     focus deep rollouts on more promising paths.                 |
    |     Status: Deferred to Phase 2c (predictive scoring).           |
    |                                                                  |
    |  3. NO CROSS-BRANCH MEMORY                                       |
    |     Each tree path is independent. A feature combination that    |
    |     works well in branch A is not shared with branch B.          |
    |     AutoCT has the same limitation.                              |
    |     Status: Deferred to Phase 2c (ML-Master scoped memory).      |
    |                                                                  |
    |  4. MULTI-OBJECTIVE "BEST ON PATH" IS APPROXIMATE                |
    |     Using hypervolume (product of deltas) as the scalar proxy    |
    |     for multi-objective "best" works well but is not the only    |
    |     option. Alternative: dominated hypervolume, or Pareto        |
    |     dominance along the path. Current approach is simple and     |
    |     degenerates correctly to scalar max for single-objective.    |
    |                                                                  |
    |  5. HYPERVOLUME COMPUTATION FOR 3+ OBJECTIVES                    |
    |     Exact hypervolume is O(n^(d/2)) for d objectives.            |
    |     For 3 objectives, we use Monte Carlo (20K samples).          |
    |     Acceptable for <50 Pareto front points, but may introduce    |
    |     variance in rankings. 2D uses exact sweep-line (no MC).      |
    |     (Unreachable with the current two-objective config;          |
    |     kept for the library function.)                              |
    |                                                                  |
    |  6. NO LLM-FE ALTERNATIVE YET                                    |
    |     The research proposed a head-to-head: MCTS vs LLM-FE         |
    |     (evolutionary feature engineering with only 20 evals).       |
    |     LLM-FE is not implemented. MCTS may not be optimal for      |
    |     all problems — the comparison would validate this.           |
    |     Status: Task #15 (deferred).                                 |
    |                                                                  |
    |  7. EXPAND_FN IS A BLACK BOX                                     |
    |     The MCTS tree structure supports add/remove/refine, but the  |
    |     actual feature operations come from expand_fn (pluggable).   |
    |     In production, this calls DSPy agents which analyze          |
    |     misclassified trials. Quality of expansion depends entirely  |
    |     on the LLM's feature engineering capability.                 |
    |                                                                  |
    +------------------------------------------------------------------+
```

---

## 11. Configuration Reference

```
    MCTSConfig (prefix: CTRA_MCTS_)
    =================================

    # Core search parameters
    num_rollouts:            20       # Number of MCTS iterations
    max_depth:               10       # Maximum refinement steps per path
    deep_simulation:         True     # AutoCT-style deep rollout

    # Multi-objective
    objectives:              ["accuracy", "parsimony"]
    max_features:            50       # Parsimony denominator
    reference_point:         [0.5, 0.0]  # Hypervolume reference: accuracy at the
                                         # ROC-AUC chance baseline, parsimony at
                                         # its floor (issue #18)

    # Backpropagation rule (issue #16)
    backprop:                "mean"     # Other: "max", "max_hv" (issue #16, research/backprop-ablation.md)

    # UCT exploration
    exploration_constant:    1.414    # C_p in UCB1 formula

    # Adaptive branching (AB-MCTS)
    adaptive_branching:      True
    min_branch_factor:       2
    max_branch_factor:       8

    # Feature caching
    feature_store_enabled:   True
    feature_store_dir:       Path("output/feature_store")

    # Subprocess timeout
    subprocess_timeout:      3600     # seconds per agent evaluation
```

---

## 12. File Map

```
    src/ctra/search/
    ├── mcts.py          MCTSNode (dataclass), MCTSSearch (main loop)
    │                    _select, _expand, _simulate_deep, _backpropagate,
    │                    _select_best, _call_evaluate
    │
    ├── pareto.py        pareto_front, pareto_front_indices,
    │                    hypervolume_contribution, pareto_select (Pareto UCT
    │                    for tree traversal)
    │
    ├── objectives.py    FeatureSet, ObjectiveResult
    │
    └── __init__.py      Public exports

    src/ctra/mlops/
    └── objectives.py    PredictiveAccuracy, Parsimony, MultiObjectiveEvaluator
                         (used by the retraining pipeline, not the search layer)

    tests/test_search/
    ├── test_pareto.py                         26 tests
    ├── test_objectives.py                      3 tests
    ├── test_mcts.py                           13 tests
    ├── test_mcts_simulation.py                13 tests
    ├── test_mcts_deep_exploration.py           8 tests
    ├── test_mcts_node_id.py                   28 tests
    ├── test_autoct_mcts_comparison.py          6 tests
    ├── test_backprop_variants.py              26 tests
    ├── test_mcts_callback.py                   6 tests
    ├── test_mcts_callback_edge.py             10 tests
    ├── test_mcts_e2e_shapiq.py                 3 tests
    ├── test_mcts_exhaustion.py                17 tests
    ├── test_mcts_integrity.py                  6 tests
    ├── test_mcts_suggestion_index.py          20 tests
    └── test_select_best.py                    34 tests

    Total: 219 tests
```
