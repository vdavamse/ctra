# Feature Extraction Optimization: Caching and Reuse Across MCTS Runs

> Deep analysis of AutoCT's feature extraction pipeline, identifying waste in LLM queries and proposing a feature value store for cross-node reuse. Produced 2026-03-26 from complete codebase analysis.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [How Feature Extraction Works Today](#2-how-feature-extraction-works-today)
3. [LLM Call Cost Model](#3-llm-call-cost-model)
4. [Current Caching Inventory](#4-current-caching-inventory)
5. [Where Waste Occurs](#5-where-waste-occurs)
6. [Proposed Solution: Global Feature Value Store](#6-proposed-solution-global-feature-value-store)
7. [Projected Savings](#7-projected-savings)
8. [Additional Optimizations](#8-additional-optimizations)
9. [Implementation Plan](#9-implementation-plan)
10. [Risks and Mitigations](#10-risks-and-mitigations)

---

## 1. Executive Summary

**Problem:** Each MCTS node exploration runs the full LLM agent pipeline in a subprocess. While AutoCT does reuse parent features within a tree path (ADD only rebuilds the new feature, REMOVE is free), there are significant inefficiencies:

1. **Feature building is the dominant cost.** For 200 train + 200 val + test trials, each new feature triggers ~600+ LLM calls (ReAct research + construction per trial per split). This dwarfs the proposer (1 call), planner (1 call), and evaluator (3-12 calls).

2. **The same feature is rebuilt when it appears on different branches.** If MCTS node `0-0` computes feature F4 for 200 trials, and later node `0-1-2` independently proposes the same F4, it rebuilds from scratch — the feature builder cache misses because the group hash differs.

3. **Simulation wastes evaluations.** `_simulate()` creates and explores temporary nodes, running full agent pipelines that are never reused.

4. **Cross-branch knowledge is lost.** Features computed on one branch are invisible to other branches, despite the feature plans being semantically identical.

**Solution:** A **global feature value store** keyed by `(nctid, feature_name, plan_hash)` — individual feature granularity instead of feature-group granularity. This allows any MCTS node to reuse any previously computed feature value regardless of which branch computed it.

**Projected savings:** 40-70% reduction in LLM calls across a 20-rollout MCTS run.

---

## 2. How Feature Extraction Works Today

### 2.1 Per-Node Execution Flow

Each MCTS node is explored by spawning a subprocess (`run_agent_as_script_v2`) that runs `AgentV2.forward()`:

```
MCTS._explore_node(node)
  → run_agent_as_script_v2(node.id, task, node.input)
    → subprocess: python scripts/run_agent.py --input <serialized OutputV2> <output_file>
      → AgentV2.forward(previous_output)
        → [Feature Proposal] → [Feature Planning] → [Feature Building] → [Model Training] → [Evaluation]
      → Serialize OutputV2 to output_file
    → Cache OutputV2 by node ID
```

### 2.2 Root Node (Depth 0)

```
AgentV2.forward(None):
  1. Initializer()                              # ~9+N LLM calls (zero-shot + factor analysis + planning)
  2. compute_features_v3(X_train, ALL features)  # 2 × num_groups × num_train_trials LLM calls
  3. compute_features_v3(X_val, ALL features)    # 2 × num_groups × num_val_trials LLM calls
  4. compute_features_v3(X_test, ALL features)   # 2 × num_groups × num_test_trials LLM calls
  5. Train 3 models (XGB, LR, RF)               # 0 LLM calls
  6. Evaluate on val + test                       # 0 LLM calls
  7. EvaluatorV2() × 3 models                    # 3-12 LLM calls (1 no-example + up to 3 ReAct each)
  → Return OutputV2 with raw_features, raw_val_features, raw_test_features, feature_plans, suggestions
```

### 2.3 Non-Root Node — ADD Operation

```
AgentV2.forward(previous_output):
  1. deepcopy(previous_output.raw_features)      # REUSE all parent features (0 LLM calls)
  2. FeatureProposerV2(previous_output)           # 1 LLM call (reads suggestion_index)
  3. FeaturePlannerV2(feature_name, idea)         # 1 LLM call
  4. compute_features_v3(X_train, {NEW feature})  # 2 × 1 × num_train_trials LLM calls
  5. compute_features_v3(X_val, {NEW feature})    # 2 × 1 × num_val_trials LLM calls
  6. compute_features_v3(X_test, {NEW feature})   # 2 × 1 × num_test_trials LLM calls
  7. Merge: current_feature_values[nctid] |= new_features[nctid]
  8. Train 3 models, evaluate, EvaluatorV2        # 3-12 LLM calls
  → Return OutputV2 with merged features
```

**Key:** Step 1 reuses ALL parent features. Steps 4-6 only build the NEW feature.

### 2.4 Non-Root Node — REMOVE Operation

```
AgentV2.forward(previous_output):
  1. deepcopy(previous_output.raw_features)
  2. FeatureProposerV2(previous_output)           # 1 LLM call
  3. Filter out removed feature (dict comprehension) # 0 LLM calls
  4. Train 3 models, evaluate, EvaluatorV2        # 3-12 LLM calls
  → Return OutputV2 with filtered features
```

**REMOVE is essentially free** — no feature building at all.

### 2.5 Non-Root Node — REFINE Operation

Same as ADD but overwrites the existing feature instead of adding a new one. Same LLM cost as ADD.

### 2.6 Feature Building Pipeline (compute_features_v3)

```python
def compute_features_v3(grouper, nctids, task, plans):
    grouped_plans = grouper(plans)  # LLM groups related features (1 call, or skip if single feature)

    product = itertools.product(nctids, grouped_plans)  # Cartesian product

    results = ProcessPoolExecutor.map(WrappedFeatureBuilderV3(task), product)
    # For each (nctid, feature_group):
    #   1. Check file cache: .cache/feature-builder/{task}--{group_hash}/{nctid}.json
    #   2. If MISS: run FeatureBuilderV3.forward(nctid, feature_group)
    #      a. Get trial info (cached in diskcache)
    #      b. dspy.ReAct research (up to 5 tool-calling iterations)  ← 1 LLM chain
    #      c. dspy.ChainOfThought construction                       ← 1 LLM call
    #      d. Type validation (no LLM)
    #   3. Write to file cache
    #   4. Return (nctid, feature_values, metadata)

    return aggregated_results
```

---

## 3. LLM Call Cost Model

### 3.1 Per-Node LLM Call Count

| Phase | Root Node | ADD/REFINE Node | REMOVE Node |
|-------|-----------|-----------------|-------------|
| Initializer | 9 + N_features | — | — |
| Proposer | — | 1 | 1 |
| Planner | N_features | 1 | 0 |
| Grouper | 1 | 0 (single feature) | 0 |
| **Feature Building** | **2 × G × T × 3** | **2 × 1 × T × 3** | **0** |
| Model Training | 0 | 0 | 0 |
| Evaluator | 3-12 | 3-12 | 3-12 |
| **Total** | **~12 + N + 6GT** | **~5 + 6T** | **~4-13** |

Where:
- G = number of feature groups (typically 2-4 for ~8 features)
- T = number of trials per split (train + val + test)
- N = number of initial features (~5-8)

### 3.2 Concrete Cost Example (200 train + 200 val + 100 test)

| Phase | Root (8 features, 3 groups) | ADD Node | REMOVE Node |
|-------|----------------------------|----------|-------------|
| Overhead (proposer/planner/evaluator) | ~25 | ~8 | ~8 |
| **Feature Building** | **2 × 3 × 500 × 3 = 9,000** | **2 × 1 × 500 × 3 = 3,000** | **0** |
| **Total LLM calls** | **~9,025** | **~3,008** | **~8** |

**Feature building is 99.7% of root node cost and 99.7% of ADD node cost.**

### 3.3 Full MCTS Run (20 rollouts, depth 7)

Estimated ~140-160 nodes explored:
- 1 root node: ~9,000 calls
- ~80 ADD/REFINE nodes: 80 × 3,000 = ~240,000 calls
- ~20 REMOVE nodes: 20 × 8 = ~160 calls
- ~40 simulation nodes (explored but temporary): 40 × 3,000 = ~120,000 calls

**Total: ~369,000 LLM calls per 20-rollout run**

At ~$0.002 per call (Opus 4.6 at 1K input + 500 output tokens average): **~$738/run**

---

## 4. Current Caching Inventory

### 4.1 Five Cache Layers

| # | Cache | Location | Type | Key | What's Cached | Scope |
|---|-------|----------|------|-----|---------------|-------|
| 1 | **Agent Script Cache** | `.cache/agent_script_cache_dir/` | dill files | `{task}--{node_id}` | Entire OutputV2 | Cross-rollout (same node ID) |
| 2 | **Feature Builder File Cache** | `.cache/feature-builder/` | JSON+dill files | `{task}--{group_hash}/{nctid}` | Feature values for group | Within same group hash |
| 3 | **diskcache FanoutCache** | `.cache/dc/` | diskcache | Function args | CT info, embedding searches | Global, persistent |
| 4 | **DSPy LLM Cache** | `.cache/dspy/` | DSPy disk cache | Full prompt | Raw LLM responses | Global, persistent |
| 5 | **Feature deepcopy** | In-memory (OutputV2) | NamedTuple field | N/A | Parent's raw_features | Parent → child only |

### 4.2 What Works Well

- **Within-path feature reuse (Cache 5):** When a child node inherits from its parent, `deepcopy(previous_output.raw_features)` provides all previously computed feature values. ADD only builds the new feature. REMOVE builds nothing. This is efficient.

- **Agent script cache (Cache 1):** If a node ID was already explored (e.g., from a previous rollout or resumed run), the entire OutputV2 is loaded from cache. Zero recomputation.

- **Data retrieval cache (Cache 3):** Trial metadata and embedding search results are globally cached. The same NCT ID or search query never hits the database twice.

### 4.3 What Doesn't Work

- **Feature builder cache (Cache 2) uses GROUP hash, not individual feature hash.** The key is `SHA-256(orjson.dumps(all plans in group))`. If the group changes (different features grouped together, or a feature plan is refined), the hash changes and ALL features in the group miss.

- **Cross-branch reuse is impossible.** Branch A computes F4 for 500 trials. Branch B independently proposes F4 with the same plan. Branch B rebuilds F4 from scratch because:
  1. The agent script cache key is `node_id`, which is different
  2. The feature builder cache key includes the group hash, which may differ
  3. Even if the group hash matches (single-feature group), it works — but only for EXACTLY the same FeaturePlanV2

- **Simulation nodes are never reused.** `_simulate()` creates new node IDs (e.g., "0-0-0-1-2-3") that don't match any previously explored node. Every simulation step runs the full agent pipeline.

---

## 5. Where Waste Occurs

### 5.1 Waste Category 1: Duplicate Feature Computation Across Branches

**Scenario:** MCTS explores two branches that converge on the same feature.

```
Root (features: [F1, F2, F3])
├─ Node 0-0: ADD F4 → features [F1, F2, F3, F4]    ← Builds F4 for 500 trials (3,000 calls)
│  └─ Node 0-0-0: ADD F5 → features [F1, F2, F3, F4, F5]
├─ Node 0-1: ADD F5 → features [F1, F2, F3, F5]    ← Builds F5 for 500 trials (3,000 calls)
│  └─ Node 0-1-0: ADD F4 → features [F1, F2, F3, F5, F4]  ← REBUILDS F4! (3,000 calls WASTED)
└─ Node 0-2: REFINE F3 → F3'
   └─ Node 0-2-0: ADD F4 → features [F1, F2, F3', F4]  ← REBUILDS F4 AGAIN! (3,000 calls WASTED)
```

Node `0-1-0` and `0-2-0` both rebuild F4 even though `0-0` already computed it. The feature builder cache misses because:
- `0-1-0`'s input is different from `0-0`'s input (different parent OutputV2)
- Even if F4's plan is identical, the group hash includes only F4 (single feature), so the cache SHOULD hit IF the plan is byte-for-byte identical
- BUT: if the proposer generates a slightly different `feature_explanation` string, the planner produces a slightly different FeaturePlanV2, and the hash changes

**Estimated waste:** With 20 rollouts and branching factor ~3, approximately 30-50% of ADD operations propose features that have been computed on other branches with similar (but not byte-identical) plans.

### 5.2 Waste Category 2: Simulation Node Exploration

**Scenario:** `_simulate()` explores temporary nodes that will never be reused.

```python
def _simulate(self, node):
    while True:
        if node.is_terminal():
            return max_reward_on_path
        potential_children = node.potential_children_inputs()
        choice_id = self.rng.choice(list(potential_children.keys()))
        next_node = self._get_or_create_node(choice_id, node, potential_children[choice_id])
        node = self._explore_node(next_node)  # FULL AGENT PIPELINE RUNS HERE
```

Each simulation step:
1. Creates a new node with a unique ID
2. Runs the full agent pipeline (3,000+ LLM calls for ADD)
3. The result is cached by node ID, but simulation paths are random — the same node ID is unlikely to be visited again

**Estimated waste:** With max depth 7, each simulation runs 1-6 additional node explorations. Over 20 rollouts, ~40-80 simulation nodes are explored, costing 120,000-240,000 LLM calls that provide information only for the current rollout's backpropagation.

### 5.3 Waste Category 3: Identical RAG Research Across Features

**Scenario:** Two different features need the same research for the same trial.

Feature "drug_target_count" and feature "drug_mechanism_type" for trial NCT00110279 both need to research the drug's molecular properties. The ReAct agent makes similar (but not identical) PubMed queries for each feature.

Currently:
- Group caching partially addresses this — if features are grouped together, a single ReAct call researches all features in the group
- BUT: the grouper doesn't always group related features, and for ADD operations, only the new feature is built (no grouping)

### 5.4 Waste Category 4: Redundant Model Training

Each node trains 3 models (XGB, LR, RF). With CTRA's plan to use only XGBoost + TabPFN (Workstream 2), this drops to 2 models. But model training is cheap (seconds, no LLM) — not a significant cost driver.

### 5.5 Waste Summary

| Category | Est. Wasted LLM Calls (20 rollouts) | % of Total |
|----------|-------------------------------------|------------|
| Duplicate features across branches | ~60,000-100,000 | 16-27% |
| Simulation node exploration | ~120,000-240,000 | 32-65% |
| Redundant RAG research | ~10,000-30,000 | 3-8% |
| **Total wasted** | **~190,000-370,000** | **~51-100%** |
| **Total useful** | **~180,000-200,000** | — |

---

## 6. Proposed Solution: Global Feature Value Store

### 6.1 Core Idea

Replace the group-hash-based feature builder cache with a **global feature value store** keyed by individual feature identity:

```
Key: (nctid, feature_name, plan_content_hash)
Value: {feature_values: dict, metadata: dict, research_results: str}
```

Where `plan_content_hash = SHA-256(canonical(FeaturePlanV2))` and `canonical()` normalizes the plan to be independent of field ordering, whitespace, and formatting.

### 6.2 Architecture

```
┌──────────────────────────────────────────────────────┐
│                    MCTS Tree                          │
│                                                       │
│  Node 0-0 (ADD F4)  ──────┐                          │
│  Node 0-1-0 (ADD F4) ─────┤  All query the same      │
│  Node 0-2-0 (ADD F4) ─────┤  store entry              │
│  Simulation nodes ─────────┘                          │
│                                                       │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│            Global Feature Value Store                 │
│                                                       │
│  Key: (NCT00110279, "drug_target_count", 0xa3f2...)  │
│  Value: {"value": 3, research: "...", meta: {...}}    │
│                                                       │
│  Key: (NCT00110280, "drug_target_count", 0xa3f2...)  │
│  Value: {"value": 5, research: "...", meta: {...}}    │
│                                                       │
│  Storage: DuckDB table or Parquet + diskcache         │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### 6.3 Implementation

#### 6.3.1 Feature Store Module

**New file: `src/lfe/impl/feature_store.py`**

```python
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import dill
import orjson

STORE_DIR = ".cache/feature-store"


def _plan_content_hash(plan) -> str:
    """Compute a canonical hash of a FeaturePlanV2's full identity.

    The hash captures the COMPLETE provenance of the feature, including the
    refinement chain (feature_idea). This is critical for REFINE operations:
    a refined feature appends to the original idea string
    (e.g., "original idea\n---\nrefinement"), so the hash MUST differ from
    the original even if the planner happens to produce similar instructions.

    SAFETY INVARIANT: same plan = same values. Two plans that produce
    different feature values MUST hash differently. This means we include
    MORE fields in the hash, not fewer — erring on the side of cache misses
    rather than false hits.

    Normalizes the plan to be independent of:
    - Field ordering in dicts
    - Leading/trailing whitespace

    Does NOT normalize:
    - feature_idea content (captures refinement chain — must differ on REFINE)
    - feature_instructions content (captures builder directive — must differ on REFINE)
    - possible_values (constrains categorical output)
    """
    canonical = {
        "feature_name": plan.feature_name,
        "feature_idea": plan.feature_idea.strip(),          # INCLUDE — captures refinement chain
        "feature_type": dict(sorted(plan.feature_type.items())),
        "data_sources": sorted([s.value for s in plan.data_sources]),
        "feature_instructions": plan.feature_instructions.strip(),
        "possible_values": _sort_nested(plan.possible_values),  # INCLUDE — constrains categorical output
        # EXCLUDE only: example_values (illustrative, not definitional)
    }
    raw = orjson.dumps(canonical, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(raw).hexdigest()[:16]


def _sort_nested(d: dict) -> dict:
    """Sort a dict of lists for canonical hashing."""
    return {k: sorted(v) for k, v in sorted(d.items())} if d else {}


def _store_path(task_name: str, feature_name: str, plan_hash: str, nctid: str) -> Path:
    """Path: .cache/feature-store/{task}/{feature_name}--{plan_hash}/{nctid}.json"""
    return Path(STORE_DIR) / task_name / f"{feature_name}--{plan_hash}" / f"{nctid}.json"


def get_cached_feature(task_name: str, nctid: str, feature_name: str, plan) -> Optional[dict]:
    """Look up a cached feature value. Returns None on miss."""
    plan_hash = _plan_content_hash(plan)
    path = _store_path(task_name, feature_name, plan_hash, nctid)
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
            return dill.loads(bytes.fromhex(data["feature_values_b"]))
    return None


def put_cached_feature(task_name: str, nctid: str, feature_name: str, plan,
                       feature_values: dict, metadata: dict):
    """Store a computed feature value."""
    plan_hash = _plan_content_hash(plan)
    path = _store_path(task_name, feature_name, plan_hash, nctid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "feature_values": feature_values,
            "feature_values_b": dill.dumps(feature_values).hex(),
            "metadata": metadata,
        }, f, indent=2)


def get_cached_features_batch(task_name: str, nctids: list[str], feature_name: str, plan) -> dict:
    """Batch lookup: returns {nctid: feature_values} for all cache hits."""
    plan_hash = _plan_content_hash(plan)
    results = {}
    for nctid in nctids:
        path = _store_path(task_name, feature_name, plan_hash, nctid)
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                results[nctid] = dill.loads(bytes.fromhex(data["feature_values_b"]))
    return results


def get_store_stats(task_name: str) -> dict:
    """Return cache statistics for monitoring."""
    store_path = Path(STORE_DIR) / task_name
    if not store_path.exists():
        return {"features": 0, "trials": 0, "total_entries": 0}
    features = set()
    total = 0
    trials = set()
    for feature_dir in store_path.iterdir():
        if feature_dir.is_dir():
            features.add(feature_dir.name.rsplit("--", 1)[0])
            for f in feature_dir.glob("*.json"):
                total += 1
                trials.add(f.stem)
    return {"features": len(features), "trials": len(trials), "total_entries": total}
```

#### 6.3.2 Modified WrappedFeatureBuilderV3

```python
class WrappedFeatureBuilderV3:
    def __init__(self, task: Task):
        self.task = task

    def __call__(self, arg):
        nctid, plans = arg

        # --- NEW: Check global feature store FIRST (per-feature granularity) ---
        cached_values = {}
        uncached_plans = {}
        for feature_name, plan in plans.items():
            cached = get_cached_feature(self.task.name, nctid, feature_name, plan)
            if cached is not None:
                cached_values[feature_name] = cached
            else:
                uncached_plans[feature_name] = plan

        # If all features are cached, return immediately (no LLM calls)
        if not uncached_plans:
            merged = {}
            for fv in cached_values.values():
                merged.update(fv)
            return (nctid, merged, {})

        # --- Build only uncached features ---
        try:
            fb = FeatureBuilderV3(task=self.task)
            fb = fb.activate_assertions(max_backtracks=5)
            values, meta = fb(nctid=nctid, feature_plan_group=uncached_plans)

            # Store each feature individually in the global store
            for feature_name, plan in uncached_plans.items():
                feature_value = {feature_name: values.get(feature_name, {})}
                put_cached_feature(
                    self.task.name, nctid, feature_name, plan,
                    feature_value, meta
                )

            # Merge cached + freshly built
            all_values = {}
            for fv in cached_values.values():
                all_values.update(fv)
            all_values.update(values)

            return (nctid, all_values, meta)
        except Exception as e:
            # ... error handling (unchanged)
```

#### 6.3.3 Semantic Plan Matching (Optional Enhancement)

The basic store uses exact `plan_content_hash` matching. For higher hit rates, add semantic matching:

```python
def find_similar_plan(task_name: str, feature_name: str, plan, similarity_threshold=0.9):
    """Find a cached plan that is semantically similar but not identical.

    Uses embedding similarity on feature_instructions to match plans that
    differ in wording but describe the same feature extraction.
    """
    store_path = Path(STORE_DIR) / task_name
    if not store_path.exists():
        return None

    target_instructions = plan.feature_instructions
    target_embedding = embed(target_instructions)  # sentence-transformers

    best_match = None
    best_similarity = 0

    for feature_dir in store_path.iterdir():
        dir_feature_name = feature_dir.name.rsplit("--", 1)[0]
        if dir_feature_name != feature_name:
            continue
        # Load one sample to get the original plan
        sample = next(feature_dir.glob("*.json"), None)
        if sample:
            with open(sample) as f:
                data = json.load(f)
            cached_embedding = embed(data["metadata"].get("feature_instructions", ""))
            similarity = cosine_similarity(target_embedding, cached_embedding)
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = feature_dir.name.rsplit("--", 1)[1]  # plan_hash

    return best_match
```

This catches cases where the proposer generates slightly different wording for the same feature concept.

### 6.5 REFINE Safety: Why the Store Never Corrupts Other Branches

**Invariant:** The global feature store is append-only. REFINE creates a NEW cache entry (different hash), never overwrites the original.

**Proof by construction:**

```
REFINE on branch A:
  Original plan.feature_idea = "Count drug targets"
  Refinement: proposer_result.feature_explanation = "Also consider off-target binding"
  Combined (agent.py line 2025-2029): "Count drug targets\n---\nAlso consider off-target binding"
  Planner receives combined idea → generates NEW feature_instructions
  plan_content_hash includes feature_idea → hash H2 ≠ H1 (guaranteed, strings differ)

Store after REFINE:
  (NCT001, "drug_targets", H1) → {value: 3}     ← Original, untouched
  (NCT001, "drug_targets", H2) → {value: 5}     ← Refined, new entry

Sibling branch B (still using original F2):
  Inherits raw_features from parent → has F2(H1) values
  Never consults the store for F2 — it already has the values
  Only consults store if it needs to BUILD a new feature
```

**When the store IS consulted for a refined feature:**
- Branch C independently proposes the same refinement of F2
- Planner generates plan with same combined idea → same H2 hash
- Store lookup for (nctid, "drug_targets", H2) → **hit** → correct reuse
- This is safe: same idea + same instructions = same values

**When the store is NOT consulted:**
- Branch D uses original F2 → has values from parent → no store lookup
- Branch E proposes a DIFFERENT refinement of F2 → idea differs → hash H3 ≠ H2 → store miss → builds fresh

**What about semantic matching (Section 6.3)?**
Semantic matching should be **disabled for features with a refinement chain** (i.e., where `feature_idea` contains `---`). A refined feature's identity depends on the exact refinement, and fuzzy matching could incorrectly return the original or a different refinement. Semantic matching is only safe for ADD operations where the feature_idea has no predecessor.

### 6.4 Research Results Caching (Separate Layer)

Currently, research and construction are coupled in `FeatureBuilderV3`. Separating them enables:

```
Trial NCT00110279 + "drug safety" research
  → Cached research results (PubMed articles, related trials, trial metadata)
    → Used by feature "adverse_event_count" (construction only)
    → Used by feature "safety_signal_strength" (construction only)  ← SAVES research LLM calls
```

**Implementation:**

```python
class FeatureBuilderV3(dspy.Module):
    def forward(self, nctid, feature_plan_group):
        nct_info = get_clinical_trial_info(nctid)

        # NEW: Check if research for this trial + feature set exists
        research_cache_key = (nctid, frozenset(feature_plan_group.keys()))
        cached_research = get_cached_research(research_cache_key)

        if cached_research is None:
            research_result = self.research_agent(
                task=self.task.value,
                nctid=nctid,
                feature_plans=simplified_plans,
            )
            put_cached_research(research_cache_key, research_result.research_results)
            research_text = research_result.research_results
        else:
            research_text = cached_research

        # Construction always runs (cheap, 1 LLM call)
        builder_result = self.constructor(
            feature_plans=serialized_plans,
            research_results=research_text,
        )
        return builder_result
```

**Savings:** The ReAct research agent (up to 5 tool-calling iterations) is ~70-80% of the feature builder cost. If the same trial has already been researched for overlapping features, skip the research and go straight to construction.

---

## 7. Projected Savings

### 7.1 Savings Model

**Assumptions:**
- 20 rollouts, depth 7, ~140 nodes explored
- 200 train + 200 val + 100 test = 500 trials per split call
- ~80 ADD/REFINE nodes, ~20 REMOVE nodes, ~40 simulation nodes
- Average 60% feature overlap between branches (same feature proposed with same or similar plan)
- 50% of simulation features are duplicates of selection-path features

### 7.2 With Global Feature Store Only

| Component | Current Calls | With Feature Store | Savings |
|-----------|---------------|-------------------|---------|
| Root node feature building | 9,000 | 9,000 (no savings, first time) | 0% |
| ADD node feature building (80 nodes) | 240,000 | 144,000 (40% cache hits from cross-branch reuse) | 40% |
| Simulation node feature building (40 nodes) | 120,000 | 36,000 (70% cache hits — most features already computed) | 70% |
| Other (proposer, planner, evaluator) | ~1,200 | ~1,200 (no change) | 0% |
| **Total** | **~370,000** | **~190,000** | **~49%** |

### 7.3 With Feature Store + Research Caching + Semantic Matching

| Component | Current Calls | Optimized | Savings |
|-----------|---------------|-----------|---------|
| Root node | 9,000 | 9,000 | 0% |
| ADD nodes (80) | 240,000 | 96,000 | 60% |
| Simulation nodes (40) | 120,000 | 18,000 | 85% |
| Other | ~1,200 | ~1,200 | 0% |
| **Total** | **~370,000** | **~124,000** | **~66%** |

### 7.4 Cost Impact

| Configuration | LLM Calls | Est. Cost (Opus 4.6) |
|---------------|-----------|----------------------|
| Current (no optimization) | ~370,000 | ~$740 |
| With feature store | ~190,000 | ~$380 |
| With feature store + research cache + semantic matching | ~124,000 | ~$248 |

**Savings per 20-rollout run: $360-$492**

Over a development campaign of ~15 runs (validation, benchmarking): **$5,400-$7,380 saved.**

---

## 8. Additional Optimizations

### 8.1 Simulation Cost Reduction (Complementary)

Beyond caching, simulation cost can be reduced:

**Option A: Skip feature building in simulation.** Use the parent's features directly and only run model training + evaluation. The simulation only needs a reward signal, not full feature extraction.

```python
def _simulate_cheap(self, node):
    """Lightweight simulation: reuse parent features, only train/evaluate."""
    while not node.is_terminal():
        child = self.rng.choice(node.potential_children_inputs())
        # Instead of running full agent, just train model with parent features
        reward = self._quick_evaluate(node.output_node)
        node = child
    return reward
```

**Savings:** Eliminates ~120,000 simulation LLM calls entirely. Combined with feature store: total calls drop from 370,000 to ~150,000.

**Risk:** Less accurate simulation rewards. Mitigation: use feature store for exact match, fall back to cheap simulation only when no cache hit.

**Option B: Predictive scoring for simulation** (from KompeteAI, see `mcts-alternatives-and-improvements.md`). Use Opus 4.6 few-shot to predict feature set quality without running the full pipeline. 1 LLM call vs 3,000.

### 8.2 Feature Plan Canonicalization

The proposer and planner are non-deterministic — they may generate semantically identical but textually different plans. Canonicalize plans before hashing:

```python
def canonicalize_plan(plan: FeaturePlanV2) -> FeaturePlanV2:
    """Normalize plan text to maximize cache hits."""
    return plan._replace(
        feature_instructions=" ".join(plan.feature_instructions.lower().split()),
        feature_idea="",  # Idea is for the LLM, not for caching
        example_values=[],  # Examples vary but don't change the feature definition
    )
```

### 8.3 Batch Feature Store Queries

For ADD operations, query the store for ALL trials at once before spawning workers:

```python
def compute_features_v3_optimized(grouper, nctids, task, plans):
    # Check which (nctid, feature) pairs are already cached
    for feature_name, plan in plans.items():
        cached = get_cached_features_batch(task.name, nctids, feature_name, plan)
        uncached_nctids = [nid for nid in nctids if nid not in cached]

    if not uncached_nctids:
        return cached_results  # 100% cache hit, 0 LLM calls

    # Only build for uncached trials
    product = itertools.product(uncached_nctids, grouped_plans)
    new_results = pool.map(WrappedFeatureBuilderV3(task), product)

    return merge(cached_results, new_results)
```

### 8.4 Warm-Start Feature Store

Pre-populate the feature store with common features before starting MCTS:

```python
def warm_start_feature_store(task, nctids, common_features):
    """Pre-compute features that are likely to be proposed by MCTS.

    Run this once before train_mcts.py. Common features can be derived from:
    - Previous MCTS runs (which features appeared most often)
    - Domain knowledge (trial_phase, enrollment_count, etc.)
    - LLM-proposed features from a quick zero-shot pass
    """
    for feature_name, plan in common_features.items():
        compute_features_v3(grouper, nctids, task, {feature_name: plan})
        # Results automatically stored in feature store
```

---

## 9. Implementation Plan

### Phase 1: Global Feature Store (Core, 4 days)

| Day | Task | Output |
|-----|------|--------|
| 1 | Implement `feature_store.py` with `get_cached_feature()`, `put_cached_feature()`, `_plan_content_hash()` | Tested module with unit tests |
| 2 | Modify `WrappedFeatureBuilderV3` to check/write feature store before/after building | Modified agent.py with per-feature caching |
| 2 | Add `get_cached_features_batch()` for efficient batch lookups | Batch query support |
| 3 | Add store statistics logging: hit rate, unique features, trial coverage | Monitoring |
| 3 | Add `--warm-start` flag to `train_mcts.py` | Warm-start support |
| 4 | Testing: verify cache correctness on TrialBench Phase II with 3 rollouts | Validated savings |

### Phase 2: Research Results Caching (3 days)

| Day | Task | Output |
|-----|------|--------|
| 5 | Separate research and construction in `FeatureBuilderV3.forward()` | Refactored builder |
| 6 | Implement research results cache keyed by `(nctid, feature_set)` | Research cache |
| 7 | Testing: measure research cache hit rate on TrialBench | Validated savings |

### Phase 3: Simulation Optimization (2 days)

| Day | Task | Output |
|-----|------|--------|
| 8 | Implement feature-store-aware simulation: check store first, fall back to cheap eval | Modified `_simulate()` |
| 9 | Testing: compare reward accuracy with/without cheap simulation | Validated approach |

### Phase 4: Semantic Matching (Optional, 2 days)

| Day | Task | Output |
|-----|------|--------|
| 10 | Implement `find_similar_plan()` with embedding similarity | Semantic matcher |
| 11 | Testing: measure incremental cache hits from semantic matching | Marginal benefit quantified |

**Total: 9-11 days**

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Stale cache entries** — plan hash collisions return wrong feature values | VERY LOW | HIGH | Use 16-char SHA-256 prefix (collision probability ~10^-19). Add validation: verify returned feature names match expected names. |
| **Cache invalidation** — data sources updated but cache not cleared | LOW | MEDIUM | Add `--clear-feature-store` flag. Store creation timestamp in metadata. Auto-expire entries older than configurable TTL. |
| **REFINE produces hash collision with original** — refined plan hashes to same key as original, returning stale values | VERY LOW | HIGH | Hash includes `feature_idea` which physically differs on REFINE (appends `\n---\n{refinement}`). This guarantees H_refined ≠ H_original. See Section 6.5 for proof. |
| **Plan canonicalization loses semantic differences** — two genuinely different features hash to the same key | LOW | MEDIUM | Hash includes feature_idea, feature_type, data_sources, feature_instructions, possible_values — all structural fields. Only example_values excluded. |
| **Disk space growth** — 500 trials × 20 features × 16 bytes hash = ~160K files | LOW | LOW | Each file is ~1-5KB. Total: ~160-800MB for a full run. Acceptable. Add periodic cleanup of unused entries. |
| **Subprocess cache visibility** — spawned workers don't see main process writes | VERY LOW | LOW | Feature store uses file system (not in-memory). File writes are atomic (write to temp + rename). ProcessPoolExecutor workers see all files. |
| **Semantic matching returns false positives** — similar-looking but different features reuse wrong values | MEDIUM | MEDIUM | Set high similarity threshold (0.95). Only enable semantic matching after manual validation of match quality on 100 samples. Default: disabled. |

---

## Appendix: Key Code Locations

| Component | File | Lines |
|-----------|------|-------|
| MCTS rollout loop | `treesearch.py` | 113-124 |
| Node exploration trigger | `treesearch.py` | 301-319 |
| Simulation (creates temporary nodes) | `treesearch.py` | 321-350 |
| Subprocess launcher | `agent.py` | 2714-2754 |
| Agent pipeline entry | `agent.py` | 1989-2173 |
| Feature deepcopy from parent | `agent.py` | 2004-2007 |
| ADD: only new feature built | `agent.py` | 2038-2047 |
| REMOVE: no building | `agent.py` | 2070-2099 |
| compute_features_v3 | `agent.py` | 1778-1801 |
| WrappedFeatureBuilderV3 (current cache) | `agent.py` | 1709-1775 |
| FeatureBuilderV3 (LLM research + construction) | `agent.py` | 1138-1338 |
| Feature builder file cache key (group hash) | `agent.py` | 1722-1730 |
| diskcache setup | `globals.py` | 13-15 |
| DSPy cache setup | `globals.py` | 36-42 |
| ProcessPoolExecutor | `globals.py` | 47-61 |
