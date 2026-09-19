# Cache-reuse measurement: feature store and agent cache

**Issue:** [#17](https://github.com/vdavamse/ctra/issues/17) — instrument cache hit rates and measure real cross-branch reuse.
**Status:** instrumentation shipped; offline mechanism validation done; **production cross-branch hit rate not measured** (requires an LLM-backed run, see [§5](#5-production-cross-branch-hit-rate-not-measured)).
**Date:** 2026-09-19

## 1. What is instrumented

Three cache-like layers exist in the MCTS training loop. Only the last two are caches; the first is listed because it is what people usually mean by "reuse" and it was never counted.

| Layer | Where it runs | What it saves | Counted by |
|---|---|---|---|
| (a) Value carry-forward | `Agent.forward` (subprocess) | On iteration N the parent's `raw_features` are deep-copied and only the one changed plan is built. Same-path reuse, by construction. | Not a counter: it is 100% of the inherited plans on every iteration-N path. |
| (b) Feature store | `compute_features` / `WrappedFeatureBuilder` (subprocess); disk at `output/feature_store/{phase}/{feature}--{plan_hash}/{nctid}.json` | A `(trial, plan text)` pair built once is served to any later caller with the byte-identical plan text — another branch, another rollout, another run. | `FeatureStoreCounters` (`src/ctra/agents/data_models.py`) |
| (c) Agent-output cache | `run_agent_as_subprocess` (parent); pickles at `<output_dir>/<phase>/agent_cache/<run_id>/` | A whole evaluation, keyed by rollout + lineage + parent plans + features. Crash recovery: within a run the key never repeats (the rollout is in it), so hits only happen on resume. | `RunCacheStats` (`src/ctra/agents/runner.py`) |

### Feature-store counters (`FeatureStoreCounters`)

| Counter | Meaning |
|---|---|
| `feature_lookups` | `(trial, feature)` pairs probed in the store. |
| `feature_hits` | ... of those, served from the store. `hit_rate = feature_hits / feature_lookups` (0.0 before any lookup). |
| `hits_from_initializer_plans` | Hits on plans minted by the Initializer (iteration 0). Within one run these are 0: the root is evaluated once. They only appear when the store persisted from an earlier run. |
| `hits_from_planner_plans` | Hits on plans minted by the FeaturePlanner on an iteration-N path. Every such plan is freshly generated, so a hit here means two independently generated plan texts collided — this is the cross-branch signal. One same-branch source exists: an evaluation whose subprocess crashed after some store writes but before its pickle landed is re-run on resume, and with dspy's disk cache at temperature 0 the planner reproduces the byte-identical text, so its remaining lookups hit; negligible in count, and lineage stamping (§7) is the fix. |
| `groups_dispatched` | `(trial, plan-group)` pairs sent to the builder (LLM calls happen). |
| `groups_skipped` | `(trial, plan-group)` pairs fully served from the store (zero LLM calls). |
| `store_writes` | Feature values persisted after a build. |

Single-owner rule: the upfront batch probe in `compute_features` counts lookups, hits and group decisions for the groups it probed; `WrappedFeatureBuilder` re-probes a partially cached group on disk but, told `batch_probed=True`, counts nothing but its own writes. A partially cached two-feature group is therefore 2 lookups, 1 hit, 1 dispatch — never 4 lookups / 2 hits (`tests/test_agents/test_cache_counters.py::test_partial_group_is_counted_once`).

### Agent-cache counters (`RunCacheStats`)

`agent_hits` / `agent_misses` at the runner; `agent_hit_rate`; `replayed_group_builds` (what a replayed pickle *would* have rebuilt: its `cache_stats.groups_dispatched`); `feature_store` (the children's counters, folded in only on a miss — a replayed pickle carries the counters of the run that wrote it); `llm_calls_made` (sum of the children's calibration figures).

### Transport

The store runs in the `scripts/run_agent.py` subprocess, so its counters ride back inside `AgentOutput.cache_stats` (a `CacheStats`: the store counters plus `llm_calls_made`). `Agent.forward` creates a fresh set per call; the three skip paths (exhausted suggestions, invalid proposer, unhandled operation) return a parent copy with a **zeroed** set so a skipped iteration never replays its parent's traffic. Outputs pickled before the field existed are backfilled with zeros on load.

## 2. "LLM calls avoided"

```
LLM_CALLS_PER_GROUP_BUILD = 6
llm_calls_avoided_estimate = groups_skipped * 6
```

Six is a **single-attempt point estimate** (k ≈ 4 ReAct steps, no Refine retry) of what one dispatched trial-group build costs, not a bound. One `FeatureBuilder.forward` attempt is k ReAct steps (`dspy.ReAct(max_iters=5)`: 1 ≤ k ≤ 5, the loop stops at `finish`) + 1 ReAct extract call + 1 ChainOfThought Construct call = k + 2, i.e. 3–7 calls (7 when ReAct exhausts its 5 steps). Under `ResettingRefine(N=3)` a group can take up to 3 attempts plus 2 `OfferFeedback` calls (one after each attempt but the last, when the reward is below the 1.0 threshold), so a dispatched group costs 3–23 calls (`feature_builder.py`, dspy `react.py` / `refine.py`); `llm_calls_avoided_estimate` therefore leans low when retries are common. The calibration below replaces it with the measured figure.

Only a fully cached group avoids a build. A hit inside a group that is still dispatched avoids nothing (the builder runs for the group anyway), so per-feature hits are deliberately **not** multiplied; `features_served_from_store` is reported separately as `feature_hits`.

**Calibration.** `scripts/run_agent.py` registers an `LMCallCounter` (a `dspy.utils.callback.BaseCallback` whose `on_lm_start` fires once per `LM.__call__`) on dspy's global callbacks and records its count at the end of each iteration into `cache_stats.llm_calls_made` — every LM call the iteration made (proposer, planner, evaluator, grouper, builder). Counting at the source keeps the figure unbounded; `len(dspy.clients.base_lm.GLOBAL_HISTORY)` would saturate at dspy's `MAX_HISTORY_SIZE` (10,000 entries) on exactly the large iteration-0 evaluations the figure is meant to calibrate, and is empty under `settings.disable_history`. Calls served by dspy's own response cache **are included** (`on_lm_end` receives the completions, not the response's `cache_hit` flag), so on a re-run against a warm dspy cache the figure counts calls, not paid calls. A real run therefore yields `llm_calls_made` alongside `groups_dispatched`; the ratio, after subtracting the per-iteration fixed cost (proposer + planner + evaluator, ~3–9 calls under Refine), is the measured multiplier to compare with 6.

## 3. Where the numbers land

- **Log:** one line per rollout from `train_mcts.py`: `Rollout k/N cache — agent hits/misses=…, feature-store hit rate=…% (hits/lookups), groups skipped=…, LLM calls avoided~…, LLM calls made=…` (running totals; `on_rollout` sees only the last node of a deep rollout, so the totals live at the runner boundary).
- **`results.json["cache"]`:** `RunCacheStats.as_dict()` for the whole run. The object is pickled in every checkpoint (it is bound into the runner partial), and a resume adopts it, so the counters continue across a crash; the evaluations the crashed process ran after its last checkpoint replay from its pickles and count as `agent_hits` on top.
- **MLflow** (opt-in, `--mlflow`): `ExperimentTracker.log_cache_stats` logs `agent_cache_hit_rate`, `feature_store_hit_rate`, `groups_skipped`, `llm_calls_avoided_estimate`, `llm_calls_made` and every raw counter, once per rollout (`step` = rollout index) and once more after the search with the run totals at `step` = `num_rollouts` (one past the last rollout, so the totals do not collide with the rollout-0 point).
- **Summary print** at the end of `train_mcts.py`.

## 4. Offline measurement (mechanism validation, synthetic plans)

`scripts/eval/measure_cache_reuse.py` drives the **production** `MCTSSearch` and the **production** `Agent.forward` — orchestrator, `compute_features`, `WrappedFeatureBuilder`, a real feature store in a temporary directory — with the five dspy modules replaced by deterministic stubs (the patch set of `tests/test_agents/test_orchestrator.py`) and the model block reduced to a dummy classifier under a synthetic ROC-AUC landscape. The agent-cache layer is an in-process mirror of `run_agent_as_subprocess` (same key, same `RunCacheStats.record_hit/record_miss`, a dict instead of the pickle directory).

**The collision rate is an input.** The initializer always mints the same three plans. On iteration N the planner stub returns, for a feature name, the canonical plan text with probability `--collision-rate`, otherwise a uniquely worded one. Two branches that both add `feat_C` collide in the store only when both drew the canonical text. Nothing here says how often Claude does that.

Configuration: 10 rollouts, depth 5, 3 suggestions per node, deep simulation on, 10 trials (6/2/2), 3 initializer plans, an 8-feature pool; 10 seeds per rate (seeds vary the landscape noise and the collision draws). Every search made 43 evaluations (1 root + 42 iteration-N), 42 planner calls and 450 store lookups (30 initializer + 420 planner). Reproduce with:

```bash
python scripts/eval/measure_cache_reuse.py --collision-rate 0,0.25,0.5,1 --seeds 10 --rollouts 10 --replay
```

| collision rate | store hit rate (mean, min–max) | planner-plan hit rate (mean, min–max) | groups skipped (mean, min–max) | groups dispatched (mean) | LLM calls avoided~ (mean) | replay: agent hits / evaluations |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0% (0.0–0.0) | 0.0% (0.0–0.0) | 0 (0–0) | 430 | 0 | 43 / 43 |
| 0.25 | 12.0% (6.7–20.0) | 12.9% (7.1–21.4) | 54 (30–90) | 376 | 324 | 43 / 43 |
| 0.50 | 31.6% (20.0–44.4) | 33.8% (21.4–47.6) | 142 (90–200) | 288 | 852 | 43 / 43 |
| 1.00 | 77.8% (77.8–77.8) | 83.3% (83.3–83.3) | 350 (350–350) | 80 | 2100 | 43 / 43 |

Reading the table:

- At collision rate 0 there are **no** planner-plan hits: every iteration-N plan is new text, and the value carry-forward (layer a) is what keeps the search from rebuilding inherited plans. The store does nothing within a single run unless plan texts collide.
- At collision rate 1 the ceiling is 83.3% of planner lookups, not 100%: the first branch to plan each feature name still builds it. The 80 in the `groups dispatched` column is the root's 10 (its one initializer group, dispatched once per trial and never a hit within a run) + 7 reachable pool names × 10 trials = 70 planner dispatches; `feat_H` is unreachable at depth 5, because the evaluator stub suggests the first three missing pool names in `POOL` order, so five additions can only reach `feat_A`–`feat_G` (420 planner lookups − 350 hits = 70 misses = 7 names). The store hit rate (77.8%) is lower than the planner-plan rate because the 30 initializer lookups never hit within one run.
- The avoided-calls estimate is `groups_skipped × 6`; a real run replaces the 6 with the calibrated multiplier.
- `--replay` (the resume scenario) replays every evaluation from the agent cache: 43 hits, 0 misses, `replayed_group_builds == groups_dispatched` of the first pass, and **nothing** folded into the feature-store counters. Within a single search the agent hit rate is 0 by construction (the rollout index is part of the node id).

### Cross-run reuse (`--shared-store`)

One store shared by the 10 seeds of a rate (the production layout: `output/feature_store` persists across runs and phases):

| collision rate | seed 0 | seeds 1–9 (each) |
|---:|---|---|
| 0.00 | 0 initializer hits, 0 planner hits | 30 initializer hits (3 plans × 10 trials), 0 planner hits → store hit rate 6.7%, 10 groups skipped |
| 1.00 | 0 initializer hits, 350 planner hits (77.8%) | 30 initializer hits, 420 planner hits → store hit rate 100%, 430 groups skipped |

With unique plan texts the persisted store serves exactly the initializer's plans to the next run; everything else depends on text collisions.

## 5. Production cross-branch hit rate: not measured

The number the issue asks for — how often two branches of a real search send byte-identical plan text to the store — depends on Claude's wording and **requires an LLM-backed run** with the datasets and API keys, which are not available in this environment. It is therefore recorded here as **not measured**. To measure it:

```bash
python scripts/train_mcts.py --task phase2 --rollouts 20 --mlflow
```

and read, in order of convenience:

1. the per-rollout `cache —` log lines (running totals);
2. `.output/phase2/results.json` → `"cache"` → `feature_store.hits_from_planner_plans`, `feature_store.feature_lookups`, `feature_store.groups_skipped`, `llm_calls_made`;
3. the MLflow run (`mlruns/`, experiment `ctra`) → metrics `feature_store_hit_rate`, `hits_from_planner_plans`, `llm_calls_avoided_estimate`, `llm_calls_made` (one point per rollout, plus the run totals at `step` = `num_rollouts`).

The planner-plan hit rate to report is `hits_from_planner_plans / planner_lookups`, where `planner_lookups = feature_lookups − |initializer plans| × (train + val + test trials)`: the root evaluation is the only one that sends initializer plans, and every lookup after it is a planner lookup. Run it twice against the same store to also read the cross-run figure (`hits_from_initializer_plans` in the second run).

## 6. Recommendation (conditional, thresholds fixed in advance)

| Measured planner-plan hit rate (single cold run, ≥ 20 rollouts) | Decision |
|---|---|
| **< 5%** | Keep the store as-is (it is still the crash-recovery and cross-run layer for initializer plans); do not invest in plan normalisation. Report the number in the README's output section. |
| **≥ 5%** | Byte-identical collisions already happen; near-identical ones are likely more frequent. Add plan normalisation to `plan_content_hash` (whitespace/case folding of `feature_idea` and `feature_instructions`, then a semantic-equivalence pass) and re-measure. |
| **Cross-phase sharing** | Only if the second-run `hits_from_initializer_plans` is high **and** stays high when the run is a different phase (today the store is namespaced per phase, so this needs a `--shared-namespace` experiment first). |

Regardless of the rate: if the calibrated multiplier (`llm_calls_made` per dispatched group, fixed cost subtracted) is far from 6, update `LLM_CALLS_PER_GROUP_BUILD` and re-state the estimate.

## 7. Deferred

- Lineage stamping (which node/rollout wrote each store entry) would let a hit be attributed to a specific branch pair rather than to "a planner plan"; it needs `--node-id` plumbing through `runner.py` and `run_agent.py` and a metadata-returning read path.
- Splitting `llm_calls_made` into paid calls and dspy-cache hits: the callback API does not expose the `cache_hit` flag, so it would need a wrapper around `LM.forward`.
