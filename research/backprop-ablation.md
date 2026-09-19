# Backpropagation Ablation: Mean Vector vs Per-Objective Max vs Max Hypervolume

> Issue [#16](https://github.com/vdavamse/ctra/issues/16). Measured 2026-09-19 on branch `feat/16-backprop-ablation` with `scripts/eval/ablate_backprop.py --regime all --seeds 20` (660 searches, 61.6 s). Decision: **the default stays `mean`.** The knob is `MCTSConfig.backprop` (`CTRA_MCTS_BACKPROP=mean|max|max_hv`).

---

## 1. The question

CTRA's MCTS backpropagates the full objective vector `[accuracy, parsimony]` of every evaluated node up the parent chain and averages it (`MCTSNode.mean_reward`), the way PMMG's vector MCTS does ([mcts-alternatives-and-improvements.md](mcts-alternatives-and-improvements.md), "How PMMG handles it"). AutoCT's `treesearch.py` instead backpropagates the **maximum** reward. Issue #16 asks whether the choice matters for what the search finds.

Backprop state reaches the search through exactly one path: `MCTSNode.ucb_scores` (mean + a scalar exploration bonus) → `pareto_select` (non-dominated sort of the siblings' UCB vectors). The final pick (`_select_best`, issue #15) reads each node's own `objective_history`, not backprop state. So a different rule can change **only which nodes get evaluated**, and only where `pareto_select` has to choose among *visited* siblings — it picks an unvisited child outright before computing any UCB.

## 2. The three rules

All three keep `visit_count` = number of evaluations through the node and store their aggregate `A` as `total_reward = A * visit_count`, so `mean_reward`, `ucb_scores` and `value()` are unchanged code and no node state was added (checkpoints round-trip). The `A * visit_count / visit_count` round-trip costs about one ulp per backpropagation, so the error grows with the visits: about visits × 1e-16, measured 8.6e-13 under `max` and 2.2e-12 under `max_hv` after 200,000 visits through one node — roughly three orders below `_TIE_ATOL` (1e-9).

| Rule | `A(node)` after backpropagating vectors `v_1..v_k` through it | Properties |
|---|---|---|
| `mean` (default) | `(v_1 + … + v_k) / k` — the subtree mean | PMMG-style; unchanged behaviour |
| `max` | elementwise `max(v_1, …, v_k)` | Order-independent, monotone. Can **synthesise a point no evaluation scored**: with ADD-only children parsimony falls with depth, so the parsimony coordinate pins to the ancestor's own value and sibling fronts collapse to an accuracy argmax |
| `max_hv` | the one realised `v_i` maximising `(point hypervolume above reference_point, accuracy)` — the key `_best_own_objectives` ranks by | Always a realised vector (the control for `max`'s synthesis). Accuracy breaks ties among vectors below the reference (all have HV 0); a higher-accuracy vector with HV 0 never displaces one above the reference |

**The AutoCT nuance.** AutoCT's `_simulate()` returns the max scalar reward along the root-to-leaf *path* of one rollout, then backpropagates that one scalar into the *selection path* where it is averaged with the other rollouts' path maxima (path-max-then-mean, one number per rollout). CTRA's replica (`tests/test_search/test_autoct_mcts_comparison.py::AutoCTMCTS`) does the same. `max` above is the stronger, vectorised reading the issue asks for: a per-node running maximum over *every* vector backpropagated through it, with every deep-path node backpropagating its own vector (CTRA already backpropagates simulation nodes, unlike AutoCT).

## 3. The harness

`scripts/eval/ablate_backprop.py` runs the production `MCTSSearch` under each rule on the synergy landscape of `test_autoct_mcts_comparison.py::make_synergy_evaluator`: `feat_A..D` add +0.03 ROC-AUC each, +0.08 at three of them, +0.20 at all four, with N(0, 0.01) noise per evaluation, beside `noise_1..3` that add nothing (pool of 7, order shuffled per seed). Root = `["start"]`, `max_depth 7`, `max_features 50`, `exploration_constant 1.0`, `reference_point [0.5, 0.0]`, `min_branch_factor 2`.

**Why the issue's own wiring could not be used.** `test_autoct_mcts_comparison.py` drives CTRA through `make_stub_runner`, which rebuilds each `AgentOutput` from the *parent's* plans and never applies the suggestion the child follows. Every node in the tree is therefore evaluated on the root's 1-feature set (`sizes seen = [1]`, 63/63 evaluations) and the landscape is never reached — the comparison is vacuous on the CTRA side. The harness carries its own **feature-aware runner** (the `_FeatureAwareRunner` shape from `test_mcts_deep_exploration.py`: suggestion `i` ADDs the `i`-th not-yet-carried pool feature, and the output's plans are the evaluated node's own set), copied rather than imported so the script does not depend on the test suite. Two in-harness assertions refuse a table the fixture cannot have produced: each regime must evaluate a feature set larger than the root and a set with all four synergy features at least once (`tests/test_scripts/test_ablate_backprop.py` fences the first with a parent-plans runner).

**Metrics per (cell, variant, seed):** evaluations; best own ROC-AUC (`best_own_objectives(best)[0]`); depth of the best node; whether a 4-synergy set was ever evaluated ("found") and after how many evaluations ("first", NaN when never); synergy features in the returned best set ("syn"); nodes; the deepest node reached (`depth_reached`); and **UCB-decided selections** — `pareto_select` calls where every candidate had been visited, the only calls that read backprop state. "= mean" counts the seeds on which the variant's search ended exactly where `mean`'s did.

**Regimes** (`--regime`): `issue` — the issue's factorial (deep × adaptive at 10 rollouts, branch 3) plus a shallow control at the deep cell's evaluation budget; `saturated` — enough revisits for UCB to decide most selections (shallow 120 rollouts at branch 2 and 3, deep 40 rollouts at branch 2); `deep-wide` — deep 40 rollouts with the production branch range (min 2, max 8), fixed and adaptive.

**Seed-axis caveat.** CTRA's own RNGs are rollout-seeded (`default_rng([rollout, 1])` for selection, `default_rng(rollout)` for the deep path), so seeds vary only the *landscape*: the noise stream and the pool order. Twenty seeds are twenty landscapes, not twenty search replicates. Rollouts are not a comparable budget across deep/shallow (63 vs 11 evaluations at 10 rollouts), so evaluations are reported too.

## 4. Results (20 seeds, per-cell means)

`found`, `syn`, `first` are over the seeds; `first` averages the seeds that found it. `UCB` = mean UCB-decided selections per search. `= mean` = seeds identical to `mean` (of 20).

### 4.1 `issue` regime

| variant | deep | adaptive | rollouts | branch | evals | AUC | depth | found | syn | first | nodes | depth reached | UCB | = mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | yes | no | 10 | 3 | 63.0 | 0.8295 | 6.25 | 1.00 | 4.0 | 7.7 | 130.0 | 7.0 | 8.0 | 20 |
| max | yes | no | 10 | 3 | 63.0 | 0.8293 | 6.25 | 1.00 | 4.0 | 7.7 | 130.0 | 7.0 | 8.0 | 18 |
| max_hv | yes | no | 10 | 3 | 63.0 | 0.8295 | 6.25 | 1.00 | 4.0 | 7.7 | 130.0 | 7.0 | 8.0 | 20 |
| mean | yes | yes | 10 | 3 | 55.0 | 0.8364 | 6.45 | 1.00 | 4.0 | 7.7 | 81.0 | 7.0 | 16.0 | 20 |
| max | yes | yes | 10 | 3 | 55.0 | 0.8360 | 6.35 | 1.00 | 4.0 | 7.7 | 81.0 | 7.0 | 16.0 | 19 |
| max_hv | yes | yes | 10 | 3 | 55.0 | 0.8364 | 6.45 | 1.00 | 4.0 | 7.7 | 81.0 | 7.0 | 16.0 | 20 |
| mean | no | no | 10 | 3 | 11.0 | 0.5639 | 2.00 | 0.00 | 1.9 | nan | 13.0 | 2.0 | 7.0 | 20 |
| max | no | no | 10 | 3 | 11.0 | 0.5638 | 2.00 | 0.00 | 1.9 | nan | 13.0 | 2.0 | 7.0 | 18 |
| max_hv | no | no | 10 | 3 | 11.0 | 0.5639 | 2.00 | 0.00 | 1.9 | nan | 13.0 | 2.0 | 7.0 | 17 |
| mean | no | yes | 10 | 3 | 11.0 | 0.5833 | 2.90 | 0.00 | 1.9 | nan | 15.0 | 3.0 | 12.0 | 20 |
| max | no | yes | 10 | 3 | 11.0 | 0.5819 | 2.75 | 0.00 | 1.9 | nan | 15.0 | 3.0 | 12.0 | 18 |
| max_hv | no | yes | 10 | 3 | 11.0 | 0.5833 | 2.90 | 0.00 | 1.9 | nan | 15.0 | 3.0 | 12.0 | 20 |
| mean | no | no | 63 (equal evals) | 3 | 64.0 | 0.6602 | 3.60 | 0.00 | 2.8 | nan | 112.0 | 4.0 | 135.0 | 20 |
| max | no | no | 63 (equal evals) | 3 | 64.0 | 0.6608 | 3.60 | 0.00 | 2.8 | nan | 111.8 | 4.0 | 135.0 | 9 |
| max_hv | no | no | 63 (equal evals) | 3 | 64.0 | 0.6602 | 3.70 | 0.00 | 2.8 | nan | 111.8 | 4.0 | 135.0 | 8 |

At the issue's budget the variants are indistinguishable by construction: 8-16 UCB-decided selections per search and identical results on 17-20 of 20 seeds. This reproduces Phase 1's finding rather than contradicting it.

### 4.2 `saturated` regime

| variant | deep | adaptive | rollouts | branch | evals | AUC | depth | found | syn | first | nodes | depth reached | UCB | = mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | no | no | 120 | 2 | 121.0 | 0.8392 | 5.95 | 1.00 | 4.0 | 78.5 | 127.0 | 6.0 | 486.1 | 20 |
| max | no | no | 120 | 2 | 121.0 | 0.8400 | 6.40 | 1.00 | 4.0 | 79.3 | 133.2 | 6.7 | 493.1 | 4 |
| max_hv | no | no | 120 | 2 | 121.0 | 0.8322 | 5.90 | 1.00 | 4.0 | 77.3 | 132.6 | 6.7 | 493.0 | 4 |
| mean | no | yes (min==max) | 120 | 2 | 121.0 | 0.8392 | 5.95 | 1.00 | 4.0 | 78.5 | 127.0 | 6.0 | 486.1 | 20 |
| max | no | yes (min==max) | 120 | 2 | 121.0 | 0.8400 | 6.40 | 1.00 | 4.0 | 79.3 | 133.2 | 6.7 | 493.1 | 4 |
| max_hv | no | yes (min==max) | 120 | 2 | 121.0 | 0.8322 | 5.90 | 1.00 | 4.0 | 77.3 | 132.6 | 6.7 | 493.0 | 4 |
| mean | no | no | 120 | 3 | 121.0 | 0.7379 | 3.95 | 0.35 | 3.4 | 68.0 | 127.3 | 5.0 | 308.1 | 20 |
| max | no | no | 120 | 3 | 121.0 | 0.7387 | 4.10 | 0.35 | 3.4 | 78.1 | 135.4 | 4.8 | 310.8 | 0 |
| max_hv | no | no | 120 | 3 | 121.0 | 0.7387 | 4.20 | 0.35 | 3.4 | 67.9 | 138.1 | 4.8 | 311.9 | 0 |
| mean | yes | no | 40 | 2 | 143.0 | 0.8427 | 6.45 | 1.00 | 4.0 | 7.7 | 167.0 | 7.0 | 138.0 | 20 |
| max | yes | no | 40 | 2 | 143.0 | 0.8425 | 6.40 | 1.00 | 4.0 | 7.7 | 167.0 | 7.0 | 138.0 | 17 |
| max_hv | yes | no | 40 | 2 | 143.0 | 0.8428 | 6.45 | 1.00 | 4.0 | 7.7 | 167.0 | 7.0 | 138.0 | 17 |

Here the variants do diverge (0-4 of 20 seeds identical on the shallow cells, 300-490 UCB-decided selections per search) — and the outcome does not move: `max` is +0.0008 AUC on both shallow cells and −0.0002 on the deep one; `max_hv` is −0.0070, +0.0008 and +0.0001. The per-evaluation noise SD is 0.01, and the 4-synergy plateau sits at ≈ 0.82-0.85, so ±0.0008 is inside a single evaluation's noise.

### 4.3 `deep-wide` regime

| variant | deep | adaptive | rollouts | branch | evals | AUC | depth | found | syn | first | nodes | depth reached | UCB | = mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | yes | no | 40 | 8 | 248.0 | 0.8374 | 5.75 | 1.00 | 4.0 | 7.5 | 650.0 | 7.0 | 33.0 | 20 |
| max | yes | no | 40 | 8 | 248.0 | 0.8367 | 5.90 | 1.00 | 4.0 | 7.5 | 650.0 | 7.0 | 33.0 | 12 |
| max_hv | yes | no | 40 | 8 | 247.9 | 0.8225 | 5.50 | 1.00 | 3.9 | 7.5 | 649.5 | 7.0 | 33.1 | 10 |
| mean | yes | yes | 40 | 8 | 143.0 | 0.8427 | 6.45 | 1.00 | 4.0 | 7.7 | 167.0 | 7.0 | 138.0 | 20 |
| max | yes | yes | 40 | 8 | 143.0 | 0.8425 | 6.40 | 1.00 | 4.0 | 7.7 | 167.0 | 7.0 | 138.0 | 17 |
| max_hv | yes | yes | 40 | 8 | 143.0 | 0.8428 | 6.45 | 1.00 | 4.0 | 7.7 | 167.0 | 7.0 | 138.0 | 17 |

### 4.4 Paired differences against `mean` (best own AUC, same seed and cell)

| cell | variant | mean Δ | wins / ties / losses | max \|Δ\| | seeds diverged |
|---|---|---|---|---|---|
| issue, deep, fixed, r10 b3 | max | −0.0002 | 1 / 18 / 1 | 0.0035 | 2 |
| issue, deep, fixed, r10 b3 | max_hv | 0.0000 | 0 / 20 / 0 | 0.0000 | 0 |
| issue, deep, adaptive, r10 b3 | max | −0.0003 | 0 / 19 / 1 | 0.0062 | 1 |
| issue, deep, adaptive, r10 b3 | max_hv | 0.0000 | 0 / 20 / 0 | 0.0000 | 0 |
| issue, shallow, fixed, r10 b3 | max | −0.0001 | 1 / 18 / 1 | 0.0035 | 2 |
| issue, shallow, fixed, r10 b3 | max_hv | −0.0000 | 2 / 17 / 1 | 0.0035 | 3 |
| issue, shallow, adaptive, r10 b3 | max | −0.0014 | 1 / 18 / 1 | 0.0322 | 2 |
| issue, shallow, adaptive, r10 b3 | max_hv | 0.0000 | 0 / 20 / 0 | 0.0000 | 0 |
| issue, shallow, fixed, r63 b3 (equal evals) | max | +0.0006 | 5 / 9 / 6 | 0.0165 | 11 |
| issue, shallow, fixed, r63 b3 (equal evals) | max_hv | −0.0000 | 5 / 8 / 7 | 0.0165 | 12 |
| saturated, shallow, r120 b2 | max | +0.0008 | 3 / 12 / 5 | 0.0155 | 16 |
| saturated, shallow, r120 b2 | max_hv | −0.0070 | 5 / 10 / 5 | 0.1530 | 16 |
| saturated, shallow, r120 b3 | max | +0.0008 | 12 / 0 / 8 | 0.0165 | 20 |
| saturated, shallow, r120 b3 | max_hv | +0.0008 | 11 / 2 / 7 | 0.0165 | 20 |
| saturated, deep, r40 b2 | max | −0.0002 | 0 / 17 / 3 | 0.0019 | 3 |
| saturated, deep, r40 b2 | max_hv | +0.0001 | 1 / 17 / 2 | 0.0043 | 3 |
| deep-wide, fixed, r40 b8 | max | −0.0007 | 4 / 12 / 4 | 0.0385 | 8 |
| deep-wide, fixed, r40 b8 | max_hv | −0.0148 | 4 / 11 / 5 | 0.1501 | 10 |

The `saturated` adaptive r120/b2 cell and the `deep-wide` adaptive cell duplicate the rows above them (section 5).

### 4.5 Per-seed rows, `saturated` regime

Best own AUC per variant (`mean / max / max_hv`), then best depth, synergy features in the best set, evaluations to the first 4-synergy set (`-` = never), and UCB-decided selections.

**Shallow, 120 rollouts, branch 2**

| seed | mean | max | max_hv | depth | syn | first | UCB |
|---|---|---|---|---|---|---|---|
| 0 | 0.8299 | 0.8342 | 0.8342 | 6/7/7 | 4/4/4 | 95/97/94 | 486/496/496 |
| 1 | 0.8375 | 0.8375 | 0.8375 | 6/7/6 | 4/4/4 | 95/94/94 | 486/498/498 |
| 2 | 0.8291 | 0.8446 | 0.8387 | 6/7/6 | 4/4/4 | 96/95/91 | 486/498/499 |
| 3 | 0.8321 | 0.8455 | 0.8455 | 6/7/6 | 4/4/4 | 96/97/96 | 486/497/496 |
| 4 | 0.8443 | 0.8443 | 0.8443 | 6/6/6 | 4/4/4 | 48/48/48 | 486/486/486 |
| 5 | 0.8390 | 0.8390 | **0.6860** | 6/6/3 | 4/4/**3** | 95/94/87 | 486/498/498 |
| 6 | 0.8381 | 0.8345 | 0.8345 | 6/5/5 | 4/4/4 | 48/48/47 | 486/486/486 |
| 7 | 0.8424 | 0.8424 | 0.8424 | 6/7/7 | 4/4/4 | 95/97/94 | 486/496/495 |
| 8 | 0.8439 | 0.8439 | 0.8454 | 6/6/6 | 4/4/4 | 48/52/48 | 486/486/486 |
| 9 | 0.8423 | 0.8423 | 0.8419 | 6/7/6 | 4/4/4 | 95/97/96 | 486/497/496 |
| 10 | 0.8407 | 0.8365 | 0.8407 | 6/7/7 | 4/4/4 | 93/96/94 | 486/496/497 |
| 11 | 0.8402 | 0.8402 | 0.8402 | 6/6/6 | 4/4/4 | 48/50/48 | 486/486/486 |
| 12 | 0.8408 | 0.8335 | 0.8408 | 6/6/6 | 4/4/4 | 93/96/95 | 487/496/497 |
| 13 | 0.8402 | 0.8382 | 0.8364 | 6/6/5 | 4/4/4 | 48/48/48 | 486/486/486 |
| 14 | 0.8343 | 0.8336 | 0.8343 | 5/6/5 | 4/4/4 | 48/52/48 | 486/486/486 |
| 15 | 0.8413 | 0.8413 | 0.8418 | 6/6/6 | 4/4/4 | 96/96/90 | 486/496/495 |
| 16 | 0.8368 | 0.8368 | 0.8280 | 6/7/6 | 4/4/4 | 96/96/96 | 486/496/496 |
| 17 | 0.8428 | 0.8428 | 0.8428 | 6/6/6 | 4/4/4 | 94/94/92 | 486/498/498 |
| 18 | 0.8412 | 0.8412 | 0.8412 | 6/7/7 | 4/4/4 | 96/93/93 | 486/498/497 |
| 19 | 0.8472 | 0.8472 | 0.8472 | 6/6/6 | 4/4/4 | 48/47/47 | 486/486/486 |

**Shallow, 120 rollouts, branch 3** (the 4-synergy set is out of reach on 13 of 20 landscapes for every variant: branch 3 at 120 rollouts spends its evaluations on width)

| seed | mean | max | max_hv | depth | syn | first | UCB |
|---|---|---|---|---|---|---|---|
| 0 | 0.6832 | 0.6857 | 0.6842 | 4/4/4 | 3/3/3 | -/-/- | 307/310/307 |
| 1 | 0.6829 | 0.6852 | 0.6875 | 3/5/4 | 3/3/3 | -/-/- | 307/310/309 |
| 2 | 0.6851 | 0.6820 | 0.6868 | 4/4/4 | 3/3/3 | -/-/- | 309/310/310 |
| 3 | 0.6859 | 0.6868 | 0.6868 | 4/4/4 | 3/3/3 | -/-/- | 309/310/310 |
| 4 | 0.8406 | 0.8443 | 0.8366 | 4/4/5 | 4/4/4 | 68/68/68 | 309/310/310 |
| 5 | 0.6943 | 0.6845 | 0.6888 | 4/4/5 | 3/3/3 | -/-/- | 308/306/307 |
| 6 | 0.8244 | 0.8342 | 0.8381 | 4/4/4 | 4/4/4 | 68/68/68 | 309/316/311 |
| 7 | 0.6815 | 0.6817 | 0.6815 | 4/4/4 | 3/3/3 | -/-/- | 308/310/309 |
| 8 | 0.8263 | 0.8357 | 0.8294 | 4/4/4 | 4/4/4 | 68/77/68 | 308/318/321 |
| 9 | 0.6923 | 0.6890 | 0.6890 | 4/5/4 | 3/3/3 | -/-/- | 308/310/311 |
| 10 | 0.6854 | 0.6873 | 0.6873 | 3/4/4 | 3/3/3 | -/-/- | 307/307/306 |
| 11 | 0.8332 | 0.8191 | 0.8230 | 5/4/5 | 4/4/4 | 68/104/68 | 308/318/328 |
| 12 | 0.6925 | 0.6882 | 0.6882 | 4/4/4 | 3/3/3 | -/-/- | 307/306/306 |
| 13 | 0.8382 | 0.8333 | 0.8308 | 4/5/4 | 4/4/4 | 68/68/68 | 309/312/312 |
| 14 | 0.8295 | 0.8282 | 0.8282 | 5/4/5 | 4/4/4 | 68/67/66 | 308/317/329 |
| 15 | 0.6866 | 0.6909 | 0.6883 | 3/4/4 | 3/3/3 | -/-/- | 309/307/306 |
| 16 | 0.6895 | 0.6926 | 0.6926 | 4/4/4 | 3/3/3 | -/-/- | 308/308/308 |
| 17 | 0.6928 | 0.6880 | 0.6928 | 4/3/4 | 3/3/3 | -/-/- | 308/306/309 |
| 18 | 0.6834 | 0.6998 | 0.6998 | 4/4/4 | 3/3/3 | -/-/- | 307/309/309 |
| 19 | 0.8300 | 0.8372 | 0.8344 | 4/4/4 | 4/4/4 | 68/95/69 | 309/316/320 |

**Deep, 40 rollouts, branch 2**

| seed | mean | max | max_hv | depth | syn | first | UCB |
|---|---|---|---|---|---|---|---|
| 0 | 0.8382 | 0.8369 | 0.8382 | 7/6/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 1 | 0.8412 | 0.8412 | 0.8455 | 6/6/6 | 4/4/4 | 8/8/8 | 138/138/138 |
| 2 | 0.8446 | 0.8446 | 0.8446 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 3 | 0.8446 | 0.8446 | 0.8446 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 4 | 0.8415 | 0.8415 | 0.8415 | 5/5/5 | 4/4/4 | 7/7/7 | 138/138/138 |
| 5 | 0.8443 | 0.8443 | 0.8443 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 6 | 0.8324 | 0.8309 | 0.8324 | 5/5/5 | 4/4/4 | 7/7/7 | 138/138/138 |
| 7 | 0.8344 | 0.8344 | 0.8344 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 8 | 0.8376 | 0.8357 | 0.8355 | 5/5/5 | 4/4/4 | 7/7/7 | 138/138/138 |
| 9 | 0.8423 | 0.8423 | 0.8423 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 10 | 0.8373 | 0.8373 | 0.8373 | 6/6/6 | 4/4/4 | 8/8/8 | 138/138/138 |
| 11 | 0.8457 | 0.8457 | 0.8457 | 7/7/7 | 4/4/4 | 7/7/7 | 138/138/138 |
| 12 | 0.8461 | 0.8461 | 0.8461 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 13 | 0.8459 | 0.8459 | 0.8450 | 6/6/6 | 4/4/4 | 7/7/7 | 138/138/138 |
| 14 | 0.8474 | 0.8474 | 0.8474 | 6/6/6 | 4/4/4 | 7/7/7 | 138/138/138 |
| 15 | 0.8418 | 0.8418 | 0.8418 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 16 | 0.8427 | 0.8427 | 0.8427 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 17 | 0.8490 | 0.8490 | 0.8490 | 6/6/6 | 4/4/4 | 8/8/8 | 138/138/138 |
| 18 | 0.8498 | 0.8498 | 0.8498 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |
| 19 | 0.8472 | 0.8472 | 0.8472 | 7/7/7 | 4/4/4 | 8/8/8 | 138/138/138 |

**What the collapses are.** On four (cell, seed) pairs the search *did* evaluate a 4-synergy set (`found` = yes) but the returned best is a lone 3-synergy set at depth 3 with AUC 0.68-0.69. That is the final pick, not the backprop rule, making the last call: `_select_best` ranks the Pareto front by hypervolume *contribution*, and a parsimonious 3-synergy point on a front crowded with several near-identical 4-synergy points can carry the larger exclusive contribution. It is an artefact of the #15 ranking that can hit any rule — on `issue` deep/fixed r10/b3 seed 15 all three rules return the same node, (0.6866, 0.92) at depth 3, although 12-13 four-synergy nodes with own AUC up to 0.838 (depth 7) had been evaluated (the seed-15 row is identical for `mean`, `max` and `max_hv` in the JSON; re-running the cell and listing every evaluated node confirms it). `max_hv` additionally produced trees where it fired on three further pairs that `mean` and `max` did not: `saturated` shallow r120/b2 seed 5 (0.6860; `mean` returned 0.8390, and the best 4-synergy node `max_hv` had evaluated scored 0.8388), and `deep-wide` fixed r40/b8 seeds 7 (0.6786 vs `mean` 0.8287; best 4-synergy node evaluated 0.8424) and 15 (0.6914 vs `mean` 0.8024; best evaluated 0.8410). On those three seeds the collapse is a loss for `max_hv` and section 6 counts it as one. The ranking itself is issue #15 territory, out of scope here, and is listed under follow-ups (section 9).

## 5. Side results

- **Deep beats shallow at an equal evaluation budget.** Deep, 10 rollouts, branch 3 (63 evaluations): AUC 0.8295, 4-synergy set found on 20/20 landscapes after 7.7 evaluations. Shallow, 63 rollouts, branch 3 (64 evaluations): AUC 0.6602, found on 0/20. Shallow reaches the plateau only with 120 rollouts at branch 2 (0.8392, 20/20, after 78.5 evaluations). Rollouts are not the budget to compare across modes.
- **Adaptive branching never adapts: `_expand` runs only at `visit_count == 1`, in shallow and deep mode alike.** `search()` expands the selected leaf as soon as `visit_count > 0`, an expanded node has children and is never returned by `_select` as a leaf again, and `_simulate_deep` likewise expands a node right after its first evaluation. Instrumenting `MCTSSearch._expand` over every cell of this sweep (11 cells × 3 rules × 20 seeds, 40,328 expansions) found no expansion at any visit count other than 1. The AB-MCTS formula `clamp(floor(log2(visit_count)) + 2, min_branch_factor, max_branch_factor)` therefore always evaluates to `max(min_branch_factor, 2)` clamped to `max_branch_factor` — branch 2 in every cell here — so the sweep's `adaptive` axis is a branch-2 vs branch-N comparison and nothing more. That is why the `deep-wide` adaptive cell (branch ≤ 8) equals the `saturated` deep r40/b2 cell on every metric of every seed (60/60 rows identical), why its "efficiency" (143 vs 248 evaluations, 167 vs 650 nodes, AUC 0.8427 vs 0.8374) is the efficiency of a narrower tree — on this landscape, where every ADD is a step toward the plateau, narrow and deep is simply better — and why the shallow r10 adaptive cell differs from its fixed neighbour (branch 2 vs 3: depth 2.90 vs 2.00). Pre-existing on `main` and independent of the backprop rule; listed under follow-ups (section 9).
- **The adaptive axis is inert at `min_branch_factor == max_branch_factor`.** The r120/b2 adaptive and fixed cells are identical on every seed and variant, by construction of the clamp. A factorial that flips `adaptive_branching` at branch 2 measures nothing.
- **Where UCB decides.** UCB-decided selections per search: 7-16 (issue regime), 33 (deep-wide fixed), 135-138 (equal-evals shallow, deep r40), 308-493 (saturated shallow). Divergence between variants tracks that count (2-3 seeds differ at 8, 16-20 seeds differ at 300+).

## 6. Decision

**Rule stated in Phase 1:** keep `mean` unless one variant wins *clearly* on the saturated regimes *and loses nowhere*.

**Applied:**

- `max` — saturated cells: +0.0008 (3 wins / 12 ties / 5 losses), +0.0008 (12 / 0 / 8), −0.0002 (0 / 17 / 3). Elsewhere: −0.0001 to −0.0014 on the issue cells, +0.0006 on the equal-evals control, −0.0007 on deep-wide fixed. Neither clear (every mean difference is under a tenth of the per-evaluation noise SD; the win/loss counts are near even) nor loss-free (deep r40/b2: 0 wins, 3 losses). **Not adopted.**
- `max_hv` — saturated cells: −0.0070 (5 / 10 / 5, one 0.153 collapse), +0.0008 (11 / 2 / 7), +0.0001 (1 / 17 / 2). Deep-wide fixed: −0.0148 (two collapses). Loses on two cells by an order of magnitude more than any win. **Not adopted.**

**The default stays `mean`.** The null at the issue's budget is the finding, not a failure of the harness: with 8-16 UCB-decided selections per search the rule is consulted too rarely to matter, and where it is consulted 300-500 times per search the three rules land on the same plateau within noise. `max`'s one structural effect — trees 5-6 % larger (133.2 vs 127.0 and 135.4 vs 127.3 nodes) and 0.15-0.45 deeper on the shallow saturated cells, from its synthesised optimistic points pulling selection down accuracy-argmax branches — did not buy accuracy or an earlier synergy hit (first 4-synergy set after 79.3 vs 78.5 and 78.1 vs 68.0 evaluations).

The knob stays for the TrialBench arm below and for anyone who wants to re-run this on a landscape with a different shape (a deceptive one where the mean of a subtree is misleading is the case in which `max` is expected to help; this landscape is monotone in the synergy count and rewards depth).

## 7. TrialBench: not run

The issue's second arm — the same three rules on a real per-phase TrialBench run — needs the LLM backbone (Claude Opus 4.6 through DSPy, an API key), the LinearRAG indexes over the seven sources (`scripts/build_rag_index.py`), and the TrialBench datasets under `datasets/`. None is available in the environment this ablation ran in, so it was not attempted. The harness side is ready; the commands are, per phase and per rule (the checkpoint keeps the rule it was started under, and `train_mcts.py` warns on a resume whose settings say otherwise):

```bash
# default rule (mean)
VIRTUAL_ENV=~/clinical-trial-risk-assesment-venv uv run --active \
    python scripts/train_mcts.py --task phase2 --rollouts 20 --output-dir .output/phase2-mean

# elementwise max
CTRA_MCTS_BACKPROP=max VIRTUAL_ENV=~/clinical-trial-risk-assesment-venv uv run --active \
    python scripts/train_mcts.py --task phase2 --rollouts 20 --output-dir .output/phase2-max

# best realised vector by hypervolume
CTRA_MCTS_BACKPROP=max_hv VIRTUAL_ENV=~/clinical-trial-risk-assesment-venv uv run --active \
    python scripts/train_mcts.py --task phase2 --rollouts 20 --output-dir .output/phase2-max_hv
```

Repeat with `--task phase1` and `--task phase3`; compare `results.json["best_objectives"]` (the best node's own score, issue #15) and the feature sets in `feature_plans.json` across the three output directories. Twenty rollouts of deep simulation at the production branch range is the `issue` regime's UCB-decided count (8-33 per search), so the stub result predicts no difference there either; a real difference would need a longer run than the production default.

## 8. Reproduction

```bash
cd /path/to/ctra
VIRTUAL_ENV=~/clinical-trial-risk-assesment-venv uv run --active \
    python scripts/eval/ablate_backprop.py --regime all --seeds 20 --json output/backprop_ablation/ablation.json
```

660 searches, about one minute. The tables above are its printed per-cell means and per-seed rows; the paired differences were computed from the JSON's `rows` (same cell and seed, variant minus `mean`).

## 9. Follow-ups

Both are independent of the backprop rule and pre-existing on `main`; each is worth its own issue.

- **`_select_best` returned a 0.687-AUC node while 0.83-AUC nodes had been evaluated.** Reproduce with `run_one("mean", 15, REGIMES["issue"][0])` from `scripts/eval/ablate_backprop.py` (or `--regime issue --seeds 16`, row `issue` deep/fixed r10/b3 seed 15): the returned best is the 3-synergy node at (0.6866, 0.92), depth 3, under all three rules, although 12-13 four-synergy nodes with own AUC up to 0.838 (depth 7) were evaluated. The hypervolume-*contribution* ranking of the Pareto front (issue #15) hands the lone parsimonious point the largest exclusive contribution once the plateau is crowded with near-identical points. The same pick occurs under `max_hv` on `saturated` shallow r120/b2 seed 5 and `deep-wide` fixed r40/b8 seeds 7 and 15 (section 4). A defect in the #15 ranking, not in #16.
- **AB-MCTS adaptive branching is inert: `_expand` only ever runs at `visit_count == 1`** (section 5), so the branch factor is always `max(min_branch_factor, 2)` clamped to `max_branch_factor`, in shallow and deep mode alike. Either the formula should read a count that can grow (the parent's visits, or a re-expansion of revisited nodes), or `adaptive_branching` should be documented as a fixed narrow branch.
