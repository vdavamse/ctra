# TabPFN Evaluation for CTRA

## Overview

TabPFN (Tabular Prior-data Fitted Network) is a pre-trained transformer for tabular data from Prior Labs. Published in Nature (2025). Instead of training on your dataset, it makes predictions via in-context learning — a single forward pass, no gradient updates, no hyperparameter tuning. Pre-trained on billions of synthetic datasets to "learn the learning process."

Paper: Hollmann et al., Nature 2025
Model report (v2.5): arXiv:2511.08667
Repository: https://github.com/PriorLabs/TabPFN
Docs: https://docs.priorlabs.ai

## Model Versions

| Version | Max Rows | Max Features | Text Support | Access | License |
|---|---|---|---|---|---|
| **TabPFN-2.5-Plus** | 100,000 | 2,000 | Yes (native) | API only (`tabpfn-client`) | Contact sales for commercial |
| **TabPFN-2.5** | 100,000 | 2,000 | No | Local (`pip install tabpfn`) | Non-commercial |
| TabPFN v2 | 10,000 | 500 | No | Local | Commercial available |
| TabPFN v1 | 1,000 | 100 | No | Local | Open |

## Benchmark Results

From the TabPFN-2.5 model report (arXiv:2511.08667):

- **100% win rate** vs default XGBoost on small-to-medium datasets (≤10K rows, ≤500 features)
- **87% classification / 85% regression** win rate on larger datasets (up to 100K rows, 2K features)
- **Leading method on TabArena** (industry-standard benchmark with datasets up to 100K rows)
- **Matches AutoGluon 1.4** accuracy — a complex 4-hour tuned ensemble that includes TabPFN v2 itself

## Fit for CTRA / AutoCT

AutoCT trains on 100-500 rows with ~10-25 features. This is TabPFN's optimal range.

| Dimension | TabPFN fit |
|---|---|
| Dataset size (100-500 rows) | Perfect — 100% win rate vs XGBoost at this scale |
| Feature count (~10-25) | Well within 2,000 feature limit |
| Missing values | Native handling (`pd.NA`) — no imputation needed |
| Categorical features | Native handling — no one-hot encoding needed |
| SHAP support | Yes — via `tabpfn-extensions[interpretability]` (`get_shap_values()`, `plot_shap()`) |
| sklearn API | Yes — `TabPFNClassifier` implements `fit()`, `predict()`, `predict_proba()` |
| Hyperparameter tuning | Not needed — pre-trained model |

### Preprocessing simplification

AutoCT currently builds a complex `ColumnTransformer` per feature type in `train_simple_model_v2()` (`agent.py:2365-2482`):
- Categorical → `OneHotEncoder`
- Multi-categorical → `CountVectorizer`
- Integer/Float → `SimpleImputer` (LR/RF) or passthrough (XGBoost)
- Boolean → bool→int + `SimpleImputer`

TabPFN handles all of these natively. The docs explicitly state: "Do not apply data scaling or one-hot encoding." For the TabPFN model type, the pipeline can skip the `ColumnTransformer` entirely.

## Pricing and Licensing

### Free tier (API)

The TabPFN API uses a credit-based system:

```
credits_per_request = max((train_rows + test_rows) × columns × n_estimators, 5000)
```

- Default `n_estimators = 8`
- **100 million credits/day** on the free tier
- For CTRA (200 rows × 25 features × 8 estimators = 40,000 credits/request): **~2,500 predictions/day**
- Rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` (daily UTC 00:00)

### Licensing tiers

| Use case | Model | License | Cost |
|---|---|---|---|
| **Phase 1-2** (research, benchmarking) | TabPFN-2.5 (local) or API free tier | Non-commercial | **$0** |
| **Phase 3** (production, internal tool) | TabPFN Enterprise | Commercial license required | **TBD — contact sales@priorlabs.ai** |

The non-commercial license explicitly allows "testing, evaluation, and internal benchmarking." Production deployment requires the enterprise license, which includes:
- Proprietary high-speed inference engine (distillation for tree/MLP-level latency)
- Large data mode (up to 10M rows)
- Dedicated support and integration tooling
- Fine-tuning capabilities

### Fallback

If the enterprise license cost is prohibitive, CTRA falls back to XGBoost for production — already integrated, no licensing constraints, and the `get_best_eval_output()` logic automatically selects the best-performing model per MCTS node.

## TabPFN-2.5-Plus Specifics

TabPFN-2.5-Plus is the premium model with native text handling. Relevant for CTRA because AutoCT features may include text-derived values (e.g., eligibility criteria summaries, drug mechanism descriptions). The Plus model can process these without separate text encoding.

**Access:** API only via `tabpfn-client` package.
**Pricing:** Not publicly listed — contact sales@priorlabs.ai.
**Advantage over 2.5:** Handles textual features natively, ranked #1 on TabArena.

## Integration Points in AutoCT

### Code changes needed

1. **`agent.py:2365-2482`** (`train_simple_model_v2`) — Add `"tabpfn"` model type with simplified pipeline (no `ColumnTransformer`)
2. **`agent.py:266-296`** (`OutputV2`) — Add `tabpfn_eval_output` and `test_tabpfn_eval_output` fields
3. **`agent.py:283-292`** (`get_best_eval_output`) — Include TabPFN in best-model comparison
4. **`agent.py:2111-2155`** (`AgentV2.forward`) — Add TabPFN training + evaluation calls
5. **`environment.yml`** — Add `tabpfn` and `tabpfn-extensions[interpretability]`

### Estimated effort: 3.5 days

| Task | Effort |
|---|---|
| Install + verify GPU compatibility | 0.5 days |
| Modify `train_simple_model_v2()` | 1 day |
| Update `OutputV2` + `AgentV2.forward()` | 1 day |
| Update `predict.py` | 0.5 days |
| Validate SHAP output | 0.5 days |

## Hardware Requirements

- **GPU recommended** for optimal performance (NVIDIA T4 minimum, A100/H100 optimal)
- **CPU feasible** for datasets ≤1,000 samples (sufficient for CTRA's 100-row training during Phase 1)
- KV-Cache mode (`fit_mode='fit_with_cache'`) reduces prediction latency at cost of ~6.1 KB GPU memory per cell

## References

- Hollmann N, Müller S, Hutter F. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. Nature. 2025.
- Prior Labs. TabPFN-2.5 Model Report. arXiv:2511.08667. 2025.
- Prior Labs Documentation: https://docs.priorlabs.ai
- TabPFN GitHub: https://github.com/PriorLabs/TabPFN
- TabPFN Client GitHub: https://github.com/PriorLabs/tabpfn-client
- TabPFN License: https://priorlabs.ai/tabpfn-license
