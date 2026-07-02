# shapiq: Shapley Interactions for Machine Learning (NeurIPS 2024)

**Paper**: [shapiq: Shapley Interactions for Machine Learning](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb3a9313405e2d4175a5a3cfcd49999b-Paper-Datasets_and_Benchmarks_Track.pdf)

**Venue**: NeurIPS 2024 (Datasets and Benchmarks Track)

**Library**: [shapiq](https://shapiq.readthedocs.io/en/latest/) (`pip install shapiq`)

## Summary

shapiq is a Python library for computing Shapley interaction indices — extensions of standard SHAP values that capture not just individual feature contributions but also synergies and redundancies between feature pairs (and higher-order groups). Standard SHAP values only tell you "feature X is important"; Shapley interactions tell you "features X and Y work together synergistically" or "features X and Y are redundant."

## Supported Interaction Indices

| Index | Full Name | Key Property | Recommended For |
|-------|-----------|-------------|-----------------|
| SV | Shapley Value | Order 1 only | Standard individual attributions |
| SII | Shapley Interaction Index | Axiomatically sound | Theoretical foundation |
| **k-SII** | k-Shapley Interaction Index | **Efficient** (sums to prediction) | **Tree models (XGBoost)** |
| STII | Shapley Taylor Interaction Index | Additive | Alternative to SII |
| **FSII** | Faithful Shapley Interaction Index | **Faithful** to model | **TabPFN** |
| FBII | Faithful Banzhaf Interaction Index | Faithful + efficient | Alternative to FSII |

## Key Explainers

- **TreeExplainer**: Exact computation for tree models (XGBoost, LightGBM, RandomForest) via TreeSHAP-IQ. No background data or sampling budget needed.
- **TabPFNExplainer**: Uses "remove-and-recontextualize" paradigm for TabPFN. Requires data + labels.
- **TabularExplainer**: General purpose for any model. Supports all indices via approximation.

## Relevance to CTRA

Used in [issue #18](https://github.com/merck-gen/clinical-trial-risk-assesment/issues/18) to replace gain-based feature importances with Shapley interaction values in the evaluator agent, providing the LLM with richer feature contribution and interaction information for more targeted improvement suggestions.
