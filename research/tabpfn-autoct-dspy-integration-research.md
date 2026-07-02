# TabPFN + AutoCT + DSPy Integration Research

> Deep-dive research for integrating TabPFN as a classifier alongside XGBoost in the AutoCT/CTRA pipeline. Produced 2026-03-27.

---

## 1. TabPFN API Details

### 1.1 Installation

```bash
# Core library (local inference, requires PyTorch)
pip install tabpfn

# Interpretability extension (SHAP support)
pip install "tabpfn-extensions[interpretability]"

# From source (latest)
pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
```

First `fit()` call downloads the model checkpoint from HuggingFace Hub. TabPFN-2.5 requires accepting the license on HuggingFace and authenticating via `huggingface-cli login`.

### 1.2 Classifier API (sklearn-compatible)

```python
from tabpfn import TabPFNClassifier

# Default constructor (uses TabPFN-2 weights, Apache 2.0 + attribution)
clf = TabPFNClassifier()

# Explicit device
clf = TabPFNClassifier(device="cuda")  # or "cpu", "mps"

# TabPFN-2.5 (non-commercial license, better performance)
from tabpfn.constants import ModelVersion
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)

# Key constructor parameters:
#   device: str          - "cuda", "cpu", "mps", or "auto" (v7+: uses all CUDA GPUs)
#   n_estimators: int    - Ensemble size (default 8 since v6+, was 4)
```

**Methods (sklearn interface):**
- `clf.fit(X_train, y_train)` -- Stores training data for in-context learning (no gradient updates)
- `clf.predict(X_test)` -- Class labels
- `clf.predict_proba(X_test)` -- Probability estimates (standard sklearn shape)
- `clf.predict_logits(X_test)` -- Raw logits (added for SHAP compatibility, unnormalized additive outputs)

**Critical performance note:** Each `predict()` call recomputes the training set. Calling predict on 100 samples separately is ~100x slower than a single batch call. Always batch predictions.

### 1.3 Constraints

| Constraint | TabPFN-2 | TabPFN-2.5 | TabPFN-2.6 |
|-----------|----------|------------|------------|
| Max samples | 10,000 | 50,000 | 100,000 |
| Max features | 500 | 2,000 | 2,000 |
| GPU memory | ~8 GB (min) | ~8 GB (min), 16 GB for large | ~8 GB |
| CPU feasibility | ≤1,000 samples | ≤1,000 samples | ≤1,000 samples |

**Missing values:** Training data (`X_train`) can contain NaN/`pd.NA`. Test data (`X_test`) historically could not contain NaN (see GitHub issue #108), though recent versions may have relaxed this. **Action item:** Verify NaN handling in test data during integration testing.

**Categorical features:** Natively handled via category shuffling for low-cardinality features. New `max_onehot_cardinality` option (v7.0.0) caps one-hot encoding expansion. **Do NOT apply manual one-hot encoding or scaling.**

**Multi-GPU:** v7.0.0+ `device="auto"` uses all available CUDA GPUs. Model cached on each device between estimators (v6.1.0+).

### 1.4 SHAP Integration (tabpfn-extensions)

The `tabpfn-extensions` interpretability module provides SHAP integration. Source code analysis (`src/tabpfn_extensions/interpretability/shap.py`, 317 lines) reveals the following:

#### `get_shap_values()` -- Main entry point

```python
from tabpfn_extensions import interpretability

shap_values = interpretability.shap.get_shap_values(
    estimator=clf,              # Fitted TabPFNClassifier
    test_x=X_test[:n_samples],  # Test data (DataFrame, ndarray, or Tensor)
    attribute_names=feature_names,  # Optional: list[str] of feature names
    algorithm="permutation",    # Passed as **kwargs to shap explainer
)
```

**Internal implementation flow (from source code):**
1. Converts tensor/array inputs to DataFrame
2. Assigns column names from `attribute_names` or auto-converts existing names
3. Determines prediction function: `predict_proba` (for classification) or `predict`
4. Routes to **TabPFN-specific explainer** (`get_tabpfn_explainer()`) which uses NaN-filled background data, or **default explainer** (`get_default_explainer()`) which uses `shap.maskers.Independent`
5. Internally uses `shap.PermutationExplainer` (not TreeSHAP or KernelSHAP)
6. Returns SHAP values (the `plot_shap` function checks `len(shap_values.shape) == 3` and indexes `[:, :, 0]`, indicating it returns either a `shap.Explanation` object or ndarray)

**Key detail:** The TabPFN-specific explainer creates a NaN-filled background dataset. This exploits TabPFN's native NaN handling to make permutation SHAP more efficient -- replacing a feature with NaN is equivalent to "removing" it from the model's perspective.

#### `plot_shap()` -- Visualization

```python
fig = interpretability.shap.plot_shap(shap_values)
```

Internally calls:
- `shap.plots.bar()` -- Aggregate feature importance bar chart
- `shap.summary_plot()` -- Beeswarm plot (per-sample feature effects)
- `plot_shap_feature()` -- Scatter plot for most important feature (if >1 sample)

#### SHAP IQ Alternative

TabPFN-extensions also supports `shapiq` (Shapley Interactions):

```python
# Alternative: shapiq explainer (captures feature interactions)
# See: examples/interpretability/shapiq_example.py
```

The `shapiq` library has a dedicated `TabPFNExplainer` class for more efficient Shapley value computation.

#### Compatibility with AutoCT's existing SHAP display pipeline

AutoCT currently does **NOT** compute SHAP values in the codebase (confirmed from source code analysis and paper). The paper shows SHAP plots in case studies (Appendix A), but these appear to be post-hoc analysis, not integrated into the training loop. The `ModelEvalResult` NamedTuple stores `feature_importance: list` which comes from the classifier's native importance (e.g., `XGBClassifier.feature_importances_`), not from SHAP.

**Implication:** There is no existing SHAP "display pipeline" to maintain compatibility with. SHAP integration is a new capability to add, and the `tabpfn-extensions` approach is the right way to do it for TabPFN. For XGBoost, we can either continue using native `feature_importances_` or add TreeSHAP via `shap.TreeExplainer(xgb_model)`.

### 1.5 License Summary

| Component | License | Commercial Use |
|-----------|---------|---------------|
| TabPFN code | Apache 2.0 + attribution (Prior Labs License) | Yes |
| TabPFN-2 weights | Apache 2.0 + attribution | Yes |
| TabPFN-2.5 weights | Non-Commercial License v1.0 | No -- enterprise license required |
| TabPFN-2.6 weights | Non-Commercial License | No -- enterprise license required |
| tabpfn-extensions code | Apache 2.0 | Yes |

**Non-commercial license prohibits:** Revenue-generating products, competitive benchmarking for procurement, client deliverables, internal commercial decision-making.

**For CTRA:** Phase 1-2 (research/benchmarking) = fine under non-commercial. Phase 3 (production/internal tool at Merck) = requires enterprise license from sales@priorlabs.ai. Fallback: XGBoost (no licensing constraints).

---

## 2. AutoCT Model Training Interface

### 2.1 Source Code Structure

Repository: https://github.com/CogComp/autoct
Key file: `src/lfe/impl/agent.py`

### 2.2 `OutputV2` NamedTuple (agent.py:266)

```python
class OutputV2(NamedTuple):
    xgb_eval_output: EvalOutput
    lr_eval_output: EvalOutput
    rf_eval_output: EvalOutput
    test_xgb_eval_output: ModelEvalResult
    test_lr_eval_output: ModelEvalResult
    test_rf_eval_output: ModelEvalResult
    operation: Optional[ProposerOutput]
    feature_plans: dict[str, FeaturePlanV2]
    df: pd.DataFrame
    val_df: pd.DataFrame
    suggestion_index: int
    raw_features: dict
    raw_val_features: dict
    raw_test_features: dict
    none_explanations: dict[str, dict[str, str]]
```

**Supporting types:**

```python
class EvalOutput(NamedTuple):
    model_eval_result: ModelEvalResult
    suggestions: List[str]

class ModelEvalResult(NamedTuple):
    roc_auc: float
    f1: float
    pr_auc: float
    feature_importance: list        # Native classifier feature importances (NOT SHAP)
    wrong_idxs: list[int]           # Indices of misclassified samples
    wrong_preds: list[int]          # Predicted labels for misclassified samples
    wrong_df: pd.DataFrame          # DataFrame of misclassified samples
    pipeline: Pipeline              # Fitted sklearn Pipeline
```

### 2.3 `get_best_eval_output()` (agent.py:283)

```python
def get_best_eval_output(self) -> tuple[EvalOutput, ModelEvalResult]:
    eval_outputs = [
        (self.xgb_eval_output, self.test_xgb_eval_output),
        (self.lr_eval_output, self.test_lr_eval_output),
        (self.rf_eval_output, self.test_rf_eval_output),
    ]
    best_eval_output = max(
        eval_outputs, key=lambda x: x[0].model_eval_result.roc_auc
    )
    return best_eval_output
```

Selects the model with the highest **validation** ROC-AUC, returns both the validation `EvalOutput` and the corresponding test `ModelEvalResult`. The MCTS node score is derived from this validation ROC-AUC.

### 2.4 `train_simple_model_v2()` (agent.py:2365)

```python
def train_simple_model_v2(
    plans: dict[str, FeaturePlanV2],
    input_X_df,
    y_train,
    model_type: Literal["rf", "logistic", "xgb"],
    skip=list(),
):
```

**Implementation (reconstructed from partial source + paper):**

1. **Build ColumnTransformer:** Iterates over `plans` dict, creating feature-specific transformers:
   - Categorical features -> `OneHotEncoder`
   - Multi-categorical -> `CountVectorizer`
   - Integer/Float -> conditional: `SimpleImputer` for LR/RF, passthrough for XGBoost
   - Boolean -> bool-to-int conversion + `SimpleImputer`
   - Features in `skip` list are excluded

2. **Select model:**
   ```python
   if model_type == "logistic":
       model = LogisticRegression()
   elif model_type == "rf":
       model = RandomForestClassifier(random_state=42)
   else:
       model = XGBClassifier()
   ```

3. **Build Pipeline:**
   ```python
   preprocessor = ColumnTransformer(transformers, remainder="drop")
   pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
   ```

4. **Fit:** `pipeline.fit(input_X_df, y_train)`

5. **Return:** The fitted pipeline and feature metadata

### 2.5 Evaluation (eval function)

The `eval()` function (called after training) computes:
- `roc_auc_score(y_true, predict_proba[:, 1])` -- primary metric for MCTS
- `f1_score(y_true, predictions)`
- `average_precision_score(y_true, predict_proba[:, 1])` -- PR-AUC
- `feature_importance` -- from `pipeline.named_steps['classifier'].feature_importances_` (XGBoost/RF) or `pipeline.named_steps['classifier'].coef_` (LR)
- `wrong_idxs`, `wrong_preds`, `wrong_df` -- misclassified samples for Error-Based Evaluator

### 2.6 SHAP in AutoCT

**Finding: AutoCT does NOT integrate SHAP into the training/evaluation loop.**

The paper shows SHAP plots in Appendix A case studies, but these are post-hoc analyses. The codebase uses `feature_importances_` (native classifier attribute), not SHAP values. The `ModelEvalResult.feature_importance` field stores native importance scores.

SHAP integration for CTRA is a **new addition**, not a modification of existing functionality.

### 2.7 AgentV2.forward() Call Pattern (agent.py:2111)

In the forward pass, all three models are trained sequentially:

```python
xgb_model, xgb_meta = train_simple_model_v2(
    current_feature_plans, df, self.y_train, model_type="xgb"
)
lr_model, lr_meta = train_simple_model_v2(
    current_feature_plans, df, self.y_train, model_type="logistic"
)
rf_model, rf_meta = train_simple_model_v2(
    current_feature_plans, df, self.y_train, model_type="rf"
)
# Then evaluate each on validation set
# Then construct OutputV2 with all results
```

---

## 3. DSPy Configuration for Anthropic Models

### 3.1 LM Backend Configuration

DSPy uses **LiteLLM** under the hood for all LLM provider communication. Model string format follows LiteLLM conventions: `{provider}/{model-name}`.

```python
import dspy

# Anthropic Claude configuration
lm = dspy.LM(
    "anthropic/claude-opus-4-6",
    api_key="sk-ant-...",         # Or set ANTHROPIC_API_KEY env var
    temperature=0.0,
    max_tokens=16384,
)
dspy.configure(lm=lm)
```

**Known model strings for Anthropic:**
- `"anthropic/claude-opus-4-6"` -- Claude Opus 4.6
- `"anthropic/claude-sonnet-4-5-20250929"` -- Claude Sonnet 4.5
- `"anthropic/claude-3-5-sonnet-20241022"` -- Claude 3.5 Sonnet

**Thread safety:** Both `dspy.configure(lm=...)` (global) and `dspy.context(lm=...)` (scoped) are thread-safe.

### 3.2 Per-Module LM Overrides

```python
with dspy.context(lm=specific_lm):
    result = module(input_data)
```

This enables per-agent-phase thinking budgets (see implementation plan Workstream 4).

### 3.3 Prompt Caching with Anthropic

```python
lm = dspy.LM(
    "anthropic/claude-opus-4-6",
    cache_control_injection_points=[
        {"location": "message", "role": "system"}
    ]
)
```

Anthropic prompt caching reduces cost for repeated system prompts (relevant for MCTS where the same Signatures are called repeatedly with different inputs).

### 3.4 ReAct Agents

```python
# Define a tool with docstring and type hints (required)
def search_pubmed(query: str) -> str:
    """Search PubMed for articles related to the query."""
    return rag.retrieve(query)

# Create ReAct agent
react = dspy.ReAct(
    signature="question -> answer",  # Or a class-based Signature
    tools=[search_pubmed, search_ctg],
    max_iters=10,
)

# Call
pred = react(question="What are the safety signals for Drug X?")
```

**Limitations:** ReAct currently supports only one output field in its signature. Tools must have docstrings and type hints for the LM to generate correct arguments.

### 3.5 Signatures (Class-Based)

```python
class FeatureProposerV2Signature(dspy.Signature):
    """Propose new features for clinical trial prediction based on model evaluation."""

    previous_output: str = dspy.InputField(desc="Previous model evaluation results and feature plans")
    task_description: str = dspy.InputField(desc="Description of the prediction task")

    proposal: str = dspy.OutputField(desc="JSON object with operation type and feature details")
```

**Supported types:** `str`, `int`, `float`, `bool`, `list[str]`, `dict[str, int]`, `Optional[float]`, Pydantic `BaseModel` subclasses, `dspy.Image`.

**Validation:** DSPy automatically validates input field types and logs warnings on mismatch.

### 3.6 AutoCT's Existing DSPy Signatures

From the codebase, AutoCT defines these Signatures:
- `FeatureProposerSingleOutputV2Signature` -- Proposes Add/Refine/Remove operations
- `FeaturePlannerV2Signature` -- Converts feature ideas into executable plans
- `EvaluatorWithExampleSignatureV2` -- Analyzes model errors
- `FeatureBuilderConstructSignature` -- Extracts feature values from retrieved data

All use `dspy.ChainOfThought` modules. Migration to Anthropic is LLM-backend only -- Signatures remain unchanged.

### 3.7 Current AutoCT DSPy Configuration

```python
# agent.py:46-51 (current)
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'), max_tokens=16000)
dspy.configure(lm=lm)
```

**Migration (one-line change + extended thinking):**
```python
lm = dspy.LM("anthropic/claude-opus-4-6", api_key=os.getenv('ANTHROPIC_API_KEY'), max_tokens=16384)
dspy.configure(lm=lm)
```

---

## 4. Integration Approach Recommendations

### 4.1 TabPFN Integration into train_simple_model_v2()

**Recommended approach: Add `"tabpfn"` as a new model_type, bypassing ColumnTransformer.**

```python
def train_simple_model_v2(
    plans: dict[str, FeaturePlanV2],
    input_X_df,
    y_train,
    model_type: Literal["xgb", "tabpfn"],  # Remove "rf" and "logistic"
    skip=list(),
):
    if model_type == "tabpfn":
        # TabPFN handles missing values, categoricals, and booleans natively
        # No ColumnTransformer needed -- pass raw DataFrame
        from tabpfn import TabPFNClassifier
        import torch

        model = TabPFNClassifier(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        # Filter columns based on plans (exclude skipped features)
        feature_cols = [k for k in plans.keys() if k not in skip]
        X = input_X_df[feature_cols]
        model.fit(X, y_train)
        return model, {"feature_cols": feature_cols}
    else:
        # Existing XGBoost path with ColumnTransformer
        # ... (existing code unchanged)
```

**Why bypass ColumnTransformer:** TabPFN docs explicitly state "Do not apply data scaling or one-hot encoding." The ColumnTransformer's OneHotEncoder, CountVectorizer, and SimpleImputer would degrade TabPFN performance by destroying the raw feature structure it expects.

### 4.2 OutputV2 Modification

```python
class OutputV2(NamedTuple):
    xgb_eval_output: EvalOutput
    tabpfn_eval_output: EvalOutput       # Replace lr_eval_output
    test_xgb_eval_output: ModelEvalResult
    test_tabpfn_eval_output: ModelEvalResult  # Replace lr
    # Remove rf_eval_output and test_rf_eval_output
    operation: Optional[ProposerOutput]
    feature_plans: dict[str, FeaturePlanV2]
    df: pd.DataFrame
    val_df: pd.DataFrame
    suggestion_index: int
    raw_features: dict
    raw_val_features: dict
    raw_test_features: dict
    none_explanations: dict[str, dict[str, str]]

def get_best_eval_output(self) -> tuple[EvalOutput, ModelEvalResult]:
    eval_outputs = [
        (self.xgb_eval_output, self.test_xgb_eval_output),
        (self.tabpfn_eval_output, self.test_tabpfn_eval_output),
    ]
    best_eval_output = max(
        eval_outputs, key=lambda x: x[0].model_eval_result.roc_auc
    )
    return best_eval_output
```

### 4.3 SHAP Integration Strategy

Since AutoCT does NOT have existing SHAP infrastructure in the training loop:

1. **During MCTS (training loop):** Continue using native `feature_importances_` for the Evaluator agent -- it is fast and sufficient for guiding feature engineering decisions.

2. **Post-MCTS (final model explanation):** Add SHAP computation for the best model:

```python
# After MCTS completes, for the winning model:
best_eval, best_test = final_output.get_best_eval_output()
pipeline = best_test.pipeline

if isinstance(pipeline, TabPFNClassifier):
    from tabpfn_extensions import interpretability
    shap_values = interpretability.shap.get_shap_values(
        estimator=pipeline,
        test_x=X_test,
        attribute_names=feature_names,
        algorithm="permutation",
    )
    fig = interpretability.shap.plot_shap(shap_values)
elif hasattr(pipeline.named_steps.get('classifier', None), 'feature_importances_'):
    import shap
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(
        pipeline.named_steps['preprocessor'].transform(X_test)
    )
```

3. **Unified SHAP output format:** Both paths should produce arrays compatible with `shap.summary_plot()` and `shap.plots.bar()`. The `tabpfn-extensions` `get_shap_values()` returns objects compatible with standard `shap` plotting functions.

### 4.4 Performance Considerations

- **TabPFN SHAP is slow:** Permutation SHAP requires O(n_features * n_samples) forward passes. For 25 features x 100 test samples = 2,500 forward passes. Each forward pass recomputes the full training set. **Mitigation:** Subsample test set to 50-100 instances for SHAP; use GPU.

- **TabPFN training is fast:** No gradient updates. `fit()` just stores data. `predict()` is the expensive part (one forward pass through the transformer per batch).

- **Parallel SHAP:** The `tabpfn-extensions` module includes `parallel_permutation_shap()` for multiprocessing, but this conflicts with CUDA (GPU memory sharing across processes). On GPU, use single-process SHAP.

### 4.5 DSPy Migration Checklist

1. Replace `dspy.LM("openai/gpt-4o-mini", ...)` with `dspy.LM("anthropic/claude-opus-4-6", ...)`
2. Set `ANTHROPIC_API_KEY` environment variable
3. Test all 4 Signature types: FeatureProposer, FeaturePlanner, Evaluator, FeatureBuilder
4. Verify structured output (JSON) parsing works with Anthropic backend
5. Add extended thinking via `extra_body` parameter (optional, for quality improvement)
6. Enable prompt caching via `cache_control_injection_points` (for cost reduction)
7. Use `dspy.context(lm=...)` for per-phase thinking budgets

---

## Sources

- [TabPFN GitHub Repository](https://github.com/PriorLabs/TabPFN)
- [TabPFN Documentation](https://docs.priorlabs.ai/quickstart)
- [TabPFN Interpretability Docs](https://docs.priorlabs.ai/capabilities/interpretability)
- [tabpfn-extensions GitHub](https://github.com/PriorLabs/tabpfn-extensions)
- [tabpfn-extensions PyPI](https://pypi.org/project/tabpfn-extensions/)
- [TabPFN Changelog](https://github.com/PriorLabs/TabPFN/blob/main/CHANGELOG.md)
- [TabPFN License](https://priorlabs.ai/tabpfn-license)
- [AutoCT Paper (arXiv:2506.04293v1)](https://arxiv.org/html/2506.04293v1)
- [AutoCT GitHub Repository](https://github.com/CogComp/autoct)
- [DSPy Language Models Documentation](https://dspy.ai/learn/programming/language_models/)
- [DSPy ReAct Documentation](https://dspy.ai/api/modules/ReAct/)
- [DSPy Signatures Documentation](https://dspy.ai/learn/programming/signatures/)
- [DSPy Tools Documentation](https://dspy.ai/learn/programming/tools/)
- [shapiq TabPFN Explainer](https://shapiq.readthedocs.io/en/latest/notebooks/tabular_notebooks/explaining_tabpfn.html)
