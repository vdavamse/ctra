"""Central agent orchestrator — iteration 0 and N dispatch.

- **Iteration 0**: Runs ``Initializer`` → ``compute_features`` for
  train/val/test → trains models → evaluates → returns ``AgentOutput``.
- **Iteration N**: Runs ``FeatureProposer`` → dispatches ADD/REMOVE/REFINE
  → computes only the changed feature → trains models → evaluates each →
  returns ``AgentOutput``.

Uses the primary LM (Opus) for proposer/planner/evaluator and the budget
LM (Sonnet) for builder (scoped inside ``FeatureBuilder``).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import dspy
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer

from ctra.agents.data_models import (
    BUILDER_EXCEPTION_PREFIX,
    BUILDER_EXCEPTION_REASON,
    BUILDER_OMITTED_PREFIX,
    BUILDER_OMITTED_REASON,
    AgentOutput,
    BuilderDiagnostics,
    CacheStats,
    EvalOutput,
    FeatureOp,
    ModelEvalResult,
    ProposerOutput,
    Task,
)
from ctra.agents.evaluator import Evaluator
from ctra.agents.feature_builder import compute_features
from ctra.agents.feature_grouper import FeatureGrouper
from ctra.agents.feature_planner import FeaturePlanner
from ctra.agents.feature_proposer import FeatureProposer
from ctra.agents.feature_utils import (
    build_feature_type_transformer,
    compute_shapiq_for_pipeline,
    eval_model,
    features_to_df,
)
from ctra.agents.initializer import Initializer
from ctra.agents.reward_fns import (
    ResettingRefine,
    grouper_reward,
    is_valid_planner,
    is_valid_proposer,
    planner_reward,
    proposer_reward,
    unwrap_planner_result,
    unwrap_proposal,
)
from ctra.config.settings import ClassifierType, get_settings
from ctra.models.model_registry import ModelRegistry

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Expected classifier class names — runtime guard against silent substitution
# (the original bug this fix addresses: issue #36).
_EXPECTED_CLASSIFIER_CLASS: dict[ClassifierType, str] = {
    ClassifierType.XGBOOST: "XGBClassifier",
    ClassifierType.TABPFN: "TabPFNClassifier",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _merge_two_level_dicts(
    target: dict[str, dict[str, Any]], source: dict[str, dict[str, Any]]
) -> None:
    """Merge a two-level nested dict ``source`` into ``target``.

    Shape: ``{outer_key: {inner_key: value}}``. For each outer key, the inner
    dict from ``source`` is merged into ``target``'s inner dict via ``|=`` —
    last-write-wins at the inner level. Inner ``value`` is **not** recursed,
    so three-level dicts (e.g. ``raw_features`` = ``{nctid: {feat: {k: v}}}``)
    would have their inner sub-dicts overwritten rather than merged — use an
    explicit loop in that case.

    Used to consolidate ``none_explanations`` and ``builder_meta`` from
    train/val/test splits without overwriting existing entries.
    """
    for key, val in source.items():
        if key not in target:
            target[key] = {}
        target[key] |= val


def _build_builder_diagnostics(
    none_explanations: dict[str, dict[str, str]],
    feature_plans: dict[str, Any],
    builder_meta: dict[str, dict[str, Any]],
) -> Any:
    """Build diagnostic summaries from none_explanations and builder metadata.

    Analyzes per-feature None rates and failure reasons to provide insights
    for distinguishing Researcher (bad feature idea) vs Builder (bad execution)
    failures.
    """
    from collections import Counter

    from ctra.agents.data_models import FeatureDiagnostic

    feature_diagnostics = []
    all_nctids = set(none_explanations.keys()) | set(builder_meta.keys())
    total_trials = max(len(all_nctids), 1)

    for feature_name in feature_plans:
        # Count None rate across trials
        none_count = sum(
            1 for nctid in all_nctids if feature_name in none_explanations.get(nctid, {})
        )
        none_rate = none_count / total_trials

        # Determine dominant failure reason from none_explanations
        reasons: list[str] = []
        for nctid in all_nctids:
            reason = none_explanations.get(nctid, {}).get(feature_name, "")
            if reason:
                reasons.append(reason)

        # Classify reasons into categories
        reason_categories: Counter[str] = Counter()
        for reason in reasons:
            r_lower = reason.lower()
            # Both sentinel arms are checked first: the exception sentinel
            # contains the word "exception" and the extraction_error branch
            # below would swallow it, collapsing "the extractor produced nothing
            # useful" into the same bucket as "the builder crashed and never
            # ran"; the omission sentinel carries the LLM's own explanation
            # appended, which can contain any of the keywords the arms further
            # down match on (a "builder_omitted: no data found" would otherwise
            # be mis-attributed to the RESEARCHER). startswith (not `in`) so an
            # LLM-authored explanation that merely mentions a builder exception
            # or omission cannot impersonate either sentinel.
            if r_lower.startswith(BUILDER_EXCEPTION_PREFIX.lower()):
                reason_categories[BUILDER_EXCEPTION_REASON] += 1
            elif r_lower.startswith(BUILDER_OMITTED_PREFIX.lower()):
                reason_categories[BUILDER_OMITTED_REASON] += 1
            elif any(
                kw in r_lower for kw in ["insufficient", "no data", "not found", "unavailable"]
            ):
                reason_categories["insufficient_data"] += 1
            elif any(kw in r_lower for kw in ["error", "failed", "timeout", "exception"]):
                reason_categories["extraction_error"] += 1
            elif any(kw in r_lower for kw in ["ambiguous", "unclear", "uncertain"]):
                reason_categories["ambiguity"] += 1
            else:
                reason_categories["other"] += 1

        dominant_reason = reason_categories.most_common(1)[0][0] if reason_categories else "none"

        # Research coverage score: proportion of trials where builder had non-empty research
        research_count = sum(
            1
            for nctid in all_nctids
            if builder_meta.get(nctid, {}).get(feature_name, {}).get("research_results", "")
        )
        research_coverage = research_count / total_trials

        feature_diagnostics.append(
            FeatureDiagnostic(
                feature_name=feature_name,
                none_rate=none_rate,
                dominant_failure_reason=dominant_reason,
                research_coverage_score=research_coverage,
            )
        )

    return BuilderDiagnostics(feature_diagnostics=feature_diagnostics)


class Agent(dspy.Module):  # type: ignore[misc]
    """Central agent orchestrator.

    Coordinates the entire feature engineering pipeline: initializes features,
    runs MCTS search to propose/refine/remove features, evaluates model performance,
    and iteratively improves the feature set. Each phase (I, II, III) runs an
    isolated instance with its own training/validation/test splits and feature set.

    Args:
        task: Task enum member or plain description string (e.g., "TRIAL_OUTCOME_PHASE_2").
        X_train: Training NCT IDs (DataFrame with nctId column, or Series).
        X_val: Validation NCT IDs (same format).
        y_train: Training labels (0 = failure, 1 = success).
        y_val: Validation labels.
        X_test: Test NCT IDs (same format).
        y_test: Test labels.
    """

    def __init__(
        self,
        task: Task | str,
        X_train: pd.DataFrame | pd.Series,
        X_val: pd.DataFrame | pd.Series,
        y_train: pd.Series | NDArray[Any],
        y_val: pd.Series | NDArray[Any],
        X_test: pd.DataFrame | pd.Series,
        y_test: pd.Series | NDArray[Any],
    ) -> None:
        super().__init__()
        self.task: Task | None
        if isinstance(task, Task):
            self.task = task
            self.task_description = task.description
        else:
            self.task = None
            self.task_description = task
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        # ResettingRefine, not dspy.Refine: these instances live for the whole MCTS
        # run, and Refine's failure budget erodes permanently across calls.
        self.proposer = ResettingRefine(
            module=FeatureProposer(self.task_description),
            N=3,
            reward_fn=proposer_reward,
            threshold=1.0,
        )
        self.planner = ResettingRefine(
            module=FeaturePlanner(self.task_description),
            N=3,
            reward_fn=planner_reward,
            threshold=1.0,
        )
        # Built after the planner so the Initializer can share it rather than
        # constructing a second wrapper around the same FeaturePlanner.
        self.initializer = Initializer(
            task_description=self.task_description,
            X_train=X_train,
            y_train=y_train,
            feature_planner=self.planner,
        )
        self.evaluator = Evaluator(self.task_description)
        self.grouper = ResettingRefine(
            module=FeatureGrouper(self.task_description),
            N=3,
            reward_fn=grouper_reward,
            threshold=1.0,
        )

    @staticmethod
    def _extract_nctids(data: pd.DataFrame | pd.Series) -> list[str]:
        """Extract NCT ID list from DataFrame or Series."""
        if isinstance(data, pd.Series):
            return data.astype(str).tolist()
        for col in ("nctId", "nct_id", "nctid"):
            if col in data.columns:
                return data[col].astype(str).tolist()
        return data.iloc[:, 0].astype(str).tolist()

    def forward(self, previous_output: AgentOutput | None = None) -> AgentOutput:
        """Run one iteration of the agent pipeline.

        Args:
            previous_output: ``None`` for iteration 0, or the ``AgentOutput``
                from the previous iteration.

        Returns:
            ``AgentOutput`` with evaluation results, feature plans, and raw
            feature values for state passing to the next iteration.
        """
        settings = get_settings()
        feature_store_dir = settings.mcts.feature_store_dir
        feature_store_enabled = settings.mcts.feature_store_enabled
        # Namespace the feature store by phase so per-phase task descriptions
        # never accidentally collide. ``Task.output_subdir`` is a @property on
        # the enum (e.g. ``"phase2"``); the ``"default"`` arm only applies when
        # no Task has been wired in (bare-string task description, unit tests).
        task_namespace = self.task.output_subdir if self.task is not None else "default"
        train_nctids = self._extract_nctids(self.X_train)
        val_nctids = self._extract_nctids(self.X_val)
        test_nctids = self._extract_nctids(self.X_test)
        # Fresh per call (issue #17): the six compute_features calls below
        # share it, and it leaves with the returned output.  The three skip
        # paths return a parent copy with its own zeroed set instead.
        cache_stats = CacheStats()

        # =================================================================
        # Iteration 0: Initialize from scratch
        # =================================================================
        if previous_output is None:
            logger.info("Iteration 0: Running Initializer")
            current_feature_plans = self.initializer()

            logger.info("Computing features for train set (%d trials)", len(train_nctids))
            current_feature_values, train_none, train_meta = compute_features(
                self.grouper,
                train_nctids,
                self.task_description,
                current_feature_plans,
                feature_store_dir=feature_store_dir,
                task_namespace=task_namespace,
                feature_store_enabled=feature_store_enabled,
                counters=cache_stats,
                plan_origin="initializer",
            )
            logger.info("Computing features for val set (%d trials)", len(val_nctids))
            current_val_feature_values, val_none, val_meta = compute_features(
                self.grouper,
                val_nctids,
                self.task_description,
                current_feature_plans,
                feature_store_dir=feature_store_dir,
                task_namespace=task_namespace,
                feature_store_enabled=feature_store_enabled,
                counters=cache_stats,
                plan_origin="initializer",
            )
            logger.info("Computing features for test set (%d trials)", len(test_nctids))
            current_test_feature_values, test_none, test_meta = compute_features(
                self.grouper,
                test_nctids,
                self.task_description,
                current_feature_plans,
                feature_store_dir=feature_store_dir,
                task_namespace=task_namespace,
                feature_store_enabled=feature_store_enabled,
                counters=cache_stats,
                plan_origin="initializer",
            )
            none_explanations: dict[str, dict[str, str]] = {}
            _merge_two_level_dicts(none_explanations, train_none)
            _merge_two_level_dicts(none_explanations, val_none)
            _merge_two_level_dicts(none_explanations, test_none)
            all_builder_meta: dict[str, dict[str, Any]] = {}
            _merge_two_level_dicts(all_builder_meta, train_meta)
            _merge_two_level_dicts(all_builder_meta, val_meta)
            _merge_two_level_dicts(all_builder_meta, test_meta)
            proposer_result: ProposerOutput | None = None

        # =================================================================
        # Iteration N: Propose → Dispatch → Compute changed features only
        # =================================================================
        else:
            # Nothing left to follow. ``get_next_suggestion`` would clamp and hand
            # the proposer a suggestion already tried and rejected, so the N=3
            # ``Refine`` attempts cannot make progress. Skip before spending them.
            #
            # Reachability — issue #7 closed the MCTS side: ``_expand`` caps a
            # node's children at its own suggestion count and ``_call_evaluate``
            # returns ``None`` instead of calling the runner when the parent
            # output it built (carrying the child's ``suggestion_index``) is
            # already exhausted. The search layer therefore never sends an
            # exhausted output here, so this branch is *less* reachable than
            # before, not more. It survives as the last line of defence for
            # direct ``Agent`` callers and for any future expander or runner
            # that bypasses ``MCTSSearch``, and it is what makes the state
            # nameable.
            #
            # Cost note: the deepcopy below is not cheaper than the branch it
            # skips — it copies all of ``AgentOutput`` (fitted pipelines and
            # DataFrames included) where the normal path copies only six dicts.
            # It matches the existing invalid-proposer skip, so it is not a
            # regression, but this path saves LM calls, not memory.
            if previous_output.suggestions_exhausted:
                logger.warning(
                    "suggestion_index=%d is past the evaluator's suggestions; "
                    "skipping iteration without calling the proposer (see issue #7).",
                    previous_output.suggestion_index,
                )
                # Index is left where it is: no suggestion was consumed here, so
                # advancing would inflate the "burned" count for work never done.
                # Zeroed counters: this iteration touched no store.
                return deepcopy(previous_output)._replace(cache_stats=CacheStats())

            current_feature_values = deepcopy(previous_output.raw_features)
            current_val_feature_values = deepcopy(previous_output.raw_val_features)
            current_test_feature_values = deepcopy(previous_output.raw_test_features)
            current_feature_plans = deepcopy(previous_output.feature_plans)
            current_none_explanations = deepcopy(previous_output.none_explanations)
            # Hydrate builder metadata from previous iteration
            all_builder_meta = deepcopy(previous_output.builder_meta)

            # Propose operation
            proposer_prediction = self.proposer(previous_output=previous_output)
            # Refine never returns None with our N/fail_count; the assert guards a contract change
            assert proposer_prediction is not None
            proposer_result = unwrap_proposal(proposer_prediction)

            # Validate proposer result; skip iteration if all retries failed
            if not is_valid_proposer({"previous_output": previous_output}, proposer_result):
                logger.warning(
                    "Proposer returned invalid op (best of 3 retries): op=%s name=%s. "
                    "Skipping iteration; advancing suggestion_index.",
                    getattr(proposer_result, "feature_operation", None),
                    getattr(proposer_result, "feature_name", None),
                )
                # Skip iteration with a deepcopy to avoid aliasing dicts; advance
                # suggestion_index to prevent infinite loop replaying the same dead suggestion.
                safe_prev = deepcopy(previous_output)
                return safe_prev._replace(
                    suggestion_index=safe_prev.suggestion_index + 1,
                    cache_stats=CacheStats(),
                )

            if proposer_result.feature_operation in (FeatureOp.ADD, FeatureOp.REFINE):
                # --- ADD or REFINE ---
                feature_idea = proposer_result.feature_explanation
                current_plan = current_feature_plans.get(proposer_result.feature_name)
                if current_plan is not None:
                    # REFINE: concatenate old + new
                    feature_idea = (
                        f"{current_plan.feature_idea}\n---\n{proposer_result.feature_explanation}"
                    )

                plan, raw = unwrap_planner_result(
                    self.planner(
                        feature_name=proposer_result.feature_name,
                        feature_idea=feature_idea,
                    )
                )

                # Validate planner result; skip planning if all retries failed (Site 2)
                if not is_valid_planner({}, (plan, raw)):
                    logger.warning(
                        "Planner returned invalid plan for feature '%s' (best of 3 retries). "
                        "Skipping planning; advancing to evaluation with unchanged feature set.",
                        proposer_result.feature_name,
                    )
                    # Skip planning: don't compute_features, fall through to eval with unchanged set
                    none_explanations = current_none_explanations
                else:
                    current_feature_plans[plan.feature_name] = plan

                    # Compute features for ONLY the new/refined plan (only if planner succeeded)
                    new_plan = {plan.feature_name: plan}
                    new_train, train_none, train_meta = compute_features(
                        self.grouper,
                        train_nctids,
                        self.task_description,
                        new_plan,
                        feature_store_dir=feature_store_dir,
                        task_namespace=task_namespace,
                        feature_store_enabled=feature_store_enabled,
                        counters=cache_stats,
                        plan_origin="planner",
                    )
                    new_val, val_none, val_meta = compute_features(
                        self.grouper,
                        val_nctids,
                        self.task_description,
                        new_plan,
                        feature_store_dir=feature_store_dir,
                        task_namespace=task_namespace,
                        feature_store_enabled=feature_store_enabled,
                        counters=cache_stats,
                        plan_origin="planner",
                    )
                    new_test, test_none, test_meta = compute_features(
                        self.grouper,
                        test_nctids,
                        self.task_description,
                        new_plan,
                        feature_store_dir=feature_store_dir,
                        task_namespace=task_namespace,
                        feature_store_enabled=feature_store_enabled,
                        counters=cache_stats,
                        plan_origin="planner",
                    )

                    # Merge into existing
                    for nctid, nf in new_train.items():
                        current_feature_values[nctid] = current_feature_values.get(nctid, {}) | nf
                    for nctid, nf in new_val.items():
                        current_val_feature_values[nctid] = (
                            current_val_feature_values.get(nctid, {}) | nf
                        )
                    for nctid, nf in new_test.items():
                        current_test_feature_values[nctid] = (
                            current_test_feature_values.get(nctid, {}) | nf
                        )

                    # Merge none_explanations from all splits
                    _merge_two_level_dicts(current_none_explanations, train_none)
                    _merge_two_level_dicts(current_none_explanations, val_none)
                    _merge_two_level_dicts(current_none_explanations, test_none)

                    # Merge builder metadata from all splits
                    _merge_two_level_dicts(all_builder_meta, train_meta)
                    _merge_two_level_dicts(all_builder_meta, val_meta)
                    _merge_two_level_dicts(all_builder_meta, test_meta)

                    none_explanations = current_none_explanations

            elif proposer_result.feature_operation == FeatureOp.REMOVE:
                # --- REMOVE ---
                removed = proposer_result.feature_name

                current_feature_values = {
                    nctid: {k: v for k, v in feats.items() if k != removed}
                    for nctid, feats in current_feature_values.items()
                }
                current_val_feature_values = {
                    nctid: {k: v for k, v in feats.items() if k != removed}
                    for nctid, feats in current_val_feature_values.items()
                }
                current_test_feature_values = {
                    nctid: {k: v for k, v in feats.items() if k != removed}
                    for nctid, feats in current_test_feature_values.items()
                }
                del current_feature_plans[removed]
                for ne in current_none_explanations.values():
                    ne.pop(removed, None)

                # Prune removed feature from builder metadata
                for nctid_meta in all_builder_meta.values():
                    nctid_meta.pop(removed, None)

                none_explanations = current_none_explanations

            else:
                # Unreachable while ``is_valid_proposer`` guards the dispatch above:
                # it rejects any operation outside ``FeatureOp``. Kept as a real
                # branch rather than an ``assert`` so that relaxing that predicate
                # (or reaching here from a future caller) degrades to a skipped
                # iteration instead of running the REMOVE arm -- or, under ``-O``,
                # silently deleting the feature the LLM asked to add.
                logger.error(
                    "Unhandled feature operation %r for %r; skipping iteration.",
                    getattr(proposer_result, "feature_operation", None),
                    getattr(proposer_result, "feature_name", None),
                )
                safe_prev = deepcopy(previous_output)
                return safe_prev._replace(
                    suggestion_index=safe_prev.suggestion_index + 1,
                    cache_stats=CacheStats(),
                )

        # =================================================================
        # Common: Convert to DataFrames, train, evaluate
        # =================================================================
        df = features_to_df(current_feature_values)
        val_df = features_to_df(current_val_feature_values)
        test_df = features_to_df(current_test_feature_values)

        # Build builder diagnostics for evaluator
        diagnostics = _build_builder_diagnostics(
            none_explanations, current_feature_plans, all_builder_meta
        )

        # Train and evaluate each model
        eval_outputs: dict[str, EvalOutput] = {}
        test_eval_outputs: dict[str, ModelEvalResult] = {}

        for classifier_type in settings.model.classifiers:
            model_name = classifier_type.value  # "xgboost" or "tabpfn"
            mt = classifier_type.short_name
            try:
                classifier = ModelRegistry.create_classifier(classifier_type)

                # Regression guard (issue #36): fail fast if the factory
                # returns the wrong classifier type.
                expected = _EXPECTED_CLASSIFIER_CLASS.get(classifier_type)
                actual = type(classifier).__name__
                if expected and actual != expected:
                    raise TypeError(
                        f"create_classifier({classifier_type}) returned "
                        f"{actual}, expected {expected}"
                    )

                # TabPFN handles NaN and categoricals natively -- skip
                # the ColumnTransformer entirely so it sees raw data.
                if classifier_type == ClassifierType.TABPFN:
                    preprocessor = FunctionTransformer(
                        feature_names_out="one-to-one",
                    )  # identity — preserves column names
                else:
                    preprocessor = build_feature_type_transformer(current_feature_plans, mt)

                pipeline = SkPipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("classifier", classifier),
                    ]
                )

                # Drop the "id" column before training
                train_cols = [c for c in df.columns if c != "id"]
                pipeline.fit(df[train_cols], self.y_train)

                # Evaluate on validation and test
                val_result = eval_model(pipeline, val_df[train_cols], self.y_val)
                test_result = eval_model(pipeline, test_df[train_cols], self.y_test)

                logger.info(
                    "Model %s: val ROC-AUC=%.4f, test ROC-AUC=%.4f",
                    model_name,
                    val_result.roc_auc,
                    test_result.roc_auc,
                )

                # Compute shapiq interaction values
                interaction_dict = compute_shapiq_for_pipeline(
                    pipeline=pipeline,
                    model_type=mt,
                    val_df=val_df,
                    train_cols=train_cols,
                    y_train=self.y_train,
                    train_df=df,
                    max_order=settings.model.shapiq_max_order,
                    max_samples=settings.model.shapiq_max_samples,
                    budget=settings.model.shapiq_budget,
                )

                # Attach interaction values to ModelEvalResult
                val_result = val_result._replace(interaction_values=interaction_dict)

                # Run evaluator for this model
                eval_output = self.evaluator(
                    feature_plans=current_feature_plans,
                    model_eval_result=val_result,
                    none_explanations=none_explanations,
                    builder_diagnostics=diagnostics,
                )
                eval_outputs[model_name] = eval_output
                test_eval_outputs[model_name] = test_result

            except Exception:
                logger.exception("Failed to train/evaluate model %s", model_name)

        return AgentOutput(
            eval_outputs=eval_outputs,
            test_eval_outputs=test_eval_outputs,
            operation=proposer_result,
            feature_plans=current_feature_plans,
            df=df,
            val_df=val_df,
            suggestion_index=0,
            raw_features=current_feature_values,
            raw_val_features=current_val_feature_values,
            raw_test_features=current_test_feature_values,
            none_explanations=none_explanations,
            builder_meta=all_builder_meta,
            builder_diagnostics=diagnostics,
            cache_stats=cache_stats,
        )
