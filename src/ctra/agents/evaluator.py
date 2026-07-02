"""Feature set evaluation agent.

**Evaluator**: ReAct-augmented per-trial error analysis with CTRA's 7
tools, ``none_explanations`` context, and separate generic + example-based
suggestions.

The evaluator uses the primary LM (Claude Opus 4.6) configured via
:func:`ctra.agents.lm_config.configure_lm`.
"""

from __future__ import annotations

import logging

import dspy
import numpy as np
import yaml

from ctra.agents.data_models import BuilderDiagnostics, EvalOutput, FeaturePlan, ModelEvalResult
from ctra.agents.feature_utils import dump_as_json, format_interactions_for_llm
from ctra.agents.signatures import (
    EvaluatorSignature,
    EvaluatorSummarizerSignature,
    EvaluatorWithExampleSignature,
)
from ctra.rag.tools import (
    get_detailed_nct_info,
    get_trial_info_dict,
    make_aact_search,
    make_chembl_search,
    make_drugsfda_search,
    make_faers_search,
    make_nct_search,
    make_primekg_search,
    make_pubmed_search,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluator — ReAct-augmented error analysis
# ---------------------------------------------------------------------------


class Evaluator(dspy.Module):  # type: ignore[misc]
    """Evaluate feature sets with tool-augmented per-trial error analysis.

    Enhanced with:
    - shapiq interaction values (main effects + pairwise) replacing gain-based importances
    - Builder diagnostics (None rates, Researcher vs Builder failure attribution)

    1. Runs a generic suggestions pass (``EvaluatorSignature``).
    2. Samples up to 3 misclassified trials.
    3. For each, creates a ``dspy.ReAct(EvaluatorWithExampleSignature)``
       with CTRA's 7 RAG tools to deeply analyze why the model was wrong.
    4. Returns combined suggestions as ``EvalOutput``.

    The evaluator receives ``none_explanations`` and ``builder_diagnostics``
    so it can tell the LLM why specific features are ``None`` for misclassified
    trials and whether failures are Researcher vs Builder issues.

    Parameters:
        task_description: Text description of the prediction task.
    """

    def __init__(self, task_description: str) -> None:
        super().__init__()
        self.task_description = task_description
        self.evaluator_no_example = dspy.ChainOfThought(EvaluatorSignature)
        self.summarizer = dspy.ChainOfThought(EvaluatorSummarizerSignature)
        self.rng = np.random.default_rng(42)

    def forward(
        self,
        feature_plans: dict[str, FeaturePlan],
        model_eval_result: ModelEvalResult,
        none_explanations: dict[str, dict[str, str]],
        builder_diagnostics: BuilderDiagnostics | None = None,
    ) -> EvalOutput:
        """Run generic + example-based evaluation.

        Parameters:
            feature_plans: Current feature plans.
            model_eval_result: Metrics and wrong predictions from a model.
            none_explanations: Per-trial per-feature reasons for None values.
            builder_diagnostics: Aggregated builder diagnostics for Researcher
                vs Builder failure attribution.

        Returns:
            ``EvalOutput`` with suggestions for the next MCTS iteration.
        """
        current_features_with_plan = [
            (fp.feature_name, dump_as_json(fp, pretty=False)) for fp in feature_plans.values()
        ]

        # Format interaction values for LLM consumption
        interactions_str = format_interactions_for_llm(model_eval_result.interaction_values)

        # Format builder diagnostics
        diag_str = (
            builder_diagnostics.format_for_llm()
            if builder_diagnostics is not None
            else "No builder diagnostics available."
        )

        # Step 1: Generic suggestions (no specific example)
        eval_result = self.evaluator_no_example(
            task=self.task_description,
            roc_auc_score=model_eval_result.roc_auc,
            current_features_with_plan=current_features_with_plan,
            feature_interactions=interactions_str,
            builder_diagnostics=diag_str,
        )
        no_example_suggestions: list[str] = eval_result.suggestions

        # Step 2: Sample up to 3 wrong predictions for per-trial analysis
        example_suggestions: list[str] = []

        if len(model_eval_result.wrong_idxs) > 0:
            picks = self.rng.choice(
                model_eval_result.wrong_idxs,
                size=min(3, len(model_eval_result.wrong_idxs)),
                replace=False,
            )

            wrong_idx_to_preds = dict(
                zip(model_eval_result.wrong_idxs, model_eval_result.wrong_preds, strict=True)
            )

            # Step 3: Per-trial ReAct analysis
            for pick in picks:
                entry = model_eval_result.wrong_df.loc[pick]
                pick_nct_id = str(entry.get("id", entry.name))
                pred = wrong_idx_to_preds[pick]
                correct = 1 if pred == 0 else 0

                # Build none_explanations context for this trial
                trial_none = none_explanations.get(pick_nct_id, {})

                example_error = (
                    f"## {pick_nct_id} Predicted {pred}, should be {correct}\n\n"
                    f"### Features\n```\n{yaml.dump(entry.to_dict())}\n```\n\n"
                    f"### Reasons for features that are None\n"
                    f"{yaml.dump(trial_none)}\n"
                )

                try:
                    # Build per-trial tools (CTRA's 7 sources)
                    nct_info = get_trial_info_dict(pick_nct_id)
                    tools = [
                        get_detailed_nct_info,
                        make_pubmed_search(nct_info),
                        make_nct_search(nct_info),
                        make_chembl_search(nct_info),
                        make_faers_search(nct_info),
                        make_aact_search(nct_info),
                        make_primekg_search(nct_info),
                        make_drugsfda_search(nct_info),
                    ]

                    example_evaluator = dspy.ReAct(
                        EvaluatorWithExampleSignature,
                        tools=tools,
                    )

                    result = example_evaluator(
                        task=self.task_description,
                        roc_auc_score=model_eval_result.roc_auc,
                        current_features_with_plan=current_features_with_plan,
                        feature_interactions=interactions_str,
                        builder_diagnostics=diag_str,
                        example=example_error,
                    )
                    example_suggestions.append(result.analysis)

                except Exception:
                    logger.warning(
                        "Failed to run evaluator for %s, skipping",
                        pick_nct_id,
                        exc_info=True,
                    )
                    continue

        # Step 4: Combine and summarize
        all_suggestions = example_suggestions + no_example_suggestions

        if all_suggestions:
            summary = self.summarizer(task=self.task_description, analyses=all_suggestions)
            final_suggestions: list[str] = summary.suggestions
        else:
            final_suggestions = []

        return EvalOutput(
            model_eval_result=model_eval_result,
            suggestions=final_suggestions,
        )
