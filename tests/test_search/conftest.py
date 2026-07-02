"""Shared test fixtures for MCTS search tests.

Provides ``make_stub_runner`` to wrap simple ``(features -> objectives)``
functions into the runner protocol expected by ``MCTSSearch``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ctra.agents.data_models import (
    AgentOutput,
    EvalOutput,
    FeaturePlan,
    FeatureSource,
    FeatureType,
    ModelEvalResult,
)
from ctra.search.objectives import ObjectiveResult

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_plan(name: str) -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[],
        possible_values={},
        feature_instructions=f"Extract {name}.",
    )


def make_stub_output(
    features: list[str],
    roc_auc: float = 0.5,
    suggestions: list[str] | None = None,
) -> AgentOutput:
    """Build a minimal AgentOutput for testing."""
    plans = {f: _make_plan(f) for f in features}
    suggestions = suggestions or [f"Suggestion {i}" for i in range(5)]
    er = ModelEvalResult(
        roc_auc=roc_auc,
        f1=0.5,
        pr_auc=0.5,
        interaction_values={},
        wrong_idxs=[],
        wrong_preds=[],
        wrong_df=pd.DataFrame(),
        pipeline=None,
    )
    return AgentOutput(
        eval_outputs={"stub": EvalOutput(model_eval_result=er, suggestions=suggestions)},
        test_eval_outputs={"stub": er},
        operation=None,
        feature_plans=plans,
        df=pd.DataFrame(),
        val_df=pd.DataFrame(),
        suggestion_index=0,
        raw_features={},
        raw_val_features={},
        raw_test_features={},
        none_explanations={},
        builder_meta={},
    )


def make_stub_runner(
    evaluate_fn: Callable[[list[str]], np.ndarray | ObjectiveResult],
    expand_fn: Callable[..., list[tuple[list[str], str, str]]] | None = None,
    initial_features: list[str] | None = None,
) -> Callable[..., AgentOutput]:
    """Wrap a simple ``(features -> objectives)`` function as a runner.

    The returned runner builds a minimal ``AgentOutput`` whose ROC-AUC
    matches the first objective value from ``evaluate_fn``.  If
    ``expand_fn`` is provided, its output is stored as suggestions
    on the ``AgentOutput`` so the default ``_suggestion_expand`` can
    generate children.

    Parameters:
        evaluate_fn: ``(features) -> ndarray[accuracy, parsimony]``
            or ``(features, fidelity=) -> ...``
        expand_fn: Optional legacy ``(node, max_children=) -> candidates``
            function.  Used to generate suggestion text for the output.
        initial_features: Features for the root node (when previous_output is None).
    """
    _initial = initial_features or ["base"]
    _last_features: dict[str, list[str]] = {"current": list(_initial)}
    _expand_results: dict[str, list] = {}

    def runner(node_id: str, task: Any, previous_output: AgentOutput | None) -> AgentOutput:
        # Determine features: from previous_output plans, or initial
        if previous_output is not None and previous_output.feature_plans:
            features = list(previous_output.feature_plans.keys())
        else:
            features = list(_initial)
        _last_features["current"] = features

        # Call the evaluate function
        try:
            result = evaluate_fn(features, fidelity=1.0)
        except TypeError:
            result = evaluate_fn(features)

        if isinstance(result, ObjectiveResult):
            roc_auc = float(result.values[0])
        else:
            arr = np.asarray(result).ravel()
            roc_auc = float(arr[0]) if len(arr) > 0 else 0.0

        # Generate suggestions from expand_fn if available
        suggestions = [f"Suggestion {i}" for i in range(5)]
        if expand_fn is not None:
            from ctra.search.mcts import MCTSNode

            dummy_node = MCTSNode(features=features)
            # Store AgentOutput on the dummy node so expand_fn can read it
            dummy_node.eval_output = make_stub_output(features, roc_auc, suggestions)
            try:
                candidates = expand_fn(dummy_node, max_children=5)
                suggestions = [f"{op}: {detail}" for _feats, op, detail in candidates]
                # Store for later reference
                key = "|".join(sorted(features))
                _expand_results[key] = candidates
            except Exception:
                pass

        return make_stub_output(features, roc_auc, suggestions)

    return runner
