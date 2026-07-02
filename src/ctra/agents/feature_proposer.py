"""DSPy ChainOfThought agent for feature proposal.

**FeatureProposer**: Takes an evaluator suggestion from ``AgentOutput`` and
proposes a single ADD/REMOVE/REFINE operation with ``ValueError``
validation (retried via ``dspy.Refine``).

The proposer uses the primary LM (Claude Opus 4.6) configured via
:func:`ctra.agents.lm_config.configure_lm`.
"""

from __future__ import annotations

import logging

import dspy

from ctra.agents.data_models import AgentOutput, FeatureOp, ProposerOutput
from ctra.agents.feature_utils import dump_as_json
from ctra.agents.signatures import FeatureProposerSignature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proposer — suggestion-driven ADD/REMOVE/REFINE
# ---------------------------------------------------------------------------


class FeatureProposer(dspy.Module):  # type: ignore[misc]
    """Propose a single feature operation based on an evaluator suggestion.

    Takes the best suggestion from ``AgentOutput`` and proposes exactly one operation:
    ADD (new feature), REMOVE (drop feature), or REFINE (improve feature).

    Validation (``ValueError`` on failure, retried via ``dspy.Refine``
    in the orchestrator):
    - ADD: ``feature_name`` must not already exist in current features.
    - REMOVE / REFINE: ``feature_name`` must exist in current features.

    Parameters:
        task_description: Text description of the prediction task.
    """

    def __init__(self, task_description: str) -> None:
        super().__init__()
        self.task_description = task_description
        self.proposer = dspy.ChainOfThought(FeatureProposerSignature)

    def forward(self, previous_output: AgentOutput) -> ProposerOutput:
        """Propose a single feature operation from the best evaluator suggestion.

        Parameters:
            previous_output: The ``AgentOutput`` from the previous iteration.

        Returns:
            ``ProposerOutput`` with operation type, feature name, and explanation.
        """
        existing_names = set(previous_output.feature_plans.keys())

        current_features_with_plan = [
            (fp.feature_name, dump_as_json(fp, pretty=False))
            for fp in previous_output.feature_plans.values()
        ]

        proposer_result = self.proposer(
            task=self.task_description,
            current_features_with_plan=current_features_with_plan,
            suggestion=previous_output.get_next_suggestion(),
        )

        # --- Validation: check operation against current feature set ---
        if proposer_result.operation.value == FeatureOp.ADD.value:
            if proposer_result.feature_name in existing_names:
                raise ValueError(
                    f"ADD: feature_name '{proposer_result.feature_name}' "
                    f"already exists. Pick a different name."
                )
        else:
            if proposer_result.feature_name not in existing_names:
                raise ValueError(
                    f"REMOVE/REFINE: feature_name "
                    f"'{proposer_result.feature_name}' not in existing "
                    f"features {existing_names}"
                )

        return ProposerOutput(
            feature_name=proposer_result.feature_name,
            feature_explanation=proposer_result.operation_description,
            feature_operation=proposer_result.operation,
        )
