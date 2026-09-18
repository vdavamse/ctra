"""DSPy ChainOfThought agent for feature proposal.

**FeatureProposer**: Takes an evaluator suggestion from ``AgentOutput`` and
proposes a single ADD/REMOVE/REFINE operation, validated via ``is_valid_proposer``
predicate.

The proposer uses the primary LM (Claude Opus 4.6) configured via
:func:`ctra.agents.lm_config.configure_lm`.

Returns a ``dspy.Prediction`` carrying the best-effort proposal. Note: ``forward()``
does not raise on invalid LLM output. Validation is deferred to the caller (e.g.,
``dspy.Refine`` rewards and post-checks in the orchestrator).
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

    Validation via ``is_valid_proposer`` (does not raise):
    - ADD: ``feature_name`` must not already exist in current features.
    - REMOVE / REFINE: ``feature_name`` must exist in current features.

    Parameters:
        task_description: Text description of the prediction task.
    """

    def __init__(self, task_description: str) -> None:
        super().__init__()
        self.task_description = task_description
        self.proposer = dspy.ChainOfThought(FeatureProposerSignature)

    def forward(self, previous_output: AgentOutput) -> dspy.Prediction:
        """Propose a single feature operation from the best evaluator suggestion.

        Parameters:
            previous_output: The ``AgentOutput`` from the previous iteration.

        Returns:
            ``dspy.Prediction`` with field ``proposal`` (a ``ProposerOutput``).
            Wrapped so ``dspy.Refine``'s ``dict(outputs)`` feedback step succeeds
            -- see ``reward_fns.unwrap_proposal``.
        """
        current_features_with_plan = [
            (fp.feature_name, dump_as_json(fp, pretty=False))
            for fp in previous_output.feature_plans.values()
        ]

        # ``get_next_suggestion()`` raises when the evaluator produced no
        # suggestions at all (reachable: ``evaluator.py`` returns ``[]`` when every
        # analysis step fails). This runs inside a ``dspy.Refine`` wrapper, where a
        # raise is retried N times and then re-raised past the caller's
        # ``is_valid_proposer`` guard -- the dead-skip-branch failure this module
        # was fixed to avoid. Degrade to an empty suggestion instead and let the
        # proposal be judged on its merits.
        # KeyError as well as ValueError: get_best_eval_output() indexes
        # test_eval_outputs by the best val model's name, which raises KeyError if
        # the two dicts ever disagree.
        try:
            suggestion = previous_output.get_next_suggestion()
        except (ValueError, KeyError):
            logger.warning(
                "No evaluator suggestions available; proposing without one.",
                exc_info=True,
            )
            suggestion = ""

        proposer_result = self.proposer(
            task=self.task_description,
            current_features_with_plan=current_features_with_plan,
            suggestion=suggestion,
        )

        # Coerce operation to FeatureOp defensively; fall back to the raw value if
        # coercion fails so forward() stays non-raising and is_valid_proposer can
        # reject it. Case/whitespace are normalised too: "Add " is a recoverable
        # formatting slip, not a different operation, and retrying it would burn
        # three LM calls to arrive at the same string.
        #
        # Note the normalised candidate goes through ``.value`` rather than
        # ``str(op)``: on Python 3.10 ``data_models`` uses a ``StrEnum(str, Enum)``
        # backport where ``str(FeatureOp.ADD)`` is "FeatureOp.ADD", not "add".
        raw_operation = proposer_result.operation
        normalised = getattr(raw_operation, "value", raw_operation)
        if isinstance(normalised, str):
            normalised = normalised.strip().lower()

        operation = raw_operation
        for candidate in (raw_operation, normalised):
            try:
                operation = FeatureOp(candidate)
                break
            except (ValueError, TypeError):
                continue

        return dspy.Prediction(
            proposal=ProposerOutput(
                feature_name=proposer_result.feature_name,
                feature_explanation=proposer_result.operation_description,
                feature_operation=operation,
            )
        )
