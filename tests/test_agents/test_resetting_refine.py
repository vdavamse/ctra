"""Regression tests for ``ResettingRefine``'s per-call failure budget.

``dspy.Refine`` decrements ``self.fail_count`` for every exception it swallows
inside its per-attempt ``try`` and never restores it. Because
``Agent.proposer``/``planner``/``grouper`` are built once and reused for a whole
MCTS run, that budget erodes until Refine re-raises instead of returning its
best attempt -- past the caller's ``is_valid_*`` guard.

Any swallowed exception erodes the budget: a raising ``forward()``, a failing
reward function, or a feedback step whose ``dict(outputs)`` cannot convert the
module's return. The production modules now return ``dspy.Prediction`` so that
last case no longer occurs for them; ``_ReturnsNamedTuple`` below deliberately
keeps a legacy non-Prediction shape so ``dict(outputs)`` raises on every
sub-threshold attempt, reproducing the erosion deterministically.
"""

from __future__ import annotations

import dspy
from dspy.utils.dummies import DummyLM

from ctra.agents.data_models import FeatureOp, ProposerOutput
from ctra.agents.reward_fns import ResettingRefine


class _Sig(dspy.Signature):
    q: str = dspy.InputField()
    a: str = dspy.OutputField()


class _ReturnsNamedTuple(dspy.Module):  # type: ignore[misc]
    """Shaped like ``FeatureProposer``: real predictor, non-Prediction return."""

    def __init__(self) -> None:
        super().__init__()
        self.p = dspy.Predict(_Sig)

    def forward(self, **kwargs):
        self.p(q=kwargs.get("q", "x"))
        return ProposerOutput(
            feature_operation=FeatureOp.ADD,
            feature_name="f",
            feature_explanation="e",
        )


def _always_invalid(kwargs, result) -> float:
    return 0.0


def _dummy_lm() -> DummyLM:
    # Supplies OfferFeedback's fields too, so the feedback path itself succeeds.
    return DummyLM([{"a": "ok", "discussion": "d", "advice": "{}", "reasoning": "r"}] * 80)


def test_repeated_invalid_outputs_do_not_exhaust_the_budget() -> None:
    """The caller's skip branch must stay reachable on every call, not just the first.

    With plain ``dspy.Refine`` this raises on call 2.
    """
    dspy.configure(lm=_dummy_lm())
    refine = ResettingRefine(
        module=_ReturnsNamedTuple(), N=3, reward_fn=_always_invalid, threshold=1.0
    )

    for call in range(1, 5):
        result = refine(q="x")
        assert isinstance(result, ProposerOutput), f"call {call} did not return its best attempt"


def test_fail_count_is_restored_between_calls() -> None:
    """The budget is per call, so it cannot erode across a long MCTS run."""
    dspy.configure(lm=_dummy_lm())
    refine = ResettingRefine(
        module=_ReturnsNamedTuple(), N=3, reward_fn=_always_invalid, threshold=1.0
    )
    initial = refine.fail_count

    refine(q="x")
    eroded = refine.fail_count
    assert eroded < initial, "expected dspy.Refine to erode the budget within a call"

    refine(q="x")
    assert refine.fail_count == eroded, "budget should be reset to the same starting point"


def test_valid_output_short_circuits_without_touching_the_budget() -> None:
    """A passing reward breaks before the feedback path, so nothing is consumed."""
    dspy.configure(lm=_dummy_lm())
    refine = ResettingRefine(
        module=_ReturnsNamedTuple(), N=3, reward_fn=lambda k, r: 1.0, threshold=1.0
    )
    initial = refine.fail_count

    result = refine(q="x")

    assert isinstance(result, ProposerOutput)
    assert refine.fail_count == initial
