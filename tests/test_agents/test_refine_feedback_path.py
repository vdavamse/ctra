"""Acceptance test for issue #5: Refine OfferFeedback path now runs.

The three Refine-wrapped modules return dspy.Prediction so that Refine's
feedback step (refine.py:153 ``dict(outputs)``) succeeds and OfferFeedback
actually runs. This file is the regression fence for "retries are guided,
not blind re-rolls".

Mechanism: refine.py:153 does ``dict(outputs)`` which used to raise for
non-Prediction returns, skipping OfferFeedback. Now it succeeds, the LM
receives advice, and hints are injected into subsequent attempts.

Verification:
- LM calls are observable via dspy.clients.base_lm.GLOBAL_HISTORY
  (module-level list, updated by BaseLM.__call__ unless history disabled)
- Per-attempt LMs are copies (refine.py:131), so DummyLM.history sees only
  OfferFeedback calls; GLOBAL_HISTORY sees all five
- Advice must be a non-empty dict (refine.py:117 `if not advice:`)
- Predictor names are from signature2name: "feature_grouper.predict",
  "planner.predict", "proposer.predict"
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

try:
    import dspy
    from dspy.clients.base_lm import GLOBAL_HISTORY

    from ctra.agents.data_models import (
        FeaturePlan,
        FeatureSource,
        FeatureType,
    )
    from ctra.agents.feature_grouper import FeatureGrouper
    from ctra.agents.feature_planner import FeaturePlanner
    from ctra.agents.feature_proposer import FeatureProposer
    from ctra.agents.reward_fns import (
        ResettingRefine,
        grouper_reward,
    )
    from tests.test_agents.conftest import (
        grouper_prediction,
        planner_prediction,
        proposer_prediction,
    )

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")

SENTINEL = "REFINE_FEEDBACK_TEST_SENTINEL"


def _make_plan(name: str) -> FeaturePlan:
    return FeaturePlan(
        feature_name=name,
        feature_idea=f"{name} idea",
        feature_type={"value": FeatureType.FLOAT},
        data_sources=[FeatureSource.PUBMED],
        example_values=[{"value": "1.0"}],
        possible_values={},
        feature_instructions=f"Extract {name}.",
    )


def _advice(predictor_name: str, text: str) -> str:
    """Build OfferFeedback advice as JSON dict keyed by predictor name.

    refine.py:117 is `if not advice:` — the advice **must** be a non-empty
    dict. The existing fixture in test_resetting_refine.py returns "{}",
    which is falsy, so it would never produce a hint even if the path ran.
    """
    return json.dumps({predictor_name: text})


@pytest.fixture
def _clear_history():
    """Clear GLOBAL_HISTORY around each end-to-end test."""
    GLOBAL_HISTORY.clear()
    yield
    GLOBAL_HISTORY.clear()


def test_all_three_modules_return_predictions():
    """All three Refine-wrapped modules return dspy.Prediction (core issue #5 fix).

    This verifies that proposer, planner, and grouper all have the right return
    type for Refine's feedback step to work.
    """
    # Test proposer
    proposer = FeatureProposer(task_description="Test task")
    with patch.object(proposer, "proposer") as mock_cot:
        mock_cot.return_value = dspy.Prediction(
            operation="add", feature_name="test", operation_description="test"
        )
        prev = MagicMock()
        prev.feature_plans = {}
        result = proposer(previous_output=prev)
        assert isinstance(result, dspy.Prediction)
        assert dict(result).keys() == {"proposal"}

    # Test planner
    planner = FeaturePlanner(task_description="Test task")
    with patch.object(planner, "planner") as mock_cot:
        mock_cot.return_value = dspy.Prediction(
            feature_type={"value": FeatureType.FLOAT},
            data_sources=[FeatureSource.PUBMED],
            example_values=[],
            possible_values={},
            feature_instructions="test",
        )
        result = planner(feature_name="test", feature_idea="idea")
        assert isinstance(result, dspy.Prediction)
        assert dict(result).keys() == {"plan", "raw"}

    # Test grouper
    grouper = FeatureGrouper(task_description="Test task")
    with patch.object(grouper, "feature_grouper") as mock_cot:
        mock_cot.return_value = dspy.Prediction(groups=[["feat_a"]])
        result = grouper(feature_plans={"feat_a": _make_plan("feat_a")}, task="Test task")
        assert isinstance(result, dspy.Prediction)
        assert dict(result).keys() == {"groups"}


def test_helpers_match_the_real_modules():
    """Drift guard: conftest builders produce same Prediction keys as real modules.

    If a module's field name changes (e.g., 'proposal' → 'output'), this
    test fails immediately in CI, preventing silent drift.
    """
    proposer = FeatureProposer(task_description="Test")

    # Mock instance attribute
    with patch.object(proposer, "proposer") as mock_proposer_module:
        mock_result = MagicMock()
        mock_result.operation = "add"
        mock_result.feature_name = "test"
        mock_result.operation_description = "desc"
        mock_proposer_module.return_value = mock_result

        prev = MagicMock()
        prev.feature_plans = {}
        result = proposer(previous_output=prev)

    # Helper must have same keys
    helper = proposer_prediction(
        feature_name="test",
        feature_explanation="desc",
        feature_operation="add",
    )
    assert set(helper.keys()) == set(result.keys())

    # Planner: builder keys must match the real module's Prediction keys
    planner = FeaturePlanner(task_description="Test")
    with patch.object(planner, "planner") as mock_planner_module:
        mock_planner_module.return_value = dspy.Prediction(
            feature_type={"value": FeatureType.FLOAT},
            data_sources=[FeatureSource.PUBMED],
            example_values=[],
            possible_values={},
            feature_instructions="test",
        )
        planner_result = planner(feature_name="test", feature_idea="idea")

    helper_plan = planner_prediction(_make_plan("test"), raw="raw")
    assert set(helper_plan.keys()) == set(planner_result.keys())

    # Grouper: builder keys must match the real module's Prediction keys
    grouper = FeatureGrouper(task_description="Test")
    with patch.object(grouper, "feature_grouper") as mock_grouper_module:
        mock_grouper_module.return_value = dspy.Prediction(groups=[["feat_a"]])
        grouper_result = grouper(feature_plans={"feat_a": _make_plan("feat_a")}, task="Test")

    helper_groups = grouper_prediction([])
    assert set(helper_groups.keys()) == set(grouper_result.keys())


# ---------------------------------------------------------------------------
# End-to-end: the feedback path actually runs (the issue's acceptance criterion)
# ---------------------------------------------------------------------------

_GROUPER_PREDICTOR = "feature_grouper.predict"


def _two_plans() -> dict[str, FeaturePlan]:
    return {"feat_a": _make_plan("feat_a"), "feat_b": _make_plan("feat_b")}


def _sub_threshold_lm(advice_json: str):
    """A DummyLM whose grouping covers only ``feat_a`` (reward 0.0 every attempt).

    The same canned dict serves both signatures: ``reasoning``/``groups`` for the
    grouper's ChainOfThought and ``discussion``/``advice`` for OfferFeedback.
    """
    from dspy.utils.dummies import DummyLM

    canned = {
        "reasoning": "r",
        "groups": json.dumps([["feat_a"]]),
        "discussion": "d",
        "advice": advice_json,
    }
    return DummyLM([canned] * 20)


def _rendered(entry: dict) -> str:
    return json.dumps(entry.get("messages") or entry.get("prompt") or "")


@pytest.mark.usefixtures("_clear_history")
def test_sub_threshold_attempts_trigger_offer_feedback_and_hints(capsys) -> None:
    """Headline: real grouper + real reward + ResettingRefine + DummyLM.

    Before #5 this run made 3 LM calls, printed ``Refine: Attempt failed`` twice
    (``dict(list[dict])`` raised), eroded ``fail_count`` from 3 to 1 and never sent
    a hint. Now it makes 5 calls (module, OfferFeedback, module, OfferFeedback,
    module), swallows nothing, keeps the budget, and the sentinel advice reaches
    attempts 2 and 3.
    """
    refine = ResettingRefine(module=FeatureGrouper(), N=3, reward_fn=grouper_reward, threshold=1.0)

    with dspy.context(lm=_sub_threshold_lm(_advice(_GROUPER_PREDICTOR, SENTINEL))):
        result = refine(feature_plans=_two_plans(), task="t")

    history = GLOBAL_HISTORY
    assert len(history) == 5, [_rendered(e)[:80] for e in history]
    assert "Attempt failed" not in capsys.readouterr().out
    assert refine.fail_count == 3
    assert list(result.keys()) == ["groups"]

    module_attempts = [history[0], history[2], history[4]]
    assert SENTINEL not in _rendered(module_attempts[0])
    for attempt in module_attempts[1:]:
        text = _rendered(attempt)
        assert "hint_" in text and SENTINEL in text, text[:300]


@pytest.mark.usefixtures("_clear_history")
def test_legacy_raw_return_still_takes_the_blind_path(capsys) -> None:
    """Contrast case proving the headline test discriminates old from new.

    A module shaped like the pre-#5 grouper (real predictor, raw ``list[dict]``
    return) makes ``dict(outputs)`` raise, so OfferFeedback never runs.
    """

    class _RawListGrouper(dspy.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.feature_grouper = FeatureGrouper().feature_grouper

        def forward(self, feature_plans, task):
            self.feature_grouper(task=task, feature_plans={})
            return [{"feat_a": feature_plans["feat_a"]}]

    refine = ResettingRefine(module=_RawListGrouper(), N=3, reward_fn=grouper_reward, threshold=1.0)

    with dspy.context(lm=_sub_threshold_lm(_advice(_GROUPER_PREDICTOR, SENTINEL))):
        refine(feature_plans=_two_plans(), task="t")

    history = GLOBAL_HISTORY
    assert len(history) == 3
    assert capsys.readouterr().out.count("Attempt failed") == 2
    assert refine.fail_count == 1
    assert all(SENTINEL not in _rendered(e) for e in history)


@pytest.mark.usefixtures("_clear_history")
def test_advice_keyed_by_wrong_predictor_yields_na_hint() -> None:
    """refine.py resolves advice via ``signature2name``; a wrong key degrades to "N/A".

    Guards the routing: asserting only that ``hint_`` is present would pass here too.
    """
    refine = ResettingRefine(module=FeatureGrouper(), N=3, reward_fn=grouper_reward, threshold=1.0)

    with dspy.context(lm=_sub_threshold_lm(_advice("nonexistent.predict", SENTINEL))):
        refine(feature_plans=_two_plans(), task="t")

    history = GLOBAL_HISTORY
    assert len(history) == 5
    for attempt in (history[2], history[4]):
        text = _rendered(attempt)
        assert "hint_" in text
        assert "N/A" in text
        assert SENTINEL not in text


@pytest.mark.usefixtures("_clear_history")
def test_empty_advice_dict_runs_feedback_but_injects_no_hint() -> None:
    """Pins refine.py's ``if not advice:``: ``"{}"`` parses falsy, so no hint is sent.

    This is why ``_advice`` must build a non-empty dict; the older fixture in
    ``test_resetting_refine.py`` returns ``"{}"`` and would never exercise the hint.
    """
    refine = ResettingRefine(module=FeatureGrouper(), N=3, reward_fn=grouper_reward, threshold=1.0)

    with dspy.context(lm=_sub_threshold_lm("{}")):
        refine(feature_plans=_two_plans(), task="t")

    history = GLOBAL_HISTORY
    assert len(history) == 5
    assert all("hint_" not in _rendered(e) for e in history)
