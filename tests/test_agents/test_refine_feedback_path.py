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

    from ctra.agents.data_models import (
        FeaturePlan,
        FeatureSource,
        FeatureType,
        ProposerOutput,
    )
    from ctra.agents.feature_grouper import FeatureGrouper
    from ctra.agents.feature_planner import FeaturePlanner
    from ctra.agents.feature_proposer import FeatureProposer
    from ctra.agents.reward_fns import (
        ResettingRefine,
        grouper_reward,
        unwrap_groups,
        unwrap_planner_result,
        unwrap_proposal,
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
    """Clear GLOBAL_HISTORY before each test."""
    try:
        from dspy.clients.base_lm import GLOBAL_HISTORY

        GLOBAL_HISTORY.clear()
        yield
        GLOBAL_HISTORY.clear()
    except ImportError:
        yield


def test_real_grouper_returns_prediction():
    """Real FeatureGrouper.forward() returns dspy.Prediction (core issue #5 fix).

    This is the headline assertion: the modules return Predictions that can be
    serialized by dict() in Refine's feedback step, enabling OfferFeedback.
    The full Refine integration is tested in the other cases below (without
    requiring full DSPy LM mocking).
    """
    grouper = FeatureGrouper(task_description="Test task")
    feature_plans = {
        "feat_a": _make_plan("feat_a"),
        "feat_b": _make_plan("feat_b"),
    }

    # Mock the inner ChainOfThought to return valid groups
    with patch.object(grouper, "feature_grouper") as mock_cot:
        mock_cot.return_value = dspy.Prediction(groups=[["feat_a", "feat_b"]])

        result = grouper.forward(feature_plans=feature_plans, task="Test task")

    # THE FIX: result is now dspy.Prediction, not a bare list
    assert isinstance(result, dspy.Prediction)
    # dict() now succeeds (this is what Refine's feedback step does)
    result_dict = dict(result)
    assert "groups" in result_dict
    # Unwrap works
    groups = unwrap_groups(result)
    assert isinstance(groups, list)


def test_all_three_modules_return_predictions():
    """All three Refine-wrapped modules return dspy.Prediction (core issue #5 fix).

    This verifies that proposer, planner, and grouper all have the right return
    type for Refine's feedback step to work.
    """
    from ctra.agents.feature_proposer import FeatureProposer

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
        assert dict(result) is not None  # dict() succeeds

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
        assert dict(result) is not None

    # Test grouper
    grouper = FeatureGrouper(task_description="Test task")
    with patch.object(grouper, "feature_grouper") as mock_cot:
        mock_cot.return_value = dspy.Prediction(groups=[["feat_a"]])
        result = grouper(feature_plans={"feat_a": _make_plan("feat_a")}, task="Test task")
        assert isinstance(result, dspy.Prediction)
        assert dict(result) is not None  # dict() succeeds - THE FIX!


def test_dict_of_each_return_shape_is_the_operation_refine_needs():
    """dict() succeeds on Prediction outputs (the core fix for refine.py:153).

    Before: dict(ProposerOutput) → ValueError
            dict((FeaturePlan, raw)) → ValueError
            dict(list[dict]) → ValueError (sometimes; [] → {}, singleton → garbage)

    After: dict(Prediction(proposal=...)) → {'proposal': ...}
           dict(Prediction(plan=..., raw=...)) → {'plan': ..., 'raw': ...}
           dict(Prediction(groups=...)) → {'groups': [...]}
    """
    # Test proposer shape
    prop_pred = proposer_prediction(
        feature_name="test",
        feature_explanation="explanation",
        feature_operation="add",
    )
    prop_dict = dict(prop_pred)
    assert "proposal" in prop_dict

    # Test planner shape
    plan = _make_plan("test")
    plan_pred = planner_prediction(plan, raw="raw_output")
    plan_dict = dict(plan_pred)
    assert "plan" in plan_dict and "raw" in plan_dict

    # Test grouper shape (including empty edge case)
    groups_pred_empty = grouper_prediction([])
    groups_dict_empty = dict(groups_pred_empty)
    assert "groups" in groups_dict_empty
    assert groups_dict_empty["groups"] == []

    groups_pred_full = grouper_prediction([{"feat_a": _make_plan("feat_a")}])
    groups_dict_full = dict(groups_pred_full)
    assert "groups" in groups_dict_full


def test_prediction_keys_match_module_contracts():
    """Verify Prediction fields match the module return contracts.

    Ensures field names are correct: proposal, plan/raw, groups
    """
    plan = _make_plan("test")

    # Proposer prediction has 'proposal' field
    prop_pred = proposer_prediction(
        feature_name="test",
        feature_explanation="test",
        feature_operation="add",
    )
    assert "proposal" in dict(prop_pred)
    assert isinstance(dict(prop_pred)["proposal"], ProposerOutput)

    # Planner prediction has 'plan' and 'raw' fields
    plan_pred = planner_prediction(plan, raw="raw_output")
    assert "plan" in dict(plan_pred)
    assert "raw" in dict(plan_pred)

    # Grouper prediction has 'groups' field
    groups_pred = grouper_prediction([{"feat_a": plan}])
    assert "groups" in dict(groups_pred)
    assert isinstance(dict(groups_pred)["groups"], list)


def test_unwrap_helpers_are_tolerant():
    """Unwrap helpers accept both Prediction and legacy shapes (design decision #2).

    This enables predicates to work with hand-built (plan, raw) calls and
    existing tests without modification.
    """
    plan = _make_plan("test")

    # unwrap_proposal is tolerant
    prop_pred = proposer_prediction(
        feature_name="test",
        feature_explanation="test",
        feature_operation="add",
    )
    unwrapped_prop = unwrap_proposal(prop_pred)
    assert isinstance(unwrapped_prop, ProposerOutput)

    legacy_prop = ProposerOutput(
        feature_name="test",
        feature_explanation="test",
        feature_operation="add",
    )
    assert unwrap_proposal(legacy_prop) is legacy_prop  # Passes through

    # unwrap_planner_result is tolerant
    plan_pred = planner_prediction(plan, raw="raw")
    unwrapped_plan, unwrapped_raw = unwrap_planner_result(plan_pred)
    assert unwrapped_raw == "raw"
    assert unwrapped_plan == plan

    legacy_tuple = (plan, "raw")
    assert unwrap_planner_result(legacy_tuple) is legacy_tuple  # Passes through

    # unwrap_groups is tolerant
    groups_pred = grouper_prediction([{"feat_a": plan}])
    unwrapped_groups = unwrap_groups(groups_pred)
    assert isinstance(unwrapped_groups, list)

    legacy_list = [{"feat_a": plan}]
    assert unwrap_groups(legacy_list) is legacy_list  # Passes through


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
    assert set(dict(helper).keys()) == set(dict(result).keys())

    # Planner: check plan and raw fields exist
    plan = _make_plan("test")
    helper_plan = planner_prediction(plan, raw="raw")
    assert "plan" in dict(helper_plan)
    assert "raw" in dict(helper_plan)

    # Grouper: check groups field exists
    helper_groups = grouper_prediction([])
    assert "groups" in dict(helper_groups)


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


def _global_history() -> list:
    from dspy.clients.base_lm import GLOBAL_HISTORY

    return GLOBAL_HISTORY


@pytest.mark.usefixtures("_clear_history")
def test_sub_threshold_attempts_trigger_offer_feedback_and_hints(capsys) -> None:
    """Headline: real grouper + real reward + ResettingRefine + DummyLM.

    Before #5 this run made 3 LM calls, printed ``Refine: Attempt failed`` twice
    (``dict(list[dict])`` raised), eroded ``fail_count`` from 3 to 1 and never sent
    a hint. Now it makes 5 calls (module, OfferFeedback, module, OfferFeedback,
    module), swallows nothing, keeps the budget, and the sentinel advice reaches
    attempts 2 and 3.
    """
    dspy.configure(lm=_sub_threshold_lm(_advice(_GROUPER_PREDICTOR, SENTINEL)))
    refine = ResettingRefine(module=FeatureGrouper(), N=3, reward_fn=grouper_reward, threshold=1.0)

    result = refine(feature_plans=_two_plans(), task="t")

    history = _global_history()
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

    dspy.configure(lm=_sub_threshold_lm(_advice(_GROUPER_PREDICTOR, SENTINEL)))
    refine = ResettingRefine(module=_RawListGrouper(), N=3, reward_fn=grouper_reward, threshold=1.0)

    refine(feature_plans=_two_plans(), task="t")

    history = _global_history()
    assert len(history) == 3
    assert capsys.readouterr().out.count("Attempt failed") == 2
    assert refine.fail_count == 1
    assert all(SENTINEL not in _rendered(e) for e in history)


@pytest.mark.usefixtures("_clear_history")
def test_advice_keyed_by_wrong_predictor_yields_na_hint() -> None:
    """refine.py resolves advice via ``signature2name``; a wrong key degrades to "N/A".

    Guards the routing: asserting only that ``hint_`` is present would pass here too.
    """
    dspy.configure(lm=_sub_threshold_lm(_advice("nonexistent.predict", SENTINEL)))
    refine = ResettingRefine(module=FeatureGrouper(), N=3, reward_fn=grouper_reward, threshold=1.0)

    refine(feature_plans=_two_plans(), task="t")

    history = _global_history()
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
    dspy.configure(lm=_sub_threshold_lm("{}"))
    refine = ResettingRefine(module=FeatureGrouper(), N=3, reward_fn=grouper_reward, threshold=1.0)

    refine(feature_plans=_two_plans(), task="t")

    history = _global_history()
    assert len(history) == 5
    assert all("hint_" not in _rendered(e) for e in history)
