"""Acceptance tests for issues #5 and #6: Refine's OfferFeedback path runs.

All four Refine-wrapped modules return dspy.Prediction so that Refine's
feedback step (refine.py:153 ``dict(outputs)``) succeeds and OfferFeedback
actually runs. This file is the regression fence for "retries are guided,
not blind re-rolls". Issue #6 added the builder: its partial build must reach
OfferFeedback, and must do so on the budget LM (see
``test_partial_build_reaches_offer_feedback_on_the_budget_lm``).

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
    from ctra.agents.feature_builder import FeatureBuilder, WrappedFeatureBuilder
    from ctra.agents.feature_grouper import FeatureGrouper
    from ctra.agents.feature_planner import FeaturePlanner
    from ctra.agents.feature_proposer import FeatureProposer
    from ctra.agents.reward_fns import (
        ResettingRefine,
        grouper_reward,
    )
    from tests.test_agents.conftest import (
        builder_prediction,
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


def _real_builder_result(
    all_feature_values: dict[str, dict[str, str]],
    plan_group: dict[str, FeaturePlan],
) -> dspy.Prediction:
    """Run the real ``FeatureBuilder.forward`` with its two LLM phases stubbed.

    ``get_trial_info_dict`` hits the RAG tools and ``dspy.ReAct`` is constructed
    inside ``forward``, so both are patched; the ``__init__``-time
    ``constructor`` is patched on the instance. Everything else -- the per-type
    validation and the ``Prediction`` return -- is production code.
    """
    builder = FeatureBuilder(task_description="Test task")
    research = MagicMock(return_value=dspy.Prediction(research_results="r", reasoning="x"))
    with (
        patch(
            "ctra.agents.feature_builder.get_trial_info_dict",
            return_value={"startDate": "2020-01-01"},
        ),
        patch.object(dspy, "ReAct", return_value=research),
        patch.object(
            builder,
            "constructor",
            return_value=dspy.Prediction(
                all_feature_values=all_feature_values,
                none_feature_explanations={},
                reasoning="",
            ),
        ),
    ):
        return builder(nctid="NCT001", feature_plan_group=plan_group)


def test_all_four_modules_return_predictions():
    """All four Refine-wrapped modules return dspy.Prediction (#5 + #6).

    This verifies that proposer, planner, grouper and builder all have the
    right return type for Refine's feedback step to work.
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

    # Test builder (issue #6)
    result = _real_builder_result({"feat_a": {"value": "1.0"}}, {"feat_a": _make_plan("feat_a")})
    assert isinstance(result, dspy.Prediction)
    assert dict(result).keys() == {"feature_values", "metadata"}
    # Reserved-name trap: ``Example.values`` is a method, so a field named
    # ``values`` would be silently broken. The field is ``feature_values``.
    assert "values" not in result
    assert result["feature_values"] == {"feat_a": {"value": 1.0}}
    assert set(result["metadata"]) == {
        "research_results",
        "research_result_reasoning",
        "builder_reasoning",
        "none_feature_explanations",
    }


def test_forward_returns_partial_coverage_without_raising():
    """Issue #6 headline, at the module boundary: a Construct output that omits
    a planned feature yields a *partial* ``feature_values`` -- no ValueError,
    and no fill. The fill belongs to ``WrappedFeatureBuilder``, above the reward
    boundary; filling here would make ``is_valid_builder`` always True and kill
    every retry.
    """
    result = _real_builder_result({"feat_a": {"value": "1.0"}}, _two_plans())

    assert isinstance(result, dspy.Prediction)
    assert set(result["feature_values"]) == {"feat_a"}
    assert result["feature_values"]["feat_a"] == {"value": 1.0}
    assert "feat_b" not in result["metadata"]["none_feature_explanations"]


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

    # Builder: builder keys must match the real module's Prediction keys
    builder_result = _real_builder_result(
        {"feat_a": {"value": "1.0"}}, {"feat_a": _make_plan("feat_a")}
    )
    assert set(builder_prediction({}).keys()) == set(builder_result.keys())


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


# ---------------------------------------------------------------------------
# Issue #6: the builder's partial build reaches OfferFeedback -- on the budget LM
# ---------------------------------------------------------------------------


class _PartialBuilder(dspy.Module):  # type: ignore[misc]
    """Shaped like FeatureBuilder: one ``__init__``-time predictor, and a
    ``Prediction`` return that covers only ``feat_a`` (reward 0.0 every attempt).

    A real ``dspy.Module`` rather than a MagicMock, because the fence is about
    which LM ``refine.py:107-108``'s ``mod.set_lm()`` pins onto that predictor.
    """

    def __init__(self, task_description: str) -> None:
        super().__init__()
        self.constructor = dspy.Predict("nctid -> value")

    def forward(self, nctid: str, feature_plan_group: dict[str, FeaturePlan]) -> dspy.Prediction:
        self.constructor(nctid=nctid)
        return builder_prediction({"feat_a": {"value": "1.0"}})


def _builder_dummy_lm():
    from dspy.utils.dummies import DummyLM

    canned = {
        "value": "1.0",
        "discussion": "d",
        "advice": _advice("constructor", SENTINEL),
    }
    return DummyLM([canned] * 20)


@pytest.mark.usefixtures("_clear_history")
def test_partial_build_reaches_offer_feedback_on_the_budget_lm(capsys) -> None:
    """Decision #2 fence + the retry half of R4.

    ``WrappedFeatureBuilder.__call__`` enters ``dspy.context(lm=self._budget_lm)``
    around the Refine call. Without it, ``refine.py:99`` reads the primary LM,
    ``:108`` pins it onto the module (defeating ``forward``'s own context) and
    ``:167`` runs OfferFeedback on it too: ``primary.history`` would hold the two
    OfferFeedback calls and ``budget.history`` would be empty.
    """
    primary = _builder_dummy_lm()
    budget = _builder_dummy_lm()
    captured: list[ResettingRefine] = []

    def _capture(**kw):
        refiner = ResettingRefine(**kw)
        captured.append(refiner)
        return refiner

    with (
        patch("ctra.agents.feature_builder.FeatureBuilder", _PartialBuilder),
        patch("ctra.agents.feature_builder.ResettingRefine", side_effect=_capture),
    ):
        wrapper = WrappedFeatureBuilder(
            task_description="t", feature_store_dir=None, feature_store_enabled=False
        )
        wrapper._budget_lm = budget
        with dspy.context(lm=primary):
            nctid, values, meta = wrapper(("NCT001", _two_plans()))

    # module, OfferFeedback, module, OfferFeedback, module
    assert len(GLOBAL_HISTORY) == 5, [_rendered(e)[:80] for e in GLOBAL_HISTORY]
    # Nothing reached the primary (Opus) LM ...
    assert primary.history == []
    # ... and both OfferFeedback calls landed on the budget LM. (Module attempts
    # run on ``lm.copy(rollout_id=...)`` whose history is fresh, hence 2 not 5.)
    assert len(budget.history) == 2
    assert "Attempt failed" not in capsys.readouterr().out
    assert len(captured) == 1
    assert captured[0].fail_count == 3
    # The retries were guided: the sentinel advice reached attempts 2 and 3.
    for attempt in (GLOBAL_HISTORY[2], GLOBAL_HISTORY[4]):
        assert SENTINEL in _rendered(attempt)
    # And the partial build survived: feat_a kept, feat_b filled above the reward.
    assert nctid == "NCT001"
    assert values["feat_a"] == {"value": "1.0"}
    assert values["feat_b"] == {"value": None}
    assert meta["none_feature_explanations"]["feat_b"].startswith("builder_omitted:")
