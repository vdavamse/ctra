"""Tests for ctra.agents.data_models — enums, NamedTuples, AgentOutput methods.

Requires dspy (which needs sqlite3) to be importable because the
``ctra.agents`` package __init__.py eagerly imports dspy modules.
"""

from __future__ import annotations

import pandas as pd
import pytest

try:
    from ctra.agents.data_models import (
        AgentOutput,
        EvalOutput,
        FeatureOp,
        FeaturePlan,
        FeatureSource,
        FeatureType,
        ModelEvalResult,
        ProposerOutput,
        Task,
    )

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


class TestFeatureOp:
    def test_values(self) -> None:
        assert FeatureOp.ADD == "add"
        assert FeatureOp.REMOVE == "remove"
        assert FeatureOp.REFINE == "refine"

    def test_from_string(self) -> None:
        assert FeatureOp("add") is FeatureOp.ADD


class TestFeatureType:
    def test_all_types(self) -> None:
        expected = {"boolean", "integer", "float", "categorical", "multi-categorical"}
        assert {t.value for t in FeatureType} == expected


class TestFeatureSource:
    def test_autoct_sources(self) -> None:
        assert FeatureSource.PUBMED == "pubmed"
        assert FeatureSource.RELATED_CLINICAL_TRIALS == "related_clinical_trials"
        assert FeatureSource.CURRENT_TRIAL_SUMMARY == "current_trial_summary"

    def test_ctra_sources(self) -> None:
        for s in ("chembl", "faers", "aact", "primekg", "drugsfda"):
            assert s in [fs.value for fs in FeatureSource]


# ---------------------------------------------------------------------------
# NamedTuples
# ---------------------------------------------------------------------------


class TestFeaturePlan:
    def test_construction(self) -> None:
        plan = FeaturePlan(
            feature_name="drug_mechanism",
            feature_idea="Type of drug mechanism",
            feature_type={"value": FeatureType.CATEGORICAL},
            data_sources=[FeatureSource.CHEMBL],
            example_values=[{"value": "kinase_inhibitor"}],
            possible_values={"value": ["kinase_inhibitor", "antibody", "other"]},
            feature_instructions="Look up drug mechanism in ChEMBL.",
        )
        assert plan.feature_name == "drug_mechanism"
        assert plan.feature_type["value"] == FeatureType.CATEGORICAL

    def test_multi_valued(self) -> None:
        plan = FeaturePlan(
            feature_name="drug_profile",
            feature_idea="Drug mechanism and target count",
            feature_type={
                "mechanism": FeatureType.CATEGORICAL,
                "target_count": FeatureType.INTEGER,
            },
            data_sources=[FeatureSource.CHEMBL, FeatureSource.PRIMEKG],
            example_values=[{"mechanism": "kinase_inhibitor", "target_count": "3"}],
            possible_values={"mechanism": ["kinase_inhibitor", "antibody"]},
            feature_instructions="Research drug profile.",
        )
        assert len(plan.feature_type) == 2
        assert "mechanism" in plan.possible_values


class TestProposerOutput:
    def test_construction(self) -> None:
        out = ProposerOutput(
            feature_operation=FeatureOp.ADD,
            feature_name="new_feat",
            feature_explanation="Add a new feature for safety signals.",
        )
        assert out.feature_operation == FeatureOp.ADD


# ---------------------------------------------------------------------------
# AgentOutput
# ---------------------------------------------------------------------------


def _make_eval_result(roc_auc: float = 0.8) -> ModelEvalResult:
    return ModelEvalResult(
        roc_auc=roc_auc,
        f1=0.75,
        pr_auc=0.7,
        interaction_values={},
        wrong_idxs=[1, 3],
        wrong_preds=[0, 1],
        wrong_df=pd.DataFrame({"id": ["NCT001", "NCT003"]}),
        pipeline=None,
    )


def _make_output(
    roc_aucs: dict[str, float] | None = None,
    suggestions: list[str] | None = None,
) -> AgentOutput:
    roc_aucs = roc_aucs or {"xgboost": 0.85, "tabpfn": 0.80}
    suggestions = suggestions or ["add safety_signal feature", "remove unused_feat"]

    eval_outputs = {}
    test_eval_outputs = {}
    for name, auc in roc_aucs.items():
        er = _make_eval_result(auc)
        eval_outputs[name] = EvalOutput(model_eval_result=er, suggestions=suggestions)
        test_eval_outputs[name] = er

    return AgentOutput(
        eval_outputs=eval_outputs,
        test_eval_outputs=test_eval_outputs,
        operation=None,
        feature_plans={},
        df=pd.DataFrame(),
        val_df=pd.DataFrame(),
        suggestion_index=0,
        raw_features={},
        raw_val_features={},
        raw_test_features={},
        none_explanations={},
        builder_meta={},
    )


class TestAgentOutput:
    def test_get_best_eval_output_selects_highest_roc_auc(self) -> None:
        output = _make_output({"xgboost": 0.85, "tabpfn": 0.90})
        best_eval, _best_test = output.get_best_eval_output()
        assert best_eval.model_eval_result.roc_auc == 0.90

    def test_get_next_suggestion(self) -> None:
        output = _make_output(suggestions=["suggest_a", "suggest_b"])
        assert output.get_next_suggestion() == "suggest_a"

    def test_get_next_suggestion_respects_index(self) -> None:
        output = _make_output(suggestions=["suggest_a", "suggest_b"])
        output2 = output._replace(suggestion_index=1)
        assert output2.get_next_suggestion() == "suggest_b"

    # -- Bounds safety ---------------------------------------------------
    #
    # ``suggestion_index`` is a monotonic "dead suggestions burned" counter,
    # not an array cursor: the orchestrator advances it past the end when a
    # proposal is rejected, and MCTS replays that value on the next rollout.
    # The read therefore has to tolerate an out-of-range index -- previously
    # it raised IndexError and killed the whole agent subprocess.

    def test_get_next_suggestion_clamps_index_past_end(self) -> None:
        output = _make_output(suggestions=["suggest_a", "suggest_b"])
        exhausted = output._replace(suggestion_index=7)
        assert exhausted.get_next_suggestion() == "suggest_b"

    def test_get_next_suggestion_clamps_single_suggestion(self) -> None:
        """The exact shape the orchestrator skip path produces."""
        output = _make_output(suggestions=["only_one"])
        assert output._replace(suggestion_index=3).get_next_suggestion() == "only_one"

    def test_get_next_suggestion_clamps_negative_index(self) -> None:
        output = _make_output(suggestions=["suggest_a", "suggest_b"])
        assert output._replace(suggestion_index=-5).get_next_suggestion() == "suggest_a"

    def test_get_next_suggestion_warns_when_out_of_range(self, caplog) -> None:
        output = _make_output(suggestions=["suggest_a"])._replace(suggestion_index=4)
        with caplog.at_level("WARNING"):
            output.get_next_suggestion()
        assert "out of range" in caplog.text

    def test_get_next_suggestion_no_suggestions_raises_value_error(self) -> None:
        """Empty suggestions is unrecoverable -- but must not surface as IndexError."""
        # Built explicitly: ``_make_output`` treats an empty list as "use the default".
        output = _make_output()
        empty = output._replace(
            eval_outputs={
                name: EvalOutput(model_eval_result=ev.model_eval_result, suggestions=[])
                for name, ev in output.eval_outputs.items()
            }
        )
        with pytest.raises(ValueError, match="No suggestions"):
            empty.get_next_suggestion()

    def test_single_model(self) -> None:
        output = _make_output({"xgboost": 0.75})
        best_eval, _ = output.get_best_eval_output()
        assert best_eval.model_eval_result.roc_auc == 0.75

    def test_replace_preserves_builder_meta(self) -> None:
        """_replace must preserve builder_meta across chained calls (MCTS sets
        suggestion_index on each child node; builder_meta must survive).
        """
        output = _make_output()
        meta = {"NCT001": {"feat_a": {"research_results": "found papers"}}}
        out_with_meta = output._replace(builder_meta=meta)
        out_mcts = out_with_meta._replace(suggestion_index=7)
        assert out_mcts.builder_meta == meta
        assert out_mcts.suggestion_index == 7


# ---------------------------------------------------------------------------
# Task enum
# ---------------------------------------------------------------------------


class TestTask:
    """Tests for the Task StrEnum."""

    def test_phase_property(self) -> None:
        assert Task.TRIAL_OUTCOME_PHASE_1.phase == 1
        assert Task.TRIAL_OUTCOME_PHASE_2.phase == 2
        assert Task.TRIAL_OUTCOME_PHASE_3.phase == 3

    def test_description_is_value(self) -> None:
        for t in Task:
            assert t.description == t.value

    def test_value_equals_description(self) -> None:
        for t in Task:
            assert t.value == t.description

    def test_from_phase_valid(self) -> None:
        assert Task.from_phase(1) is Task.TRIAL_OUTCOME_PHASE_1
        assert Task.from_phase(2) is Task.TRIAL_OUTCOME_PHASE_2
        assert Task.from_phase(3) is Task.TRIAL_OUTCOME_PHASE_3

    def test_from_phase_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unsupported phase"):
            Task.from_phase(4)

    def test_from_cli_arg_valid(self) -> None:
        assert Task.from_cli_arg("phase1") is Task.TRIAL_OUTCOME_PHASE_1
        assert Task.from_cli_arg("phase2") is Task.TRIAL_OUTCOME_PHASE_2
        assert Task.from_cli_arg("phase3") is Task.TRIAL_OUTCOME_PHASE_3

    def test_from_cli_arg_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown task arg"):
            Task.from_cli_arg("phase4")

    def test_output_subdir(self) -> None:
        assert Task.TRIAL_OUTCOME_PHASE_1.output_subdir == "phase1"
        assert Task.TRIAL_OUTCOME_PHASE_2.output_subdir == "phase2"
        assert Task.TRIAL_OUTCOME_PHASE_3.output_subdir == "phase3"
