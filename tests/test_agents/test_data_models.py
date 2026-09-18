"""Tests for ctra.agents.data_models — enums, NamedTuples, AgentOutput methods."""

from __future__ import annotations

import pandas as pd
import pytest

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

    # -- Exhaustion ------------------------------------------------------
    #
    # The clamp above keeps the read safe but makes exhaustion invisible: the
    # last suggestion is replayed forever, so the proposer burns N=3 LM calls
    # per rollout re-proposing against a suggestion already rejected. This flag
    # is what lets ``Agent.forward`` tell a replay from a fresh suggestion.

    def test_not_exhausted_within_range(self) -> None:
        output = _make_output(suggestions=["suggest_a", "suggest_b"])
        assert output.suggestions_exhausted is False
        assert output._replace(suggestion_index=1).suggestions_exhausted is False

    def test_exhausted_at_and_past_end(self) -> None:
        """Index == len is already past the last valid cursor (0-based)."""
        output = _make_output(suggestions=["suggest_a", "suggest_b"])
        assert output._replace(suggestion_index=2).suggestions_exhausted is True
        assert output._replace(suggestion_index=9).suggestions_exhausted is True

    def test_exhaustion_agrees_with_the_clamp(self) -> None:
        """The flag must be True exactly when the read starts replaying.

        Without this pairing the two could drift, which is the failure mode the
        whole reward/check-drift refactor exists to prevent.
        """
        output = _make_output(suggestions=["suggest_a", "suggest_b"])
        for idx in range(6):
            at = output._replace(suggestion_index=idx)
            replaying = at.get_next_suggestion() == "suggest_b" and idx != 1
            assert at.suggestions_exhausted is replaying

    def test_empty_suggestions_is_not_exhaustion(self) -> None:
        """Empty != exhausted.

        ``FeatureProposer.forward`` deliberately degrades an empty suggestion
        list to ``""`` so the proposal is still judged on its merits. Reporting
        exhaustion here would short-circuit that and skip the iteration instead.
        """
        output = _make_output()
        empty = output._replace(
            eval_outputs={
                name: EvalOutput(model_eval_result=ev.model_eval_result, suggestions=[])
                for name, ev in output.eval_outputs.items()
            },
            suggestion_index=5,
        )
        assert empty.suggestions_exhausted is False

    def test_unreadable_state_is_not_exhaustion(self) -> None:
        """A diagnostic failure must never silently halt the search."""
        output = _make_output(suggestions=["suggest_a"])
        assert output._replace(eval_outputs={}).suggestions_exhausted is False
        # eval_outputs / test_eval_outputs disagreeing -> KeyError, not a crash.
        assert output._replace(test_eval_outputs={}).suggestions_exhausted is False

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


class TestBuilderDiagnosticsAttribution:
    """Tests for builder exception attribution in format_for_llm.

    Note: the three legacy attribution cases (RESEARCHER, BUILDER, UNCLEAR)
    are already covered at test_shapiq_helpers.py:210-244. These tests
    focus on the new builder_exception override only.
    """

    def test_builder_exception_beats_researcher_heuristic(self) -> None:
        """Builder exception should override RESEARCHER heuristic."""
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        fd = FeatureDiagnostic(
            feature_name="feat",
            none_rate=1.0,
            dominant_failure_reason="builder_exception",
            research_coverage_score=0.0,
        )
        diag = BuilderDiagnostics(feature_diagnostics=[fd])
        formatted = diag.format_for_llm()

        # Without the override, this would be RESEARCHER (high none_rate + low coverage)
        assert "attribution=BUILDER" in formatted
        # Should include note about crash
        assert "note=builder crashed" in formatted

    def test_builder_exception_beats_unclear_heuristic(self) -> None:
        """Builder exception should override UNCLEAR heuristic."""
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        fd = FeatureDiagnostic(
            feature_name="feat",
            none_rate=0.5,
            dominant_failure_reason="builder_exception",
            research_coverage_score=0.1,
        )
        diag = BuilderDiagnostics(feature_diagnostics=[fd])
        formatted = diag.format_for_llm()

        # Without the override, this would be UNCLEAR (moderate none_rate + low coverage)
        assert "attribution=BUILDER" in formatted

    def test_builder_omitted_beats_researcher_heuristic(self) -> None:
        """An omission after retries is attributed BUILDER, never RESEARCHER.

        none_rate=1.0 with coverage=0.0 is exactly the RESEARCHER arm; without
        the override the omission would be charged to the feature idea even
        though research ran and the Construct LLM simply skipped it.
        """
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        fd = FeatureDiagnostic(
            feature_name="feat",
            none_rate=1.0,
            dominant_failure_reason="builder_omitted",
            research_coverage_score=0.0,
        )
        formatted = BuilderDiagnostics(feature_diagnostics=[fd]).format_for_llm()

        assert "attribution=BUILDER" in formatted
        assert "attribution=RESEARCHER" not in formatted

    def test_builder_omitted_note_differs_from_the_crash_note(self) -> None:
        """The omission line carries its own note, not the crash note."""
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        fd = FeatureDiagnostic(
            feature_name="feat",
            none_rate=1.0,
            dominant_failure_reason="builder_omitted",
            research_coverage_score=1.0,
        )
        formatted = BuilderDiagnostics(feature_diagnostics=[fd]).format_for_llm()

        assert "note=construct LLM omitted" in formatted
        assert "note=builder crashed" not in formatted

    def test_attribution_vocabulary_unchanged(self) -> None:
        """All emitted attributions must be in {RESEARCHER, BUILDER, UNCLEAR}."""
        import re

        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        # Test cases: (none_rate, research_coverage, reason, expected_attribution)
        test_cases = [
            # builder_exception overrides all heuristics -> BUILDER
            (0.9, 0.1, "builder_exception", "BUILDER"),
            # builder_omitted overrides all heuristics -> BUILDER
            (0.9, 0.1, "builder_omitted", "BUILDER"),
            # High none_rate + low coverage -> RESEARCHER
            (0.9, 0.1, "extraction_error", "RESEARCHER"),
            # Moderate none_rate + good coverage -> BUILDER
            (0.5, 0.6, "insufficient_data", "BUILDER"),
            # Moderate none_rate + low coverage -> UNCLEAR
            (0.3, 0.2, "ambiguity", "UNCLEAR"),
            # Low none_rate + medium coverage -> UNCLEAR
            (0.4, 0.5, "other", "UNCLEAR"),
            # Low none_rate + low coverage -> UNCLEAR
            (0.1, 0.1, "insufficient_data", "UNCLEAR"),
        ]

        valid_attributions = {"RESEARCHER", "BUILDER", "UNCLEAR"}

        for i, (none_rate, coverage, reason, expected) in enumerate(test_cases):
            fd = FeatureDiagnostic(
                feature_name=f"feat_{i}",
                none_rate=none_rate,
                dominant_failure_reason=reason,
                research_coverage_score=coverage,
            )
            diag = BuilderDiagnostics(feature_diagnostics=[fd])
            formatted = diag.format_for_llm()

            feat_key = f"feat_{i}"
            if f"**{feat_key}**" not in formatted:
                # Feature was skipped due to none_rate < 0.05
                continue

            # Extract all attribution values from formatted output
            emitted = re.findall(r"attribution=(\w+)", formatted)
            assert emitted == [expected], f"Expected {[expected]}, got {emitted} in: {formatted}"
            # Verify only valid attributions are used
            assert set(emitted) <= valid_attributions

    def test_low_rate_builder_exception_is_still_skipped(self) -> None:
        """Builder exception with none_rate < 0.05 should still be skipped."""
        from ctra.agents.data_models import BuilderDiagnostics, FeatureDiagnostic

        fd = FeatureDiagnostic(
            feature_name="myfeature",
            none_rate=0.02,
            dominant_failure_reason="builder_exception",
            research_coverage_score=1.0,
        )
        diag = BuilderDiagnostics(feature_diagnostics=[fd])
        formatted = diag.format_for_llm()

        # Should report the default message
        assert "All features have low None rates" in formatted
        # Should not contain the specific feature (use ** markers to identify features in the output)
        assert "**myfeature**" not in formatted
