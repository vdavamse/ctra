"""Shared data models for the CTRA agent pipeline.

Mirrors AutoCT's NamedTuples and StrEnums from ``agent.py:111-296``,
adapted for CTRA's configurable model roster and expanded data sources.

All agent modules import from this file to avoid circular dependencies.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import MISSING, dataclass, field, fields, replace

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of ``enum.StrEnum`` for Python 3.10."""


from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FeatureOp(StrEnum):
    """Feature modification operations (AutoCT line 168)."""

    ADD = "add"
    REMOVE = "remove"
    REFINE = "refine"


class FeatureType(StrEnum):
    """Supported feature value types (AutoCT line 171)."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    MULTICATEGORICAL = "multi-categorical"


class FeatureSource(StrEnum):
    """Data sources available for feature extraction.

    AutoCT's 3 sources (pubmed, related_clinical_trials, current_trial_summary)
    plus CTRA's additional sources (chembl, faers, aact, primekg, drugsfda).
    """

    PUBMED = "pubmed"
    RELATED_CLINICAL_TRIALS = "related_clinical_trials"
    CURRENT_TRIAL_SUMMARY = "current_trial_summary"
    CHEMBL = "chembl"
    FAERS = "faers"
    AACT = "aact"
    PRIMEKG = "primekg"
    DRUGSFDA = "drugsfda"


class Task(StrEnum):
    """Prediction tasks with phase-specific descriptions.

    Each member's value is the task description string passed to LLM agents.
    Mirrors AutoCT's Task enum (agent.py:111-156).  With per-phase isolation,
    each phase gets its own MCTS tree, feature set, and trained model.
    """

    TRIAL_OUTCOME_PHASE_1 = (
        "Predict whether a Phase 1 clinical trial will succeed or fail "
        "based on publicly available information from ClinicalTrials.gov, "
        "PubMed, ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA."
    )
    TRIAL_OUTCOME_PHASE_2 = (
        "Predict whether a Phase 2 clinical trial will succeed or fail "
        "based on publicly available information from ClinicalTrials.gov, "
        "PubMed, ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA."
    )
    TRIAL_OUTCOME_PHASE_3 = (
        "Predict whether a Phase 3 clinical trial will succeed or fail "
        "based on publicly available information from ClinicalTrials.gov, "
        "PubMed, ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA."
    )

    @property
    def phase(self) -> int | None:
        """Return the phase number (1, 2, 3) or None for the generic task."""
        _phase_map = {
            "TRIAL_OUTCOME_PHASE_1": 1,
            "TRIAL_OUTCOME_PHASE_2": 2,
            "TRIAL_OUTCOME_PHASE_3": 3,
        }
        return _phase_map.get(self.name)

    @property
    def description(self) -> str:
        """Return the task description string (the enum value)."""
        return self.value

    @property
    def output_subdir(self) -> str:
        """Return the phase-specific output subdirectory name."""
        return f"phase{self.phase}"

    @classmethod
    def from_phase(cls, phase: int) -> Task:
        """Look up the Task member for a given phase number."""
        _map = {
            1: cls.TRIAL_OUTCOME_PHASE_1,
            2: cls.TRIAL_OUTCOME_PHASE_2,
            3: cls.TRIAL_OUTCOME_PHASE_3,
        }
        if phase not in _map:
            raise ValueError(f"Unsupported phase: {phase}. Must be 1, 2, or 3.")
        return _map[phase]

    @classmethod
    def from_cli_arg(cls, arg: str) -> Task:
        """Convert CLI arg like 'phase1' to a Task enum member."""
        _map = {
            "phase1": cls.TRIAL_OUTCOME_PHASE_1,
            "phase2": cls.TRIAL_OUTCOME_PHASE_2,
            "phase3": cls.TRIAL_OUTCOME_PHASE_3,
        }
        if arg not in _map:
            raise ValueError(f"Unknown task arg: {arg!r}. Must be one of {list(_map)}")
        return _map[arg]


# ---------------------------------------------------------------------------
# Dataclasses for diagnostics
# ---------------------------------------------------------------------------

# Sentinels for the two distinct builder failure modes, written by
# ``WrappedFeatureBuilder.__call__`` and read by ``_build_builder_diagnostics``
# (reason classification) and ``BuilderDiagnostics.format_for_llm``
# (attribution). Defined here -- the leaf module both sides already import --
# so the writer and the two readers cannot drift apart.
#
# 1. A trial-group build that crashed inside ``WrappedFeatureBuilder.__call__``
#    (the exception path).
BUILDER_EXCEPTION_REASON = "builder_exception"
BUILDER_EXCEPTION_PREFIX = f"{BUILDER_EXCEPTION_REASON}:"
BUILDER_EXCEPTION_RESEARCH_SENTINEL = "[builder_exception]"
# Caps either builder reason string (the exception detail, or the LLM's own
# explanation appended after ``BUILDER_OMITTED_PREFIX``).
BUILDER_EXCEPTION_MSG_MAXLEN = 200

# 2. A feature the Construct step never produced, after ``ResettingRefine``
#    exhausted its retries. Distinct from ``BUILDER_EXCEPTION_*``: the builder
#    did not crash, it researched the trial and then silently skipped this
#    feature. Written by ``WrappedFeatureBuilder.__call__``'s omission fill.
#    No research sentinel: research genuinely ran on this path and
#    ``research_results`` keeps its real value.
BUILDER_OMITTED_REASON = "builder_omitted"
BUILDER_OMITTED_PREFIX = f"{BUILDER_OMITTED_REASON}:"


@dataclass
class FeatureDiagnostic:
    """Per-feature diagnostic summary for evaluator analysis."""

    feature_name: str
    none_rate: float  # fraction of trials where this feature was None
    dominant_failure_reason: str  # most common none_explanation category
    research_coverage_score: float  # 0.0-1.0, how well research covered needed data


@dataclass
class BuilderDiagnostics:
    """Aggregated builder diagnostics for the evaluator.

    Provides compact summaries to help the evaluator distinguish between
    Researcher failures (bad feature idea) and Builder failures (correct
    idea, bad execution).
    """

    feature_diagnostics: list[FeatureDiagnostic] = field(default_factory=list)

    def format_for_llm(self) -> str:
        """Format diagnostics as a concise LLM-readable string."""
        if not self.feature_diagnostics:
            return "No builder diagnostics available."

        lines = ["## Builder Diagnostics\n"]
        for fd in sorted(self.feature_diagnostics, key=lambda x: x.none_rate, reverse=True):
            if fd.none_rate < 0.05:
                continue  # skip features with very low None rates

            # Attribution heuristic:
            # - BUILDER, unconditionally, when the dominant failure is a builder
            #   crash: the feature idea was never actually tested, so it must not
            #   be charged to the RESEARCHER, and research_coverage is meaningless
            #   on that path (nothing was researched).
            # - BUILDER, unconditionally, when the Construct LLM omitted the
            #   feature after retries: research ran but the value was never
            #   attempted, so again not evidence against the plan.
            # - RESEARCHER if very high none rate and low research coverage.
            # - BUILDER if moderate none rate but good research coverage.
            # - else UNCLEAR.
            if fd.dominant_failure_reason in (BUILDER_EXCEPTION_REASON, BUILDER_OMITTED_REASON):
                attribution = "BUILDER"
            elif fd.none_rate > 0.8 and fd.research_coverage_score < 0.3:
                attribution = "RESEARCHER"
            elif fd.none_rate > 0.3 and fd.research_coverage_score > 0.5:
                attribution = "BUILDER"
            else:
                attribution = "UNCLEAR"
            line = (
                f"- **{fd.feature_name}**: None rate={fd.none_rate:.0%}, "
                f"failure='{fd.dominant_failure_reason}', "
                f"research_coverage={fd.research_coverage_score:.0%}, "
                f"attribution={attribution}"
            )
            # Add note for builder crashes / omissions
            if fd.dominant_failure_reason == BUILDER_EXCEPTION_REASON:
                line += (
                    " note=builder crashed; feature never evaluated, not evidence against the plan"
                )
            elif fd.dominant_failure_reason == BUILDER_OMITTED_REASON:
                line += (
                    " note=construct LLM omitted this feature; retries exhausted, "
                    "not evidence against the plan"
                )
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else "All features have low None rates."


# ---------------------------------------------------------------------------
# NamedTuples — matching AutoCT exactly (with CTRA adaptations noted)
# ---------------------------------------------------------------------------


class ProposerOutput(NamedTuple):
    """Output from FeatureProposer (AutoCT line 179)."""

    feature_operation: FeatureOp
    feature_name: str
    feature_explanation: str


class FeaturePlan(NamedTuple):
    """Multi-valued feature schema (AutoCT line 202).

    Unlike CTRA's single-valued ``FeatureSpec``, this supports sub-features
    via ``feature_type: dict[str, FeatureType]``.  A single feature can
    produce multiple typed sub-values, e.g.::

        feature_type = {"mechanism": FeatureType.CATEGORICAL,
                        "target_count": FeatureType.INTEGER}
    """

    feature_name: str
    feature_idea: str
    feature_type: dict[str, FeatureType]
    data_sources: list[FeatureSource]
    example_values: list[dict[str, str]]
    possible_values: dict[str, list[str]]
    feature_instructions: str


class ModelEvalResult(NamedTuple):
    """Evaluation result for a single model (AutoCT line 216).

    The ``interaction_values`` field contains SHAPLEY interaction values computed
    via shapiq (k-SII for XGBoost, FSII for TabPFN), structured as a dict with
    keys: 'main_effects', 'interactions', 'feature_names', 'index_type', 'max_order'.
    """

    roc_auc: float
    f1: float
    pr_auc: float
    interaction_values: dict[str, Any]  # REPLACES feature_importance (SHAPLEY interaction values)
    wrong_idxs: list[int]
    wrong_preds: list[int]
    wrong_df: pd.DataFrame
    pipeline: Any  # sklearn Pipeline or wrapper


class EvalOutput(NamedTuple):
    """Evaluation output with suggestions (AutoCT line 227)."""

    model_eval_result: ModelEvalResult
    suggestions: list[str]


# ---------------------------------------------------------------------------
# Cache instrumentation (issue #17)
# ---------------------------------------------------------------------------

#: LLM calls one dispatched trial-group build is taken to cost: a
#: single-attempt point estimate (k ~ 4 ReAct steps, no Refine retry), not
#: a bound.  One ``FeatureBuilder.forward`` attempt is k ReAct steps
#: (``dspy.ReAct(max_iters=5)``: 1 <= k <= 5, the loop stops at ``finish``)
#: + 1 ReAct extract call + 1 ChainOfThought Construct call = k + 2, i.e.
#: 3-7 calls (7 when ReAct exhausts its 5 steps).  Under
#: ``ResettingRefine(N=3)`` a group can take up to 3 attempts plus 2
#: ``OfferFeedback`` calls, so a dispatched group costs 3-23 calls;
#: ``llm_calls_avoided_estimate`` therefore leans low when retries are
#: common.  A real run calibrates it: ``scripts/run_agent.py`` records the
#: process-wide LLM call count in ``CacheStats.llm_calls_made``.
LLM_CALLS_PER_GROUP_BUILD = 6

#: Where a plan sent to the feature store was minted (see ``Agent.forward``).
PLAN_ORIGINS = ("initializer", "planner")


@dataclass
class FeatureStoreCounters:
    """Runtime feature-store counters for one process (issue #17).

    Plain ints only: the object is JSON-friendly and crosses the
    ``scripts/run_agent.py`` subprocess boundary inside ``AgentOutput`` without
    a new dill site.  Single-owner rule: every (trial, feature) lookup and
    every trial-group decision is counted exactly once -- by the upfront batch
    probe in ``compute_features`` when it ran, otherwise by
    ``WrappedFeatureBuilder`` -- so the wrapper's re-probe of a partially
    cached group is never counted twice.

    ``feature_hits`` is split by the origin of the plan that hit:
    ``hits_from_planner_plans`` are hits on plans minted fresh by the planner
    on an iteration-N path, which is exactly what cross-branch reuse looks
    like (two independently generated plan texts collided in the store);
    ``hits_from_initializer_plans`` are hits on iteration-0 plans, which a
    store persisted across runs serves.
    """

    feature_lookups: int = 0
    feature_hits: int = 0
    groups_dispatched: int = 0
    groups_skipped: int = 0
    hits_from_initializer_plans: int = 0
    hits_from_planner_plans: int = 0
    store_writes: int = 0

    @property
    def hit_rate(self) -> float:
        """``feature_hits / feature_lookups``; ``0.0`` before any lookup."""
        if self.feature_lookups <= 0:
            return 0.0
        return self.feature_hits / self.feature_lookups

    @property
    def llm_calls_avoided_estimate(self) -> int:
        """``groups_skipped * LLM_CALLS_PER_GROUP_BUILD``.

        Only a fully cached trial-group avoids a build; a hit inside a group
        that is still dispatched avoids nothing (the builder runs anyway), so
        per-feature hits are deliberately not multiplied.
        """
        return self.groups_skipped * LLM_CALLS_PER_GROUP_BUILD

    def record_hits(self, n: int, plan_origin: str) -> None:
        """Add ``n`` hits and attribute them to ``plan_origin``."""
        if plan_origin not in PLAN_ORIGINS:
            raise ValueError(f"unknown plan_origin {plan_origin!r}; expected one of {PLAN_ORIGINS}")
        self.feature_hits += n
        if plan_origin == "initializer":
            self.hits_from_initializer_plans += n
        else:
            self.hits_from_planner_plans += n

    def merge(self, other: FeatureStoreCounters) -> None:
        """Fold another counter set into this one (in place).

        Only the ``FeatureStoreCounters`` fields are summed, so a
        ``CacheStats`` can be folded into a plain counter set; a non-int value
        (a test stand-in) is skipped rather than raising.
        """
        for f in fields(FeatureStoreCounters):
            value = getattr(other, f.name, 0)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, f.name, getattr(self, f.name) + value)

    def as_dict(self) -> dict[str, int | float]:
        """Counters plus the derived ``hit_rate`` and ``llm_calls_avoided_estimate``."""
        out: dict[str, int | float] = {f.name: getattr(self, f.name) for f in fields(self)}
        out["hit_rate"] = self.hit_rate
        out["llm_calls_avoided_estimate"] = self.llm_calls_avoided_estimate
        return out


@dataclass
class CacheStats(FeatureStoreCounters):
    """One agent iteration's cache counters, carried in ``AgentOutput.cache_stats``.

    ``llm_calls_made`` is the calibration field: ``scripts/run_agent.py`` sets
    it to the number of LLM calls the whole iteration made (every agent, not
    only the builder), counted at the source by a ``dspy`` callback on every
    ``LM.__call__`` -- unbounded, unlike the 10,000-entry global history --
    so a real run can check ``LLM_CALLS_PER_GROUP_BUILD`` against
    ``groups_dispatched``.  Calls served by dspy's own response cache are
    included.  It stays ``0`` when the iteration ran in-process.
    """

    llm_calls_made: int = 0


@dataclass(eq=False)
class AgentOutput:
    """Full iteration state container (adapted from AutoCT line 266).

    AutoCT uses positional fields ``xgb_eval_output``, ``lr_eval_output``,
    ``rf_eval_output``.  CTRA uses ``dict[str, EvalOutput]`` keyed by model
    name because the model roster is configurable via ``ModelRegistry``
    (XGBoost + TabPFN, not fixed XGB + LR + RF).

    Migrated from NamedTuple to dataclass to support mutable fields with defaults
    (builder_meta) and to maintain compatibility with existing code that uses
    _replace() and _asdict() methods.

    ``eq=False`` opts out of the auto-generated ``__eq__``: it would
    element-wise compare every field including the ``pd.DataFrame`` fields
    (``df``, ``val_df``), and ``DataFrame.__eq__`` returns a DataFrame that
    raises ``ValueError: The truth value of a DataFrame is ambiguous`` when
    coerced to bool. Identity comparison is the intended semantics.
    """

    eval_outputs: dict[str, EvalOutput]
    test_eval_outputs: dict[str, ModelEvalResult]
    operation: ProposerOutput | None
    feature_plans: dict[str, FeaturePlan]
    df: pd.DataFrame
    val_df: pd.DataFrame
    suggestion_index: int
    raw_features: dict[str, dict[str, dict[str, Any]]]
    raw_val_features: dict[str, dict[str, dict[str, Any]]]
    raw_test_features: dict[str, dict[str, dict[str, Any]]]
    none_explanations: dict[str, dict[str, str]]
    builder_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    builder_diagnostics: BuilderDiagnostics = field(default_factory=BuilderDiagnostics)
    # Counters for the compute_features calls *this* iteration made (issue
    # #17).  Zeroed on every skipped iteration, so an output copied from its
    # parent never replays the parent's counts; read with ``getattr`` by
    # consumers that may see an output pickled before the field existed.
    cache_stats: CacheStats = field(default_factory=CacheStats)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Backfill ``default_factory`` fields missing from an older pickle.

        Pickles and deepcopies restore instances through this hook.  An
        ``AgentOutput`` dumped before ``cache_stats`` (or any later
        defaulted field) existed would otherwise come back without the
        attribute and break ``_replace`` -- ``dataclasses.replace`` reads
        every field -- on the resume and skip paths.
        """
        self.__dict__.update(state)
        for f in fields(self):
            if f.name not in self.__dict__ and f.default_factory is not MISSING:
                self.__dict__[f.name] = f.default_factory()

    def _replace(self, **kwargs: Any) -> AgentOutput:
        """Return a copy with specified fields replaced (NamedTuple compat).

        NOTE: like ``dataclasses.replace``, non-overridden mutable fields are
        **aliased** — the returned instance shares references to
        ``builder_meta``, ``builder_diagnostics``, ``cache_stats``,
        ``none_explanations``, ``raw_features``, ``raw_val_features``,
        ``raw_test_features``, and ``feature_plans`` with ``self``. In-place
        mutation (``|=``, ``.pop``, ``.update``, ``+=`` on a counter,
        assignment into nested dicts) on any of those fields will silently
        propagate across sibling copies — e.g. across sibling MCTS nodes
        created by ``parent_output._replace(suggestion_index=...)``, or from
        the probe/child copies ``mcts.py`` makes of a parent's output into
        the parent's own ``cache_stats``.

        Callers must ``deepcopy`` before mutating. ``forward()`` already
        does this at the top of the iter-N branch
        (``deepcopy(previous_output.raw_features)`` etc.) to preserve the
        invariant.
        """
        return replace(self, **kwargs)

    def _asdict(self) -> dict[str, Any]:
        """Return fields as a shallow dict (NamedTuple compat).

        Matches NamedTuple's ``_asdict`` semantics: keys map to raw field
        values, **not** recursively converted. ``dataclasses.asdict`` would
        deep-copy the ``df``/``val_df`` fields via ``copy.deepcopy`` —
        expensive for large DataFrames and unnecessary: the
        ``_NamedTupleEncoder`` in ``feature_utils`` already recurses as
        needed when serializing. (``dataclasses.asdict`` does not raise on
        DataFrame; it just copies it expensively.)
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def get_best_eval_output(self) -> tuple[EvalOutput, ModelEvalResult]:
        """Select best model by validation ROC-AUC (AutoCT line 282)."""
        if not self.eval_outputs:
            raise ValueError("No eval outputs available — all models may have failed to train")
        best_name = max(
            self.eval_outputs,
            key=lambda k: self.eval_outputs[k].model_eval_result.roc_auc,
        )
        return self.eval_outputs[best_name], self.test_eval_outputs[best_name]

    @property
    def suggestions_exhausted(self) -> bool:
        """True once ``suggestion_index`` has run past the best model's suggestions.

        ``suggestion_index`` is a monotonic "dead suggestions burned" counter, not
        an array cursor: ``Agent.forward`` advances it on every skipped iteration
        and deliberately never bounds it. ``get_next_suggestion`` therefore clamps,
        which means that once the counter passes the end it replays the *same* final
        suggestion forever — the proposer re-proposes against a suggestion already
        rejected, ``is_valid_proposer`` rejects it again, the counter advances, and
        the cycle repeats at ``N=3`` LM calls per rollout with zero forward progress.
        The clamp warning fires each time, but nothing downstream can distinguish a
        fresh suggestion from a replay without this flag.

        Callers should check it before spending those calls. ``Agent.forward`` does.

        Empty suggestions are **not** exhaustion: ``FeatureProposer.forward``
        deliberately degrades to an empty suggestion so the proposal is still judged
        on its merits, and short-circuiting here would undo that.

        Returns ``False`` when the underlying state cannot be read (no eval outputs,
        or ``eval_outputs``/``test_eval_outputs`` disagreeing), so a diagnostic
        failure never silently halts the search.
        """
        try:
            eval_output, _ = self.get_best_eval_output()
        except (ValueError, KeyError):
            return False
        suggestions = eval_output.suggestions
        return bool(suggestions) and self.suggestion_index >= len(suggestions)

    def get_next_suggestion(self) -> str:
        """Return the suggestion at ``suggestion_index`` from the best model (AutoCT line 294).

        Bounds-safe: if ``suggestion_index`` is out of range, clamp to [0, len-1]
        and log a warning. Raise ``ValueError`` if there are no suggestions at all.
        """
        eval_output, _ = self.get_best_eval_output()
        suggestions = eval_output.suggestions

        if not suggestions:
            raise ValueError("No suggestions available to advance")

        # Clamp index to valid range [0, len-1]
        clamped_index = max(0, min(self.suggestion_index, len(suggestions) - 1))

        if clamped_index != self.suggestion_index:
            logger.warning(
                "suggestion_index=%d out of range [0, %d); clamping to %d",
                self.suggestion_index,
                len(suggestions),
                clamped_index,
            )

        return suggestions[clamped_index]
