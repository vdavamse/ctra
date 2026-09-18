"""DSPy ReAct agents for feature extraction via RAG.

- **FeatureBuilder**: Two-phase grouped builder (ReAct research → CoT
  construct) with disk caching and ``soft_assert`` validation. Returns a
  ``dspy.Prediction(feature_values=..., metadata=...)`` that may cover only
  **part** of the requested group: completeness is judged by
  ``reward_fns.is_valid_builder`` (which drives the ``ResettingRefine``
  retries), and gap-filling is ``WrappedFeatureBuilder``'s job, not the
  module's.
- **WrappedFeatureBuilder**: Disk-cached wrapper keyed by plan hash. After
  Refine returns, fills any feature the builder never produced with all-None
  sub-values and a ``builder_omitted`` explanation -- after the store writes
  (never negatively cached) and outside the reward boundary.
- **compute_features**: Parallel grouped feature computation.

The builder uses the **budget LM** (Claude Sonnet 4.6) via
:func:`ctra.agents.lm_config.configure_budget_lm` to reduce cost for
high-volume extraction calls. ``FeatureBuilder.forward`` enters
``dspy.context(lm=<budget>)`` itself, but under ``ResettingRefine`` that is
not enough: ``refine.py:107-108`` deepcopies the module and ``set_lm()``s
``dspy.settings.lm`` onto every ``__init__``-time predictor, which overrides
the inner context. ``WrappedFeatureBuilder`` therefore enters the budget-LM
context *around* the Refine call, so both the Construct phase and Refine's
own ``OfferFeedback`` call run on the budget LM.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import dspy
import numpy as np

from ctra.agents.data_models import (
    BUILDER_EXCEPTION_MSG_MAXLEN,
    BUILDER_EXCEPTION_PREFIX,
    BUILDER_EXCEPTION_RESEARCH_SENTINEL,
    BUILDER_OMITTED_PREFIX,
    FeaturePlan,
    FeatureType,
)
from ctra.agents.feature_store import (
    get_cached_feature,
    get_cached_features_batch,
    put_cached_feature,
)
from ctra.agents.feature_utils import dump_as_json, soft_assert
from ctra.agents.lm_config import configure_budget_lm
from ctra.agents.reward_fns import (
    ResettingRefine,
    builder_reward,
    is_valid_grouper,
    unwrap_builder_result,
    unwrap_groups,
)
from ctra.agents.signatures import (
    FeatureBuilderConstructSignature,
    FeatureBuilderResearchMultiSignature,
)
from ctra.rag.tools import (
    get_trial_info_dict,
    make_aact_search,
    make_chembl_search,
    make_drugsfda_search,
    make_faers_search,
    make_nct_search,
    make_primekg_search,
    make_pubmed_search,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Builder — 2-phase grouped build
# ---------------------------------------------------------------------------


class FeatureBuilder(dspy.Module):  # type: ignore[misc]
    """Two-phase grouped feature builder with per-type validation.

    Phase 1 (Research): ``dspy.ReAct(FeatureBuilderResearchMultiSignature)``
    with CTRA's 7 tools gathers data for all features in a group.

    Phase 2 (Construct): ``dspy.ChainOfThought(FeatureBuilderConstructSignature)``
    converts research results into typed feature values with
    ``none_feature_explanations`` for missing data.

    Both phases run under the budget LM (Sonnet) via ``dspy.context`` when
    the module is called directly. Under ``ResettingRefine`` the wrapper's
    outer ``dspy.context`` is what keeps the Construct phase on the budget LM
    (see the module docstring).

    Parameters:
        task_description: Text description of the prediction task.
    """

    def __init__(self, task_description: str) -> None:
        super().__init__()
        self.task_description = task_description
        self.constructor = dspy.ChainOfThought(FeatureBuilderConstructSignature)
        self._budget_lm = configure_budget_lm()

    def forward(
        self,
        nctid: str,
        feature_plan_group: dict[str, FeaturePlan],
    ) -> dspy.Prediction:
        """Build features for a single trial from a group of plans.

        Parameters:
            nctid: NCT ID of the trial.
            feature_plan_group: Grouped feature plans (max 5).

        Returns:
            ``dspy.Prediction`` with fields ``feature_values``
            (``{feature_name: {sub: value}}``, covering **only** the features the
            Construct step actually produced -- coverage may be partial) and
            ``metadata`` (the four keys ``research_results``,
            ``research_result_reasoning``, ``builder_reasoning``,
            ``none_feature_explanations``). Completeness is validated by
            ``reward_fns.is_valid_builder`` via the ``ResettingRefine`` wrapper in
            ``WrappedFeatureBuilder``, not here. Unwrap with
            ``reward_fns.unwrap_builder_result`` -- never ``values, meta = ...``,
            which binds the field-name strings.
        """
        nct_info = get_trial_info_dict(nctid)

        # Serialize plans (strip heavy fields for research prompt)
        serialized = {
            name: json.loads(dump_as_json(plan)) for name, plan in feature_plan_group.items()
        }
        simplified = {
            fn: {
                k: v
                for k, v in d.items()
                if k not in ("data_sources", "example_values", "possible_values")
            }
            for fn, d in serialized.items()
        }

        # Phase 1: Research (ReAct with 7 tools)
        tools = [
            make_pubmed_search(nct_info),
            make_nct_search(nct_info),
            make_chembl_search(nct_info),
            make_faers_search(nct_info),
            make_aact_search(nct_info),
            make_primekg_search(nct_info),
            make_drugsfda_search(nct_info),
        ]
        research = dspy.ReAct(
            FeatureBuilderResearchMultiSignature,
            tools=tools,
            max_iters=5,
        )

        with dspy.context(lm=self._budget_lm):
            research_result = research(
                task=self.task_description,
                nctid=nctid,
                feature_plans=simplified,
            )

            # Phase 2: Construct from research
            builder_result = self.constructor(
                task=self.task_description,
                feature_plans=serialized,
                research_results=research_result.research_results,
            )

        feature_values = deepcopy(builder_result.all_feature_values)

        # Incomplete coverage is reported, not raised. Raising here burned every
        # ResettingRefine attempt and re-raised on the last one (refine.py:172),
        # discarding the features that *were* built. is_valid_builder scores the
        # partial result 0.0 so Refine retries with OfferFeedback guidance; if the
        # budget runs out, WrappedFeatureBuilder fills the stragglers with all-None
        # values and a builder_omitted explanation.
        missing_features = set(feature_plan_group.keys()) - set(feature_values.keys())
        if missing_features:
            logger.warning(
                "Construct step omitted %d/%d feature(s) for %s: %s",
                len(missing_features),
                len(feature_plan_group),
                nctid,
                sorted(missing_features),
            )

        # Per-type validation (AutoCT lines 1214-1330)
        values: dict[str, dict[str, Any]] = {}
        for feature_name, feature_value in feature_values.items():
            if feature_name not in feature_plan_group:
                continue

            feature_type = feature_plan_group[feature_name].feature_type
            possible_values = feature_plan_group[feature_name].possible_values

            if len(feature_value) == 0:
                logger.warning(
                    "Empty output for %s / %s",
                    feature_name,
                    nctid,
                )
                values[feature_name] = {k: None for k in feature_type}
                continue

            for key, value in feature_value.items():
                value = soft_assert(
                    value,
                    key in feature_type,
                    f"Unexpected sub-feature key: {key} for {feature_name}.  "
                    f"Expected one of {list(feature_type.keys())}",
                )

                if isinstance(value, list):
                    value = json.dumps(value)
                if not isinstance(value, str):
                    value = str(value)

                # Strip quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                if value == "None":
                    value = None

                if key not in feature_type:
                    continue

                ft = feature_type[key]

                if ft == FeatureType.BOOLEAN:
                    if value is not None:
                        value = value.lower()
                        value = soft_assert(
                            value,
                            value in ("true", "false"),
                            f"'boolean' features must have values 'True' or "
                            f"'False'.  Got '{value}' in {feature_name}/{key}.",
                        )
                        value = np.nan if value is None else value == "true"
                    else:
                        value = np.nan

                elif ft == FeatureType.CATEGORICAL:
                    pv = possible_values.get(key, [])
                    if value is not None:
                        value = soft_assert(
                            value,
                            value in pv,
                            f"'categorical' feature must be a single value in "
                            f"{pv}.  Got '{value}' in {feature_name}/{key}.",
                        )

                elif ft == FeatureType.MULTICATEGORICAL:
                    pv = possible_values.get(key, [])
                    pv_set = set(pv)
                    if value is not None:
                        try:
                            as_list = json.loads(value)
                        except (ValueError, TypeError):
                            as_list = None

                        as_list = soft_assert(
                            as_list,
                            as_list is not None and isinstance(as_list, list),
                            f"'multi-categorical' feature must be returned as "
                            f"a JSON array of values from {pv}.  "
                            f"Got {value} in {feature_name}/{key}",
                        )
                        if as_list is None or (
                            len(as_list) == 1 and (as_list[0] == "None" or as_list[0] is None)
                        ):
                            value = None
                        else:
                            as_list = soft_assert(
                                as_list,
                                set(as_list).issubset(pv_set),
                                f"'multi-categorical' feature must be values "
                                f"from {pv}.  Invalid: "
                                f"{set(as_list) - pv_set} in {feature_name}/{key}",
                            )
                            value = as_list

                elif ft == FeatureType.INTEGER:
                    if value is not None:
                        value = soft_assert(
                            value,
                            re.fullmatch(r"-?\d+", value) is not None,
                            f"'integer' feature must return an integer value.  "
                            f"Got {value} in {feature_name}/{key}",
                        )
                        value = np.nan if value is None else float(int(value))
                    else:
                        value = np.nan

                elif ft == FeatureType.FLOAT:
                    if value is not None:
                        value = soft_assert(
                            value,
                            re.match(r"^-?\d+(?:\.\d+)?$", value) is not None,
                            f"'float' feature must return a floating point "
                            f"value.  Got {value} in {feature_name}/{key}",
                        )
                        value = np.nan if value is None else float(value)
                    else:
                        value = np.nan

                feature_value[key] = value
            values[feature_name] = feature_value

        return dspy.Prediction(
            feature_values=values,
            metadata={
                "research_results": research_result.research_results,
                "research_result_reasoning": getattr(research_result, "reasoning", ""),
                "builder_reasoning": getattr(builder_result, "reasoning", ""),
                "none_feature_explanations": getattr(
                    builder_result, "none_feature_explanations", {}
                ),
            },
        )


# ---------------------------------------------------------------------------
# Disk-cached wrapper (AutoCT agent.py:1709-1775)
# ---------------------------------------------------------------------------


class WrappedFeatureBuilder:
    """Wrapper around FeatureBuilder with global feature value store.

    Per-feature granularity: before building, probes the store for every
    (nctid, feature_name, plan_hash) key. Only uncached plans are dispatched
    to FeatureBuilder. After a successful build, each freshly built feature
    is persisted individually.

    On exception, returns all-None values and sentinel metadata documenting
    the crash. No store writes on the failure path — no negative caching.

    After Refine returns, any uncached plan the builder never produced (the
    Construct step omitted it on every attempt) is filled with all-``None``
    sub-values and a ``builder_omitted`` explanation. The fill happens **after**
    the store writes, so an omission is never negatively cached and is retried
    on the next run, and **outside** the reward boundary, so Refine still sees
    (and retries) the partial result.
    """

    def __init__(
        self,
        task_description: str,
        feature_store_dir: Path | None = None,
        task_namespace: str = "default",
        feature_store_enabled: bool = True,
    ) -> None:
        self.task_description = task_description
        self._feature_store_dir = feature_store_dir
        self._task_namespace = task_namespace
        self._feature_store_enabled = feature_store_enabled and feature_store_dir is not None
        # Entered around the ResettingRefine call in __call__. refine.py:107-108
        # deepcopies the module and pins ``dspy.settings.lm`` onto every named
        # predictor via ``mod.set_lm()``, which OVERRIDES FeatureBuilder.forward's
        # own ``dspy.context(lm=...)`` for the __init__-time ``constructor``.
        # Deliberately not ``builder._budget_lm``: FeatureBuilder is patched as a
        # MagicMock class in tests, and a MagicMock must never reach
        # ``dspy.context(lm=...)``. Constructed once per wrapper (one per
        # compute_features call), not per trial-group.
        self._budget_lm = configure_budget_lm()

    def __call__(
        self, arg: tuple[str, dict[str, FeaturePlan]]
    ) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
        """Build features for a single (nctid, plan_group) pair.

        Parameters:
            arg: Tuple of ``(nctid, plan_group_dict)``.

        Returns:
            Tuple of ``(nctid, values, metadata)``.
        """
        nctid, plans = arg

        # Partition into cached / uncached via the feature store
        cached_values: dict[str, dict[str, Any]] = {}
        uncached_plans: dict[str, FeaturePlan] = {}
        if self._feature_store_enabled:
            assert self._feature_store_dir is not None
            for feature_name, plan in plans.items():
                hit = get_cached_feature(
                    self._feature_store_dir,
                    self._task_namespace,
                    nctid,
                    feature_name,
                    plan,
                )
                if hit is not None:
                    # Canonical stored shape is {feature_name: sub_dict} (see
                    # put_cached_feature call sites below). Use explicit access
                    # so a shape drift bug fails loudly instead of silently
                    # feeding the wrong dict shape to downstream code.
                    cached_values[feature_name] = hit[feature_name]
                else:
                    uncached_plans[feature_name] = plan
        else:
            uncached_plans = dict(plans)

        # All features cached -> short-circuit, zero LLM calls
        if not uncached_plans:
            return (nctid, dict(cached_values), {})

        # Build only the uncached subset
        try:
            builder = FeatureBuilder(task_description=self.task_description)
            # Constructed per call, so Refine's failure budget cannot erode across
            # calls here -- ResettingRefine for consistency with the other sites.
            refiner = ResettingRefine(
                module=builder,
                N=3,
                reward_fn=builder_reward,
                threshold=1.0,
            )
            # Refine call under the budget LM. refine.py:99 reads
            # dspy.settings.lm and :108 pins it onto the deepcopied module's
            # predictors, so without this context the Construct phase leaks onto
            # the primary (Opus) LM. Entering it here routes the Construct phase
            # *and* the now-live OfferFeedback call (refine.py:167, resolved at
            # call time) to the budget LM instead.
            with dspy.context(lm=self._budget_lm):
                result = refiner(nctid=nctid, feature_plan_group=uncached_plans)

            # Never ``values, meta = result``: a Prediction unpacks into its KEY
            # STRINGS with no error. The helper also passes a legacy tuple through.
            values, meta = unwrap_builder_result(result)

            # Persist each freshly built feature individually. The
            # ``feature_name not in values`` guard is what keeps the omission
            # fill below out of the store: omitted names are not in ``values``
            # yet, so nothing negative is ever cached.
            if self._feature_store_enabled:
                assert self._feature_store_dir is not None
                for feature_name, plan in uncached_plans.items():
                    if feature_name not in values:
                        continue
                    put_cached_feature(
                        self._feature_store_dir,
                        self._task_namespace,
                        nctid,
                        feature_name,
                        plan,
                        {feature_name: values[feature_name]},
                        metadata={
                            "builder_reasoning": meta.get("builder_reasoning", ""),
                        },
                    )

            # Fill omissions -- AFTER the writes (no negative caching), BEFORE the
            # merge (the column must exist downstream: features_to_df names
            # columns from what the rows contain, and the orchestrator slices
            # val/test by the train columns). Outside the reward boundary so
            # is_valid_builder still saw the partial result and Refine retried.
            # The explanation entry is what keeps the omission visible: none_rate
            # in _build_builder_diagnostics counts explanation-map membership.
            omitted = [name for name in uncached_plans if name not in values]
            if omitted:
                explanations = dict(meta.get("none_feature_explanations", {}))
                for name in omitted:
                    values[name] = {k: None for k in uncached_plans[name].feature_type}
                    detail = str(explanations.get(name) or "no explanation provided")
                    if len(detail) > BUILDER_EXCEPTION_MSG_MAXLEN:
                        detail = detail[: BUILDER_EXCEPTION_MSG_MAXLEN - 3] + "..."
                    explanations[name] = f"{BUILDER_OMITTED_PREFIX} {detail}"
                # Rebuilt, not mutated: a shared metadata dict from a test double
                # must not be corrupted; the happy path returns meta untouched.
                meta = {**meta, "none_feature_explanations": explanations}
                logger.warning(
                    "Builder omitted %d feature(s) for %s after retries: %s",
                    len(omitted),
                    nctid,
                    sorted(omitted),
                )

            merged = {**cached_values, **values}
            return (nctid, merged, meta)
        except Exception as e:
            logger.warning("Failed to build features for %s: %s", nctid, e)
            traceback.print_exc()
            # All uncached plans come back as None. Cached values are still
            # returned -- they are independent successful builds.
            all_none = {
                k: {kk: None for kk in uncached_plans[k].feature_type} for k in uncached_plans
            }
            # Report the crash instead of returning empty metadata. ``none_rate``
            # in _build_builder_diagnostics counts membership in the explanation
            # map, so a feature that crashed on *every* trial used to report
            # none_rate=0.0 and be dropped by format_for_llm as healthy. Mirrors
            # the "[cached]" rehydration in compute_features: a sentinel entry
            # keeps the feature visible instead of erasing it.
            detail = f"{type(e).__name__}: {e}"
            if len(detail) > BUILDER_EXCEPTION_MSG_MAXLEN:
                detail = detail[: BUILDER_EXCEPTION_MSG_MAXLEN - 3] + "..."
            reason = f"{BUILDER_EXCEPTION_PREFIX} {detail}"
            return (
                nctid,
                {**cached_values, **all_none},
                {
                    "research_results": BUILDER_EXCEPTION_RESEARCH_SENTINEL,
                    "builder_reasoning": reason,
                    "none_feature_explanations": dict.fromkeys(uncached_plans, reason),
                },
            )


# ---------------------------------------------------------------------------
# Parallel grouped computation (AutoCT agent.py:1778-1801)
# ---------------------------------------------------------------------------


def compute_features(
    grouper: Any,
    nctids: list[str],
    task_description: str,
    plans: dict[str, FeaturePlan],
    feature_store_dir: Path | None = None,
    task_namespace: str = "default",
    feature_store_enabled: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    """Compute features for all trials using grouped parallel execution.

    Parameters:
        grouper: ``FeatureGrouper`` module (or any callable with same API). May return
            either a ``dspy.Prediction(groups=...)`` or the bare list; both are accepted.
        nctids: List of NCT IDs to process.
        task_description: Task description for the builder.
        plans: All feature plans.
        feature_store_dir: Feature store root directory.
        task_namespace: Namespace within store (e.g., "phase2").
        feature_store_enabled: Enable the global feature value store.

    Returns:
        Tuple of ``(raw_features, none_explanations, builder_metadata)`` where:
        - ``raw_features``: ``{nctid: {feature: {sub: val}}}``
        - ``none_explanations``: ``{nctid: {feature: reason}}``
        - ``builder_metadata``: ``{nctid: {feature: {research_results, builder_reasoning}}}``
          where ``research_results`` may carry ``"[cached]"`` or ``"[builder_exception]"`` sentinels.
    """
    # Group features (skip grouping for single plan)
    if len(plans) == 1:
        grouped_feature_plans = [plans]
    else:
        grouped_feature_plans = unwrap_groups(grouper(feature_plans=plans, task=task_description))

    # Site 3 validation. ``FeatureGrouper.forward()`` no longer raises on a bad
    # partition -- it filters and returns whatever survives -- so without a repair
    # here any feature the grouper dropped would silently never be built.
    #
    # Repair rather than discard: rebuilding the whole partition as
    # one-feature-per-group would undo the ~5x batching this module exists for
    # (one missing feature out of 20 would cost 20 research passes instead of ~4,
    # for every trial, every iteration). Keep the groups that came back and add
    # the unassigned features as singletons.
    if plans and not is_valid_grouper({"feature_plans": plans}, grouped_feature_plans):
        assigned: set[str] = set()
        repaired: list[dict[str, FeaturePlan]] = []
        # The guard tolerates degenerate shapes (None, a bare list of names) so
        # that they are *caught* here -- so the repair itself must not assume the
        # output is well-formed. Anything unusable contributes nothing and every
        # feature is recovered as a singleton below.
        for group in grouped_feature_plans or []:
            if not isinstance(group, dict):
                continue
            kept = {k: v for k, v in group.items() if k in plans and k not in assigned}
            assigned.update(kept)
            if kept:
                repaired.append(kept)

        missing = [k for k in plans if k not in assigned]
        repaired.extend({k: plans[k]} for k in missing)

        logger.warning(
            "Grouper partition invalid: %d group(s), %d/%d features assigned, "
            "missing=%s; repairing into %d group(s).",
            len(grouped_feature_plans or []),
            len(assigned),
            len(plans),
            missing,
            len(repaired),
        )
        grouped_feature_plans = repaired

    agged: dict[str, dict[str, Any]] = defaultdict(dict)
    none_feature_reasons: dict[str, dict[str, str]] = defaultdict(dict)
    builder_meta: dict[str, dict[str, Any]] = defaultdict(dict)

    # --- NEW: upfront batch short-circuit ---
    product: list[tuple[str, dict[str, FeaturePlan]]] = []
    fully_cached_count = 0
    if feature_store_enabled and feature_store_dir is not None:
        for group in grouped_feature_plans:
            # Per-feature batch probe across all nctids
            per_feature_hits: dict[str, dict[str, dict[str, Any]]] = {}
            for feature_name, plan in group.items():
                per_feature_hits[feature_name] = get_cached_features_batch(
                    feature_store_dir, task_namespace, nctids, feature_name, plan
                )
            for nctid in nctids:
                hit_names = {fn for fn, hits in per_feature_hits.items() if nctid in hits}
                if hit_names == set(group.keys()):
                    # Fully cached for this nctid: accumulate and skip dispatch
                    for fn in group:
                        agged[nctid][fn] = per_feature_hits[fn][nctid][fn]
                        # Rehydrate builder_meta so fully-cached trials do not
                        # look like RESEARCHER failures to
                        # _build_builder_diagnostics. A non-empty
                        # research_results string is all that
                        # research_coverage_score checks for — we mark the
                        # entry as cached instead of dropping it, which would
                        # push research_coverage below the 50% BUILDER
                        # attribution threshold as the store warms up.
                        builder_meta[nctid][fn] = {
                            "research_results": "[cached]",
                            "builder_reasoning": "[cached]",
                        }
                    fully_cached_count += 1
                else:
                    product.append((nctid, group))
        logger.info(
            "Feature store: %d of %d trial-group pairs served from store",
            fully_cached_count,
            len(nctids) * len(grouped_feature_plans),
        )
    else:
        product = list(itertools.product(nctids, grouped_feature_plans))

    # Execute (sequential for now; ProcessPoolExecutor can be added later
    # when dspy serialization issues are resolved)
    wrapper = WrappedFeatureBuilder(
        task_description=task_description,
        feature_store_dir=feature_store_dir,
        task_namespace=task_namespace,
        feature_store_enabled=feature_store_enabled,
    )

    for arg in product:
        nctid, feature_values, metadata = wrapper(arg)
        agged[nctid] = agged[nctid] | feature_values
        none_feature_reasons[nctid] = none_feature_reasons[nctid] | metadata.get(
            "none_feature_explanations", {}
        )
        # Preserve per-feature builder metadata for diagnostic analysis.
        # When the builder crashed (exception sentinel), only stamp exception metadata
        # on the features that actually failed; give cached siblings the "[cached]" sentinel.
        is_exception = metadata.get("research_results") == BUILDER_EXCEPTION_RESEARCH_SENTINEL
        crashed_features = (
            set(metadata.get("none_feature_explanations", {})) if is_exception else set()
        )
        group_meta = {
            "research_results": metadata.get("research_results", ""),
            "builder_reasoning": metadata.get("builder_reasoning", ""),
        }
        cached_meta = {"research_results": "[cached]", "builder_reasoning": "[cached]"}
        builder_meta[nctid] = builder_meta[nctid] | {
            feat_name: (
                dict(cached_meta)
                if is_exception and feat_name not in crashed_features
                else dict(group_meta)
            )
            for feat_name in feature_values
        }

    return dict(agged), dict(none_feature_reasons), dict(builder_meta)
