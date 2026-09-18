"""Multi-stage feature initialization for MCTS iteration 0.

Generates a comprehensive initial feature set via five stages:

1. Zero-shot feature ideas from task description alone.
2. Factor analysis on sample success/failure trial pairs using ReAct + tools.
3. Factor-based feature ideas from discovered factors.
4. Combine + deduplicate from both idea sources.
5. Plan each unique feature idea via ``FeaturePlanner``.

Uses the primary LM (Claude Opus 4.6) for all stages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import dspy
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ctra.agents.data_models import FeaturePlan

from ctra.agents.feature_planner import FeaturePlanner
from ctra.agents.reward_fns import (
    ResettingRefine,
    is_valid_planner,
    planner_reward,
    unwrap_planner_result,
)
from ctra.agents.signatures import (
    FactorAnalystSignature,
    FeatureInitializerCombinedSignature,
    FeatureInitializerWithFactorsSignature,
    FeatureInitializerZeroShotSignature,
)
from ctra.rag.tools import (
    get_detailed_nct_info,
    get_trial_info_dict,
    make_aact_search,
    make_chembl_search,
    make_drugsfda_search,
    make_faers_search,
    make_nct_search,
    make_primekg_search,
    make_pubmed_search,
)

logger = logging.getLogger(__name__)

# LLM runtime exception tuple for Stage 5 planner error handling.
# Start with base timeout/connection exceptions, then probe for dspy/litellm
# exceptions. This ensures real LLM failures (rate limits, malformed JSON, 5xx)
# are caught and logged distinctly from non-LLM schema validation errors.
_LLM_RUNTIME_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
)
try:
    from dspy.utils.exceptions import AdapterParseError
    _LLM_RUNTIME_EXCEPTIONS = (*_LLM_RUNTIME_EXCEPTIONS, AdapterParseError)
except ImportError:
    logger.debug("dspy.utils.exceptions.AdapterParseError not available")
try:
    from litellm.exceptions import APIError as _LiteLLMAPIError
    _LLM_RUNTIME_EXCEPTIONS = (*_LLM_RUNTIME_EXCEPTIONS, _LiteLLMAPIError)
except ImportError:
    logger.debug("litellm.exceptions.APIError not available")


def _make_tools_for_trial(nct_info: dict[str, Any]) -> list[Any]:
    """Build LinearRAG-powered tools for feature research on a specific trial.

    Creates DSPy-compatible tool definitions for the FeatureInitializer agents,
    each wrapping a LinearRAG query for a specific data source. Tools include:
    PubMed (literature), ClinicalTrials.gov (trial registry), ChEMBL (drug targets),
    FAERS/OpenFDA (adverse events), AACT (trial statistics), PrimeKG (knowledge
    graph), and Drugs@FDA (approval history).

    Args:
        nct_info: Dict with trial metadata (e.g., drug, disease, phase).

    Returns:
        List of DSPy tool definitions, one per data source.
    """
    return [
        make_pubmed_search(nct_info),
        make_nct_search(nct_info),
        make_chembl_search(nct_info),
        make_faers_search(nct_info),
        make_aact_search(nct_info),
        make_primekg_search(nct_info),
        make_drugsfda_search(nct_info),
    ]


class Initializer(dspy.Module):  # type: ignore[misc]
    """Generate initial feature plans for MCTS iteration 0 (zero-shot + factor-based).

    Bootstraps feature engineering by running three initialization paths in parallel:
    (1) zero-shot feature proposal, (2) factor-based analysis from example trials,
    and (3) merging the proposals into a single coherent feature set.
    This initialization populates the feature space with diverse initial hypotheses
    for the MCTS search to refine.

    Args:
        task_description: Text description of the prediction task.
        X_train: Training set — DataFrame with nctId (or nct_id) column, or Series of NCT IDs.
        y_train: Training labels (0 = failure, 1 = success).
        seed: Random seed for sampling trial pairs for factor analysis.
        num_examples: Number of success/failure trial pairs to analyze.
    """

    def __init__(
        self,
        task_description: str,
        X_train: pd.DataFrame | pd.Series,
        y_train: pd.Series | NDArray[Any],
        seed: int = 42,
        num_examples: int = 3,
        feature_planner: dspy.Module | None = None,
    ) -> None:
        super().__init__()
        self.task_description = task_description
        self.X_train = X_train
        self.y_train = np.asarray(y_train)
        self.rng = np.random.default_rng(seed)
        self.num_examples = num_examples

        self.feature_initializer_zero_shot = dspy.ChainOfThought(
            FeatureInitializerZeroShotSignature
        )
        self.feature_initializer_from_factors = dspy.ChainOfThought(
            FeatureInitializerWithFactorsSignature
        )
        self.feature_initializer_combined = dspy.ChainOfThought(FeatureInitializerCombinedSignature)
        # Reuse the caller's planner when given one, so ``Agent`` and its
        # ``Initializer`` share a single wrapped planner rather than each keeping
        # its own (and each eroding its own failure budget).
        #
        # Explicit ``is None``, not truthiness: a ``dspy.Module`` subclass
        # defining ``__bool__``/``__len__`` would otherwise be discarded and
        # replaced with a second wrapper, silently undoing the shared-planner
        # contract while the identity test still passed on a truthy MagicMock.
        if feature_planner is None:
            # ResettingRefine, not dspy.Refine: Refine's failure budget erodes
            # permanently across calls, and Stage 5 calls this once per feature.
            feature_planner = ResettingRefine(
                module=FeaturePlanner(task_description),
                N=3,
                reward_fn=planner_reward,
                threshold=1.0,
            )
        self.feature_planner = feature_planner

    def _get_nctids(self) -> pd.Series:
        """Extract NCT IDs from X_train."""
        if isinstance(self.X_train, pd.Series):
            return self.X_train
        for col in ("nctId", "nct_id", "nctid"):
            if col in self.X_train.columns:
                return self.X_train[col]
        return self.X_train.iloc[:, 0]

    def forward(self) -> dict[str, FeaturePlan]:
        """Run the 5-stage initialization pipeline.

        Returns:
            Dictionary of feature plans keyed by feature name.
        """
        nctids = self._get_nctids()
        success_mask = self.y_train == 1
        success_ids = nctids[success_mask]
        failure_ids = nctids[~success_mask]

        # Stage 1: Zero-shot feature ideas
        logger.info("Initializer Stage 1: Zero-shot feature ideas")
        zero_shot_result = self.feature_initializer_zero_shot(
            task=self.task_description,
        )

        # Stage 2: Factor analysis on sample trial pairs
        logger.info(
            "Initializer Stage 2: Factor analysis on %d trial pairs",
            self.num_examples,
        )
        example_factors: list[dict[str, str]] = []

        for i in range(self.num_examples):
            # Sample one success and one failure
            success_nctid = str(self.rng.choice(success_ids.values))
            failure_nctid = str(self.rng.choice(failure_ids.values))

            nct_info_success = get_trial_info_dict(success_nctid)
            nct_info_failure = get_trial_info_dict(failure_nctid)
            context_success = get_detailed_nct_info(success_nctid)
            context_failure = get_detailed_nct_info(failure_nctid)

            # ReAct factor analyst for success trial
            factor_analyst_success = dspy.ReAct(
                FactorAnalystSignature,
                tools=_make_tools_for_trial(nct_info_success),
            )
            # ReAct factor analyst for failure trial
            factor_analyst_failure = dspy.ReAct(
                FactorAnalystSignature,
                tools=_make_tools_for_trial(nct_info_failure),
            )

            try:
                result_success = factor_analyst_success(
                    task=self.task_description,
                    sample_clinical_trial=context_success,
                    trial_task_label=1,
                )
                example_factors += result_success.factors
            except Exception:
                logger.warning(
                    "Factor analysis failed for success trial %s",
                    success_nctid,
                    exc_info=True,
                )

            try:
                result_failure = factor_analyst_failure(
                    task=self.task_description,
                    sample_clinical_trial=context_failure,
                    trial_task_label=0,
                )
                example_factors += result_failure.factors
            except Exception:
                logger.warning(
                    "Factor analysis failed for failure trial %s",
                    failure_nctid,
                    exc_info=True,
                )

            logger.debug(
                "Pair %d: success=%s failure=%s factors=%d",
                i,
                success_nctid,
                failure_nctid,
                len(example_factors),
            )

        # Stage 3: Factor-based feature ideas
        logger.info(
            "Initializer Stage 3: Factor-based ideas from %d factors",
            len(example_factors),
        )
        factor_result = self.feature_initializer_from_factors(
            task=self.task_description,
            factors=example_factors,
        )

        # Stage 4: Combine + deduplicate
        combined_feature_ideas = zero_shot_result.feature_ideas + factor_result.feature_ideas
        logger.info(
            "Initializer Stage 4: Combining %d ideas",
            len(combined_feature_ideas),
        )
        combined_result = self.feature_initializer_combined(
            task=self.task_description,
            combined_feature_ideas=combined_feature_ideas,
        )

        # Stage 5: Plan each feature
        logger.info(
            "Initializer Stage 5: Planning %d features",
            len(combined_result.feature_ideas),
        )
        feature_plans: dict[str, FeaturePlan] = {}
        for feature_name, feature_idea in combined_result.feature_ideas.items():
            try:
                plan, raw = unwrap_planner_result(
                    self.feature_planner(
                        feature_name=feature_name,
                        feature_idea=feature_idea,
                    )
                )
            except _LLM_RUNTIME_EXCEPTIONS:
                # LLM-origin error: timeout, connection, rate limit, malformed JSON, etc.
                # Skip the feature and continue to the next.
                logger.warning(
                    "Initializer Stage 5: LLM error planning feature '%s', skipping",
                    feature_name,
                    exc_info=True,
                )
                continue
            except Exception:
                # Non-LLM error (e.g. attribute error in the pipeline): log with a
                # traceback and skip this feature -- one bad plan must not abort
                # initialization for every other feature.
                logger.warning(
                    "Initializer Stage 5: unexpected error planning feature '%s', skipping",
                    feature_name,
                    exc_info=True,
                )
                continue

            # Validate schema consistency: re-apply is_valid_planner check.
            if not is_valid_planner({}, (plan, raw)):
                logger.warning(
                    "Initializer Stage 5: planner returned invalid plan for feature '%s' "
                    "(best of 3 retries), skipping",
                    feature_name,
                )
                continue

            feature_plans[feature_name] = plan

        logger.info(
            "Initialization complete: %d features planned",
            len(feature_plans),
        )
        return feature_plans
