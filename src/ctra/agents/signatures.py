"""DSPy Signatures for the CTRA agent pipeline.

Mirrors AutoCT's signatures from ``agent.py:298-930``, adapted for CTRA's
7 data sources (ClinicalTrials.gov, PubMed, ChEMBL, FAERS, AACT, PrimeKG,
Drugs@FDA) instead of AutoCT's 2 (ClinicalTrials.gov + PubMed).

All signatures are defined as ``dspy.Signature`` subclasses.  The DSPy
module type (ChainOfThought vs ReAct) is chosen at instantiation time in
the agent modules, not here.
"""

from __future__ import annotations

from typing import Any

import dspy

from ctra.agents.data_models import FeatureOp, FeatureSource, FeatureType  # noqa: TC001

# ---------------------------------------------------------------------------
# Initialization signatures (iteration 0)
# ---------------------------------------------------------------------------


# FeatureInitializerZeroShotSignature — Used by: FeatureInitializer agent (initializer.py)
# Purpose: Bootstrap the feature engineering process by asking the LLM to propose
# an initial set of feature ideas from scratch (zero-shot, no prior features or factors).
# Input: task description → Output: list of feature name/description dicts
class FeatureInitializerZeroShotSignature(dspy.Signature):  # type: ignore[misc]
    """You are an experienced clinical researcher skilled at proposing
    features for a machine learning model.

    Your task is to propose a comprehensive list of feature ideas (at least
    10) for this model.  Be as exhaustive and as detailed as possible in
    describing the feature.

    The features should be built off data from ClinicalTrials.gov, PubMed,
    ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA.

    The features should
    - be one of integer, float, boolean, categorical or multicategorical
    - NOT be a composite of multiple factors or features
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from the available sources
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_ideas: list[dict[str, str]] = dspy.OutputField(
        desc=(
            "A list of factors with a key 'feature_name' containing the "
            "snake-cased feature name and a key 'description' with a brief "
            "description of the feature idea."
        )
    )


# FactorAnalystSignature — Used by: FeatureInitializer agent (initializer.py)
# Purpose: Analyze specific trial successes/failures to extract generalizable factors
# that inform feature generation. Provides the "learning from examples" path of initialization.
# Input: task, sample trial with outcome → Output: list of factor name/description dicts
class FactorAnalystSignature(dspy.Signature):  # type: ignore[misc]
    """You are an experienced clinical researcher.

    You are analyzing clinical trials to deduce factors to help with building
    a machine learning model for a given task.  You are given detailed
    information of a clinical trial and the label for the task.  Your task is
    to analyze the key factors that contributed to the particular outcome that
    can be used to inform future trials.

    You should provide at least 5 factors that are generalizable to other
    trials.  Keep your analysis concise.

    Your factors can be from the trial context, or from historical data in
    PubMed, ChEMBL, FAERS, AACT, PrimeKG, Drugs@FDA, and other clinical
    trials in the NCT database.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    sample_clinical_trial: str = dspy.InputField()
    trial_task_label: int = dspy.InputField(desc="The task label")

    factors: list[dict[str, str]] = dspy.OutputField(
        desc=(
            "A list of factors, with each factor having a 'name' key and a "
            "'description' key containing brief explanation of how it can "
            "contribute to the outcome of a trial."
        )
    )


# FeatureInitializerWithFactorsSignature — Used by: FeatureInitializer agent (initializer.py)
# Purpose: Convert extracted factors into a set of machine-learning-ready features,
# ensuring they are measurable and constructible from available data sources.
# Input: task, factor analyses → Output: list of feature name/description dicts
class FeatureInitializerWithFactorsSignature(dspy.Signature):  # type: ignore[misc]
    """You are an experienced biomedical data scientist.

    You are given analyses of key factors that might have influenced the label
    of a particular task for past clinical trials.  Suppose we need to build
    a machine learning model for this prediction task.  Based on the analyses
    provided, summarize a comprehensive list (at least 8) of features that
    can help with the prediction.

    The features should be built off data from ClinicalTrials.gov, PubMed,
    ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA.

    If any of the factors cannot be made into a feature with these data
    sources, you should skip and move on to the next factor.

    Your features should
    - be one of integer, float, boolean, categorical or multicategorical
    - be generic enough to apply to most clinical trials
    - NOT be a composite of multiple factors or features
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from the available sources
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    factors: list[dict[str, str]] = dspy.InputField()
    feature_ideas: list[dict[str, str]] = dspy.OutputField(
        desc=(
            "A list of factors with a key 'feature_name' containing the "
            "snake-cased feature name and a key 'description' with a brief "
            "description of the feature idea."
        )
    )


# FeatureInitializerCombinedSignature — Used by: FeatureInitializer agent (initializer.py)
# Purpose: Merge and deduplicate feature proposals from multiple initialization paths
# (zero-shot + factor-based) into a single coherent feature set for the MCTS search.
# Input: task, list of feature idea dicts → Output: dict mapping feature_name → description
class FeatureInitializerCombinedSignature(dspy.Signature):  # type: ignore[misc]
    """You are an experienced biomedical data scientist.

    You are given a combined list of feature ideas from different experts for
    a clinical trial machine learning task.  Summarize and refine them as
    needed to produce a new list of features that is comprehensive,
    well-defined and non-overlapping.

    Make sure that all features can be built off data from ClinicalTrials.gov,
    PubMed, ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA.

    Your features should
    - be one of integer, float, boolean, categorical or multicategorical
    - be generic enough to apply to most clinical trials
    - NOT be a composite of multiple factors or features
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from the available sources
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    combined_feature_ideas: list[dict[str, str]] = dspy.InputField()
    feature_ideas: dict[str, str] = dspy.OutputField(
        desc=(
            "A dict where the key is a snake-cased feature name and the "
            "value is a description of the feature idea."
        )
    )


# ---------------------------------------------------------------------------
# Proposer signature (iteration N)
# ---------------------------------------------------------------------------


# FeatureProposerSignature — Used by: FeatureProposer agent (feature_proposer.py)
# Purpose: Generate ADD/REMOVE/REFINE operations on features based on expert suggestions
# or error analyses. Drives the iterative feature engineering in MCTS rollouts.
# Input: task, current features, suggestion → Output: operation type, feature name, description
class FeatureProposerSignature(dspy.Signature):  # type: ignore[misc]
    """You are an experienced clinical researcher skilled at proposing
    features for a machine learning model.

    An initial model has been built, and you are working to improve the model
    further by incorporating a suggestion from an expert for an update to the
    features.  The suggestion can be either a generic suggestion, or a
    detailed analysis of a prediction that a model got wrong.

    Your job is to propose a SINGLE OPERATION that is ONE OF
    - 'ADD' adding a new feature
    - 'REMOVE' removing a feature
    - 'REFINE' refining an existing feature from the model

    The feature should be built off data from ClinicalTrials.gov, PubMed,
    ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA.

    The feature should
    - be simple
    - be explainable
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from the available sources
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    current_features_with_plan: list[tuple[str, str]] = dspy.InputField(
        desc="The current features with their current plans"
    )
    suggestion: str = dspy.InputField(desc="The suggestion from the data scientist")

    operation: FeatureOp = dspy.OutputField(desc="The feature operation to perform")
    feature_name: str = dspy.OutputField(
        desc=(
            "If REFINE or REMOVE: The name of the feature to apply this "
            "operation on.  If ADD: the snake_cased name of this feature"
        )
    )
    operation_description: str = dspy.OutputField(
        desc=(
            "The description of the feature add, removal, or refinement.  "
            "This should be as detailed as possible to fully explain the "
            "operation."
        )
    )


# ---------------------------------------------------------------------------
# Planner signature
# ---------------------------------------------------------------------------


# FeaturePlannerSignature — Used by: FeaturePlanner agent (feature_planner.py)
# Purpose: Define the schema, instructions, data sources, and examples for constructing
# a specific feature. Converts LLM-proposed feature ideas into executable plans.
# Input: task, feature_name, feature_idea → Output: feature type, data sources, instructions, examples
class FeaturePlannerSignature(dspy.Signature):  # type: ignore[misc]
    """You are an expert data scientist.

    You are given an idea for a single feature to be used in a machine
    learning model for a clinical trial task.

    For this single feature, you are defining a feature schema for your
    co-workers to construct the feature for each clinical trial.

    The final built feature should be a JSON object
    - If there's only a single value, it should be a JSON with a single key
      "value" and the value.
    - If there are multiple values, it should be a JSON with multiple keys,
      each key corresponding to a sub-feature name, and the value
      corresponding to the sub-feature value.

    The schema and instruction should be as simple as possible to represent
    the feature idea.  Your instruction should be clear, and allow for the
    feature to be computed consistently and reliably.  The instruction needs
    to be explicit and avoid ambiguity since multiple teams are working
    together.
    - For e.g., if weights need to be assigned, they should be explicitly
      defined in the instructions.

    The feature should be built off data from ClinicalTrials.gov, PubMed,
    ChEMBL, FAERS, AACT, PrimeKG, and Drugs@FDA.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_name: str = dspy.InputField()
    feature_idea: str = dspy.InputField()

    feature_type: dict[str, FeatureType] = dspy.OutputField(
        desc=(
            "The type of the feature.\n\n"
            "If this is a single valued feature, this should be a dict with "
            "a single key 'value' and the expected type.\n"
            "If this is a multi-valued feature, each key should correspond "
            "to the sub-feature name, and the value should be the expected "
            "type.\n\n"
            "The allowed types are:\n"
            "'boolean' features are for 'True' / 'False' questions\n"
            "'integer' features are for integral numbers\n"
            "'float' features are for floating point numbers\n"
            "'categorical' features are for single-valued categorical "
            "features, returned as a String from `possible_values`\n"
            "'multi-categorical' features are for multi-valued categorical "
            "features, returned as one or more Strings from "
            "`possible_values` in a JSON array"
        )
    )
    data_sources: list[FeatureSource] = dspy.OutputField(
        desc=(
            "a list of data sources to retrieve data from.  This can include "
            "'pubmed', 'current_trial_summary', 'related_clinical_trials', "
            "'chembl', 'faers', 'aact', 'primekg', 'drugsfda'"
        )
    )
    example_values: list[dict[str, Any]] = dspy.OutputField(
        desc=(
            "one or more examples of an expected value for this data.  This "
            "should be correctly formatted as the output JSON"
        )
    )
    possible_values: dict[str, list[str]] = dspy.OutputField(
        desc=(
            "(for 'categorical' and 'multi-categorical' only) a list of "
            "possible values for this data\n\n"
            "If this is a single valued feature, this should be a dict with "
            "a single key 'value' and the possible values.\n"
            "If this is a multi-valued feature, each key should correspond "
            "to the sub-feature name, and the value should be the possible "
            "values."
        )
    )
    feature_instructions: str = dspy.OutputField(
        desc=(
            "Instructions for researching and building this feature.  The "
            "instructions should be clear, explicitly define the kind of "
            "research and data to look up, and make sure it is as precise "
            "as possible."
        )
    )


# ---------------------------------------------------------------------------
# Feature grouping signature
# ---------------------------------------------------------------------------


# FeatureGroupingSignature — Used by: FeatureGrouper agent (feature_grouper.py)
# Purpose: Partition feature plans into groups (max 5 per group) that can be researched
# together, minimizing redundant data retrieval and optimizing feature builder efficiency.
# Input: task, feature_plans (dict) → Output: list of feature groups (list of lists)
class FeatureGroupingSignature(dspy.Signature):  # type: ignore[misc]
    """You are part of a clinical research team creating features for
    clinical trial machine learning models.

    You have a list of features that need to be constructed.

    Your job is to identify groups of features that rely on similar data,
    and can be researched together based on their plan instructions.
    The grouping should be based on the feature plans and the kind of data
    the features require to enable efficient research.

    You should return a list of groups, where each group is a list of
    feature names that can be constructed together.
    There should be a maximum of 5 features in each group.
    Each feature should only be in one group.
    If a feature cannot be grouped with any other feature, it should be in
    its own group.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_plans: dict[str, dict[str, Any]] = dspy.InputField(
        desc="A dict of feature names to their corresponding plans."
    )
    groups: list[list[str]] = dspy.OutputField(
        desc=(
            "A list of groups, where each group is a list of feature names "
            "that can be constructed together."
        )
    )


# ---------------------------------------------------------------------------
# Builder signatures (research + construct)
# ---------------------------------------------------------------------------


# FeatureBuilderResearchMultiSignature — Used by: FeatureBuilder agent (feature_builder.py)
# Purpose: Conduct multi-source research (LinearRAG + tool calls) for a group of features,
# gathering data from ClinicalTrials.gov, PubMed, ChEMBL, FAERS, AACT, PrimeKG, Drugs@FDA.
# Input: task, nctid, feature_plans → Output: research_results (summarized data)
class FeatureBuilderResearchMultiSignature(dspy.Signature):  # type: ignore[misc]
    """You are part of a clinical research team creating features for
    clinical trial machine learning models.

    You are investigating a particular clinical trial.  You are given a dict
    of features that your team needs to do research on.  You should make use
    of the given tools to do deep research, gather information and provide
    the data necessary to build all the features.

    Do not focus on formatting the features correctly, instead focus on
    making sure you have a full and complete set of data.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    nctid: str = dspy.InputField(desc="The NCT ID of the clinical trial you are looking at")
    feature_plans: dict[str, dict[str, Any]] = dspy.InputField(
        desc=(
            "A dict of feature names to their corresponding plans.  This is "
            "the ultimate set of features we're doing research for."
        )
    )
    research_results: str = dspy.OutputField(
        desc=(
            "Your summarized research results that can be used to build the "
            "features.  You should produce enough results that are "
            "sufficient to build all the features."
        )
    )


# FeatureBuilderConstructSignature — Used by: FeatureBuilder agent (feature_builder.py)
# Purpose: Format research results into correctly-typed feature values (JSON, arrays, etc.)
# following the feature plan specifications. Handles missing data gracefully (None values).
# Input: task, feature_plans, research_results → Output: all_feature_values, none_explanations
class FeatureBuilderConstructSignature(dspy.Signature):  # type: ignore[misc]
    """You are part of a clinical research team creating features for
    clinical trial machine learning models.

    You are investigating a particular clinical trial.  You are given a dict
    of features and their corresponding plans that your team needs to
    construct.  A previous step has already gathered the necessary research
    results for these features, your job is to CORRECTLY construct these in
    the format prescribed by the feature plan.

    If there is
    - insufficient information
    - missing information
    - uncertainty/ambiguity
    for any of the features, you should return the value 'None' for that
    feature (or sub-feature) and provide explanations for the feature you
    can't build.

    YOU MUST HAVE AN OUTPUT FOR EACH FEATURE
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_plans: dict[str, dict[str, Any]] = dspy.InputField(
        desc="A dict of feature names to their corresponding plans."
    )
    research_results: str = dspy.InputField(
        desc=("The research results.  The data should be sufficient to build all the features")
    )
    all_feature_values: dict[str, dict[str, Any]] = dspy.OutputField(
        desc=(
            "A dict containing the feature name of each feature from "
            "feature_plans, and the output for that feature as a STRING.  "
            "For multi-categorical features, this should be JSON array of "
            "String"
        )
    )
    none_feature_explanations: dict[str, str] = dspy.OutputField(
        desc=(
            "A dict mapping feature name to the reason its value is None. Include "
            "an entry for any feature you cannot produce at all."
        )
    )


# ---------------------------------------------------------------------------
# Evaluator signatures
# ---------------------------------------------------------------------------


# EvaluatorSignature — Used by: Evaluator agent (evaluator.py)
# Purpose: Generate generic improvement suggestions (add/refine/remove features) based
# on model performance metrics and feature importances. Drives MCTS reward signal.
# Input: task, roc_auc, current_features, importances → Output: list of suggestions
class EvaluatorSignature(dspy.Signature):  # type: ignore[misc]
    """You are an experienced biomedical data scientist.

    You are supervising the construction of a machine learning model for a
    specific clinical trial task.  The model must be built with features from
    data from ClinicalTrials.gov, PubMed, ChEMBL, FAERS, AACT, PrimeKG, and
    Drugs@FDA.  A version of the model has been trained, and you are provided
    the current performance.

    You are also given:
    - Feature interaction analysis showing main effects (individual feature
      contributions) and pairwise interactions (synergies/redundancies between
      feature pairs).  Use these to identify which features are most/least
      useful and which feature combinations create value.
    - Builder diagnostics showing which features frequently fail to compute
      (high None rates) and whether the failure is likely due to a bad feature
      idea (RESEARCHER issue) or bad extraction logic (BUILDER issue).

    Please provide suggestions for
    - additional features (especially ones that would interact well with
      existing high-value features)
    - refinements to existing features (especially those with high None rates
      attributed to BUILDER issues)
    - features to remove (especially those with low main effects AND no
      significant interactions, or those with high None rates attributed to
      RESEARCHER issues)

    Keep your suggestions concise, and limit to a maximum of 2-3 suggestions.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    roc_auc_score: float = dspy.InputField()
    current_features_with_plan: list[tuple[str, str]] = dspy.InputField()
    feature_interactions: str = dspy.InputField(
        desc="Shapley interaction analysis: main effects and pairwise feature interactions"
    )
    builder_diagnostics: str = dspy.InputField(
        desc="Per-feature diagnostic summary: None rates, failure reasons, researcher vs builder attribution"
    )

    suggestions: list[str] = dspy.OutputField()


# EvaluatorWithExampleSignature — Used by: Evaluator agent (evaluator.py)
# Purpose: Conduct deep analysis of a specific model error (misclassified trial) to
# extract actionable, generalizable insights about missing/incorrect features.
# Input: task, roc_auc, current_features, error_example → Output: analysis (narrative)
class EvaluatorWithExampleSignature(dspy.Signature):  # type: ignore[misc]
    """You are an experienced clinical researcher.

    You are supervising the construction of a machine learning model for a
    specific clinical trial task.  The model must be built with features from
    data from ClinicalTrials.gov, PubMed, ChEMBL, FAERS, AACT, PrimeKG, and
    Drugs@FDA.  A version of the model has been trained, and you are provided
    the current performance, and an example of an incorrect prediction from
    the current model.

    You are also given feature interaction analysis (main effects and pairwise
    interactions) and builder diagnostics (None rates, failure attribution).

    Based on the example and using the tools provided to help with further
    research, please conduct some analysis on why the model made the
    incorrect prediction.

    You should consider
    - features that were missed, and could have helped with the prediction
    - features that were not useful (low main effect, no interactions)
    - misconstructed features (high None rate with BUILDER attribution)
    - feature plans that are not properly set up (e.g. missing instructions /
      missing categories) — indicated by RESEARCHER attribution
    - feature pairs that show strong interactions (synergy or redundancy)

    Your analysis should be generalizable to other trials where possible.
    Keep your analysis concise.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    roc_auc_score: float = dspy.InputField()
    current_features_with_plan: list[tuple[str, str]] = dspy.InputField()
    feature_interactions: str = dspy.InputField(
        desc="Shapley interaction analysis: main effects and pairwise feature interactions"
    )
    builder_diagnostics: str = dspy.InputField(
        desc="Per-feature diagnostic summary: None rates, failure reasons, researcher vs builder attribution"
    )
    example: str = dspy.InputField()

    analysis: str = dspy.OutputField()


# EvaluatorSummarizerSignature — Used by: Orchestrator agent (orchestrator.py)
# Purpose: Aggregate multiple expert evaluations into a consolidated list of feature
# improvement suggestions. Deduplicates and prioritizes suggestions for MCTS efficiency.
# Input: task, list of analyses → Output: list of unified suggestions (4-5 max)
class EvaluatorSummarizerSignature(dspy.Signature):  # type: ignore[misc]
    """You are an experienced clinical researcher.

    You are supervising the construction of a machine learning model for a
    specific clinical trial task.  The model must be built with features from
    data from ClinicalTrials.gov, PubMed, ChEMBL, FAERS, AACT, PrimeKG, and
    Drugs@FDA.

    You are given ideas and analyses from various experts that have analyzed
    the model results.

    Your job is to summarize and generalize these into a list of a maximum of
    4-5 suggestions that can be used to improve the model.

    Each suggestion should either
    - add a new feature
    - refine an existing feature
    - remove an existing feature
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    analyses: list[str] = dspy.InputField(desc="The suggestions from other experts")

    suggestions: list[str] = dspy.OutputField()
