"""DSPy-based LLM agents for AutoCT feature engineering.

Full orchestrator with iteration dispatch::

    from ctra.agents import configure_lm, Agent, run_agent_as_subprocess
    lm = configure_lm()
    agent = Agent(task, X_train, X_val, y_train, y_val, X_test, y_test)
    output = agent(previous_output=None)  # iteration 0

LM Configuration
-----------------

All agents use Claude models via DSPy's LiteLLM backend.  Call
:func:`configure_lm` once at pipeline startup to set the global LM
(Claude Opus 4.6).  The builder automatically uses the budget LM
(Claude Sonnet 4.6) via ``dspy.context(lm=...)`` for cost efficiency.
"""

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
from ctra.agents.evaluator import Evaluator
from ctra.agents.feature_builder import FeatureBuilder, compute_features
from ctra.agents.feature_grouper import FeatureGrouper
from ctra.agents.feature_planner import FeaturePlanner
from ctra.agents.feature_proposer import FeatureProposer
from ctra.agents.feature_utils import (
    dump_as_json,
    features_to_df,
    parse_date,
    soft_assert,
)
from ctra.agents.initializer import Initializer
from ctra.agents.lm_config import configure_budget_lm, configure_lm
from ctra.agents.orchestrator import Agent
from ctra.agents.runner import extract_objectives, run_agent_as_subprocess

__all__ = [
    "Agent",
    "AgentOutput",
    "EvalOutput",
    "Evaluator",
    "FeatureBuilder",
    "FeatureGrouper",
    "FeatureOp",
    "FeaturePlan",
    "FeaturePlanner",
    "FeatureProposer",
    "FeatureSource",
    "FeatureType",
    "Initializer",
    "ModelEvalResult",
    "ProposerOutput",
    "Task",
    "compute_features",
    "configure_budget_lm",
    "configure_lm",
    "dump_as_json",
    "extract_objectives",
    "features_to_df",
    "parse_date",
    "run_agent_as_subprocess",
    "soft_assert",
]
