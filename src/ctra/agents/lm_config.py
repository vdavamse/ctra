"""DSPy Language Model configuration for CTRA agents.

Provides centralized LM setup so every agent module uses the same model
configuration.  Two LM instances are available:

* **Primary LM** (``configure_lm``) -- Claude Opus 4.6 for the feature
  proposer, planner, and evaluator agents.  These are low-volume, high-
  reasoning calls where model quality dominates cost.

* **Budget LM** (``configure_budget_lm``) -- Claude Sonnet 4.6 for the
  feature builder (ReAct) agent.  The builder issues many calls per trial
  (one per feature), so using a cheaper model ($3/$15 per MTok vs $5/$25)
  reduces cost while still providing strong extraction quality.

Both LMs use LiteLLM model-ID format (``anthropic/model-name``), which
DSPy passes directly to LiteLLM.  API keys are read from the environment
variable specified in ``settings.llm.api_key_env_var`` (default:
``ANTHROPIC_API_KEY``).

Usage::

    from ctra.agents.lm_config import configure_lm, configure_budget_lm

    # Set the global DSPy LM (call once at pipeline startup)
    lm = configure_lm()

    # For the builder agent, use the budget model in a scoped context
    budget_lm = configure_budget_lm()
    with dspy.context(lm=budget_lm):
        result = builder_react_module(...)
"""

from __future__ import annotations

import logging
import os

import dspy

from ctra.config.settings import get_settings

logger = logging.getLogger(__name__)


def configure_lm() -> dspy.LM:
    """Configure and activate the primary DSPy language model.

    Uses ``settings.llm.model_name`` (default ``anthropic/claude-opus-4-6``)
    with deterministic temperature (0.0) for reproducible feature engineering.

    The API key is resolved from the environment variable named in
    ``settings.llm.api_key_env_var``. If the variable is not set, LiteLLM
    will still attempt to resolve credentials via its own chain (e.g.,
    ``~/.litellm/config.yaml`` or cloud provider metadata).

    Returns:
        The configured language model instance, already set as the global
        DSPy default via ``dspy.configure(lm=...)``.
    """
    settings = get_settings()

    api_key = os.environ.get(settings.llm.api_key_env_var)
    if api_key is None:
        logger.warning(
            "Environment variable %s is not set. "
            "LiteLLM will attempt fallback credential resolution.",
            settings.llm.api_key_env_var,
        )

    lm = dspy.LM(
        model=settings.llm.model_name,
        api_key=api_key,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        cache=settings.llm.cache_control,
    )
    dspy.configure(lm=lm)

    logger.info(
        "Configured primary LM: model=%s max_tokens=%d temperature=%.1f cache=%s",
        settings.llm.model_name,
        settings.llm.max_tokens,
        settings.llm.temperature,
        settings.llm.cache_control,
    )
    return lm


def configure_budget_lm() -> dspy.LM:
    """Configure a cheaper LM for high-volume operations.

    Uses ``settings.llm.budget_model_name`` (default
    ``anthropic/claude-sonnet-4-6``) for the feature builder agent,
    which issues many extraction calls per MCTS node.

    This LM is **not** set as the global default -- callers should use it
    via ``dspy.context(lm=budget_lm)`` to scope its usage to specific
    agent calls.

    Returns:
        A budget language model instance (not globally configured).
    """
    settings = get_settings()

    api_key = os.environ.get(settings.llm.api_key_env_var)

    lm = dspy.LM(
        model=settings.llm.budget_model_name,
        api_key=api_key,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        cache=settings.llm.cache_control,
    )

    logger.info(
        "Configured budget LM: model=%s max_tokens=%d temperature=%.1f cache=%s",
        settings.llm.budget_model_name,
        settings.llm.max_tokens,
        settings.llm.temperature,
        settings.llm.cache_control,
    )
    return lm
