"""Ungated import smoke tests: every first-party package must import.

This module deliberately has **no** ``try``/``except ImportError`` and no
``pytest.mark.skipif``.  If a first-party module stops importing, these tests
must go red -- a skip here would reproduce the false green of issue #9.

Scope is every package that imports with the *required* dependency set only
(``pyproject.toml`` ``[project] dependencies``).  That includes ``ctra.rag`` and
``ctra.dashboard``, verified by importing them with every extra blocked: their
package ``__init__`` never reaches an optional third-party import.  Only
``ctra.api`` is excluded -- its ``__init__`` imports ``fastapi`` at module scope
and needs the ``[api]`` extra, so a minimal install would fail here for the
wrong reason.  Genuinely optional third-party imports belong behind
``pytest.importorskip``.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

CORE_PACKAGES = (
    "ctra",
    "ctra.agents",
    "ctra.config",
    "ctra.dashboard",
    "ctra.data",
    "ctra.mlops",
    "ctra.models",
    "ctra.rag",
    "ctra.search",
)


def _agent_submodules() -> list[str]:
    """Every module under ``ctra.agents`` -- discovered, never hardcoded."""

    def _raise(name: str) -> None:
        raise ImportError(f"could not walk {name}")

    agents = importlib.import_module("ctra.agents")
    return sorted(
        info.name
        for info in pkgutil.walk_packages(agents.__path__, prefix="ctra.agents.", onerror=_raise)
    )


@pytest.mark.parametrize("module_name", CORE_PACKAGES)
def test_core_package_imports(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", _agent_submodules())
def test_agents_submodule_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_agents_submodule_discovery_is_not_empty() -> None:
    """A silently empty walk would make the parametrised test vacuous."""
    found = _agent_submodules()
    assert len(found) >= 10, found
    assert "ctra.agents.reward_fns" in found
    assert "ctra.agents.orchestrator" in found
