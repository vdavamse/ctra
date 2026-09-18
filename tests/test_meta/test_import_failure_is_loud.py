"""Test-of-the-tests: a broken first-party module must turn the suite RED.

Issue #9: ``try: import ... except ImportError: _HAS_DSPY = False`` +
``pytestmark = skipif(...)`` converted first-party breakage into silent skips,
which is how issue #2 shipped green.  These tests run pytest in a **subprocess**
with ``ctra.agents.reward_fns`` made unimportable and assert the run fails and
*names the module*.

A subprocess is required: the blocker must be installed before ``ctra.agents``
is imported, and this parent process has already imported it.

Every test here spawns a full pytest run, so all are marked ``slow``.  They
still run by default; deselect with ``-m "not slow"``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
assert (REPO_ROOT / "pyproject.toml").is_file(), REPO_ROOT
BLOCKED_MODULE = "ctra.agents.reward_fns"
AGENTS_CONFTEST = "tests/test_agents/conftest.py"

_PLUGIN_SOURCE = f'''\
"""pytest plugin: make {BLOCKED_MODULE} unimportable (test fixture for issue #9)."""
import sys

BLOCKED = {BLOCKED_MODULE!r}


class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == BLOCKED or fullname.startswith(BLOCKED + "."):
            raise ModuleNotFoundError(f"No module named {{BLOCKED!r}}", name=BLOCKED)
        return None


sys.modules.pop(BLOCKED, None)
sys.meta_path.insert(0, _Blocker())
'''


@pytest.fixture(scope="module")
def blocker_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding the meta-path blocker, loaded via ``-p``."""
    d = tmp_path_factory.mktemp("ctra_blocker")
    (d / "ctra_block_reward_fns.py").write_text(_PLUGIN_SOURCE, encoding="utf-8")
    return d


def _run_pytest(
    blocker_dir: Path, *args: str, blocked: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run pytest in a subprocess; ``blocked`` adds ``-p ctra_block_reward_fns``.

    The environment is built identically on both paths, so the control run
    differs from the blocked runs only by that one ``-p`` flag.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(blocker_dir), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    env.pop("PYTEST_ADDOPTS", None)
    # Inherit normal plugin autoload (pytest-asyncio etc.); the blocker is loaded by -p.
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    # This repo tracks __pycache__/*.pyc; the subprocess must not dirty the tree.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    plugin = ["-p", "ctra_block_reward_fns"] if blocked else []
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *plugin,
            "-p",
            "no:cacheprovider",
            "-q",
            "--no-header",
            *args,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )


@pytest.mark.slow
def test_blocked_first_party_module_fails_collection_of_the_agent_tests(
    blocker_dir: Path,
) -> None:
    """Collecting tests/test_agents must fail and name the blocked module.

    Today ``tests/test_agents/conftest.py`` imports ``ctra`` and raises before
    any module is collected, so this overlaps with the conftest test below.  It
    is insurance for a future conftest without first-party imports: then every
    collected module that imports the blocked name must fail on its own.
    """
    proc = _run_pytest(blocker_dir, "--collect-only", "tests/test_agents")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"suite stayed green with {BLOCKED_MODULE} blocked:\n{output}"
    assert BLOCKED_MODULE in output, f"failure never named {BLOCKED_MODULE}:\n{output}"
    assert " skipped" not in output, f"breakage was converted into skips:\n{output}"


@pytest.mark.slow
def test_blocked_first_party_module_fails_in_the_agents_conftest(blocker_dir: Path) -> None:
    """A single-module run must fail in the agents conftest, not skip.

    ``pytest tests/test_agents/test_reward_fns.py`` is the per-file invocation
    that issue #9 turned into a false green (exit 0, every test skipped).  It is
    now loud because ``tests/test_agents/conftest.py`` imports ``ctra`` first
    and raises the real ``ModuleNotFoundError`` before the module is collected.
    """
    proc = _run_pytest(blocker_dir, "tests/test_agents/test_reward_fns.py")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"module run stayed green with {BLOCKED_MODULE} blocked:\n{output}"
    assert BLOCKED_MODULE in output, f"failure never named {BLOCKED_MODULE}:\n{output}"
    assert AGENTS_CONFTEST in output, f"failure was not raised by {AGENTS_CONFTEST}:\n{output}"
    assert " skipped" not in output, f"breakage was converted into skips:\n{output}"


@pytest.mark.slow
def test_blocked_first_party_module_fails_the_import_smoke_test(blocker_dir: Path) -> None:
    """The ungated smoke test is the tripwire that needs no other test to exist."""
    proc = _run_pytest(blocker_dir, "tests/test_import_smoke.py")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"smoke test stayed green:\n{output}"
    assert BLOCKED_MODULE in output, f"failure never named {BLOCKED_MODULE}:\n{output}"


@pytest.mark.slow
def test_control_run_without_blocker_is_green(blocker_dir: Path) -> None:
    """Positive control: the same command and environment pass when nothing is blocked."""
    proc = _run_pytest(blocker_dir, "tests/test_import_smoke.py", blocked=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
