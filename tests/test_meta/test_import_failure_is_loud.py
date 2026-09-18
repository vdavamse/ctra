"""Test-of-the-tests: a broken first-party module must turn the suite RED.

Issue #9: ``try: import ... except ImportError: _HAS_DSPY = False`` +
``pytestmark = skipif(...)`` converted first-party breakage into silent skips,
which is how issue #2 shipped green.  These tests run pytest in a **subprocess**
with ``ctra.agents.reward_fns`` made unimportable and assert the run fails and
*names the module*.

A subprocess is required: the blocker must be installed before ``ctra.agents``
is imported, and this parent process has already imported it.
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


def _run_pytest(blocker_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(blocker_dir), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = ""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "ctra_block_reward_fns",
            "-p",
            "no:cacheprovider",
            "-q",
            "--no-header",
            *args,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_blocked_first_party_module_fails_the_agent_tests(blocker_dir: Path) -> None:
    """Collecting tests/test_agents must fail and name the blocked module."""
    proc = _run_pytest(blocker_dir, "--collect-only", "tests/test_agents")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"suite stayed green with {BLOCKED_MODULE} blocked:\n{output}"
    assert BLOCKED_MODULE in output, f"failure never named {BLOCKED_MODULE}:\n{output}"
    assert " skipped" not in output, f"breakage was converted into skips:\n{output}"


def test_blocked_first_party_module_fails_the_module_that_imports_it(blocker_dir: Path) -> None:
    """Running only the module that imports the blocked name must fail, not skip.

    This is the per-file false green of issue #9: ``pytestmark = skipif(not
    _HAS_DSPY)`` made this exact invocation exit 0 with every test skipped.
    """
    proc = _run_pytest(blocker_dir, "tests/test_agents/test_reward_fns.py")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"module run stayed green with {BLOCKED_MODULE} blocked:\n{output}"
    assert BLOCKED_MODULE in output, f"failure never named {BLOCKED_MODULE}:\n{output}"
    assert " skipped" not in output, f"breakage was converted into skips:\n{output}"


def test_blocked_first_party_module_fails_the_import_smoke_test(blocker_dir: Path) -> None:
    """The ungated smoke test is the tripwire that needs no other test to exist."""
    proc = _run_pytest(blocker_dir, "tests/test_import_smoke.py")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"smoke test stayed green:\n{output}"
    assert BLOCKED_MODULE in output, f"failure never named {BLOCKED_MODULE}:\n{output}"


def test_control_run_without_blocker_is_green() -> None:
    """Positive control: the same command passes when nothing is blocked."""
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--no-header",
         "tests/test_import_smoke.py"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
