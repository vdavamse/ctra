"""Test-of-the-tests: a broken first-party module must turn the suite RED.

Issue #9: ``try: import ... except ImportError: _HAS_DSPY = False`` +
``pytestmark = skipif(...)`` converted first-party breakage into silent skips,
which is how issue #2 shipped green.  These tests run pytest in a **subprocess**
with ``ctra.agents.reward_fns`` made unimportable and assert the run fails and
*names the module*.

A subprocess is required: the blocker must be installed before ``ctra.agents``
is imported, and this parent process has already imported it.

Each subprocess run happens once per module through a module-scoped fixture
(``blocked_agents_run``, ``blocked_smoke_run``, ``control_run``); the tests
assert on the shared ``CompletedProcess``.  All are marked ``slow`` and still
run by default; deselect with ``-m "not slow"``.
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


@pytest.fixture(scope="module")
def blocked_agents_run(blocker_dir: Path) -> subprocess.CompletedProcess[str]:
    """One blocked ``pytest tests/test_agents/test_reward_fns.py`` run.

    This is the per-file invocation issue #9 turned into a false green (exit
    0, every test skipped).  It is a real run, not ``--collect-only``: skips
    are not reported under ``--collect-only -q``, so the ``skipped`` assertion
    would be vacuous.  Today ``tests/test_agents/conftest.py`` raises on
    import, so the run ends before collection; with a future conftest that
    has no first-party imports it becomes a real run of the one module that
    imports the blocked name, and that module must then fail on its own.  The
    whole directory is not run because it holds a legitimate skip
    (``test_orchestrator.py``, TabPFN without CUDA) that would trip the
    assertion in that scenario.
    """
    return _run_pytest(blocker_dir, "tests/test_agents/test_reward_fns.py")


@pytest.fixture(scope="module")
def blocked_smoke_run(blocker_dir: Path) -> subprocess.CompletedProcess[str]:
    """One blocked ``pytest tests/test_import_smoke.py`` run."""
    return _run_pytest(blocker_dir, "tests/test_import_smoke.py")


@pytest.fixture(scope="module")
def control_run(blocker_dir: Path) -> subprocess.CompletedProcess[str]:
    """The smoke run again with nothing blocked -- same command and environment."""
    return _run_pytest(blocker_dir, "tests/test_import_smoke.py", blocked=False)


@pytest.mark.slow
def test_blocked_first_party_module_fails_the_agent_tests(
    blocked_agents_run: subprocess.CompletedProcess[str],
) -> None:
    """The issue #9 invocation must fail, name the blocked module and skip nothing."""
    proc = blocked_agents_run
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"run stayed green with {BLOCKED_MODULE} blocked:\n{output}"
    assert BLOCKED_MODULE in output, f"failure never named {BLOCKED_MODULE}:\n{output}"
    assert " skipped" not in output, f"breakage was converted into skips:\n{output}"


@pytest.mark.slow
def test_blocked_first_party_module_fails_while_loading_a_conftest(
    blocked_agents_run: subprocess.CompletedProcess[str],
) -> None:
    """The failure must be raised while a conftest loads, before any module is collected.

    The per-file run is loud because ``tests/test_agents/conftest.py`` imports
    ``ctra`` first and raises the real ``ModuleNotFoundError``.  The assertion
    is on pytest's message, not the conftest path, so a root
    ``tests/conftest.py`` that imports ``ctra`` would satisfy it too.
    """
    proc = blocked_agents_run
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"run stayed green with {BLOCKED_MODULE} blocked:\n{output}"
    assert "ImportError while loading conftest" in output, (
        f"failure was not raised while loading a conftest:\n{output}"
    )


@pytest.mark.slow
def test_blocked_first_party_module_fails_the_import_smoke_test(
    blocked_smoke_run: subprocess.CompletedProcess[str],
) -> None:
    """The ungated smoke test is the tripwire that needs no other test to exist."""
    proc = blocked_smoke_run
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"smoke test stayed green:\n{output}"
    assert BLOCKED_MODULE in output, f"failure never named {BLOCKED_MODULE}:\n{output}"


@pytest.mark.slow
def test_control_run_without_blocker_is_green(
    control_run: subprocess.CompletedProcess[str],
) -> None:
    """Positive control: the same command and environment pass when nothing is blocked."""
    assert control_run.returncode == 0, control_run.stdout + control_run.stderr
