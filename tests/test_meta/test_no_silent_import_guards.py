"""Regression fence for issue #9: no test module may guard a required import.

Every ``[project] dependencies`` entry in ``pyproject.toml`` (``dspy``,
``shapiq``, ``xgboost``, ...) is **required**.  Wrapping its import -- or any
first-party ``ctra.*`` import -- in ``try: ... except ImportError`` and
skipping turns real breakage into a green run.  That is how issue #2 shipped.

Sanctioned alternative for a genuinely optional third-party dependency:
``pytest.importorskip("datasets")`` inside the test (see
``tests/test_rag/test_ner_eval.py``).  It names the dependency, it cannot go
stale, and it cannot be defeated by a partially-imported package.

Caught at module scope and inside class bodies, including when nested in
``if`` / ``for`` / ``while`` / ``with`` / ``try`` blocks (function bodies are
not entered by this walker):

* ``try`` (or ``try``/``except*``) whose body contains an import and whose
  handler is ``except ImportError`` / ``except ModuleNotFoundError``, named
  directly, in a tuple, or as an attribute (``builtins.ImportError``);
* ``try`` whose body contains an import and whose handler is ``except
  Exception``, ``except BaseException`` or a bare ``except:``;
* ``with contextlib.suppress(ImportError)`` / ``with suppress(ModuleNotFoundError)``;
* the stale skip-reason strings of the removed guards.

Caught anywhere in the file, function bodies included:

* ``pytest.importorskip(...)`` on a required or first-party module, matched on
  the top-level package: ``"ctra.agents.reward_fns"``, ``"dspy.utils.dummies"``
  and ``"xgboost"`` are all rejected;
* an ``except ImportError`` / ``except ModuleNotFoundError`` handler that calls
  ``pytest.skip`` or ``pytest.importorskip``.  In a fixture or helper this
  skips every test that uses it, which is the same false green.  A handler
  that ``pytest.fail``s is loud and stays allowed (see
  ``test_dspy_api_surface.py``).
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError"})
BROAD_EXCEPTIONS = frozenset({"Exception", "BaseException"})
SKIP_CALLS = frozenset({"skip", "importorskip"})

# Top-level import names of every ``[project] dependencies`` entry in
# ``pyproject.toml``, plus the first-party package.  Distribution names that
# differ from their import name: dspy-ai -> dspy, scikit-learn -> sklearn,
# pyyaml -> yaml, pydantic-settings -> pydantic_settings.  Keep in sync with
# ``pyproject.toml`` when a required dependency is added or removed.
REQUIRED_DEPS = frozenset(
    {
        "ctra",  # first-party
        "dspy",
        "xgboost",
        "shap",
        "shapiq",
        "sklearn",
        "polars",
        "pandas",
        "pyarrow",
        "numpy",
        "scipy",
        "mlflow",
        "joblib",
        "pydantic",
        "pydantic_settings",
        "httpx",
        "tenacity",
        "tqdm",
        "yaml",
        "diskcache",
        "pubchempy",
        "dill",
    }
)

# ``except*`` (Python 3.11+) has the same shape as ``try`` and guards the same way.
_TRY_TYPES: tuple[type[ast.stmt], ...] = (ast.Try,) + (
    (ast.TryStar,) if hasattr(ast, "TryStar") else ()
)

# Reason strings from the guards issue #9 removed; they must not come back.
STALE_SKIP_REASONS = (
    "dspy/sqlite3 not available",
    "dspy not available",
    "dependencies not available",
)


def _test_sources() -> list[Path]:
    return sorted(p for p in TESTS_ROOT.rglob("*.py") if p.resolve() != SELF)


def _exception_names(node: ast.expr | None) -> set[str]:
    """Exception names in a handler type or ``suppress`` argument.

    ``ImportError`` -> {"ImportError"}; ``builtins.ImportError`` -> {"ImportError"};
    ``(OSError, ImportError)`` -> {"OSError", "ImportError"}.
    """
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Tuple):
        return set().union(*(_exception_names(e) for e in node.elts))
    return set()


def _callee_name(call: ast.Call) -> str | None:
    """``suppress(...)`` and ``contextlib.suppress(...)`` both give "suppress"."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _contains_import(stmts: list[ast.stmt]) -> bool:
    return any(
        isinstance(node, (ast.Import, ast.ImportFrom)) for s in stmts for node in ast.walk(s)
    )


def _try_guard(stmt: ast.Try) -> str | None:
    """Describe the guard a ``try`` (or ``try``/``except*``) statement is, or ``None``."""
    if not _contains_import(stmt.body):
        return None
    for handler in stmt.handlers:
        names = _exception_names(handler.type)
        if names & IMPORT_ERRORS:
            return "'try/except ImportError' import guard"
        if handler.type is None or names & BROAD_EXCEPTIONS:
            return "broad 'except' around an import"
    return None


def _with_guard(stmt: ast.With | ast.AsyncWith) -> str | None:
    for item in stmt.items:
        call = item.context_expr
        if (
            isinstance(call, ast.Call)
            and _callee_name(call) == "suppress"
            and any(_exception_names(arg) & IMPORT_ERRORS for arg in call.args)
        ):
            return "'suppress(ImportError)' import guard"
    return None


def _scope_violations(path: Path, tree: ast.Module) -> list[str]:
    """Guards at module scope or in class bodies; function bodies are not entered.

    Compound statements (``if``/``for``/``while``/``with``/``try`` and their
    async forms) are descended so nesting a guard in one does not hide it.
    Each finding names the scope it was found in.
    """
    out: list[str] = []
    stack: list[tuple[ast.stmt, str]] = [(s, "module-level") for s in tree.body]
    while stack:
        stmt, scope = stack.pop()
        if isinstance(stmt, _TRY_TYPES):
            if (what := _try_guard(stmt)) is not None:
                out.append(f"{path}:{stmt.lineno}: {scope} {what}")
            children = stmt.body + stmt.orelse + stmt.finalbody
            for handler in stmt.handlers:
                children = children + handler.body
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            if (what := _with_guard(stmt)) is not None:
                out.append(f"{path}:{stmt.lineno}: {scope} {what}")
            children = stmt.body
        elif isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            children = stmt.body + stmt.orelse
        elif isinstance(stmt, ast.ClassDef):
            stack.extend((s, f"class-level (in class {stmt.name})") for s in stmt.body)
            continue
        else:
            continue
        stack.extend((s, scope) for s in children)
    return out


def _importorskip_violations(path: Path, tree: ast.Module) -> list[str]:
    """``importorskip`` on a required or first-party module, anywhere in the file.

    Matched on the top-level package, so submodules (``"ctra.agents.reward_fns"``,
    ``"dspy.utils.dummies"``) are rejected along with the package itself.
    """
    out: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _callee_name(node) == "importorskip"):
            continue
        first = node.args[0] if node.args else None
        for kw in node.keywords:
            if kw.arg == "modname":
                first = kw.value
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        if first.value.split(".")[0] in REQUIRED_DEPS:
            out.append(
                f"{path}:{node.lineno}: importorskip({first.value!r}) "
                "on a required/first-party module"
            )
    return out


def _skipping_handler_violations(path: Path, tree: ast.Module) -> list[str]:
    """``except ImportError`` handlers that skip instead of failing, at any depth.

    Function bodies included: an autouse fixture or a helper that catches
    ``ImportError`` and calls ``pytest.skip`` skips every test that uses it.
    A handler that ``pytest.fail``s is loud and is not reported.
    """
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _exception_names(node.type) & IMPORT_ERRORS:
            continue
        for call in (n for s in node.body for n in ast.walk(s) if isinstance(n, ast.Call)):
            if _callee_name(call) in SKIP_CALLS:
                out.append(
                    f"{path}:{call.lineno}: 'except ImportError' handler calls "
                    f"{_callee_name(call)}() instead of failing"
                )
                break
    return out


def test_no_module_level_import_guards_in_tests() -> None:
    sources = _test_sources()
    assert len(sources) > 50, f"scan found only {len(sources)} test sources -- glob broken?"

    violations: list[str] = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        violations.extend(_scope_violations(path, tree))
        violations.extend(_importorskip_violations(path, tree))
        violations.extend(_skipping_handler_violations(path, tree))
        for reason in STALE_SKIP_REASONS:
            if reason in text:
                violations.append(f"{path}: stale skip reason {reason!r}")

    assert not violations, (
        "Import guards re-introduced in tests/ (issue #9).\n"
        "Every [project] dependency and every ctra.* module is REQUIRED; a failed "
        "import must turn the suite RED, not skip it.\n"
        "For a genuinely optional dependency use pytest.importorskip(...) inside "
        "the test instead.\n\n" + "\n".join(sorted(violations))
    )
