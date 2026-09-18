"""Regression fence for issue #9: no test module may guard a required import.

``dspy`` and ``shapiq`` are **required** dependencies (``pyproject.toml``
``[project] dependencies``).  Wrapping their import -- or any first-party
``ctra.*`` import -- in ``try: ... except ImportError`` and skipping the module
turns real breakage into a green run.  That is how issue #2 shipped.

Sanctioned alternative for a genuinely optional third-party dependency:
``pytest.importorskip("datasets")`` inside the test (see
``tests/test_rag/test_ner_eval.py``).  It names the dependency, it cannot go
stale, and it cannot be defeated by a partially-imported package.

Caught at module scope and inside class bodies (function bodies are exempt: a
guard inside a test is loud by nature, see ``test_dspy_api_surface.py``):

* ``try`` with an ``except ImportError`` / ``except ModuleNotFoundError``
  handler, named directly, in a tuple, or as an attribute
  (``builtins.ImportError``);
* ``try`` whose body contains an import and whose handler is ``except
  Exception``, ``except BaseException`` or a bare ``except:``;
* ``with contextlib.suppress(ImportError)`` / ``with suppress(ModuleNotFoundError)``;
* ``_HAS_*`` availability flags, including tuple targets;
* the stale skip-reason strings of the removed guards.

Caught anywhere in the file, function bodies included:

* ``pytest.importorskip("dspy")`` / ``pytest.importorskip("shapiq")`` -- these
  are required dependencies, so skipping on them is the same false green.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError"})
BROAD_EXCEPTIONS = frozenset({"Exception", "BaseException"})
REQUIRED_DEPS = frozenset({"dspy", "shapiq"})

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
    """Describe the guard a ``try`` statement is, or ``None``."""
    for handler in stmt.handlers:
        names = _exception_names(handler.type)
        if names & IMPORT_ERRORS:
            return "'try/except ImportError' import guard"
        broad = handler.type is None or bool(names & BROAD_EXCEPTIONS)
        if broad and _contains_import(stmt.body):
            return "broad 'except' around an import"
    return None


def _with_guard(stmt: ast.With) -> str | None:
    for item in stmt.items:
        call = item.context_expr
        if (
            isinstance(call, ast.Call)
            and _callee_name(call) == "suppress"
            and any(_exception_names(arg) & IMPORT_ERRORS for arg in call.args)
        ):
            return "'suppress(ImportError)' import guard"
    return None


def _availability_flags(stmt: ast.Assign | ast.AnnAssign) -> list[str]:
    """``_HAS_*`` names assigned, including inside tuple targets."""
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    return [
        node.id
        for target in targets
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and node.id.startswith("_HAS_")
    ]


def _scope_violations(path: Path, tree: ast.Module) -> list[str]:
    """Guards at module scope or in class bodies; function bodies are not entered."""
    out: list[str] = []
    stack: list[ast.stmt] = list(tree.body)
    while stack:
        stmt = stack.pop()
        if isinstance(stmt, ast.Try):
            if (what := _try_guard(stmt)) is not None:
                out.append(f"{path}:{stmt.lineno}: module-level {what}")
            stack.extend(stmt.body + stmt.orelse + stmt.finalbody)
            for handler in stmt.handlers:
                stack.extend(handler.body)
        elif isinstance(stmt, ast.With):
            if (what := _with_guard(stmt)) is not None:
                out.append(f"{path}:{stmt.lineno}: module-level {what}")
            stack.extend(stmt.body)
        elif isinstance(stmt, ast.If):
            stack.extend(stmt.body + stmt.orelse)
        elif isinstance(stmt, ast.ClassDef):
            stack.extend(stmt.body)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            for name in _availability_flags(stmt):
                out.append(f"{path}:{stmt.lineno}: module-level '{name}' availability flag")
    return out


def _importorskip_violations(path: Path, tree: ast.Module) -> list[str]:
    """``importorskip`` on a required dependency, anywhere in the file."""
    out: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _callee_name(node) == "importorskip"):
            continue
        first = node.args[0] if node.args else None
        for kw in node.keywords:
            if kw.arg == "modname":
                first = kw.value
        if isinstance(first, ast.Constant) and first.value in REQUIRED_DEPS:
            out.append(
                f"{path}:{node.lineno}: importorskip({first.value!r}) on a required dependency"
            )
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
        for reason in STALE_SKIP_REASONS:
            if reason in text:
                violations.append(f"{path}: stale skip reason {reason!r}")

    assert not violations, (
        "Import guards re-introduced in tests/ (issue #9).\n"
        "dspy and shapiq are REQUIRED dependencies; a failed first-party import "
        "must turn the suite RED, not skip it.\n"
        "For a genuinely optional dependency use pytest.importorskip(...) inside "
        "the test instead.\n\n" + "\n".join(sorted(violations))
    )
