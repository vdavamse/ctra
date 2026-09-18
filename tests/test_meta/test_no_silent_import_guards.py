"""Regression fence for issue #9: no test module may guard a required import.

``dspy`` and ``shapiq`` are **required** dependencies (``pyproject.toml``
``[project] dependencies``).  Wrapping their import -- or any first-party
``ctra.*`` import -- in ``try: ... except ImportError`` and skipping the module
turns real breakage into a green run.  That is how issue #2 shipped.

Sanctioned alternative for a genuinely optional third-party dependency:
``pytest.importorskip("datasets")`` inside the test (see
``tests/test_rag/test_ner_eval.py``).  It names the dependency, it cannot go
stale, and it cannot be defeated by a partially-imported package.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Reason strings from the guards issue #9 removed; they must not come back.
STALE_SKIP_REASONS = (
    "dspy/sqlite3 not available",
    "dspy not available",
    "dependencies not available",
)


def _test_sources() -> list[Path]:
    return sorted(p for p in TESTS_ROOT.rglob("*.py") if p.resolve() != SELF)


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    node = handler.type
    names: list[str] = []
    if isinstance(node, ast.Name):
        names = [node.id]
    elif isinstance(node, ast.Tuple):
        names = [e.id for e in node.elts if isinstance(e, ast.Name)]
    return any(n in {"ImportError", "ModuleNotFoundError"} for n in names)


def _module_level_violations(path: Path, tree: ast.Module) -> list[str]:
    out: list[str] = []
    stack: list[ast.stmt] = list(tree.body)
    while stack:
        stmt = stack.pop()
        # Module level includes bodies of module-level if/try, but NOT function
        # or class bodies -- a guard inside a test function is loud by nature.
        if isinstance(stmt, ast.Try):
            if any(_catches_import_error(h) for h in stmt.handlers):
                out.append(
                    f"{path}:{stmt.lineno}: module-level 'try/except ImportError' import guard"
                )
            stack.extend(stmt.body + stmt.orelse + stmt.finalbody)
            for h in stmt.handlers:
                stack.extend(h.body)
        elif isinstance(stmt, ast.If):
            stack.extend(stmt.body + stmt.orelse)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id.startswith("_HAS_"):
                    out.append(
                        f"{path}:{stmt.lineno}: module-level '{t.id}' availability flag"
                    )
    return out


def test_no_module_level_import_guards_in_tests() -> None:
    sources = _test_sources()
    assert len(sources) > 50, f"scan found only {len(sources)} test sources -- glob broken?"

    violations: list[str] = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        violations.extend(_module_level_violations(path, ast.parse(text)))
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
