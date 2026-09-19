"""Global per-feature value store for cross-branch reuse in MCTS.

Layout on disk:

    {store_dir}/{task_namespace}/{feature_name}--{plan_hash}/{nctid}.json

SAFETY INVARIANT (from research/feature-extraction-optimization.md §6.3.1):
same plan = same values. Two plans that produce different feature values
MUST hash differently. We err on the side of cache MISSES rather than
false hits: the canonical hash INCLUDES `feature_idea` (which carries the
REFINE chain "original\\n---\\nrefinement"), `feature_instructions`,
`possible_values`, `feature_type`, and `data_sources`. It EXCLUDES only
`example_values` (illustrative, not definitional).

Atomicity: writes go to a sibling `*.tmp.<pid>.<rnd>` file then os.replace
onto the final path. No file locks — idempotent writes under the safety
invariant make racing writers write identical content.

Schema: every stored JSON carries `schema_version: 1`. Future breaking
changes bump the version; mismatched-version entries are treated as misses.

Trust boundary: this module calls ``dill.loads`` on the stored payload,
which is equivalent to ``pickle.loads`` and is a remote code execution
primitive if a store file is adversarially controlled. The store directory
is therefore a TRUST BOUNDARY. Only the CTRA pipeline writes to it, and it
must live under a path the pipeline owns (e.g. ``output/feature_store/``
on the process host). Never point ``feature_store_dir`` at a directory
writable by untrusted users, and never mount shared stores across security
domains. A future PR may migrate to a JSON-native representation to
eliminate this primitive entirely.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import dill

from ctra.agents.data_models import FeaturePlan  # noqa: TC001

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _sort_nested(d: dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Sort a dict of lists for canonical hashing. Empty/None -> {}."""
    if not d:
        return {}
    return {k: sorted(v) for k, v in sorted(d.items())}


def plan_content_hash(plan: FeaturePlan) -> str:
    """Canonical 16-char SHA-256 hex digest of a FeaturePlan's identity.

    Included fields (definitional):
        - feature_name
        - feature_idea (carries REFINE chain)
        - feature_instructions (builder directive)
        - feature_type (field-order independent)
        - data_sources (order independent)
        - possible_values (constrains categorical output)

    Excluded fields (illustrative / non-definitional):
        - example_values

    Returns:
        A 16-character lowercase hex string. 64-bit key space; effective
        collision domain is 2^64 within the (task, feature_name, nctid)
        namespace, which keeps collisions astronomically unlikely.
    """
    canonical = {
        "feature_name": plan.feature_name,
        "feature_idea": plan.feature_idea.strip(),
        "feature_instructions": plan.feature_instructions.strip(),
        "feature_type": {
            k: (v.value if hasattr(v, "value") else str(v))
            for k, v in sorted(plan.feature_type.items())
        },
        "data_sources": sorted(
            [s.value if hasattr(s, "value") else str(s) for s in plan.data_sources]
        ),
        "possible_values": _sort_nested(plan.possible_values),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# Backwards-compatible alias: ``plan_content_hash`` is the public name (the
# canonical FeaturePlan digest, reusable outside this module); the underscored
# name is kept for existing importers and tests.
_plan_content_hash = plan_content_hash


def _store_path(
    store_dir: Path,
    task_namespace: str,
    feature_name: str,
    plan_hash: str,
    nctid: str,
) -> Path:
    """Resolve the on-disk path for a single stored feature value."""
    return Path(store_dir) / task_namespace / f"{feature_name}--{plan_hash}" / f"{nctid}.json"


def get_cached_feature(
    store_dir: Path,
    task_namespace: str,
    nctid: str,
    feature_name: str,
    plan: FeaturePlan,
) -> dict[str, Any] | None:
    """Return the cached feature values dict for a single (nctid, plan) pair.

    Returns None on miss or on schema-version mismatch. A malformed JSON
    entry is logged at WARNING and treated as a miss (does not crash).
    """
    plan_hash = plan_content_hash(plan)
    path = _store_path(store_dir, task_namespace, feature_name, plan_hash, nctid)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Corrupt feature store entry %s: %s", path, e)
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    try:
        return dill.loads(bytes.fromhex(data["feature_values_b"]))  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("Failed to decode feature values at %s: %s", path, e)
        return None


def put_cached_feature(
    store_dir: Path,
    task_namespace: str,
    nctid: str,
    feature_name: str,
    plan: FeaturePlan,
    feature_values: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a single built feature value.

    Atomic: writes to a `*.tmp.<pid>.<rnd>` sibling file, then os.replace.
    Safe under concurrent writers because the safety invariant guarantees
    identical content for the same key.
    """
    plan_hash = plan_content_hash(plan)
    path = _store_path(store_dir, task_namespace, feature_name, plan_hash, nctid)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only the dill-hex copy is read back (see get_cached_feature). A raw
    # JSON copy would crash the put for common value types (np.nan,
    # np.float64, datetime, ...) since json.dump has no default encoder for
    # them, and the resulting "silent LLM failure" (caught upstream by the
    # wrapper's blanket except) would be indistinguishable from a real
    # builder error. Keep a single source of truth.
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_name": feature_name,
        "nctid": nctid,
        "plan_hash": plan_hash,
        "feature_values_b": dill.dumps(feature_values).hex(),
        "metadata": metadata or {},
    }
    fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def get_cached_features_batch(
    store_dir: Path,
    task_namespace: str,
    nctids: list[str],
    feature_name: str,
    plan: FeaturePlan,
) -> dict[str, dict[str, Any]]:
    """Bulk probe: returns {nctid: feature_values} for all nctids that hit."""
    plan_hash = plan_content_hash(plan)
    results: dict[str, dict[str, Any]] = {}
    for nctid in nctids:
        path = _store_path(store_dir, task_namespace, feature_name, plan_hash, nctid)
        if path.exists():
            cached = get_cached_feature(store_dir, task_namespace, nctid, feature_name, plan)
            if cached is not None:
                results[nctid] = cached
    return results


def get_store_stats(store_dir: Path, task_namespace: str | None = None) -> dict[str, int]:
    """Return counts of features, trials, and total entries for monitoring.

    If `task_namespace` is None, aggregates across all namespaces under
    `store_dir`.
    """
    root = Path(store_dir) / task_namespace if task_namespace else Path(store_dir)
    if not root.exists():
        return {"features": 0, "trials": 0, "total_entries": 0}
    features: set[str] = set()
    trials: set[str] = set()
    total = 0
    iter_roots = [root] if task_namespace else [p for p in root.iterdir() if p.is_dir()]
    for ns_root in iter_roots:
        if not ns_root.is_dir():
            continue
        for feature_dir in ns_root.iterdir():
            if not feature_dir.is_dir():
                continue
            name = feature_dir.name.rsplit("--", 1)[0]
            features.add(name)
            for f in feature_dir.glob("*.json"):
                total += 1
                trials.add(f.stem)
    return {"features": len(features), "trials": len(trials), "total_entries": total}
