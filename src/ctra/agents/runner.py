"""Agent execution via subprocess for process/memory isolation.

Provides ``run_agent_as_subprocess`` — the default runner for MCTS node
evaluation.  Each call spawns ``scripts/run_agent.py`` as a child process
that creates its own ``Agent``, loads data, and returns ``AgentOutput``
via dill serialization.

Also provides:
- ``extract_objectives`` — extracts [accuracy, parsimony] from ``AgentOutput``
- ``load_feature_plans_from_json`` — reconstructs ``FeaturePlan`` from JSON
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dill
import numpy as np

from ctra.agents.data_models import (
    AgentOutput,
    FeaturePlan,
    FeatureSource,
    FeatureType,
    Task,
)
from ctra.config.settings import get_settings

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Objective extraction
# ---------------------------------------------------------------------------


def extract_objectives(
    output: AgentOutput,
    n_features: int,
    max_features: int = 50,
) -> NDArray[Any]:
    """Extract [accuracy, parsimony] objectives from an ``AgentOutput``.

    Parameters:
        output: Agent evaluation result.
        n_features: Number of features in the evaluated set.
        max_features: Maximum feature count for parsimony normalization.

    Returns:
        ndarray of shape ``(2,)`` with ``[accuracy, parsimony]``.
    """
    if output.eval_outputs:
        best_eval, _ = output.get_best_eval_output()
        accuracy = best_eval.model_eval_result.roc_auc
    else:
        accuracy = 0.0

    parsimony = max(0.0, 1.0 - n_features / max_features)
    return np.array([accuracy, parsimony])


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def run_agent_as_subprocess(
    node_id: str,
    task: Task | str,
    previous_output: AgentOutput | None,
    cache_dir: Path | None = None,
) -> AgentOutput:
    """Run a single Agent iteration as a subprocess.

    Serializes ``previous_output`` to a temp file, spawns
    ``scripts/run_agent.py``, and deserializes the result.  Results are
    cached on disk by ``node_id`` for crash recovery.

    Parameters:
        node_id: Unique identifier for this node.
        task: ``Task`` enum member or CLI arg string (e.g. ``"phase2"``).
        previous_output: ``None`` for iteration 0, or previous ``AgentOutput``.
        cache_dir: Directory for caching results.  Defaults to
            ``settings.output_dir / "agent_cache"``.

    Returns:
        The ``AgentOutput`` from the subprocess.
    """
    # Derive CLI arg from Task enum if needed
    if isinstance(task, Task):
        if task.phase is None:
            raise ValueError(
                f"Cannot run subprocess for generic task {task.name!r} — "
                f"a phase-specific task is required (e.g. Task.TRIAL_OUTCOME_PHASE_2)."
            )
        task_cli = f"phase{task.phase}"
    else:
        task_cli = task
    settings = get_settings()
    cache_dir = cache_dir or settings.output_dir / "agent_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check disk cache
    cached_path = cache_dir / f"{task_cli}--{node_id}.output.pkl"
    if cached_path.exists():
        logger.info("Node %s: loading from cache %s", node_id, cached_path)
        with open(cached_path, "rb") as f:
            return dill.load(f)  # type: ignore[no-any-return]

    # Serialize input
    with (
        tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as input_file,
        tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as output_file,
    ):
        input_path = input_file.name
        output_path = output_file.name

        if previous_output is not None:
            dill.dump(previous_output, input_file)
            input_file.flush()

    # Build command
    cmd = [
        sys.executable,
        "scripts/run_agent.py",
        "--task",
        task_cli,
        "--output",
        output_path,
    ]
    if previous_output is not None:
        cmd.extend(["--input", input_path])

    logger.info("Node %s: spawning subprocess: %s", node_id, " ".join(cmd))
    start = time.monotonic()

    timeout = settings.mcts.subprocess_timeout
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Node %s: subprocess failed: %s", node_id, e)
        raise

    # Deserialize output
    with open(output_path, "rb") as f:
        output: AgentOutput = dill.load(f)

    # Cache for crash recovery
    with open(cached_path, "wb") as f:
        dill.dump(output, f)

    elapsed = time.monotonic() - start
    logger.info("Node %s: subprocess completed in %.1fs", node_id, elapsed)

    # Clean up temp files
    Path(input_path).unlink(missing_ok=True)
    Path(output_path).unlink(missing_ok=True)

    return output


# ---------------------------------------------------------------------------
# Feature plan serialization
# ---------------------------------------------------------------------------


def load_feature_plans_from_json(path: Path) -> dict[str, FeaturePlan]:
    """Load feature plans from a JSON file saved by ``dump_as_json``.

    Reconstructs ``FeaturePlan`` NamedTuples from the dict representation
    produced by ``_asdict()``.

    Parameters:
        path: Path to the JSON file.

    Returns:
        Dictionary of feature plans keyed by feature name.
    """
    with open(path) as f:
        raw = json.load(f)

    plans: dict[str, FeaturePlan] = {}
    for name, data in raw.items():
        plans[name] = FeaturePlan(
            feature_name=data["feature_name"],
            feature_idea=data["feature_idea"],
            feature_type={k: FeatureType(v) for k, v in data["feature_type"].items()},
            data_sources=[FeatureSource(s) for s in data["data_sources"]],
            example_values=data["example_values"],
            possible_values=data["possible_values"],
            feature_instructions=data["feature_instructions"],
        )
    return plans
