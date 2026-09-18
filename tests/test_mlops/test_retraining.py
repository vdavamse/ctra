"""Tests for the retraining pipeline's MCTS wiring.

``RetrainingPipeline.retrain`` builds a remove-one-feature ``expand_fn`` and
hands it to ``MCTSSearch``.  Issue #7 caps ``_expand``'s candidate list at the
parent's suggestion count, which must be a no-op for this pipeline: its runner
returns an ``ObjectiveResult`` (no suggestion list), so the branch factor is
whatever the closure produces.  The test binds to the real closure by
capturing the ``expand_fn`` kwarg at construction time.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from ctra.config.settings import MCTSConfig
from ctra.mlops.retraining import RetrainingPipeline
from ctra.search.mcts import MCTSNode

if TYPE_CHECKING:
    from pathlib import Path


class _StopSearchError(Exception):
    """Sentinel raised by the recorder's ``search()`` to end ``retrain`` early."""


def _write_splits(splits_dir: Path, features: list[str]) -> None:
    rng = np.random.default_rng(0)
    rows = 8
    x = pl.DataFrame({f: rng.normal(size=rows) for f in features})
    y = pl.DataFrame({"label": [0, 1] * (rows // 2)})
    x.write_parquet(splits_dir / "X_train.parquet")
    y.write_parquet(splits_dir / "y_train.parquet")
    x.write_parquet(splits_dir / "X_val.parquet")
    y.write_parquet(splits_dir / "y_val.parquet")
    plans = [{"name": f, "dtype": "float", "description": f"{f} desc"} for f in features]
    (splits_dir / "feature_plans.json").write_text(json.dumps(plans))


def test_expand_fn_branch_factor_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The remove-one-feature expander still yields ``max_children`` candidates."""
    features = [f"f{i}" for i in range(6)]
    splits_dir = tmp_path / "retrain_task"
    splits_dir.mkdir()
    _write_splits(splits_dir, features)

    captured: dict[str, Any] = {}

    class _RecordingSearch:
        def __init__(self, runner: Any, task: Any, **kwargs: Any) -> None:
            captured["runner"] = runner
            captured["expand_fn"] = kwargs["expand_fn"]

        def search(self, initial_features: list[str]) -> MCTSNode:
            captured["initial_features"] = list(initial_features)
            raise _StopSearchError

    import ctra.search.mcts as mcts_module

    monkeypatch.setattr(mcts_module, "MCTSSearch", _RecordingSearch)

    tracker = MagicMock()
    tracker.start_mcts_run.return_value = "run-1"
    pipeline = RetrainingPipeline(tracker=tracker, store=MagicMock())
    config = MCTSConfig(
        num_rollouts=1,
        objectives=["accuracy", "parsimony"],
        reference_point=[0.0, 0.0],
        min_branch_factor=4,
        max_branch_factor=4,
    )

    with pytest.raises(_StopSearchError):
        pipeline.retrain(str(splits_dir), mcts_config=config)

    tracker.end_run.assert_called_once()
    assert captured["initial_features"] == features

    expand_fn = captured["expand_fn"]
    node = MCTSNode(features=list(features), total_reward=np.zeros(2))
    candidates = expand_fn(node, max_children=4)

    assert len(candidates) == 4
    assert [op for _feats, op, _detail in candidates] == ["remove"] * 4
    assert [detail for _feats, _op, detail in candidates] == [f"remove:{f}" for f in features[:4]]
    for (child_features, _op, detail), removed in zip(candidates, features[:4], strict=True):
        assert removed not in child_features
        assert len(child_features) == len(features) - 1
        assert detail == f"remove:{removed}"

    # The runner is the ObjectiveResult-returning evaluate_fn: no suggestion
    # list is attached, so the issue #7 cap in ``_expand`` cannot apply.
    assert not hasattr(captured["runner"], "get_best_eval_output")
