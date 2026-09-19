"""Tests for scripts/eval/measure_cache_reuse.py (issue #17).

A tiny grid drives the real search and the real orchestrator with the
harness's stubs; the rendering and I/O tests use hand-made rows.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))

import measure_cache_reuse as mcr

TINY = {"rollouts": 3, "depth": 3, "branch": 3, "deep": True}


def _row(**overrides):
    base = dict.fromkeys(mcr.COLUMNS, 0)
    base.update(collision_rate=0.5, seed=0, rollouts=3, evaluations=10, seconds=0.1)
    base.update(overrides)
    assert set(base) == set(mcr.COLUMNS)
    return base


@pytest.fixture(scope="module")
def tiny_rows() -> list[dict]:
    started = time.perf_counter()
    rows = mcr.run_sweep([0.0, 1.0], [0], shared_store=False, replay=True, **TINY)
    elapsed = time.perf_counter() - started
    # A budget, not an assertion: wall-clock time on a loaded CI box or under
    # WSL is not a test outcome.
    if elapsed > 5.0:
        logging.getLogger(__name__).warning("tiny grid took %.1fs (budget 5s)", elapsed)
    return rows


class TestTinyGrid:
    def test_one_row_per_rate_with_every_column(self, tiny_rows: list[dict]) -> None:
        assert [r["collision_rate"] for r in tiny_rows] == [0.0, 1.0]
        assert all(set(r) == set(mcr.COLUMNS) for r in tiny_rows)
        for row in tiny_rows:
            mcr.check_row(row)
            assert row["evaluations"] > 1
            assert row["builds"] == row["groups_dispatched"]
            # One write per built feature: the root's single 3-plan group
            # writes three, every iteration-N singleton group writes one.
            extra_root_writes = (len(mcr.INITIAL) - 1) * mcr.N_TRIALS
            assert row["store_writes"] == row["groups_dispatched"] + extra_root_writes

    def test_collision_rate_0_gives_no_planner_plan_hits(self, tiny_rows: list[dict]) -> None:
        row = tiny_rows[0]
        assert row["planner_calls"] > 0
        assert row["hits_from_planner_plans"] == 0
        assert row["planner_plan_hit_rate"] == 0.0
        assert row["groups_skipped"] == 0
        assert row["llm_calls_avoided_estimate"] == 0

    def test_collision_rate_1_gives_planner_plan_hits(self, tiny_rows: list[dict]) -> None:
        row = tiny_rows[1]
        assert row["hits_from_planner_plans"] > 0
        assert row["planner_plan_hit_rate"] > 0.0
        assert row["groups_skipped"] > 0
        assert row["llm_calls_avoided_estimate"] == 6 * row["groups_skipped"]
        assert row["feature_store_hit_rate"] > tiny_rows[0]["feature_store_hit_rate"]

    def test_initializer_plans_never_hit_within_one_search(self, tiny_rows: list[dict]) -> None:
        """Iteration 0 runs once per search, so its plans only hit across runs."""
        assert all(r["hits_from_initializer_plans"] == 0 for r in tiny_rows)

    def test_agent_cache_never_hits_within_a_search_but_replays_fully(
        self, tiny_rows: list[dict]
    ) -> None:
        for row in tiny_rows:
            assert row["agent_hits"] == 0
            assert row["agent_hit_rate"] == 0.0
            assert row["replay_evaluations"] == row["evaluations"]
            assert row["replay_agent_hits"] == row["evaluations"]
            assert row["replay_group_builds"] == row["groups_dispatched"]

    def test_shared_store_serves_the_initializer_plans_to_the_next_seed(self) -> None:
        rows = mcr.run_sweep([0.0], [0, 1], shared_store=True, replay=False, **TINY)
        assert rows[0]["hits_from_initializer_plans"] == 0
        assert rows[1]["hits_from_initializer_plans"] == len(mcr.INITIAL) * mcr.N_TRIALS


class TestSummaryAndOutput:
    def test_summarise_means_per_rate(self) -> None:
        rows = [
            _row(collision_rate=0.0, seed=0, feature_store_hit_rate=0.0),
            _row(collision_rate=1.0, seed=0, feature_store_hit_rate=0.4, groups_skipped=10),
            _row(collision_rate=1.0, seed=1, feature_store_hit_rate=0.6, groups_skipped=20),
        ]
        summary = mcr.summarise(rows)
        assert [s["collision_rate"] for s in summary] == [0.0, 1.0]
        assert summary[1]["seeds"] == 2
        assert summary[1]["feature_store_hit_rate"] == pytest.approx(0.5)
        assert summary[1]["groups_skipped"] == pytest.approx(15.0)
        assert set(summary[1]) == {"collision_rate", "seeds", *mcr.MEAN_COLUMNS}

    def test_format_table_has_a_line_per_rate(self) -> None:
        table = mcr.format_table(
            mcr.summarise([_row(collision_rate=0.25), _row(collision_rate=1.0)])
        )
        lines = table.splitlines()
        assert len(lines) == 4
        assert lines[2].startswith(" 0.25")
        assert lines[3].startswith(" 1.00")

    def test_write_json_shape(self, tmp_path: Path) -> None:
        rows = [_row(collision_rate=0.0), _row(collision_rate=1.0)]
        args = mcr.parse_args(["--collision-rate", "0,1", "--seeds", "1", "--rollouts", "3"])
        path = tmp_path / "out" / "measurement.json"
        mcr.write_json(path, rows, mcr.summarise(rows), args)
        payload = json.loads(path.read_text())
        assert set(payload) == {"config", "rows", "summary"}
        assert payload["config"]["collision_rates"] == [0.0, 1.0]
        assert payload["config"]["seeds"] == 1
        assert payload["config"]["pool"] == list(mcr.POOL)
        assert len(payload["rows"]) == 2
        assert [s["collision_rate"] for s in payload["summary"]] == [0.0, 1.0]

    def test_parse_args_defaults_and_rates(self) -> None:
        args = mcr.parse_args([])
        assert args.collision_rate == [0.0, 0.25, 0.5, 1.0]
        assert args.seeds == 10
        assert args.deep is True
        assert args.replay is False
        args = mcr.parse_args(["--collision-rate", "0,0.5", "--no-deep", "--replay"])
        assert args.collision_rate == [0.0, 0.5]
        assert args.deep is False
        assert args.replay is True

    def test_check_row_rejects_a_builder_count_that_disagrees(self) -> None:
        with pytest.raises(RuntimeError, match="builder ran"):
            mcr.check_row(_row(builds=3, groups_dispatched=4))
