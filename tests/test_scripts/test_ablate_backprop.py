"""Tests for scripts/eval/ablate_backprop.py — the backprop ablation harness (issue #16).

A tiny grid (one seed, few rollouts) exercises the real search; the
rendering and I/O tests use hand-made rows so they stay independent of the
landscape.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))

import ablate_backprop as ab

TINY_ROLLOUTS = 4


def _row(**overrides):
    base = {
        "regime": "issue",
        "variant": "mean",
        "deep": True,
        "adaptive": False,
        "rollouts": 4,
        "branch": 3,
        "note": "",
        "seed": 0,
        "evaluations": 20,
        "best_auc": 0.81,
        "best_depth": 4,
        "synergy_found": False,
        "best_synergy_count": 3,
        "evals_to_first_synergy": float("nan"),
        "nodes": 30,
        "depth_reached": 6,
        "ucb_decided": 2,
        "max_size_seen": 6,
        "max_synergy_seen": 4,
        "seconds": 0.01,
    }
    base.update(overrides)
    assert set(base) == set(ab.COLUMNS)
    return base


class TestTinyGrid:
    def test_one_row_per_cell_and_variant_with_every_column(self):
        rows = ab.run_sweep(["issue"], ["mean", "max"], seeds=[0], rollouts=TINY_ROLLOUTS)

        cells = ab.REGIMES["issue"]
        assert len(rows) == len(cells) * 2
        assert all(set(r) == set(ab.COLUMNS) for r in rows)
        assert {(r["deep"], r["adaptive"], r["branch"]) for r in rows} == {
            (c.deep, c.adaptive, c.branch) for c in cells
        }
        assert all(r["rollouts"] == TINY_ROLLOUTS for r in rows)
        assert all(r["evaluations"] > 0 and r["nodes"] > 0 for r in rows)
        # The fixture check passed inside run_sweep: the landscape was reached.
        assert max(r["max_size_seen"] for r in rows) > 1
        assert max(r["max_synergy_seen"] for r in rows) == 4

        summary = ab.summarise(rows)
        assert len(summary) == len(rows)
        assert all(e["n"] == 1 for e in summary)
        assert all(e["same_as_mean"] == 1 for e in summary if e["variant"] == "mean")

    def test_main_prints_the_table_and_writes_json(self, tmp_path, capsys):
        code = ab.main(
            [
                *("--regime", "issue", "--variants", "mean", "--seeds", "1"),
                *("--rollouts", str(TINY_ROLLOUTS), "--output-dir", str(tmp_path)),
            ]
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "regime" in out and "UCB" in out and "=mean" in out
        payload = json.loads((tmp_path / "ablation_issue.json").read_text())
        assert len(payload["rows"]) == len(ab.REGIMES["issue"])
        assert payload["variants"] == ["mean"]
        assert payload["rollouts_override"] == TINY_ROLLOUTS

    def test_unknown_variant_is_rejected(self, tmp_path):
        assert ab.main(["--variants", "sum", "--output-dir", str(tmp_path)]) == 2


class TestFixtureFence:
    def test_a_runner_regressed_to_parent_plans_trips_the_sizes_assertion(self, monkeypatch):
        """The wiring the issue named scores the root's set for every node; refuse it."""

        class ParentPlansRunner(ab.FeatureAwareRunner):
            def __call__(self, node_id, task, previous_output):
                if previous_output is None or not previous_output.feature_plans:
                    features = list(ab.INITIAL)
                else:
                    features = list(previous_output.feature_plans)  # the parent's plans
                self.seen.append(list(features))
                remaining = [f for f in self.pool if f not in features][: self.max_suggestions]
                return ab.stub_output(
                    features, self.evaluate(features), [f"add {f}" for f in remaining]
                )

        monkeypatch.setattr(ab, "FeatureAwareRunner", ParentPlansRunner)

        with pytest.raises(RuntimeError, match="sizes > 1 were never evaluated"):
            ab.run_sweep(["issue"], ["mean"], seeds=[0], rollouts=2)

    def test_an_unreachable_bonus_trips_the_synergy_assertion(self):
        rows = [_row(max_synergy_seen=3), _row(seed=1, max_synergy_seen=2)]

        with pytest.raises(RuntimeError, match="synergy count 4 was never evaluated"):
            ab.check_fixture(rows, "issue")

    def test_a_regime_without_rows_is_an_error(self):
        with pytest.raises(RuntimeError, match="no rows"):
            ab.check_fixture([_row()], "saturated")


class TestRendering:
    def test_nan_first_synergy_renders_as_nan_in_both_tables(self):
        rows = [_row(), _row(seed=1)]
        summary = ab.summarise(rows)

        assert math.isnan(summary[0]["evals_to_first_synergy"])
        assert "nan" in ab.format_table(summary)
        assert "nan" in ab.format_rows(rows)

    def test_first_synergy_mean_ignores_seeds_that_never_found_it(self):
        rows = [_row(), _row(seed=1, synergy_found=True, evals_to_first_synergy=12.0)]

        (entry,) = ab.summarise(rows)
        assert entry["evals_to_first_synergy"] == 12.0
        assert entry["synergy_found"] == 0.5
        assert entry["n"] == 2

    def test_same_as_mean_counts_identical_seeds(self):
        rows = [
            _row(),
            _row(seed=1),
            _row(variant="max"),  # identical to mean on seed 0
            _row(variant="max", seed=1, best_auc=0.9),  # diverged on seed 1
        ]

        by_variant = {e["variant"]: e for e in ab.summarise(rows)}
        assert by_variant["mean"]["same_as_mean"] == 2
        assert by_variant["max"]["same_as_mean"] == 1

    def test_json_round_trip_turns_nan_into_null(self, tmp_path):
        rows = [_row(), _row(seed=1, synergy_found=True, evals_to_first_synergy=7.0)]
        summary = ab.summarise(rows)
        path = tmp_path / "nested" / "ablation.json"

        ab.write_json(path, rows, summary, regimes=["issue"])
        payload = json.loads(path.read_text())

        assert payload["regimes"] == ["issue"]
        assert payload["rows"][0]["evals_to_first_synergy"] is None
        assert payload["rows"][1]["evals_to_first_synergy"] == 7.0
        for key in ab.COLUMNS:
            if key != "evals_to_first_synergy":
                assert payload["rows"][1][key] == rows[1][key]
        assert payload["summary"][0]["evals_to_first_synergy"] == 7.0
