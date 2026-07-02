"""Tests for scripts/predict.py — prediction entry point.

Pure unit tests with all external dependencies mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_plans():
    """Create a mock feature plans dict."""
    return {
        "drug_mechanism": MagicMock(),
        "trial_size": MagicMock(),
    }


def _make_raw_features(nctid: str, empty: bool = False):
    """Create mock raw features dict."""
    if empty:
        return {}, {}, {}
    return (
        {nctid: {"drug_mechanism": {"value": "kinase_inhibitor"}, "trial_size": {"value": 200}}},
        {},
        {},
    )


def _make_features_df():
    """Create a mock features DataFrame."""
    return pd.DataFrame(
        {
            "id": ["NCT00110279"],
            "drug_mechanism": ["kinase_inhibitor"],
            "trial_size": [200],
        }
    )


def _make_pipeline(p_success: float = 0.72):
    """Create a mock sklearn pipeline."""
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = np.array([[1 - p_success, p_success]])
    return pipeline


def _write_plans_json(path: Path):
    """Write minimal feature_plans.json."""
    plans = {
        "drug_mechanism": {
            "feature_name": "drug_mechanism",
            "feature_idea": "drug mechanism type",
            "feature_type": {"value": "categorical"},
            "data_sources": ["chembl"],
            "example_values": [{"value": "kinase_inhibitor"}],
            "possible_values": {},
            "feature_instructions": "look it up",
        },
    }
    path.write_text(json.dumps(plans))


# ---------------------------------------------------------------------------
# Tests: parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["predict.py"])
        from predict import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_model_dir_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--nctid",
                "NCT00110279",
            ],
        )
        from predict import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_nctid_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                "/tmp/model",
            ],
        )
        from predict import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_valid_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                "/tmp/model",
                "--nctid",
                "NCT00110279",
            ],
        )
        from predict import parse_args

        args = parse_args()
        assert args.model_dir == "/tmp/model"
        assert args.nctid == "NCT00110279"
        assert args.format == "text"  # default

    def test_format_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                "/tmp/model",
                "--nctid",
                "NCT00110279",
                "--format",
                "json",
            ],
        )
        from predict import parse_args

        args = parse_args()
        assert args.format == "json"

    def test_format_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                "/tmp/model",
                "--nctid",
                "NCT00110279",
                "--format",
                "csv",
            ],
        )
        from predict import parse_args

        with pytest.raises(SystemExit):
            parse_args()


# ---------------------------------------------------------------------------
# Tests: main() — missing files
# ---------------------------------------------------------------------------


class TestMainMissingFiles:
    def test_missing_feature_plans(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Should sys.exit(1) when feature_plans.json is missing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        # No feature_plans.json created

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                str(model_dir),
                "--nctid",
                "NCT00110279",
            ],
        )

        from predict import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_missing_best_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Should sys.exit(1) when best_model.pkl is missing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        # Create feature_plans.json but not best_model.pkl
        _write_plans_json(model_dir / "feature_plans.json")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                str(model_dir),
                "--nctid",
                "NCT00110279",
            ],
        )

        from predict import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Tests: main() — successful prediction
# ---------------------------------------------------------------------------


class TestMainPrediction:
    def _setup_main(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fmt: str = "text",
        p_success: float = 0.72,
        empty_features: bool = False,
    ):
        """Common setup for main() tests. Returns model_dir."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_plans_json(model_dir / "feature_plans.json")
        (model_dir / "best_model.pkl").touch()

        nctid = "NCT00110279"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                str(model_dir),
                "--nctid",
                nctid,
                "--format",
                fmt,
            ],
        )

        # Mock configure_lm
        monkeypatch.setattr("ctra.agents.lm_config.configure_lm", MagicMock())

        # Mock load_feature_plans_from_json
        monkeypatch.setattr(
            "ctra.agents.runner.load_feature_plans_from_json",
            MagicMock(return_value=_make_feature_plans()),
        )

        # Mock dill.load (for pipeline)
        mock_pipeline = _make_pipeline(p_success)
        monkeypatch.setattr("dill.load", MagicMock(return_value=mock_pipeline))

        # Mock FeatureGrouper
        monkeypatch.setattr(
            "ctra.agents.feature_grouper.FeatureGrouper",
            MagicMock(),
        )

        # Mock compute_features
        monkeypatch.setattr(
            "ctra.agents.feature_builder.compute_features",
            MagicMock(return_value=_make_raw_features(nctid, empty=empty_features)),
        )

        # Mock features_to_df
        monkeypatch.setattr(
            "ctra.agents.feature_utils.features_to_df",
            MagicMock(return_value=_make_features_df()),
        )

        return model_dir, mock_pipeline

    def test_text_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys,
    ) -> None:
        self._setup_main(monkeypatch, tmp_path, fmt="text", p_success=0.72)

        from predict import main

        main()

        captured = capsys.readouterr()
        assert "NCT00110279" in captured.out
        assert "0.7200" in captured.out
        assert "success" in captured.out

    def test_json_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys,
    ) -> None:
        self._setup_main(monkeypatch, tmp_path, fmt="json", p_success=0.72)

        from predict import main

        main()

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["trial_id"] == "NCT00110279"
        assert result["probability_success"] == 0.72
        assert result["prediction"] == "success"
        assert "n_features" in result
        assert "features_used" in result

    def test_failure_prediction(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys,
    ) -> None:
        """P(success) < 0.5 should predict 'failure'."""
        self._setup_main(monkeypatch, tmp_path, fmt="json", p_success=0.35)

        from predict import main

        main()

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["prediction"] == "failure"
        assert result["probability_success"] == 0.35

    def test_pipeline_predict_proba_called(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _, mock_pipeline = self._setup_main(monkeypatch, tmp_path)

        from predict import main

        main()

        mock_pipeline.predict_proba.assert_called_once()

    def test_feature_building_called_with_nctid(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_plans_json(model_dir / "feature_plans.json")
        (model_dir / "best_model.pkl").touch()

        nctid = "NCT00110279"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                str(model_dir),
                "--nctid",
                nctid,
            ],
        )

        monkeypatch.setattr("ctra.agents.lm_config.configure_lm", MagicMock())
        monkeypatch.setattr(
            "ctra.agents.runner.load_feature_plans_from_json",
            MagicMock(return_value=_make_feature_plans()),
        )
        monkeypatch.setattr("dill.load", MagicMock(return_value=_make_pipeline()))
        monkeypatch.setattr("ctra.agents.feature_grouper.FeatureGrouper", MagicMock())

        mock_compute = MagicMock(return_value=_make_raw_features(nctid))
        monkeypatch.setattr("ctra.agents.feature_builder.compute_features", mock_compute)
        monkeypatch.setattr(
            "ctra.agents.feature_utils.features_to_df",
            MagicMock(return_value=_make_features_df()),
        )

        from predict import main

        main()

        # Verify compute_features was called with the correct nctid list
        call_kwargs = mock_compute.call_args
        assert nctid in call_kwargs.kwargs.get("nctids", call_kwargs[1].get("nctids", []))


# ---------------------------------------------------------------------------
# Tests: main() — empty features error
# ---------------------------------------------------------------------------


class TestMainEmptyFeatures:
    def test_empty_features_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Should sys.exit(1) when feature building returns empty."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_plans_json(model_dir / "feature_plans.json")
        (model_dir / "best_model.pkl").touch()

        nctid = "NCT00110279"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                str(model_dir),
                "--nctid",
                nctid,
            ],
        )

        monkeypatch.setattr("ctra.agents.lm_config.configure_lm", MagicMock())
        monkeypatch.setattr(
            "ctra.agents.runner.load_feature_plans_from_json",
            MagicMock(return_value=_make_feature_plans()),
        )
        monkeypatch.setattr("dill.load", MagicMock(return_value=_make_pipeline()))
        monkeypatch.setattr("ctra.agents.feature_grouper.FeatureGrouper", MagicMock())

        # Return empty raw_features
        monkeypatch.setattr(
            "ctra.agents.feature_builder.compute_features",
            MagicMock(return_value=({}, {}, {})),
        )

        from predict import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_nctid_present_but_empty_features(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """nctid key present in raw_features but maps to empty dict."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_plans_json(model_dir / "feature_plans.json")
        (model_dir / "best_model.pkl").touch()

        nctid = "NCT00110279"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                str(model_dir),
                "--nctid",
                nctid,
            ],
        )

        monkeypatch.setattr("ctra.agents.lm_config.configure_lm", MagicMock())
        monkeypatch.setattr(
            "ctra.agents.runner.load_feature_plans_from_json",
            MagicMock(return_value=_make_feature_plans()),
        )
        monkeypatch.setattr("dill.load", MagicMock(return_value=_make_pipeline()))
        monkeypatch.setattr("ctra.agents.feature_grouper.FeatureGrouper", MagicMock())

        # nctid key exists but maps to empty dict
        monkeypatch.setattr(
            "ctra.agents.feature_builder.compute_features",
            MagicMock(return_value=({nctid: {}}, {}, {})),
        )

        from predict import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Tests: results.json phase detection
# ---------------------------------------------------------------------------


class TestPhaseDetection:
    def test_reads_phase_from_results_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys,
    ) -> None:
        """When results.json exists, phase should be read from it."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_plans_json(model_dir / "feature_plans.json")
        (model_dir / "best_model.pkl").touch()
        (model_dir / "results.json").write_text(
            json.dumps(
                {
                    "task": "phase3",
                    "rollouts": 10,
                }
            )
        )

        nctid = "NCT00110279"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                "--model-dir",
                str(model_dir),
                "--nctid",
                nctid,
            ],
        )

        monkeypatch.setattr("ctra.agents.lm_config.configure_lm", MagicMock())
        monkeypatch.setattr(
            "ctra.agents.runner.load_feature_plans_from_json",
            MagicMock(return_value=_make_feature_plans()),
        )
        monkeypatch.setattr("dill.load", MagicMock(return_value=_make_pipeline()))
        monkeypatch.setattr("ctra.agents.feature_grouper.FeatureGrouper", MagicMock())

        mock_compute = MagicMock(return_value=_make_raw_features(nctid))
        monkeypatch.setattr("ctra.agents.feature_builder.compute_features", mock_compute)
        monkeypatch.setattr(
            "ctra.agents.feature_utils.features_to_df",
            MagicMock(return_value=_make_features_df()),
        )

        from predict import main

        main()

        # task_description should mention Phase 3
        call_kwargs = mock_compute.call_args
        task_desc = call_kwargs.kwargs.get(
            "task_description",
            call_kwargs[1].get("task_description", ""),
        )
        assert "Phase 3" in task_desc
