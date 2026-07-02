"""Tests for PredictionMonitor: recording, flushing, drift detection, metrics.

Uses ``tmp_path`` for Parquet storage and generates synthetic prediction
data for drift detection scenarios.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from ctra.mlops.monitoring import DriftReport, MonitoringMetrics, PredictionMonitor

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_n_predictions(
    monitor: PredictionMonitor,
    n: int,
    probability_fn: Any = None,
    feature_fn: Any = None,
    model: str = "xgboost",
    base_trial_id: str = "NCT",
) -> None:
    """Record *n* predictions with optional custom probability/feature callables."""
    for i in range(n):
        prob = probability_fn(i) if probability_fn else 0.5 + 0.01 * i
        features = feature_fn(i) if feature_fn else {"enrollment": 100 + i, "n_arms": 2}
        monitor.record_prediction(
            trial_id=f"{base_trial_id}{i:08d}",
            prediction="success" if prob > 0.5 else "failure",
            probability=prob,
            model=model,
            features=features,
            latency_ms=50.0 + i,
        )


def _make_monitor(tmp_path: Path, **kwargs: Any) -> PredictionMonitor:
    """Create a PredictionMonitor pointed at tmp_path."""
    return PredictionMonitor(store_dir=str(tmp_path), **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecordPrediction:
    """record_prediction buffers records in memory."""

    def test_buffer_grows(self, tmp_path: Path) -> None:
        monitor = _make_monitor(tmp_path)
        assert len(monitor._buffer) == 0

        _record_n_predictions(monitor, 5)

        assert len(monitor._buffer) == 5

    def test_buffer_contains_expected_keys(self, tmp_path: Path) -> None:
        monitor = _make_monitor(tmp_path)
        monitor.record_prediction(
            trial_id="NCT00000001",
            prediction="success",
            probability=0.85,
            model="xgboost",
            features={"enrollment": 200},
            latency_ms=42.0,
        )
        record = monitor._buffer[0]
        assert record["trial_id"] == "NCT00000001"
        assert record["prediction"] == "success"
        assert record["probability"] == 0.85
        assert record["model"] == "xgboost"
        assert record["latency_ms"] == 42.0
        assert "timestamp_utc" in record


class TestFlush:
    """flush() writes buffered records to a Parquet file."""

    def test_flush_writes_parquet(self, tmp_path: Path) -> None:
        monitor = _make_monitor(tmp_path)
        _record_n_predictions(monitor, 3)

        monitor.flush()

        parquet_files = list(tmp_path.glob("predictions_*.parquet"))
        assert len(parquet_files) == 1

        import polars as pl

        df = pl.read_parquet(parquet_files[0])
        assert len(df) == 3

    def test_flush_clears_buffer(self, tmp_path: Path) -> None:
        monitor = _make_monitor(tmp_path)
        _record_n_predictions(monitor, 2)
        monitor.flush()

        assert len(monitor._buffer) == 0

    def test_flush_appends_to_existing(self, tmp_path: Path) -> None:
        monitor = _make_monitor(tmp_path)
        _record_n_predictions(monitor, 2)
        monitor.flush()
        _record_n_predictions(monitor, 3, base_trial_id="NCT2")
        monitor.flush()

        import polars as pl

        parquet_files = list(tmp_path.glob("predictions_*.parquet"))
        assert len(parquet_files) == 1
        df = pl.read_parquet(parquet_files[0])
        assert len(df) == 5


class TestCheckDrift:
    """Drift detection via Kolmogorov--Smirnov tests."""

    def test_insufficient_data_returns_no_drift(self, tmp_path: Path) -> None:
        """With fewer than 10 records, drift should be False."""
        monitor = _make_monitor(tmp_path)
        _record_n_predictions(monitor, 5)
        monitor.flush()

        report = monitor.check_drift(window_days=30)
        assert isinstance(report, DriftReport)
        assert report.has_drift is False
        assert "Insufficient data" in report.recommendations[0]

    def test_identical_distributions_no_drift(self, tmp_path: Path) -> None:
        """When both halves have identical distributions, no drift detected."""
        monitor = _make_monitor(tmp_path)

        # Generate enough records with constant probability to avoid drift
        now = datetime.datetime.now(datetime.timezone.utc)
        for i in range(40):
            # Manually create records with timestamps spread across the window
            ts = now - datetime.timedelta(hours=40 - i)
            record = {
                "timestamp_utc": ts.isoformat(),
                "trial_id": f"NCT{i:08d}",
                "prediction": "success",
                "probability": 0.60,  # constant
                "model": "xgboost",
                "latency_ms": 50.0,
                "feat_enrollment": 200,
                "feat_n_arms": 2,
            }
            monitor._buffer.append(record)

        monitor._buffer_date = now.strftime("%Y-%m-%d")
        monitor.flush()

        report = monitor.check_drift(window_days=30)
        assert report.has_drift is False

    def test_shifted_distributions_detect_drift(self, tmp_path: Path) -> None:
        """When probability distributions differ between halves, drift is detected."""
        monitor = _make_monitor(tmp_path)
        rng = np.random.default_rng(42)

        now = datetime.datetime.now(datetime.timezone.utc)
        for i in range(60):
            ts = now - datetime.timedelta(hours=60 - i)
            # First half: low probability; second half: high probability
            if i < 30:
                prob = float(rng.normal(loc=0.3, scale=0.05))
            else:
                prob = float(rng.normal(loc=0.9, scale=0.05))
            prob = max(0.0, min(1.0, prob))

            record = {
                "timestamp_utc": ts.isoformat(),
                "trial_id": f"NCT{i:08d}",
                "prediction": "success" if prob > 0.5 else "failure",
                "probability": prob,
                "model": "xgboost",
                "latency_ms": 50.0,
                "feat_enrollment": 200,
            }
            monitor._buffer.append(record)

        monitor._buffer_date = now.strftime("%Y-%m-%d")
        monitor.flush()

        report = monitor.check_drift(window_days=30)
        assert report.has_drift is True
        assert report.prediction_distribution_pvalue < 0.05


class TestGetMetrics:
    """get_metrics returns summary statistics."""

    def test_empty_store_returns_zero(self, tmp_path: Path) -> None:
        monitor = _make_monitor(tmp_path)
        metrics = monitor.get_metrics(window_days=7)
        assert isinstance(metrics, MonitoringMetrics)
        assert metrics.total_predictions == 0
        assert metrics.avg_probability_success == 0.0
        assert metrics.avg_latency_ms == 0.0

    def test_metrics_reflect_recorded_data(self, tmp_path: Path) -> None:
        monitor = _make_monitor(tmp_path)
        _record_n_predictions(monitor, 10)
        monitor.flush()

        metrics = monitor.get_metrics(window_days=7)
        assert metrics.total_predictions == 10
        assert metrics.unique_trials == 10
        assert metrics.avg_latency_ms > 0.0
        assert "xgboost" in metrics.model_distribution


class TestFeatureColumnsPrefix:
    """Feature dict keys become 'feat_*' columns in the Parquet file."""

    def test_feature_prefix(self, tmp_path: Path) -> None:
        monitor = _make_monitor(tmp_path)
        monitor.record_prediction(
            trial_id="NCT00000001",
            prediction="success",
            probability=0.75,
            model="xgboost",
            features={"enrollment": 200, "n_arms": 3, "phase_numeric": 2},
            latency_ms=55.0,
        )
        monitor.flush()

        import polars as pl

        parquet_files = list(tmp_path.glob("predictions_*.parquet"))
        df = pl.read_parquet(parquet_files[0])

        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        assert "feat_enrollment" in feat_cols
        assert "feat_n_arms" in feat_cols
        assert "feat_phase_numeric" in feat_cols
        assert len(feat_cols) == 3

    def test_raw_feature_keys_not_present(self, tmp_path: Path) -> None:
        """Raw feature names (without 'feat_' prefix) must not appear as columns."""
        monitor = _make_monitor(tmp_path)
        monitor.record_prediction(
            trial_id="NCT00000002",
            prediction="failure",
            probability=0.3,
            model="tabpfn",
            features={"enrollment": 100},
            latency_ms=30.0,
        )
        monitor.flush()

        import polars as pl

        df = pl.read_parquet(next(iter(tmp_path.glob("predictions_*.parquet"))))
        # "enrollment" should NOT be a column; only "feat_enrollment"
        non_prefixed = [c for c in df.columns if c == "enrollment"]
        assert len(non_prefixed) == 0
