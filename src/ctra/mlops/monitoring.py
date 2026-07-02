"""Production prediction monitoring with drift detection.

Records every prediction to append-only Parquet files (rotated daily)
and detects distribution drift via Kolmogorov--Smirnov tests on numeric
features and prediction probabilities.

All timestamps are in UTC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Result of a drift check over a monitoring window."""

    has_drift: bool
    drifted_features: list[str]
    prediction_distribution_pvalue: float
    avg_latency_ms: float
    missing_feature_rate: float
    recommendations: list[str]


@dataclass
class MonitoringMetrics:
    """Aggregate metrics for the monitoring dashboard."""

    total_predictions: int
    avg_probability_success: float
    avg_latency_ms: float
    p99_latency_ms: float
    unique_trials: int
    model_distribution: dict[str, int]  # model_name -> count
    daily_counts: list[dict[str, Any]]  # [{"date": "2026-03-27", "count": 42}, ...]


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class PredictionMonitor:
    """Monitors prediction quality and drift in production.

    Predictions are logged to daily Parquet files under ``store_dir``::

        monitoring/
            predictions_2026-03-27.parquet
            predictions_2026-03-28.parquet
            ...

    Args:
        store_dir: Directory for monitoring logs. Defaults to ``"monitoring/"``.
        drift_threshold: p-value threshold below which a KS-test signals drift.
            Defaults to ``0.05``.
    """

    def __init__(
        self,
        store_dir: str = "monitoring/",
        drift_threshold: float = 0.05,
    ) -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._drift_threshold = drift_threshold

        # In-memory buffer for the current day; flushed on day rollover
        # or when explicitly requested.
        self._buffer: list[dict[str, Any]] = []
        self._buffer_date: str | None = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        trial_id: str,
        prediction: str,
        probability: float,
        model: str,
        features: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Record a production prediction for monitoring.

        Args:
            trial_id: NCT ID or internal trial identifier.
            prediction: ``"success"`` or ``"failure"``.
            probability: Predicted probability of success.
            model: Classifier name (e.g. ``"xgboost"``).
            features: Feature dict used for this prediction (for drift detection).
            latency_ms: End-to-end prediction latency in milliseconds.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # Flush buffer if the day has rolled over
        if self._buffer_date is not None and self._buffer_date != today:
            self._flush_buffer()

        self._buffer_date = today

        record: dict[str, Any] = {
            "timestamp_utc": now.isoformat(),
            "trial_id": trial_id,
            "prediction": prediction,
            "probability": probability,
            "model": model,
            "latency_ms": latency_ms,
        }

        # Flatten features into the record with a prefix
        for feat_name, feat_value in features.items():
            record[f"feat_{feat_name}"] = feat_value

        self._buffer.append(record)

    def flush(self) -> None:
        """Force-flush the in-memory buffer to Parquet."""
        if self._buffer:
            self._flush_buffer()

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(self, window_days: int = 30) -> DriftReport:
        """Check for prediction drift over a time window.

        Detects:
            - Feature distribution shift (KS test on numeric features)
            - Prediction probability distribution shift
            - Missing feature rate increase
            - Latency degradation

        Args:
            window_days: Number of days to look back.

        Returns:
            Drift analysis results with recommendations.
        """
        from scipy import stats

        # Flush any pending records
        self.flush()

        df = self._load_window(window_days)

        if df.empty or len(df) < 10:
            return DriftReport(
                has_drift=False,
                drifted_features=[],
                prediction_distribution_pvalue=1.0,
                avg_latency_ms=0.0,
                missing_feature_rate=0.0,
                recommendations=["Insufficient data for drift detection."],
            )

        # Split the window into two halves for comparison
        midpoint = df["timestamp_utc"].median()
        older = df[df["timestamp_utc"] <= midpoint]
        newer = df[df["timestamp_utc"] > midpoint]

        if len(older) < 5 or len(newer) < 5:
            return DriftReport(
                has_drift=False,
                drifted_features=[],
                prediction_distribution_pvalue=1.0,
                avg_latency_ms=float(df["latency_ms"].mean()),
                missing_feature_rate=0.0,
                recommendations=["Insufficient data in each half-window for drift detection."],
            )

        recommendations: list[str] = []
        drifted_features: list[str] = []

        # 1. Prediction probability distribution shift
        _ks_stat, pred_pvalue = stats.ks_2samp(
            np.asarray(older["probability"].dropna(), dtype=float),
            np.asarray(newer["probability"].dropna(), dtype=float),
        )
        if pred_pvalue < self._drift_threshold:
            recommendations.append(
                f"Prediction probability distribution has shifted "
                f"(KS p-value={pred_pvalue:.4f}). Consider retraining."
            )

        # 2. Feature distribution shifts (numeric feat_ columns)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        numeric_feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]

        for col in numeric_feat_cols:
            old_vals = np.asarray(older[col].dropna(), dtype=float)
            new_vals = np.asarray(newer[col].dropna(), dtype=float)
            if len(old_vals) < 5 or len(new_vals) < 5:
                continue
            _, pval = stats.ks_2samp(old_vals, new_vals)
            if pval < self._drift_threshold:
                feat_name = col.replace("feat_", "")
                drifted_features.append(feat_name)

        if drifted_features:
            recommendations.append(
                f"Feature drift detected in: {', '.join(drifted_features)}. "
                f"Review data pipeline or consider retraining."
            )

        # 3. Missing feature rate
        total_feat_cells = len(df) * len(feat_cols) if feat_cols else 1
        missing_cells = sum(df[c].isna().sum() for c in feat_cols)
        missing_rate = float(missing_cells / total_feat_cells) if total_feat_cells > 0 else 0.0

        if missing_rate > 0.1:
            recommendations.append(
                f"Missing feature rate is {missing_rate:.1%}. Check data source availability."
            )

        # 4. Latency degradation
        avg_latency = float(df["latency_ms"].mean())
        old_latency = float(older["latency_ms"].mean())
        new_latency = float(newer["latency_ms"].mean())
        if old_latency > 0 and new_latency / max(old_latency, 1e-6) > 1.5:
            recommendations.append(
                f"Average latency increased from {old_latency:.0f}ms to "
                f"{new_latency:.0f}ms (+{(new_latency / old_latency - 1) * 100:.0f}%). "
                f"Investigate infrastructure or model complexity."
            )

        has_drift = bool(pred_pvalue < self._drift_threshold or len(drifted_features) > 0)

        return DriftReport(
            has_drift=has_drift,
            drifted_features=drifted_features,
            prediction_distribution_pvalue=float(pred_pvalue),
            avg_latency_ms=avg_latency,
            missing_feature_rate=missing_rate,
            recommendations=recommendations if recommendations else ["No drift detected."],
        )

    # ------------------------------------------------------------------
    # Dashboard metrics
    # ------------------------------------------------------------------

    def get_metrics(self, window_days: int = 7) -> MonitoringMetrics:
        """Get aggregate metrics for the monitoring dashboard.

        Args:
            window_days: Number of days to look back.

        Returns:
            Summary statistics for the monitoring dashboard.
        """
        self.flush()

        df = self._load_window(window_days)

        if df.empty:
            return MonitoringMetrics(
                total_predictions=0,
                avg_probability_success=0.0,
                avg_latency_ms=0.0,
                p99_latency_ms=0.0,
                unique_trials=0,
                model_distribution={},
                daily_counts=[],
            )

        # Model distribution
        model_counts = {str(k): v for k, v in df["model"].value_counts().to_dict().items()}

        # Daily counts
        df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.date.astype(str)
        daily: list[dict[str, Any]] = (
            df.groupby("date")
            .size()
            .reset_index(name="count")
            .sort_values("date")
            .to_dict("records")  # type: ignore[assignment]
        )

        return MonitoringMetrics(
            total_predictions=len(df),
            avg_probability_success=float(df["probability"].mean()),
            avg_latency_ms=float(df["latency_ms"].mean()),
            p99_latency_ms=float(df["latency_ms"].quantile(0.99)),
            unique_trials=int(df["trial_id"].nunique()),
            model_distribution=model_counts,
            daily_counts=daily,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_buffer(self) -> None:
        """Write the in-memory prediction buffer to a daily Parquet file.

        Appends the buffer to ``predictions_YYYY-MM-DD.parquet`` in the
        store directory. If the file exists, new records are concatenated
        to it. Clears the buffer after writing.

        Used by ``record_prediction`` and ``flush`` to persist predictions
        to disk for later drift analysis.
        """
        if not self._buffer:
            return

        date_str = self._buffer_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        parquet_path = self._store_dir / f"predictions_{date_str}.parquet"

        new_df = pl.DataFrame(self._buffer)

        if parquet_path.exists():
            existing = pl.read_parquet(parquet_path)
            combined = pl.concat([existing, new_df])
        else:
            combined = new_df

        combined.write_parquet(parquet_path)

        logger.debug(
            "Flushed %d records to %s (total: %d)",
            len(self._buffer),
            parquet_path,
            len(combined),
        )
        self._buffer.clear()

    def _load_window(self, window_days: int) -> pd.DataFrame:
        """Load monitoring data for the last ``window_days`` days from Parquet.

        Reads all daily ``predictions_YYYY-MM-DD.parquet`` files within the
        window and concatenates them into a single pandas DataFrame. The
        DataFrame format is used because drift detection requires scipy
        and pandas-specific analytics (groupby, value_counts, quantile).

        Args:
            window_days: Number of days to look back from today.

        Returns:
            Concatenated pandas DataFrame of all predictions in the window,
            or an empty DataFrame if no matching files are found.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        cutoff_date = cutoff.strftime("%Y-%m-%d")

        frames: list[pl.DataFrame] = []
        for parquet_path in sorted(self._store_dir.glob("predictions_*.parquet")):
            # Extract date from filename: predictions_YYYY-MM-DD.parquet
            stem = parquet_path.stem  # predictions_2026-03-27
            file_date = stem.replace("predictions_", "")
            if file_date < cutoff_date:
                continue
            try:
                df = pl.read_parquet(parquet_path)
                frames.append(df)
            except Exception:
                logger.warning("Failed to read %s; skipping", parquet_path)

        if not frames:
            return pd.DataFrame()

        combined = pl.concat(frames)

        # Convert to pandas for drift detection analytics (scipy, groupby, etc.).
        # polars .to_pandas() can produce object-dtype columns for nullable
        # integers/booleans; convert_dtypes() infers proper pandas nullable
        # types so that downstream is_numeric_dtype checks work correctly.
        combined_pd = combined.to_pandas()
        combined_pd = combined_pd.convert_dtypes()

        # Filter by exact timestamp cutoff
        if "timestamp_utc" in combined_pd.columns:
            combined_pd["timestamp_utc"] = pd.to_datetime(combined_pd["timestamp_utc"])
            combined_pd = combined_pd[combined_pd["timestamp_utc"] >= cutoff]

        return combined_pd
