"""Tests for parse_date utility (extracted from V1 FeatureBuilder).

Covers: None, NaN, date, datetime, ISO string, malformed inputs.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

try:
    from ctra.agents.feature_utils import parse_date

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason="dependencies not available")


class TestParseDate:
    """Tests for parse_date utility function."""

    def test_none_returns_none(self) -> None:
        assert parse_date(None) is None

    def test_nan_returns_none(self) -> None:
        assert parse_date(float("nan")) is None

    def test_numpy_nan_returns_none(self) -> None:
        assert parse_date(np.nan) is None

    def test_date_object_returns_date(self) -> None:
        d = datetime.date(2023, 6, 15)
        result = parse_date(d)
        assert result == datetime.date(2023, 6, 15)

    def test_datetime_object_returns_date(self) -> None:
        """datetime.datetime input should be converted to datetime.date."""
        dt = datetime.datetime(2023, 6, 15, 14, 30, 0)
        result = parse_date(dt)
        assert result == datetime.date(2023, 6, 15)
        assert type(result) is datetime.date

    def test_iso_string(self) -> None:
        result = parse_date("2023-06-15")
        assert result == datetime.date(2023, 6, 15)

    def test_iso_string_with_time(self) -> None:
        """ISO string with time component — should truncate to first 10 chars."""
        result = parse_date("2023-06-15T14:30:00")
        assert result == datetime.date(2023, 6, 15)

    def test_malformed_string_returns_none(self) -> None:
        assert parse_date("not-a-date") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_date("") is None

    def test_integer_returns_none(self) -> None:
        """An integer is not a valid date — fromisoformat will fail."""
        result = parse_date(12345)
        # str(12345)[:10] = "12345" which is not a valid ISO date
        assert result is None
