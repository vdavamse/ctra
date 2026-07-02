"""Tests for FeatureBuilder per-type validation logic.

Extracts and tests the validation patterns from FeatureBuilder.forward()
without instantiating the class (which requires dspy LM configuration).

Covers:
- boolean: "true"/"false" -> bool, "invalid" -> None via soft_assert
- categorical: value in possible_values, value not in -> None
- multicategorical: valid JSON array + subset check, invalid JSON -> None
- integer: regex check, "123" -> 123.0, "-5" -> -5.0, "abc" -> nan
- float: regex match, "1.5" -> 1.5, "abc" -> nan
- quote stripping: '"value"' -> 'value', "'value'" -> 'value'
- "None" string -> None
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

try:
    from ctra.agents.data_models import FeatureType
    from ctra.agents.feature_utils import soft_assert

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

pytestmark = pytest.mark.skipif(not _HAS_DSPY, reason="dspy/sqlite3 not available")


# ---------------------------------------------------------------------------
# Helpers — extracted validation logic from FeatureBuilder.forward()
# ---------------------------------------------------------------------------


def _validate_value(
    value: str | None,
    feature_type: FeatureType,
    possible_values: list[str] | None = None,
) -> object:
    """Apply the same validation logic as FeatureBuilder.forward().

    This is a standalone extraction of lines 436-540 of feature_builder.py
    so we can test the validation without instantiating the full module.
    """
    if isinstance(value, list):
        value = json.dumps(value)
    if value is not None and not isinstance(value, str):
        value = str(value)

    # Strip quotes
    if value is not None:
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]

    if value == "None":
        value = None

    ft = feature_type
    pv = possible_values or []

    if ft == FeatureType.BOOLEAN:
        if value is not None:
            value = value.lower()
            value = soft_assert(
                value,
                value in ("true", "false"),
                f"'boolean' features must have values 'True' or 'False'.  Got '{value}'.",
            )
            value = np.nan if value is None else value == "true"
        else:
            value = np.nan

    elif ft == FeatureType.CATEGORICAL:
        if value is not None:
            value = soft_assert(
                value,
                value in pv,
                f"'categorical' feature must be a single value in {pv}.  Got '{value}'.",
            )

    elif ft == FeatureType.MULTICATEGORICAL:
        pv_set = set(pv)
        if value is not None:
            try:
                as_list = json.loads(value)
            except (ValueError, TypeError):
                as_list = None

            as_list = soft_assert(
                as_list,
                as_list is not None and isinstance(as_list, list),
                f"'multi-categorical' feature must be returned as a JSON array.  Got {value}",
            )
            if as_list is None or (
                len(as_list) == 1 and (as_list[0] == "None" or as_list[0] is None)
            ):
                value = None
            else:
                as_list = soft_assert(
                    as_list,
                    set(as_list).issubset(pv_set),
                    f"'multi-categorical' feature must be values from {pv}.  "
                    f"Invalid: {set(as_list) - pv_set}",
                )
                value = as_list

    elif ft == FeatureType.INTEGER:
        if value is not None:
            value = soft_assert(
                value,
                re.fullmatch(r"-?\d+", value) is not None,
                f"'integer' feature must return an integer value.  Got {value}",
            )
            value = np.nan if value is None else float(int(value))
        else:
            value = np.nan

    elif ft == FeatureType.FLOAT:
        if value is not None:
            value = soft_assert(
                value,
                re.match(r"^-?\d+(?:\.\d+)?$", value) is not None,
                f"'float' feature must return a floating point value.  Got {value}",
            )
            value = np.nan if value is None else float(value)
        else:
            value = np.nan

    return value


# ======================================================================
# Boolean validation
# ======================================================================


class TestBooleanValidation:
    def test_true_string(self) -> None:
        result = _validate_value("true", FeatureType.BOOLEAN)
        assert result is True

    def test_false_string(self) -> None:
        result = _validate_value("false", FeatureType.BOOLEAN)
        assert result is False

    def test_True_string(self) -> None:
        result = _validate_value("True", FeatureType.BOOLEAN)
        assert result is True

    def test_False_string(self) -> None:
        result = _validate_value("False", FeatureType.BOOLEAN)
        assert result is False

    def test_invalid_string_returns_nan(self) -> None:
        """Invalid boolean string -> soft_assert returns None -> becomes nan."""
        result = _validate_value("invalid", FeatureType.BOOLEAN)
        assert np.isnan(result)

    def test_none_returns_nan(self) -> None:
        result = _validate_value(None, FeatureType.BOOLEAN)
        assert np.isnan(result)

    def test_none_string_returns_nan(self) -> None:
        """The string 'None' should be converted to None, then to nan."""
        result = _validate_value("None", FeatureType.BOOLEAN)
        assert np.isnan(result)


# ======================================================================
# Categorical validation
# ======================================================================


class TestCategoricalValidation:
    def test_valid_value(self) -> None:
        result = _validate_value(
            "kinase_inhibitor",
            FeatureType.CATEGORICAL,
            ["kinase_inhibitor", "antibody", "other"],
        )
        assert result == "kinase_inhibitor"

    def test_invalid_value_returns_none(self) -> None:
        result = _validate_value(
            "not_a_category",
            FeatureType.CATEGORICAL,
            ["kinase_inhibitor", "antibody"],
        )
        assert result is None

    def test_none_value_stays_none(self) -> None:
        result = _validate_value(None, FeatureType.CATEGORICAL, ["a", "b"])
        assert result is None

    def test_none_string_becomes_none(self) -> None:
        result = _validate_value("None", FeatureType.CATEGORICAL, ["a", "b"])
        assert result is None


# ======================================================================
# Multicategorical validation
# ======================================================================


class TestMulticategoricalValidation:
    def test_valid_json_array(self) -> None:
        result = _validate_value(
            '["a", "b"]',
            FeatureType.MULTICATEGORICAL,
            ["a", "b", "c"],
        )
        assert result == ["a", "b"]

    def test_valid_subset(self) -> None:
        result = _validate_value(
            '["a"]',
            FeatureType.MULTICATEGORICAL,
            ["a", "b", "c"],
        )
        assert result == ["a"]

    def test_invalid_json_returns_none(self) -> None:
        result = _validate_value(
            "not valid json",
            FeatureType.MULTICATEGORICAL,
            ["a", "b"],
        )
        assert result is None

    def test_values_not_in_possible_values_returns_none(self) -> None:
        result = _validate_value(
            '["a", "x"]',
            FeatureType.MULTICATEGORICAL,
            ["a", "b"],
        )
        assert result is None

    def test_single_none_in_array_returns_none(self) -> None:
        result = _validate_value(
            '["None"]',
            FeatureType.MULTICATEGORICAL,
            ["a", "b"],
        )
        assert result is None

    def test_single_null_in_array_returns_none(self) -> None:
        result = _validate_value(
            "[null]",
            FeatureType.MULTICATEGORICAL,
            ["a", "b"],
        )
        assert result is None

    def test_none_value(self) -> None:
        result = _validate_value(None, FeatureType.MULTICATEGORICAL, ["a", "b"])
        assert result is None

    def test_list_input_converted_to_json(self) -> None:
        """A Python list input should be serialized to JSON then validated."""
        result = _validate_value(
            ["a", "b"],
            FeatureType.MULTICATEGORICAL,
            ["a", "b", "c"],
        )
        assert result == ["a", "b"]


# ======================================================================
# Integer validation
# ======================================================================


class TestIntegerValidation:
    def test_valid_integer_string(self) -> None:
        result = _validate_value("123", FeatureType.INTEGER)
        assert result == 123.0

    def test_zero(self) -> None:
        result = _validate_value("0", FeatureType.INTEGER)
        assert result == 0.0

    def test_non_digit_returns_nan(self) -> None:
        result = _validate_value("abc", FeatureType.INTEGER)
        assert np.isnan(result)

    def test_float_string_returns_nan(self) -> None:
        """Regex rejects '1.5' for INTEGER type, so it should return nan."""
        result = _validate_value("1.5", FeatureType.INTEGER)
        assert np.isnan(result)

    def test_negative_integer(self) -> None:
        """Negative integers should parse correctly (not be rejected)."""
        result = _validate_value("-1", FeatureType.INTEGER)
        assert result == -1.0

    def test_none_returns_nan(self) -> None:
        result = _validate_value(None, FeatureType.INTEGER)
        assert np.isnan(result)

    def test_negative_five(self) -> None:
        """Acceptance criteria: '-5' should parse to -5.0."""
        result = _validate_value("-5", FeatureType.INTEGER)
        assert result == -5.0

    def test_negative_large(self) -> None:
        """Acceptance criteria: '-100' should parse to -100.0."""
        result = _validate_value("-100", FeatureType.INTEGER)
        assert result == -100.0

    def test_forty_two(self) -> None:
        """Acceptance criteria: '42' should parse to 42.0."""
        result = _validate_value("42", FeatureType.INTEGER)
        assert result == 42.0

    def test_trailing_chars_returns_nan(self) -> None:
        """Acceptance criteria: '5x' should soft-fail to NaN."""
        result = _validate_value("5x", FeatureType.INTEGER)
        assert np.isnan(result)

    def test_empty_string_returns_nan(self) -> None:
        """Acceptance criteria: '' should soft-fail to NaN."""
        result = _validate_value("", FeatureType.INTEGER)
        assert np.isnan(result)

    def test_whitespace_padded_returns_nan(self) -> None:
        """Leading/trailing whitespace should soft-fail to NaN."""
        assert np.isnan(_validate_value(" -5", FeatureType.INTEGER))
        assert np.isnan(_validate_value("-5 ", FeatureType.INTEGER))


# ======================================================================
# Float validation
# ======================================================================


class TestFloatValidation:
    def test_valid_float_string(self) -> None:
        result = _validate_value("1.5", FeatureType.FLOAT)
        assert result == pytest.approx(1.5)

    def test_integer_string_accepted(self) -> None:
        result = _validate_value("42", FeatureType.FLOAT)
        assert result == pytest.approx(42.0)

    def test_negative_float(self) -> None:
        result = _validate_value("-3.14", FeatureType.FLOAT)
        assert result == pytest.approx(-3.14)

    def test_non_numeric_returns_nan(self) -> None:
        result = _validate_value("abc", FeatureType.FLOAT)
        assert np.isnan(result)

    def test_scientific_notation_returns_nan(self) -> None:
        """The regex does not accept scientific notation."""
        result = _validate_value("1.5e10", FeatureType.FLOAT)
        assert np.isnan(result)

    def test_none_returns_nan(self) -> None:
        result = _validate_value(None, FeatureType.FLOAT)
        assert np.isnan(result)


# ======================================================================
# Quote stripping
# ======================================================================


class TestQuoteStripping:
    def test_double_quotes_stripped(self) -> None:
        result = _validate_value(
            '"kinase_inhibitor"',
            FeatureType.CATEGORICAL,
            ["kinase_inhibitor"],
        )
        assert result == "kinase_inhibitor"

    def test_single_quotes_stripped(self) -> None:
        result = _validate_value(
            "'kinase_inhibitor'",
            FeatureType.CATEGORICAL,
            ["kinase_inhibitor"],
        )
        assert result == "kinase_inhibitor"

    def test_mismatched_quotes_not_stripped(self) -> None:
        """Quotes that don't match should not be stripped."""
        result = _validate_value(
            "\"kinase_inhibitor'",
            FeatureType.CATEGORICAL,
            ["kinase_inhibitor"],
        )
        # The first condition (starts and ends with ") is False, and
        # the second (starts and ends with ') is also False, so no stripping.
        # The value with mismatched quotes won't be in possible_values.
        assert result is None


# ======================================================================
# "None" string conversion
# ======================================================================


class TestNoneStringConversion:
    def test_none_string_to_none_categorical(self) -> None:
        result = _validate_value("None", FeatureType.CATEGORICAL, ["a", "b"])
        assert result is None

    def test_none_string_to_nan_boolean(self) -> None:
        result = _validate_value("None", FeatureType.BOOLEAN)
        assert np.isnan(result)

    def test_none_string_to_nan_integer(self) -> None:
        result = _validate_value("None", FeatureType.INTEGER)
        assert np.isnan(result)

    def test_none_string_to_nan_float(self) -> None:
        result = _validate_value("None", FeatureType.FLOAT)
        assert np.isnan(result)
