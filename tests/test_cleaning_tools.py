"""Unit tests for cleaning tools.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8
"""

import numpy as np
import pandas as pd
import pytest

from src.models import CleaningLogEntry
from src.tools.cleaning import (
    convert_dtypes,
    drop_duplicates,
    drop_useless_columns,
    fill_missing,
    normalize_columns,
    remove_outliers,
    strip_string_values,
)


# ---------------------------------------------------------------------------
# drop_duplicates
# ---------------------------------------------------------------------------


class TestDropDuplicates:
    """Requirement 3.1: remove duplicate rows."""

    def test_removes_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [10, 10, 20, 20, 30]})
        cleaned, log = drop_duplicates(df)
        assert len(cleaned) == 3
        assert cleaned.duplicated().sum() == 0

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        cleaned, log = drop_duplicates(df)
        assert len(cleaned) == 3

    def test_all_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1]})
        cleaned, log = drop_duplicates(df)
        assert len(cleaned) == 1

    def test_log_entry(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        cleaned, log = drop_duplicates(df)
        assert isinstance(log, CleaningLogEntry)
        assert log.operation == "drop_duplicates"
        assert log.rows_before == 3
        assert log.rows_after == 2
        assert log.timestamp
        assert "1" in log.description

    def test_empty_df(self):
        df = pd.DataFrame({"a": pd.Series(dtype="int64")})
        cleaned, log = drop_duplicates(df)
        assert len(cleaned) == 0


# ---------------------------------------------------------------------------
# fill_missing
# ---------------------------------------------------------------------------


class TestFillMissing:
    """Requirement 3.2: fill missing values with various strategies."""

    def test_mean_strategy(self):
        df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0]})
        cleaned, log = fill_missing(df, "a", "mean")
        assert cleaned["a"].isna().sum() == 0
        assert cleaned["a"].iloc[2] == pytest.approx(7 / 3)

    def test_median_strategy(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0, 5.0]})
        cleaned, log = fill_missing(df, "a", "median")
        assert cleaned["a"].isna().sum() == 0
        assert cleaned["a"].iloc[1] == pytest.approx(3.0)

    def test_mode_strategy(self):
        df = pd.DataFrame({"a": ["x", "x", "y", None]})
        cleaned, log = fill_missing(df, "a", "mode")
        assert cleaned["a"].isna().sum() == 0
        assert cleaned["a"].iloc[3] == "x"

    def test_ffill_strategy(self):
        df = pd.DataFrame({"a": [1.0, None, None, 4.0]})
        cleaned, log = fill_missing(df, "a", "ffill")
        assert cleaned["a"].isna().sum() == 0
        assert cleaned["a"].iloc[1] == 1.0
        assert cleaned["a"].iloc[2] == 1.0

    def test_ffill_leading_nan(self):
        df = pd.DataFrame({"a": [None, None, 3.0, 4.0]})
        cleaned, log = fill_missing(df, "a", "ffill")
        assert cleaned["a"].isna().sum() == 0
        assert cleaned["a"].iloc[0] == 3.0

    def test_knn_strategy(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        cleaned, log = fill_missing(df, "a", "knn")
        assert cleaned["a"].isna().sum() == 0

    def test_invalid_strategy_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Invalid strategy"):
            fill_missing(df, "a", "invalid")

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            fill_missing(df, "nonexistent", "mean")

    def test_log_entry(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        cleaned, log = fill_missing(df, "a", "median")
        assert log.operation == "fill_missing"
        assert log.columns_affected == ["a"]
        assert log.parameters == {"strategy": "median"}


# ---------------------------------------------------------------------------
# convert_dtypes
# ---------------------------------------------------------------------------


class TestConvertDtypes:
    """Requirement 3.3: convert column types with coercion."""

    def test_to_float(self):
        df = pd.DataFrame({"a": ["1", "2.5", "3"]})
        cleaned, log = convert_dtypes(df, "a", "float")
        assert pd.api.types.is_float_dtype(cleaned["a"])
        assert cleaned["a"].iloc[1] == pytest.approx(2.5)

    def test_to_int(self):
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        cleaned, log = convert_dtypes(df, "a", "int")
        assert cleaned["a"].dtype.name == "Int64"

    def test_coercion_on_invalid(self):
        df = pd.DataFrame({"a": ["1", "hello", "3"]})
        cleaned, log = convert_dtypes(df, "a", "float")
        assert pd.isna(cleaned["a"].iloc[1])

    def test_to_str(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        cleaned, log = convert_dtypes(df, "a", "str")
        assert pd.api.types.is_string_dtype(cleaned["a"])

    def test_to_datetime(self):
        df = pd.DataFrame({"a": ["2024-01-01", "2024-06-15"]})
        cleaned, log = convert_dtypes(df, "a", "datetime")
        assert pd.api.types.is_datetime64_any_dtype(cleaned["a"])

    def test_to_category(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        cleaned, log = convert_dtypes(df, "a", "category")
        assert cleaned["a"].dtype.name == "category"

    def test_unsupported_type_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unsupported target_type"):
            convert_dtypes(df, "a", "complex")

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            convert_dtypes(df, "b", "float")

    def test_log_entry(self):
        df = pd.DataFrame({"a": ["1", "2"]})
        cleaned, log = convert_dtypes(df, "a", "float")
        assert log.operation == "convert_dtypes"
        assert log.parameters["target_type"] == "float"
        assert "original_dtype" in log.parameters


# ---------------------------------------------------------------------------
# remove_outliers
# ---------------------------------------------------------------------------


class TestRemoveOutliers:
    """Requirement 3.4: remove/cap outliers."""

    def test_iqr_caps_outliers(self):
        data = list(range(1, 11)) + [1000]
        df = pd.DataFrame({"a": data})
        cleaned, log = remove_outliers(df, "a", "iqr")
        q1 = pd.Series(data).quantile(0.25)
        q3 = pd.Series(data).quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        assert cleaned["a"].max() <= upper + 1e-9

    def test_iqr_no_outliers(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        cleaned, log = remove_outliers(df, "a", "iqr")
        assert len(cleaned) == 5
        pd.testing.assert_series_equal(cleaned["a"], df["a"])

    def test_zscore_removes_rows(self):
        data = [10] * 100 + [10000]
        df = pd.DataFrame({"a": data})
        cleaned, log = remove_outliers(df, "a", "zscore")
        assert len(cleaned) < len(df)

    def test_zscore_preserves_nan(self):
        data = [1.0, 2.0, 3.0, None, 5.0]
        df = pd.DataFrame({"a": data})
        cleaned, log = remove_outliers(df, "a", "zscore")
        assert cleaned["a"].isna().sum() == 1

    def test_invalid_method_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Invalid method"):
            remove_outliers(df, "a", "invalid")

    def test_non_numeric_raises(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="not numeric"):
            remove_outliers(df, "a", "iqr")

    def test_log_entry_iqr(self):
        df = pd.DataFrame({"a": [1, 2, 3, 100]})
        cleaned, log = remove_outliers(df, "a", "iqr")
        assert log.operation == "remove_outliers"
        assert log.parameters["method"] == "iqr"
        assert "lower_bound" in log.parameters

    def test_log_entry_zscore(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        cleaned, log = remove_outliers(df, "a", "zscore")
        assert log.parameters["method"] == "zscore"


# ---------------------------------------------------------------------------
# normalize_columns
# ---------------------------------------------------------------------------


class TestNormalizeColumns:
    """Requirement 3.5: strip whitespace and lowercase column names."""

    def test_lowercases_columns(self):
        df = pd.DataFrame({"Name": [1], "AGE": [2]})
        cleaned, log = normalize_columns(df)
        assert list(cleaned.columns) == ["name", "age"]

    def test_strips_whitespace(self):
        df = pd.DataFrame({" name ": [1], "  age": [2]})
        cleaned, log = normalize_columns(df)
        assert list(cleaned.columns) == ["name", "age"]

    def test_already_normalized(self):
        df = pd.DataFrame({"name": [1], "age": [2]})
        cleaned, log = normalize_columns(df)
        assert list(cleaned.columns) == ["name", "age"]
        assert "already normalized" in log.description

    def test_log_entry(self):
        df = pd.DataFrame({"Name": [1]})
        cleaned, log = normalize_columns(df)
        assert log.operation == "normalize_columns"


# ---------------------------------------------------------------------------
# strip_string_values
# ---------------------------------------------------------------------------


class TestStripStringValues:
    """Requirement 3.7: strip whitespace from string values."""

    def test_strips_whitespace(self):
        df = pd.DataFrame({"a": ["  hello  ", " world ", "test"]})
        cleaned, log = strip_string_values(df)
        assert list(cleaned["a"]) == ["hello", "world", "test"]

    def test_numeric_columns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        cleaned, log = strip_string_values(df)
        pd.testing.assert_frame_equal(cleaned, df)

    def test_mixed_columns(self):
        df = pd.DataFrame({"text": [" hi ", " bye "], "num": [1, 2]})
        cleaned, log = strip_string_values(df)
        assert list(cleaned["text"]) == ["hi", "bye"]
        assert list(cleaned["num"]) == [1, 2]

    def test_log_entry(self):
        df = pd.DataFrame({"a": [" x "]})
        cleaned, log = strip_string_values(df)
        assert log.operation == "strip_string_values"
        assert "a" in log.columns_affected


# ---------------------------------------------------------------------------
# drop_useless_columns
# ---------------------------------------------------------------------------


class TestDropUselessColumns:
    """Requirement 3.6: drop columns with >90% missing or zero variance."""

    def test_drops_high_missing(self):
        # 9/10 = 90% missing -> exactly at threshold, not dropped (> not >=)
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "bad": [None] * 9 + [1],
        })
        cleaned, log = drop_useless_columns(df, threshold=0.9)
        assert "good" in cleaned.columns

    def test_drops_above_threshold(self):
        # 10/11 > 90% missing -> dropped
        df = pd.DataFrame({
            "good": list(range(11)),
            "bad": [None] * 10 + [1],
        })
        cleaned, log = drop_useless_columns(df, threshold=0.9)
        assert "bad" not in cleaned.columns

    def test_drops_zero_variance(self):
        df = pd.DataFrame({"a": [1, 2, 3], "constant": [5, 5, 5]})
        cleaned, log = drop_useless_columns(df)
        assert "constant" not in cleaned.columns
        assert "a" in cleaned.columns

    def test_drops_all_nan_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "empty": [None, None, None]})
        cleaned, log = drop_useless_columns(df)
        assert "empty" not in cleaned.columns

    def test_keeps_good_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cleaned, log = drop_useless_columns(df)
        assert list(cleaned.columns) == ["a", "b"]

    def test_custom_threshold(self):
        # 6/10 = 60% missing
        df = pd.DataFrame({
            "a": list(range(10)),
            "b": [None] * 6 + [7, 8, 9, 10],
        })
        cleaned, log = drop_useless_columns(df, threshold=0.5)
        assert "b" not in cleaned.columns

    def test_log_entry(self):
        df = pd.DataFrame({"a": [1, 2, 3], "const": [5, 5, 5]})
        cleaned, log = drop_useless_columns(df)
        assert log.operation == "drop_useless_columns"
        assert "const" in log.columns_affected
        assert log.parameters["threshold"] == 0.9


# ---------------------------------------------------------------------------
# Cross-cutting: CleaningLogEntry structure (Requirement 3.8)
# ---------------------------------------------------------------------------


class TestCleaningLogStructure:
    """Requirement 3.8: each operation produces a complete log entry."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "a": [1.0, 2.0, None, 4.0],
            "b": ["x", "y", "z", "w"],
        })

    @pytest.mark.parametrize("func,args", [
        (drop_duplicates, []),
        (normalize_columns, []),
        (strip_string_values, []),
    ])
    def test_log_has_required_fields(self, sample_df, func, args):
        cleaned, log = func(sample_df, *args)
        assert isinstance(log, CleaningLogEntry)
        assert log.timestamp
        assert log.operation
        assert isinstance(log.columns_affected, list)
        assert isinstance(log.parameters, dict)
        assert isinstance(log.rows_before, int)
        assert isinstance(log.rows_after, int)
        assert log.description

    def test_fill_missing_log(self, sample_df):
        cleaned, log = fill_missing(sample_df, "a", "mean")
        assert log.operation == "fill_missing"
        assert log.columns_affected == ["a"]
        assert log.parameters["strategy"] == "mean"
        assert log.rows_before == 4
        assert log.rows_after == 4

    def test_remove_outliers_log(self):
        df = pd.DataFrame({"a": [1, 2, 3, 100]})
        cleaned, log = remove_outliers(df, "a", "iqr")
        assert log.operation == "remove_outliers"
        assert log.rows_before == 4

    def test_drop_useless_log(self):
        df = pd.DataFrame({"a": [1, 2, 3], "c": [5, 5, 5]})
        cleaned, log = drop_useless_columns(df)
        assert log.operation == "drop_useless_columns"
        assert "c" in log.columns_affected
