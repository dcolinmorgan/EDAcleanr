"""Unit and property tests for inspection tools.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

import numpy as np
import pandas as pd
import pytest

from src.tools.inspection import detect_issues, get_df_info


# ---------------------------------------------------------------------------
# Unit tests for get_df_info
# ---------------------------------------------------------------------------


class TestGetDfInfo:
    """Requirement 2.1: profile contains shape, dtypes, head, tail, sample."""

    def test_basic_profile(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        info = get_df_info(df)
        assert "3 rows x 2 columns" in info
        assert "a:" in info
        assert "b:" in info

    def test_contains_head_tail_sample(self):
        df = pd.DataFrame({"val": range(20)})
        info = get_df_info(df)
        assert "First 10 rows" in info
        assert "Last 10 rows" in info
        assert "Random sample of 5 rows" in info

    def test_small_df_sample_capped(self):
        df = pd.DataFrame({"x": [1, 2]})
        info = get_df_info(df)
        assert "Random sample of 2 rows" in info

    def test_single_row(self):
        df = pd.DataFrame({"col": [42]})
        info = get_df_info(df)
        assert "1 rows x 1 columns" in info
        assert "First 1 rows" in info
        assert "Last 1 rows" in info
        assert "Random sample of 1 rows" in info


# ---------------------------------------------------------------------------
# Unit tests for detect_issues
# ---------------------------------------------------------------------------


class TestDetectMissing:
    """Requirement 2.2: missing value percentage per column."""

    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = detect_issues(df)
        assert result["missing_pct"]["a"] == 0.0
        assert result["missing_pct"]["b"] == 0.0

    def test_known_missing(self):
        df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
        result = detect_issues(df)
        assert result["missing_pct"]["a"] == pytest.approx(50.0)
        assert result["missing_pct"]["b"] == 0.0

    def test_all_missing(self):
        df = pd.DataFrame({"a": [None, None, None]})
        result = detect_issues(df)
        assert result["missing_pct"]["a"] == pytest.approx(100.0)


class TestDetectDuplicates:
    """Requirement 2.3: duplicate row count."""

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = detect_issues(df)
        assert result["duplicate_count"] == 0

    def test_known_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [10, 10, 20, 20, 30]})
        result = detect_issues(df)
        assert result["duplicate_count"] == 2

    def test_all_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1]})
        result = detect_issues(df)
        assert result["duplicate_count"] == 2


class TestDetectOutliers:
    """Requirement 2.4: outlier detection via IQR."""

    def test_no_outliers(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = detect_issues(df)
        assert "a" not in result["outliers"] or result["outliers"]["a"] == []

    def test_clear_outlier(self):
        # Values 1-10 with an extreme outlier at 1000
        df = pd.DataFrame({"a": list(range(1, 11)) + [1000]})
        result = detect_issues(df)
        assert "a" in result["outliers"]
        # Index 10 is the outlier (value 1000)
        assert 10 in result["outliers"]["a"]

    def test_non_numeric_skipped(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result = detect_issues(df)
        assert "text" not in result["outliers"]

    def test_nan_not_outlier(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, None, 1000]})
        result = detect_issues(df)
        # NaN index (5) should NOT be in outliers
        if "a" in result["outliers"]:
            assert 5 not in result["outliers"]["a"]


class TestDetectInconsistentTypes:
    """Requirement 2.5: mixed numeric/string columns."""

    def test_pure_numeric_not_flagged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = detect_issues(df)
        assert "a" not in result["inconsistent_types"]

    def test_pure_string_not_flagged(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = detect_issues(df)
        assert "a" not in result["inconsistent_types"]

    def test_mixed_types_flagged(self):
        df = pd.DataFrame({"a": ["1", "2", "hello", "4"]})
        result = detect_issues(df)
        assert "a" in result["inconsistent_types"]

    def test_all_numeric_strings_not_flagged(self):
        # All values parseable as numbers — not inconsistent
        df = pd.DataFrame({"a": ["1", "2.5", "3"]})
        result = detect_issues(df)
        assert "a" not in result["inconsistent_types"]


class TestDetectHighCardinality:
    """Requirement 2.6: high-cardinality categorical columns."""

    def test_low_cardinality(self):
        df = pd.DataFrame({"cat": ["a", "a", "b", "b", "a", "b", "a", "b", "a", "b"]})
        result = detect_issues(df)
        assert "cat" not in result["high_cardinality"]

    def test_high_cardinality(self):
        # All unique values -> ratio = 1.0 > 0.5
        df = pd.DataFrame({"cat": ["a", "b", "c", "d", "e"]})
        result = detect_issues(df)
        assert "cat" in result["high_cardinality"]

    def test_numeric_not_considered(self):
        df = pd.DataFrame({"num": [1, 2, 3, 4, 5]})
        result = detect_issues(df)
        assert "num" not in result["high_cardinality"]


class TestDetectZeroVariance:
    """Requirement 2.6: zero-variance columns."""

    def test_constant_column(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5]})
        result = detect_issues(df)
        assert "a" in result["zero_variance"]

    def test_varying_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = detect_issues(df)
        assert "a" not in result["zero_variance"]

    def test_all_nan_is_zero_variance(self):
        df = pd.DataFrame({"a": [None, None, None]})
        result = detect_issues(df)
        assert "a" in result["zero_variance"]

    def test_single_value_with_nan(self):
        df = pd.DataFrame({"a": [5, None, 5, None]})
        result = detect_issues(df)
        assert "a" in result["zero_variance"]


class TestDetectIssuesEmptyDf:
    """Edge case: empty DataFrame."""

    def test_empty_df(self):
        df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})
        result = detect_issues(df)
        assert result["missing_pct"] == {"a": 0.0, "b": 0.0}
        assert result["duplicate_count"] == 0
        assert result["outliers"] == {}
        assert result["inconsistent_types"] == []
        assert result["high_cardinality"] == []
        assert result["zero_variance"] == ["a", "b"]
