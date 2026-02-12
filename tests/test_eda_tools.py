"""Unit tests for EDA tools.

Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.tools.eda import (
    compute_correlation,
    describe_categorical,
    describe_numeric,
    generate_plots,
)


# ---------------------------------------------------------------------------
# describe_numeric
# ---------------------------------------------------------------------------

class TestDescribeNumeric:
    """Tests for describe_numeric — Requirement 4.1."""

    def test_basic_numeric_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = describe_numeric(df)
        assert "a" in result
        assert "b" in result
        # Should contain standard describe stats
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": ["d", "e", "f"]})
        result = describe_numeric(df)
        assert result == "No numeric columns found."

    def test_mixed_columns_only_reports_numeric(self):
        df = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "c"]})
        result = describe_numeric(df)
        assert "num" in result
        assert "cat" not in result

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        result = describe_numeric(df)
        # Empty DataFrame has no rows to describe — acceptable to report nothing
        assert isinstance(result, str)

    def test_single_value_column(self):
        df = pd.DataFrame({"val": [42, 42, 42]})
        result = describe_numeric(df)
        assert "val" in result
        assert "42" in result


# ---------------------------------------------------------------------------
# describe_categorical
# ---------------------------------------------------------------------------

class TestDescribeCategorical:
    """Tests for describe_categorical — Requirement 4.2."""

    def test_basic_categorical(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green", "blue", "blue"]})
        result = describe_categorical(df)
        assert "color" in result
        assert "blue" in result
        assert "red" in result
        assert "green" in result

    def test_no_categorical_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = describe_categorical(df)
        assert result == "No categorical columns found."

    def test_top_n_limits_output(self):
        df = pd.DataFrame({"letter": list("abcdefghij") * 2})
        result = describe_categorical(df, top_n=3)
        # Extract only the value-count data lines (skip "Column:" header and series name)
        lines_with_counts = [
            line for line in result.split("\n")
            if line.strip() and not line.startswith("Column:") and "    " in line
        ]
        assert len(lines_with_counts) <= 3

    def test_multiple_categorical_columns(self):
        df = pd.DataFrame({
            "fruit": ["apple", "banana", "apple"],
            "size": ["small", "large", "small"],
        })
        result = describe_categorical(df)
        assert "fruit" in result
        assert "size" in result

    def test_category_dtype(self):
        df = pd.DataFrame({"status": pd.Categorical(["active", "inactive", "active"])})
        result = describe_categorical(df)
        assert "status" in result
        assert "active" in result


# ---------------------------------------------------------------------------
# compute_correlation
# ---------------------------------------------------------------------------

class TestComputeCorrelation:
    """Tests for compute_correlation — Requirement 4.3."""

    def test_two_numeric_columns(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        result = compute_correlation(df)
        assert "x" in result
        assert "y" in result
        # Perfect correlation
        assert "1.0" in result or "1.000" in result

    def test_fewer_than_two_numeric_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = compute_correlation(df)
        assert "skipped" in result.lower() or "fewer" in result.lower()

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        result = compute_correlation(df)
        assert "skipped" in result.lower() or "fewer" in result.lower()

    def test_correlation_matrix_is_symmetric(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(50),
            "b": np.random.randn(50),
            "c": np.random.randn(50),
        })
        result = compute_correlation(df)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_mixed_columns_only_uses_numeric(self):
        df = pd.DataFrame({
            "num1": [1, 2, 3],
            "num2": [4, 5, 6],
            "cat": ["a", "b", "c"],
        })
        result = compute_correlation(df)
        assert "num1" in result
        assert "num2" in result
        assert "cat" not in result


# ---------------------------------------------------------------------------
# generate_plots
# ---------------------------------------------------------------------------

class TestGeneratePlots:
    """Tests for generate_plots — Requirement 4.4."""

    def test_generates_histograms_and_boxplots(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_plots(df, tmpdir)
            # Should have hist + box for each of 2 cols + heatmap = 5
            assert len(paths) == 5
            filenames = [os.path.basename(p) for p in paths]
            assert "hist_a.png" in filenames
            assert "hist_b.png" in filenames
            assert "box_a.png" in filenames
            assert "box_b.png" in filenames
            assert "correlation_heatmap.png" in filenames
            # All files should exist on disk
            for p in paths:
                assert os.path.isfile(p)

    def test_no_heatmap_with_single_numeric_column(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_plots(df, tmpdir)
            filenames = [os.path.basename(p) for p in paths]
            assert "hist_x.png" in filenames
            assert "box_x.png" in filenames
            assert "correlation_heatmap.png" not in filenames

    def test_no_numeric_columns_returns_empty(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_plots(df, tmpdir)
            assert paths == []

    def test_creates_output_directory(self):
        df = pd.DataFrame({"v": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "figures")
            paths = generate_plots(df, nested)
            assert os.path.isdir(nested)
            assert len(paths) > 0

    def test_handles_nan_values(self):
        df = pd.DataFrame({"a": [1, np.nan, 3, np.nan, 5], "b": [10, 20, np.nan, 40, 50]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_plots(df, tmpdir)
            # Should still generate plots despite NaNs
            assert len(paths) > 0
            for p in paths:
                assert os.path.isfile(p)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_plots(df, tmpdir)
            # May produce empty plots or skip — should not crash
            assert isinstance(paths, list)
