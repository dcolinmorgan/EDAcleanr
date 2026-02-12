"""Unit tests for the report generator module."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.models import CleaningLogEntry
from src.report_generator import generate_report


@pytest.fixture
def sample_issue_report() -> dict:
    return {
        "missing_pct": {"age": 15.0, "income": 5.5},
        "duplicate_count": 12,
        "outliers": {"income": [3, 7, 42]},
        "inconsistent_types": ["mixed_col"],
        "high_cardinality": ["city"],
        "zero_variance": ["constant_col"],
    }


@pytest.fixture
def sample_cleaning_log() -> list[str]:
    return [
        "Dropped 12 duplicate rows",
        "Filled missing values in 'age' using median strategy",
        "Removed 3 outliers from 'income' using IQR method",
    ]


@pytest.fixture
def sample_eda_results() -> dict:
    return {
        "numeric_stats": "       age    income\nmean   35.2   50000\nstd    10.1   15000",
        "categorical_stats": {"city": {"NYC": 50, "LA": 30, "SF": 20}},
        "correlation_matrix": "         age  income\nage     1.00   0.45\nincome  0.45   1.00",
    }


class TestGenerateReport:
    """Tests for the generate_report function."""

    def test_returns_report_path(self, sample_issue_report, sample_cleaning_log, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(100, 10),
                cleaned_shape=(88, 9),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=["Insight 1", "Insight 2", "Insight 3"],
                figure_paths=[],
                output_dir=tmpdir,
            )
            assert path == os.path.join(tmpdir, "report.md")
            assert os.path.isfile(path)

    def test_creates_output_dir_if_missing(self, sample_issue_report, sample_cleaning_log, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "output")
            path = generate_report(
                original_shape=(50, 5),
                cleaned_shape=(50, 5),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=[],
                figure_paths=[],
                output_dir=nested,
            )
            assert os.path.isfile(path)

    def test_report_contains_dataset_overview(self, sample_issue_report, sample_cleaning_log, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(100, 10),
                cleaned_shape=(88, 9),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=["Insight 1"],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "## Dataset Overview" in content
            assert "100 rows" in content
            assert "10 columns" in content
            assert "88 rows" in content
            assert "9 columns" in content

    def test_report_contains_issues_found(self, sample_issue_report, sample_cleaning_log, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(100, 10),
                cleaned_shape=(88, 9),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=[],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "## Issues Found" in content
            assert "15.0%" in content
            assert "12" in content  # duplicate count
            assert "mixed_col" in content

    def test_report_contains_cleaning_actions(self, sample_issue_report, sample_cleaning_log, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(100, 10),
                cleaned_shape=(88, 9),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=[],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "## Cleaning Actions" in content
            for entry in sample_cleaning_log:
                assert entry in content

    def test_report_contains_key_statistics(self, sample_issue_report, sample_cleaning_log, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(100, 10),
                cleaned_shape=(88, 9),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=[],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "## Key Statistics" in content
            assert "35.2" in content  # from numeric_stats

    def test_report_contains_insights(self, sample_issue_report, sample_cleaning_log, sample_eda_results):
        insights = ["Revenue is trending up", "Age correlates with income", "Missing data concentrated in Q1"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(100, 10),
                cleaned_shape=(88, 9),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=insights,
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "## Insights" in content
            for insight in insights:
                assert insight in content

    def test_report_embeds_figure_references_relative(self, sample_issue_report, sample_cleaning_log, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            fig_dir = os.path.join(tmpdir, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            fig1 = os.path.join(fig_dir, "histogram.png")
            fig2 = os.path.join(fig_dir, "correlation_heatmap.png")
            # Create dummy files
            for f in [fig1, fig2]:
                open(f, "w").close()

            path = generate_report(
                original_shape=(100, 10),
                cleaned_shape=(88, 9),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=["Insight"],
                figure_paths=[fig1, fig2],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "## Figures" in content
            # Paths should be relative to the output dir
            assert "figures/histogram.png" in content or "figures\\histogram.png" in content
            assert "figures/correlation_heatmap.png" in content or "figures\\correlation_heatmap.png" in content
            # Should use markdown image syntax
            assert "![" in content

    def test_handles_cleaning_log_entry_objects(self, sample_issue_report, sample_eda_results):
        log_entries = [
            CleaningLogEntry(
                timestamp="2024-01-01T00:00:00",
                operation="drop_duplicates",
                columns_affected=["all"],
                parameters={},
                rows_before=100,
                rows_after=88,
                description="Removed 12 duplicate rows",
            ),
            CleaningLogEntry(
                timestamp="2024-01-01T00:00:01",
                operation="fill_missing",
                columns_affected=["age"],
                parameters={"strategy": "median"},
                rows_before=88,
                rows_after=88,
                description="Filled missing values in age",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(100, 10),
                cleaned_shape=(88, 9),
                issue_report=sample_issue_report,
                cleaning_log=log_entries,
                eda_results=sample_eda_results,
                insights=[],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "drop_duplicates" in content
            assert "fill_missing" in content
            assert "age" in content
            assert "median" in content

    def test_handles_mixed_cleaning_log_types(self, sample_issue_report, sample_eda_results):
        log_entries = [
            "Dropped 5 duplicate rows",
            CleaningLogEntry(
                timestamp="2024-01-01T00:00:00",
                operation="fill_missing",
                columns_affected=["score"],
                parameters={"strategy": "mean"},
                rows_before=95,
                rows_after=95,
                description="Filled missing values in score",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(100, 5),
                cleaned_shape=(95, 5),
                issue_report=sample_issue_report,
                cleaning_log=log_entries,
                eda_results=sample_eda_results,
                insights=[],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "Dropped 5 duplicate rows" in content
            assert "fill_missing" in content

    def test_empty_cleaning_log(self, sample_issue_report, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(50, 5),
                cleaned_shape=(50, 5),
                issue_report=sample_issue_report,
                cleaning_log=[],
                eda_results=sample_eda_results,
                insights=[],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "## Cleaning Actions" in content
            assert "No cleaning actions were performed" in content

    def test_empty_issue_report(self, sample_cleaning_log, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(50, 5),
                cleaned_shape=(50, 5),
                issue_report={},
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=[],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "## Issues Found" in content

    def test_empty_eda_results(self, sample_issue_report, sample_cleaning_log):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(50, 5),
                cleaned_shape=(50, 5),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results={},
                insights=[],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            assert "## Key Statistics" in content

    def test_all_required_sections_present(self, sample_issue_report, sample_cleaning_log, sample_eda_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(
                original_shape=(100, 10),
                cleaned_shape=(88, 9),
                issue_report=sample_issue_report,
                cleaning_log=sample_cleaning_log,
                eda_results=sample_eda_results,
                insights=["Insight 1", "Insight 2"],
                figure_paths=[],
                output_dir=tmpdir,
            )
            content = open(path, encoding="utf-8").read()
            required_sections = [
                "## Dataset Overview",
                "## Issues Found",
                "## Cleaning Actions",
                "## Key Statistics",
                "## Insights",
            ]
            for section in required_sections:
                assert section in content, f"Missing section: {section}"
