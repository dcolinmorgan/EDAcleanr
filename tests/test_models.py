"""Tests for core data models."""

from dataclasses import fields

import pandas as pd

from src.models import AgentState, CleaningLogEntry, EDAResults, IssueReport


class TestAgentState:
    """Tests for the AgentState TypedDict."""

    def test_create_minimal_state(self):
        state: AgentState = {"file_path": "data.csv"}
        assert state["file_path"] == "data.csv"

    def test_create_full_state(self):
        df = pd.DataFrame({"a": [1, 2]})
        state: AgentState = {
            "file_path": "data.csv",
            "df": df,
            "original_shape": (100, 5),
            "profile": "shape: (100, 5)",
            "issue_report": {"missing_pct": {}},
            "cleaning_log": ["removed duplicates"],
            "cleaning_iteration": 1,
            "max_cleaning_iterations": 3,
            "needs_more_cleaning": False,
            "eda_results": {"numeric_stats": "..."},
            "insights": ["insight 1"],
            "figure_paths": ["fig.png"],
            "report_path": "output/report.md",
            "errors": [],
            "reasoning_log": [{"timestamp": "2024-01-01", "agent": "inspector", "reasoning": "..."}],
        }
        assert state["file_path"] == "data.csv"
        assert state["original_shape"] == (100, 5)
        assert state["cleaning_iteration"] == 1
        assert state["max_cleaning_iterations"] == 3
        assert state["needs_more_cleaning"] is False
        assert len(state["reasoning_log"]) == 1

    def test_state_has_expected_keys(self):
        expected_keys = {
            "file_path", "df", "original_shape", "profile", "issue_report",
            "cleaning_log", "cleaning_iteration", "max_cleaning_iterations",
            "needs_more_cleaning", "eda_results", "insights", "figure_paths",
            "report_path", "errors", "reasoning_log",
        }
        assert set(AgentState.__annotations__.keys()) == expected_keys


class TestIssueReport:
    """Tests for the IssueReport dataclass."""

    def test_default_values(self):
        report = IssueReport()
        assert report.missing_pct == {}
        assert report.duplicate_count == 0
        assert report.outliers == {}
        assert report.inconsistent_types == []
        assert report.high_cardinality == []
        assert report.zero_variance == []
        assert report.severity_ranking == []

    def test_custom_values(self):
        report = IssueReport(
            missing_pct={"col_a": 15.0, "col_b": 0.5},
            duplicate_count=42,
            outliers={"col_a": [3, 7, 99]},
            inconsistent_types=["col_c"],
            high_cardinality=["col_d"],
            zero_variance=["col_e"],
            severity_ranking=[{"issue": "missing", "severity": "high", "priority": 1}],
        )
        assert report.missing_pct["col_a"] == 15.0
        assert report.duplicate_count == 42
        assert report.outliers["col_a"] == [3, 7, 99]
        assert report.inconsistent_types == ["col_c"]
        assert report.high_cardinality == ["col_d"]
        assert report.zero_variance == ["col_e"]
        assert len(report.severity_ranking) == 1

    def test_has_expected_fields(self):
        field_names = {f.name for f in fields(IssueReport)}
        expected = {
            "missing_pct", "duplicate_count", "outliers",
            "inconsistent_types", "high_cardinality", "zero_variance",
            "severity_ranking",
        }
        assert field_names == expected


class TestCleaningLogEntry:
    """Tests for the CleaningLogEntry dataclass."""

    def test_create_entry(self):
        entry = CleaningLogEntry(
            timestamp="2024-01-01T00:00:00",
            operation="drop_duplicates",
            columns_affected=["all"],
            parameters={},
            rows_before=100,
            rows_after=95,
            description="Removed 5 duplicate rows",
        )
        assert entry.operation == "drop_duplicates"
        assert entry.rows_before == 100
        assert entry.rows_after == 95
        assert entry.columns_affected == ["all"]

    def test_has_expected_fields(self):
        field_names = {f.name for f in fields(CleaningLogEntry)}
        expected = {
            "timestamp", "operation", "columns_affected",
            "parameters", "rows_before", "rows_after", "description",
        }
        assert field_names == expected


class TestEDAResults:
    """Tests for the EDAResults dataclass."""

    def test_default_values(self):
        results = EDAResults()
        assert results.numeric_stats == ""
        assert results.categorical_stats == {}
        assert results.correlation_matrix is None
        assert results.figure_paths == []
        assert results.insights == []

    def test_custom_values(self):
        results = EDAResults(
            numeric_stats="count  mean  std",
            categorical_stats={"color": {"red": 10, "blue": 5}},
            correlation_matrix="col_a  col_b\ncol_a  1.0  0.5\ncol_b  0.5  1.0",
            figure_paths=["hist.png", "box.png"],
            insights=["Strong correlation between A and B"],
        )
        assert "count" in results.numeric_stats
        assert results.categorical_stats["color"]["red"] == 10
        assert results.correlation_matrix is not None
        assert len(results.figure_paths) == 2
        assert len(results.insights) == 1

    def test_has_expected_fields(self):
        field_names = {f.name for f in fields(EDAResults)}
        expected = {
            "numeric_stats", "categorical_stats", "correlation_matrix",
            "figure_paths", "insights",
        }
        assert field_names == expected
