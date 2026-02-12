"""Unit tests for graph nodes in src/graph.py.

Each node is tested with mocked LLM responses to keep tests deterministic.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.graph import (
    _append_error,
    _append_reasoning,
    _ensure_list,
    clean_decision_node,
    clean_node,
    eda_node,
    inspect_node,
    load_csv_node,
    report_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Return a minimal valid AgentState dict with sensible defaults."""
    state: dict = {
        "file_path": "",
        "df": None,
        "original_shape": None,
        "profile": None,
        "issue_report": None,
        "cleaning_log": [],
        "cleaning_iteration": 0,
        "max_cleaning_iterations": 3,
        "needs_more_cleaning": False,
        "eda_results": None,
        "insights": [],
        "figure_paths": [],
        "report_path": None,
        "errors": [],
        "reasoning_log": [],
    }
    state.update(overrides)
    return state


def _mock_llm(content: str = "Mock LLM response", tool_calls=None):
    """Return a mock LLM whose invoke returns an AIMessage-like object."""
    response = MagicMock()
    response.content = content
    response.tool_calls = tool_calls or []

    llm = MagicMock()
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm
    return llm


def _sample_df() -> pd.DataFrame:
    """Return a small sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Alice", "Eve"],
            "age": [30, 25, 35, 30, 28],
            "score": [88.5, 92.0, 76.3, 88.5, 95.1],
        }
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_ensure_list_creates_missing_key(self):
        state: dict = {}
        _ensure_list(state, "errors")
        assert state["errors"] == []

    def test_ensure_list_replaces_none(self):
        state: dict = {"errors": None}
        _ensure_list(state, "errors")
        assert state["errors"] == []

    def test_ensure_list_preserves_existing(self):
        state: dict = {"errors": ["existing"]}
        _ensure_list(state, "errors")
        assert state["errors"] == ["existing"]

    def test_append_error(self):
        state: dict = {"errors": []}
        _append_error(state, "something broke")
        assert state["errors"] == ["something broke"]

    def test_append_error_initialises_list(self):
        state: dict = {}
        _append_error(state, "oops")
        assert state["errors"] == ["oops"]

    def test_append_reasoning(self):
        state: dict = {"reasoning_log": []}
        _append_reasoning(state, "test_agent", "did a thing")
        assert len(state["reasoning_log"]) == 1
        entry = state["reasoning_log"][0]
        assert entry["agent"] == "test_agent"
        assert entry["reasoning"] == "did a thing"
        assert "timestamp" in entry

    def test_append_reasoning_initialises_list(self):
        state: dict = {}
        _append_reasoning(state, "agent", "reason")
        assert len(state["reasoning_log"]) == 1


# ---------------------------------------------------------------------------
# load_csv_node
# ---------------------------------------------------------------------------


class TestLoadCsvNode:
    def test_successful_load(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        state = _make_state(file_path=str(csv_file))
        result = load_csv_node(state)

        assert result["df"] is not None
        assert result["original_shape"] == (2, 2)
        assert len(result["errors"]) == 0
        assert len(result["reasoning_log"]) == 1
        assert "load_csv_node" in result["reasoning_log"][0]["agent"]

    def test_missing_file(self):
        state = _make_state(file_path="/nonexistent/path.csv")
        result = load_csv_node(state)

        assert result["df"] is None
        assert result["original_shape"] is None
        assert len(result["errors"]) == 1
        assert "CSV load error" in result["errors"][0]
        assert len(result["reasoning_log"]) == 1

    def test_empty_file(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        state = _make_state(file_path=str(csv_file))
        result = load_csv_node(state)

        assert result["df"] is None
        assert len(result["errors"]) == 1

    def test_reasoning_log_has_timestamp(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x\n1\n")

        state = _make_state(file_path=str(csv_file))
        result = load_csv_node(state)

        entry = result["reasoning_log"][0]
        assert "timestamp" in entry
        assert "agent" in entry
        assert "reasoning" in entry


# ---------------------------------------------------------------------------
# inspect_node
# ---------------------------------------------------------------------------


class TestInspectNode:
    def test_successful_inspection(self):
        llm = _mock_llm("Found 1 duplicate and 10% missing in col_a.")
        state = _make_state(df=_sample_df())
        result = inspect_node(state, llm)

        assert result["profile"] is not None
        assert result["issue_report"] is not None
        assert "missing_pct" in result["issue_report"]
        assert len(result["reasoning_log"]) == 1
        assert result["reasoning_log"][0]["agent"] == "inspect_node"

    def test_no_dataframe(self):
        llm = _mock_llm()
        state = _make_state(df=None)
        result = inspect_node(state, llm)

        assert result["profile"] is None
        assert result["issue_report"] is None
        assert len(result["errors"]) == 1
        assert "No DataFrame" in result["errors"][0]

    def test_llm_exception_degrades_gracefully(self):
        llm = MagicMock()
        llm.bind_tools.side_effect = Exception("LLM unavailable")

        state = _make_state(df=_sample_df())
        result = inspect_node(state, llm)

        # Should still have profile and issue_report from degraded path
        assert result["profile"] is not None
        assert result["issue_report"] is not None
        assert len(result["errors"]) == 1
        assert "exception" in result["errors"][0].lower()

    def test_reasoning_log_entry_format(self):
        llm = _mock_llm("Inspection complete.")
        state = _make_state(df=_sample_df())
        result = inspect_node(state, llm)

        entry = result["reasoning_log"][0]
        assert "timestamp" in entry
        assert entry["agent"] == "inspect_node"
        assert isinstance(entry["reasoning"], str)


# ---------------------------------------------------------------------------
# clean_decision_node
# ---------------------------------------------------------------------------


class TestCleanDecisionNode:
    def test_decides_cleaning_needed(self):
        llm = _mock_llm('{"needs_cleaning": true, "plan": "drop duplicates"}')
        state = _make_state(
            issue_report={"missing_pct": {"a": 10.0}, "duplicate_count": 5}
        )
        result = clean_decision_node(state, llm)

        assert result["needs_more_cleaning"] is True
        assert len(result["reasoning_log"]) == 1

    def test_decides_no_cleaning(self):
        llm = _mock_llm('{"needs_cleaning": false, "plan": "data is clean"}')
        state = _make_state(
            issue_report={"missing_pct": {}, "duplicate_count": 0}
        )
        result = clean_decision_node(state, llm)

        assert result["needs_more_cleaning"] is False

    def test_no_issue_report_skips(self):
        llm = _mock_llm()
        state = _make_state(issue_report=None)
        result = clean_decision_node(state, llm)

        assert result["needs_more_cleaning"] is False
        assert len(result["reasoning_log"]) == 1

    def test_llm_exception_defaults_to_cleaning(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM down")

        state = _make_state(
            issue_report={"missing_pct": {"a": 50.0}, "duplicate_count": 10}
        )
        result = clean_decision_node(state, llm)

        assert result["needs_more_cleaning"] is True
        assert len(result["errors"]) == 1

    def test_unparseable_llm_response_defaults_to_cleaning(self):
        llm = _mock_llm("I think we should clean the data but no JSON here.")
        state = _make_state(
            issue_report={"missing_pct": {"a": 5.0}, "duplicate_count": 1}
        )
        result = clean_decision_node(state, llm)

        # Default is True when JSON parsing fails
        assert result["needs_more_cleaning"] is True


# ---------------------------------------------------------------------------
# clean_node
# ---------------------------------------------------------------------------


class TestCleanNode:
    def test_increments_iteration(self):
        llm = _mock_llm("Cleaned the data.")
        state = _make_state(df=_sample_df(), cleaning_iteration=0)
        result = clean_node(state, llm)

        assert result["cleaning_iteration"] == 1

    def test_no_dataframe(self):
        llm = _mock_llm()
        state = _make_state(df=None)
        result = clean_node(state, llm)

        assert result["needs_more_cleaning"] is False
        assert len(result["errors"]) == 1

    def test_max_iterations_stops_cleaning(self):
        llm = _mock_llm("Applied cleaning.")
        state = _make_state(
            df=_sample_df(),
            cleaning_iteration=2,
            max_cleaning_iterations=3,
        )
        result = clean_node(state, llm)

        assert result["cleaning_iteration"] == 3
        assert result["needs_more_cleaning"] is False

    def test_llm_exception_stops_cleaning(self):
        llm = MagicMock()
        llm.bind_tools.side_effect = Exception("LLM error")

        state = _make_state(df=_sample_df())
        result = clean_node(state, llm)

        assert result["needs_more_cleaning"] is False
        assert len(result["errors"]) == 1

    def test_reasoning_log_includes_iteration(self):
        llm = _mock_llm("Done cleaning.")
        state = _make_state(df=_sample_df(), cleaning_iteration=0)
        result = clean_node(state, llm)

        entry = result["reasoning_log"][0]
        assert "Iteration 1" in entry["reasoning"]

    def test_tool_calls_executed(self):
        """When the LLM returns tool calls, they should be executed."""
        tool_call = {
            "name": "tool_drop_duplicates",
            "args": {},
        }
        llm = _mock_llm("Dropping duplicates.", tool_calls=[tool_call])
        df = _sample_df()
        # Add a duplicate row
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        state = _make_state(df=df, cleaning_iteration=0)
        result = clean_node(state, llm)

        # The tool should have been called and log entry added
        assert result["cleaning_iteration"] == 1
        assert len(result["reasoning_log"]) == 1


# ---------------------------------------------------------------------------
# eda_node
# ---------------------------------------------------------------------------


class TestEdaNode:
    def test_successful_eda(self, tmp_path, monkeypatch):
        # Redirect output dir to tmp_path
        monkeypatch.setattr(
            "src.graph.generate_plots",
            lambda df, output_dir: [],
        )
        llm = _mock_llm("Insight 1: data is good\nInsight 2: no anomalies")
        state = _make_state(df=_sample_df())
        result = eda_node(state, llm)

        assert result["eda_results"] is not None
        assert "numeric_stats" in result["eda_results"]
        assert len(result["insights"]) > 0
        assert len(result["reasoning_log"]) == 1

    def test_no_dataframe(self):
        llm = _mock_llm()
        state = _make_state(df=None)
        result = eda_node(state, llm)

        assert result["eda_results"] is None
        assert len(result["errors"]) == 1

    def test_llm_exception_degrades_gracefully(self, monkeypatch):
        monkeypatch.setattr(
            "src.graph.generate_plots",
            lambda df, output_dir: [],
        )
        llm = MagicMock()
        llm.bind_tools.side_effect = Exception("LLM unavailable")

        state = _make_state(df=_sample_df())
        result = eda_node(state, llm)

        # Degraded path should still produce eda_results
        assert result["eda_results"] is not None
        assert len(result["errors"]) == 1

    def test_insights_extracted_from_response(self, monkeypatch):
        monkeypatch.setattr(
            "src.graph.generate_plots",
            lambda df, output_dir: [],
        )
        llm = _mock_llm("Line one\nLine two\nLine three")
        state = _make_state(df=_sample_df())
        result = eda_node(state, llm)

        assert len(result["insights"]) == 3


# ---------------------------------------------------------------------------
# report_node
# ---------------------------------------------------------------------------


class TestReportNode:
    def test_successful_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.graph.generate_report",
            lambda **kwargs: str(tmp_path / "report.md"),
        )
        state = _make_state(
            df=_sample_df(),
            original_shape=(5, 3),
            issue_report={"missing_pct": {}, "duplicate_count": 0},
            cleaning_log=[],
            eda_results={"numeric_stats": "stats"},
            insights=["insight 1"],
            figure_paths=[],
        )
        result = report_node(state)

        assert result["report_path"] is not None
        assert len(result["reasoning_log"]) == 1
        assert result["reasoning_log"][0]["agent"] == "report_node"

    def test_report_with_real_generator(self, tmp_path, monkeypatch):
        # Use the real generate_report but redirect output_dir
        import src.graph as graph_mod

        original_report_node = graph_mod.report_node

        state = _make_state(
            df=_sample_df(),
            original_shape=(5, 3),
            issue_report={"missing_pct": {"age": 0.0}, "duplicate_count": 1},
            cleaning_log=["Removed 1 duplicate"],
            eda_results={"numeric_stats": "count 5"},
            insights=["Data looks clean"],
            figure_paths=[],
        )

        # Monkey-patch the output dir used inside report_node
        def patched_report_node(s):
            import src.report_generator as rg

            path = rg.generate_report(
                original_shape=s.get("original_shape") or (0, 0),
                cleaned_shape=(s["df"].shape[0], s["df"].shape[1]),
                issue_report=s.get("issue_report") or {},
                cleaning_log=s.get("cleaning_log") or [],
                eda_results=s.get("eda_results") or {},
                insights=s.get("insights") or [],
                figure_paths=s.get("figure_paths") or [],
                output_dir=str(tmp_path),
            )
            s["report_path"] = path
            graph_mod._append_reasoning(s, "report_node", f"Report at {path}")
            return s

        result = patched_report_node(state)
        assert result["report_path"].endswith("report.md")
        assert os.path.exists(result["report_path"])

    def test_no_dataframe_still_generates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.graph.generate_report",
            lambda **kwargs: str(tmp_path / "report.md"),
        )
        state = _make_state(df=None)
        result = report_node(state)

        assert result["report_path"] is not None

    def test_exception_logged(self, monkeypatch):
        monkeypatch.setattr(
            "src.graph.generate_report",
            MagicMock(side_effect=Exception("disk full")),
        )
        state = _make_state(df=_sample_df(), original_shape=(5, 3))
        result = report_node(state)

        assert result["report_path"] is None
        assert len(result["errors"]) == 1
        assert "disk full" in result["errors"][0]


# ---------------------------------------------------------------------------
# Cross-cutting: error resilience & reasoning log format
# ---------------------------------------------------------------------------


class TestCrossCutting:
    def test_all_nodes_return_state(self, tmp_path, monkeypatch):
        """Every node must return a dict (the state)."""
        monkeypatch.setattr(
            "src.graph.generate_plots",
            lambda df, output_dir: [],
        )
        monkeypatch.setattr(
            "src.graph.generate_report",
            lambda **kwargs: "output/report.md",
        )

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n1,2\n3,4\n")

        llm = _mock_llm("ok")
        state = _make_state(file_path=str(csv_file))

        state = load_csv_node(state)
        assert isinstance(state, dict)

        state = inspect_node(state, llm)
        assert isinstance(state, dict)

        state = clean_decision_node(state, llm)
        assert isinstance(state, dict)

        state = clean_node(state, llm)
        assert isinstance(state, dict)

        state = eda_node(state, llm)
        assert isinstance(state, dict)

        state = report_node(state)
        assert isinstance(state, dict)

    def test_reasoning_log_entries_have_required_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.graph.generate_plots",
            lambda df, output_dir: [],
        )
        monkeypatch.setattr(
            "src.graph.generate_report",
            lambda **kwargs: "output/report.md",
        )

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        llm = _mock_llm("reasoning text")
        state = _make_state(file_path=str(csv_file))

        state = load_csv_node(state)
        state = inspect_node(state, llm)
        state = clean_decision_node(state, llm)
        state = clean_node(state, llm)
        state = eda_node(state, llm)
        state = report_node(state)

        for entry in state["reasoning_log"]:
            assert "timestamp" in entry, f"Missing timestamp in {entry}"
            assert "agent" in entry, f"Missing agent in {entry}"
            assert "reasoning" in entry, f"Missing reasoning in {entry}"
            assert isinstance(entry["timestamp"], str)
            assert isinstance(entry["agent"], str)
            assert isinstance(entry["reasoning"], str)
