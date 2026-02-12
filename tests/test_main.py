"""Tests for the CLI entry point (src/main.py).

Validates: Requirements 1.1, 6.1, 7.1
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.main import main, parse_args


# ---------------------------------------------------------------------------
# parse_args tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Unit tests for CLI argument parsing."""

    def test_positional_csv_file(self):
        args = parse_args(["data.csv"])
        assert args.csv_file == "data.csv"

    def test_defaults(self):
        args = parse_args(["data.csv"])
        assert args.provider == "openai"
        assert args.model is None
        assert args.output_dir == "output"
        assert args.max_iterations == 3

    def test_provider_flag(self):
        args = parse_args(["data.csv", "--provider", "anthropic"])
        assert args.provider == "anthropic"

    def test_bedrock_provider_flag(self):
        args = parse_args(["data.csv", "--provider", "bedrock"])
        assert args.provider == "bedrock"

    def test_model_flag(self):
        args = parse_args(["data.csv", "--model", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_output_dir_flag(self):
        args = parse_args(["data.csv", "--output-dir", "/tmp/results"])
        assert args.output_dir == "/tmp/results"

    def test_max_iterations_flag(self):
        args = parse_args(["data.csv", "--max-iterations", "5"])
        assert args.max_iterations == 5

    def test_all_flags_combined(self):
        args = parse_args([
            "input.csv",
            "--provider", "groq",
            "--model", "llama-3.1-70b-versatile",
            "--output-dir", "results",
            "--max-iterations", "10",
        ])
        assert args.csv_file == "input.csv"
        assert args.provider == "groq"
        assert args.model == "llama-3.1-70b-versatile"
        assert args.output_dir == "results"
        assert args.max_iterations == 10

    def test_invalid_provider_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["data.csv", "--provider", "invalid"])

    def test_missing_csv_file_rejected(self):
        with pytest.raises(SystemExit):
            parse_args([])


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    """Unit tests for the main() entry point."""

    def test_nonexistent_file_exits(self):
        """main() should exit with code 1 when the CSV file doesn't exist."""
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent_file_abc123.csv"])
        assert exc_info.value.code == 1

    def test_successful_run(self, tmp_path):
        """main() should print the report path on a successful run."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        output_dir = tmp_path / "output"

        fake_result = {"report_path": str(output_dir / "report.md"), "errors": []}

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = fake_result

        mock_llm = MagicMock()

        with (
            patch("src.llm_config.get_llm", return_value=mock_llm),
            patch("src.graph.build_graph", return_value=mock_graph),
            patch("builtins.print") as mock_print,
        ):
            main([
                str(csv_file),
                "--output-dir", str(output_dir),
            ])

        mock_print.assert_called_once_with(
            f"Report saved to: {output_dir / 'report.md'}"
        )

    def test_no_report_path_exits(self, tmp_path):
        """main() should exit with code 1 when no report is generated."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        fake_result = {"report_path": None, "errors": ["something went wrong"]}

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = fake_result

        mock_llm = MagicMock()

        with (
            patch("src.llm_config.get_llm", return_value=mock_llm),
            patch("src.graph.build_graph", return_value=mock_graph),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main([str(csv_file), "--output-dir", str(tmp_path / "out")])
            assert exc_info.value.code == 1

    def test_initial_state_structure(self, tmp_path):
        """The initial AgentState passed to graph.invoke should have the right keys."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y\n1,2\n")

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"report_path": "output/report.md"}

        mock_llm = MagicMock()

        with (
            patch("src.llm_config.get_llm", return_value=mock_llm),
            patch("src.graph.build_graph", return_value=mock_graph),
        ):
            main([str(csv_file), "--max-iterations", "7"])

        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["file_path"] == str(csv_file)
        assert call_args["cleaning_iteration"] == 0
        assert call_args["max_cleaning_iterations"] == 7
        assert call_args["cleaning_log"] == []
        assert call_args["errors"] == []
        assert call_args["reasoning_log"] == []

    def test_output_dirs_created(self, tmp_path):
        """main() should create the output and figures directories."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        output_dir = tmp_path / "custom_output"

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"report_path": "report.md"}

        mock_llm = MagicMock()

        with (
            patch("src.llm_config.get_llm", return_value=mock_llm),
            patch("src.graph.build_graph", return_value=mock_graph),
        ):
            main([str(csv_file), "--output-dir", str(output_dir)])

        assert (output_dir / "figures").is_dir()

    def test_exception_in_pipeline_exits(self, tmp_path):
        """main() should catch unexpected exceptions and exit with code 1."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        with patch("src.llm_config.get_llm", side_effect=RuntimeError("boom")):
            with pytest.raises(SystemExit) as exc_info:
                main([str(csv_file)])
            assert exc_info.value.code == 1
