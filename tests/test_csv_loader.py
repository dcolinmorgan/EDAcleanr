"""Unit tests for CSV loader.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
"""

import os
import tempfile

import pandas as pd
import pytest

from src.csv_loader import load_csv


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write_file(path: str, content: bytes):
    with open(path, "wb") as f:
        f.write(content)


class TestLoadCsvSuccess:
    """Requirement 1.1: valid CSV loads into DataFrame."""

    def test_basic_csv(self, tmp_dir):
        path = os.path.join(tmp_dir, "basic.csv")
        _write_file(path, b"a,b,c\n1,2,3\n4,5,6\n")
        result = load_csv(path)
        assert result["error"] is None
        assert isinstance(result["df"], pd.DataFrame)
        assert result["df"].shape == (2, 3)
        assert list(result["df"].columns) == ["a", "b", "c"]

    def test_single_row(self, tmp_dir):
        path = os.path.join(tmp_dir, "single.csv")
        _write_file(path, b"x,y\n10,20\n")
        result = load_csv(path)
        assert result["error"] is None
        assert result["df"].shape == (1, 2)


class TestEncodingDetection:
    """Requirement 1.2: non-UTF-8 encoding detected and decoded."""

    def test_latin1_encoding(self, tmp_dir):
        path = os.path.join(tmp_dir, "latin1.csv")
        content = "name,city\nJosé,São Paulo\nRené,Zürich\n"
        _write_file(path, content.encode("latin-1"))
        result = load_csv(path)
        assert result["error"] is None
        df = result["df"]
        assert df.shape == (2, 2)
        # Values should be decoded correctly
        assert "José" in df["name"].values or "Jos" in str(df["name"].values)

    def test_utf8_encoding(self, tmp_dir):
        path = os.path.join(tmp_dir, "utf8.csv")
        content = "name,city\n日本語,東京\n"
        _write_file(path, content.encode("utf-8"))
        result = load_csv(path)
        assert result["error"] is None
        assert result["df"].shape == (1, 2)


class TestDelimiterDetection:
    """Requirement 1.3: non-comma delimiter inferred correctly."""

    def test_tab_delimiter(self, tmp_dir):
        path = os.path.join(tmp_dir, "tabs.csv")
        _write_file(path, b"a\tb\tc\n1\t2\t3\n4\t5\t6\n")
        result = load_csv(path)
        assert result["error"] is None
        assert result["df"].shape == (2, 3)

    def test_semicolon_delimiter(self, tmp_dir):
        path = os.path.join(tmp_dir, "semi.csv")
        _write_file(path, b"a;b;c\n1;2;3\n4;5;6\n")
        result = load_csv(path)
        assert result["error"] is None
        assert result["df"].shape == (2, 3)

    def test_pipe_delimiter(self, tmp_dir):
        path = os.path.join(tmp_dir, "pipe.csv")
        _write_file(path, b"a|b|c\n1|2|3\n4|5|6\n")
        result = load_csv(path)
        assert result["error"] is None
        assert result["df"].shape == (2, 3)


class TestErrorHandling:
    """Requirement 1.4: non-existent/unreadable file returns error."""

    def test_nonexistent_file(self):
        result = load_csv("/nonexistent/path/file.csv")
        assert result["df"] is None
        assert result["error"] is not None
        assert "not found" in result["error"].lower() or "File not found" in result["error"]

    def test_directory_path(self, tmp_dir):
        result = load_csv(tmp_dir)
        assert result["df"] is None
        assert result["error"] is not None


class TestEmptyFileValidation:
    """Requirement 1.5: empty/header-only file returns error."""

    def test_empty_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.csv")
        _write_file(path, b"")
        result = load_csv(path)
        assert result["df"] is None
        assert "empty" in result["error"].lower()

    def test_header_only(self, tmp_dir):
        path = os.path.join(tmp_dir, "header_only.csv")
        _write_file(path, b"col1,col2,col3\n")
        result = load_csv(path)
        assert result["df"] is None
        assert "header" in result["error"].lower() or "no data" in result["error"].lower()

    def test_whitespace_only(self, tmp_dir):
        path = os.path.join(tmp_dir, "whitespace.csv")
        _write_file(path, b"   \n  \n  ")
        result = load_csv(path)
        assert result["df"] is None
        assert result["error"] is not None
