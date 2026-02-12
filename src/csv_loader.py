"""CSV Loader with automatic encoding and delimiter detection.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
"""

from __future__ import annotations

import csv
import io
import os
from typing import Optional

import chardet
import pandas as pd


def _detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet, falling back to utf-8."""
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
        if not raw:
            return "utf-8"
        result = chardet.detect(raw)
        encoding = result.get("encoding")
        if encoding:
            return encoding
    except Exception:
        pass
    return "utf-8"


def _detect_delimiter(text: str) -> str:
    """Detect CSV delimiter using csv.Sniffer, falling back to comma."""
    try:
        sample = text[:8192]
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _read_with_encoding(file_path: str, encoding: str) -> Optional[str]:
    """Try reading a file with the given encoding. Returns text or None."""
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except (UnicodeDecodeError, LookupError):
        return None


def _try_parse(text: str, delimiter: str) -> Optional[pd.DataFrame]:
    """Try parsing text as CSV with the given delimiter. Returns DataFrame or None."""
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")
        return df
    except Exception:
        return None


def load_csv(file_path: str) -> dict:
    """Load a CSV file with automatic encoding/delimiter detection.

    Args:
        file_path: Path to the CSV file.

    Returns:
        dict with keys:
            - "df": pd.DataFrame or None
            - "error": Optional[str] error message if loading failed
    """
    # Check file exists and is readable
    if not os.path.exists(file_path):
        return {"df": None, "error": f"File not found: {file_path}"}

    if not os.path.isfile(file_path):
        return {"df": None, "error": f"Path is not a file: {file_path}"}

    try:
        if os.path.getsize(file_path) == 0:
            return {"df": None, "error": "File is empty"}
    except OSError as e:
        return {"df": None, "error": f"Cannot read file: {e}"}

    # Detect encoding with fallback chain
    encoding = _detect_encoding(file_path)
    text = _read_with_encoding(file_path, encoding)

    if text is None:
        # Fallback: try utf-8
        text = _read_with_encoding(file_path, "utf-8")
    if text is None:
        # Fallback: try latin-1 (never fails for byte sequences)
        text = _read_with_encoding(file_path, "latin-1")
    if text is None:
        return {"df": None, "error": "Failed to decode file with any supported encoding"}

    # Check for content after decoding
    stripped = text.strip()
    if not stripped:
        return {"df": None, "error": "File is empty"}

    # Detect delimiter
    delimiter = _detect_delimiter(text)

    # Parse CSV
    df = _try_parse(text, delimiter)

    if df is None:
        return {"df": None, "error": "Failed to parse CSV file"}

    # If we got a single-column DataFrame with many rows, the delimiter
    # detection may have failed — retry with common delimiters
    if len(df.columns) == 1 and len(df) > 1:
        for alt_delim in ["\t", ";", "|", ","]:
            if alt_delim == delimiter:
                continue
            alt_df = _try_parse(text, alt_delim)
            if alt_df is not None and len(alt_df.columns) > 1:
                df = alt_df
                break

    # Validate at least one data row
    if len(df) == 0:
        return {"df": None, "error": "File contains only headers with no data rows"}

    return {"df": df, "error": None}
