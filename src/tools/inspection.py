"""Inspection tools for data profiling and issue detection.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_df_info(df: pd.DataFrame) -> str:
    """Return shape, dtypes, head(10), tail(10), sample(5) as formatted string.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A formatted string containing the DataFrame profile.
    """
    parts: list[str] = []

    # Shape
    parts.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Dtypes
    parts.append("\nColumn Data Types:")
    for col in df.columns:
        parts.append(f"  {col}: {df[col].dtype}")

    # Head
    parts.append(f"\nFirst {min(10, len(df))} rows:")
    parts.append(df.head(10).to_string())

    # Tail
    parts.append(f"\nLast {min(10, len(df))} rows:")
    parts.append(df.tail(10).to_string())

    # Sample
    sample_n = min(5, len(df))
    parts.append(f"\nRandom sample of {sample_n} rows:")
    parts.append(df.sample(n=sample_n, random_state=42).to_string())

    return "\n".join(parts)


def detect_issues(df: pd.DataFrame) -> dict:
    """Detect data quality issues in the DataFrame.

    Returns dict with:
        - missing_pct: dict[str, float] — column -> % missing
        - duplicate_count: int — number of duplicate rows
        - outliers: dict[str, list] — column -> outlier row indices (IQR method)
        - inconsistent_types: list[str] — columns with mixed numeric/string values
        - high_cardinality: list[str] — categorical cols with >50% unique ratio
        - zero_variance: list[str] — columns with single unique value
    """
    result: dict = {
        "missing_pct": _detect_missing(df),
        "duplicate_count": _detect_duplicates(df),
        "outliers": _detect_outliers_iqr(df),
        "inconsistent_types": _detect_inconsistent_types(df),
        "high_cardinality": _detect_high_cardinality(df),
        "zero_variance": _detect_zero_variance(df),
    }
    return result


def _detect_missing(df: pd.DataFrame) -> dict[str, float]:
    """Return percentage of missing values per column."""
    if len(df) == 0:
        return {col: 0.0 for col in df.columns}
    pct = df.isna().sum() / len(df) * 100
    return {col: round(float(pct[col]), 4) for col in df.columns}


def _detect_duplicates(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows."""
    return int(df.duplicated().sum())


def _detect_outliers_iqr(df: pd.DataFrame) -> dict[str, list]:
    """Detect outliers using IQR method [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Returns dict mapping numeric column names to lists of row indices
    where outlier values occur.
    """
    outliers: dict[str, list] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Find indices where values are outside bounds
        mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        # Exclude NaN from outlier detection — NaN is not an outlier
        mask = mask & df[col].notna()
        outlier_indices = df.index[mask].tolist()

        if outlier_indices:
            outliers[col] = outlier_indices

    return outliers


def _is_string_dtype(series: pd.Series) -> bool:
    """Check if a series has a string-like dtype (object or StringDtype)."""
    return pd.api.types.is_string_dtype(series) and not pd.api.types.is_numeric_dtype(series)


def _detect_inconsistent_types(df: pd.DataFrame) -> list[str]:
    """Detect columns with mixed numeric and string values.

    A column is inconsistent if it has string/object dtype and contains a mix
    of values that can be parsed as numbers and values that cannot.
    """
    inconsistent: list[str] = []

    for col in df.columns:
        if not _is_string_dtype(df[col]):
            continue

        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        numeric_count = 0
        string_count = 0

        for val in non_null:
            if not isinstance(val, str):
                # Already a non-string type in an object column
                numeric_count += 1
                continue
            try:
                float(val)
                numeric_count += 1
            except (ValueError, TypeError):
                string_count += 1

        if numeric_count > 0 and string_count > 0:
            inconsistent.append(col)

    return inconsistent


def _detect_high_cardinality(df: pd.DataFrame) -> list[str]:
    """Detect categorical columns with high unique count (>50% unique ratio).

    Only considers string/object/categorical dtype columns.
    """
    high_card: list[str] = []

    for col in df.columns:
        if not (_is_string_dtype(df[col]) or df[col].dtype.name == "category"):
            continue

        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        unique_ratio = non_null.nunique() / len(non_null)
        if unique_ratio > 0.5:
            high_card.append(col)

    return high_card


def _detect_zero_variance(df: pd.DataFrame) -> list[str]:
    """Detect columns with a single unique value (zero variance)."""
    zero_var: list[str] = []

    for col in df.columns:
        if df[col].dropna().nunique() <= 1:
            zero_var.append(col)

    return zero_var
