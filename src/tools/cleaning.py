"""Cleaning tools for autonomous data cleaning operations.

Each function takes a DataFrame (and relevant parameters), performs a cleaning
operation, and returns a tuple of (cleaned_df, CleaningLogEntry).

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.models import CleaningLogEntry


def _now() -> str:
    """Return current timestamp as ISO format string."""
    return datetime.now().isoformat()


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Remove duplicate rows from the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, cleaning log entry).
    """
    rows_before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    rows_after = len(cleaned)
    duplicates_removed = rows_before - rows_after

    log = CleaningLogEntry(
        timestamp=_now(),
        operation="drop_duplicates",
        columns_affected=list(df.columns),
        parameters={},
        rows_before=rows_before,
        rows_after=rows_after,
        description=f"Removed {duplicates_removed} duplicate rows.",
    )
    return cleaned, log


def fill_missing(
    df: pd.DataFrame, column: str, strategy: str
) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Fill missing values in a column using the specified strategy.

    Args:
        df: Input DataFrame.
        column: Column name to fill.
        strategy: One of 'mean', 'median', 'mode', 'ffill', 'knn'.

    Returns:
        Tuple of (cleaned DataFrame, cleaning log entry).

    Raises:
        ValueError: If column not in DataFrame or strategy is invalid.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    valid_strategies = {"mean", "median", "mode", "ffill", "knn"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}."
        )

    rows_before = len(df)
    cleaned = df.copy()

    if strategy == "mean":
        fill_value = cleaned[column].mean()
        cleaned[column] = cleaned[column].fillna(fill_value)
    elif strategy == "median":
        fill_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(fill_value)
    elif strategy == "mode":
        mode_values = cleaned[column].mode()
        if len(mode_values) > 0:
            cleaned[column] = cleaned[column].fillna(mode_values.iloc[0])
    elif strategy == "ffill":
        cleaned[column] = cleaned[column].ffill()
        # Back-fill any remaining NaNs at the start
        cleaned[column] = cleaned[column].bfill()
    elif strategy == "knn":
        cleaned = _fill_knn(cleaned, column)

    log = CleaningLogEntry(
        timestamp=_now(),
        operation="fill_missing",
        columns_affected=[column],
        parameters={"strategy": strategy},
        rows_before=rows_before,
        rows_after=len(cleaned),
        description=f"Filled missing values in '{column}' using {strategy} strategy.",
    )
    return cleaned, log


def _fill_knn(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill missing values using KNN imputation, falling back to median.

    Uses sklearn's KNNImputer on numeric columns. If sklearn is not available
    or the target column is non-numeric, falls back to median imputation.
    """
    # Only KNN-impute numeric columns
    if not pd.api.types.is_numeric_dtype(df[column]):
        mode_values = df[column].mode()
        if len(mode_values) > 0:
            df[column] = df[column].fillna(mode_values.iloc[0])
        return df

    try:
        from sklearn.impute import KNNImputer

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if column not in numeric_cols:
            # Shouldn't happen given the check above, but be safe
            df[column] = df[column].fillna(df[column].median())
            return df

        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    except ImportError:
        # sklearn not available — fall back to median
        df[column] = df[column].fillna(df[column].median())

    return df


def convert_dtypes(
    df: pd.DataFrame, column: str, target_type: str
) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Convert a column to the specified data type with coercion.

    Uses ``errors='coerce'`` for numeric conversions so unparseable values
    become NaN rather than raising.

    Args:
        df: Input DataFrame.
        column: Column name to convert.
        target_type: Target type string — 'int', 'float', 'str', 'datetime', 'bool', 'category'.

    Returns:
        Tuple of (cleaned DataFrame, cleaning log entry).

    Raises:
        ValueError: If column not in DataFrame or target_type is unsupported.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    rows_before = len(df)
    cleaned = df.copy()
    original_dtype = str(cleaned[column].dtype)

    if target_type in ("int", "float"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if target_type == "int":
            # Use nullable integer to preserve NaN
            cleaned[column] = cleaned[column].astype("Int64")
    elif target_type == "str":
        cleaned[column] = cleaned[column].astype(str)
    elif target_type == "datetime":
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
    elif target_type == "bool":
        cleaned[column] = cleaned[column].astype(bool)
    elif target_type == "category":
        cleaned[column] = cleaned[column].astype("category")
    else:
        raise ValueError(
            f"Unsupported target_type '{target_type}'. "
            "Must be one of: int, float, str, datetime, bool, category."
        )

    log = CleaningLogEntry(
        timestamp=_now(),
        operation="convert_dtypes",
        columns_affected=[column],
        parameters={"target_type": target_type, "original_dtype": original_dtype},
        rows_before=rows_before,
        rows_after=len(cleaned),
        description=(
            f"Converted '{column}' from {original_dtype} to {target_type}."
        ),
    )
    return cleaned, log


def remove_outliers(
    df: pd.DataFrame, column: str, method: str = "iqr"
) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Remove or cap outliers in a numeric column.

    Args:
        df: Input DataFrame.
        column: Numeric column name.
        method: 'iqr' (cap at bounds) or 'zscore' (remove rows with |z| > 3).

    Returns:
        Tuple of (cleaned DataFrame, cleaning log entry).

    Raises:
        ValueError: If column not in DataFrame, not numeric, or method invalid.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")
    if method not in ("iqr", "zscore"):
        raise ValueError(f"Invalid method '{method}'. Must be 'iqr' or 'zscore'.")

    rows_before = len(df)
    cleaned = df.copy()

    if method == "iqr":
        series = cleaned[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
        description = (
            f"Capped outliers in '{column}' using IQR method "
            f"(bounds: [{lower:.4f}, {upper:.4f}])."
        )
        params = {"method": "iqr", "lower_bound": lower, "upper_bound": upper}
    else:
        # z-score: remove rows where |z| > 3
        mean = cleaned[column].mean()
        std = cleaned[column].std()
        if std == 0 or pd.isna(std):
            # No variance — nothing to remove
            description = (
                f"No outliers removed from '{column}' (zero variance)."
            )
            params = {"method": "zscore", "threshold": 3}
        else:
            z_scores = (cleaned[column] - mean) / std
            mask = z_scores.abs() <= 3
            # Keep rows where z-score is within bounds OR value is NaN
            mask = mask | cleaned[column].isna()
            cleaned = cleaned[mask].reset_index(drop=True)
            description = (
                f"Removed rows with |z-score| > 3 in '{column}' "
                f"(mean={mean:.4f}, std={std:.4f})."
            )
            params = {"method": "zscore", "threshold": 3, "mean": mean, "std": std}

    log = CleaningLogEntry(
        timestamp=_now(),
        operation="remove_outliers",
        columns_affected=[column],
        parameters=params,
        rows_before=rows_before,
        rows_after=len(cleaned),
        description=description,
    )
    return cleaned, log


def normalize_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Strip whitespace and lowercase all column names.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, cleaning log entry).
    """
    rows_before = len(df)
    cleaned = df.copy()
    original_columns = list(cleaned.columns)
    cleaned.columns = [str(c).strip().lower() for c in cleaned.columns]
    new_columns = list(cleaned.columns)

    changed = [
        f"'{orig}' -> '{new}'"
        for orig, new in zip(original_columns, new_columns)
        if orig != new
    ]

    log = CleaningLogEntry(
        timestamp=_now(),
        operation="normalize_columns",
        columns_affected=new_columns,
        parameters={},
        rows_before=rows_before,
        rows_after=len(cleaned),
        description=(
            f"Normalized {len(changed)} column name(s): {', '.join(changed)}"
            if changed
            else "All column names already normalized."
        ),
    )
    return cleaned, log


def strip_string_values(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Strip leading/trailing whitespace from all string columns.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, cleaning log entry).
    """
    rows_before = len(df)
    cleaned = df.copy()
    string_cols: list[str] = []

    for col in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].str.strip()
            string_cols.append(col)

    log = CleaningLogEntry(
        timestamp=_now(),
        operation="strip_string_values",
        columns_affected=string_cols,
        parameters={},
        rows_before=rows_before,
        rows_after=len(cleaned),
        description=(
            f"Stripped whitespace from {len(string_cols)} string column(s)."
        ),
    )
    return cleaned, log


def drop_useless_columns(
    df: pd.DataFrame, threshold: float = 0.9
) -> tuple[pd.DataFrame, CleaningLogEntry]:
    """Drop columns with >threshold fraction missing or zero variance.

    Args:
        df: Input DataFrame.
        threshold: Fraction of missing values above which a column is dropped.
            Defaults to 0.9 (90%).

    Returns:
        Tuple of (cleaned DataFrame, cleaning log entry).
    """
    rows_before = len(df)
    cleaned = df.copy()
    dropped: list[str] = []

    for col in list(cleaned.columns):
        if len(cleaned) == 0:
            break
        missing_frac = cleaned[col].isna().sum() / len(cleaned)
        if missing_frac > threshold:
            dropped.append(col)
            cleaned = cleaned.drop(columns=[col])
            continue

        # Zero variance: single unique non-null value (or all null)
        n_unique = cleaned[col].dropna().nunique()
        if n_unique <= 1:
            dropped.append(col)
            cleaned = cleaned.drop(columns=[col])

    log = CleaningLogEntry(
        timestamp=_now(),
        operation="drop_useless_columns",
        columns_affected=dropped,
        parameters={"threshold": threshold},
        rows_before=rows_before,
        rows_after=len(cleaned),
        description=(
            f"Dropped {len(dropped)} useless column(s): {', '.join(dropped)}"
            if dropped
            else "No useless columns found."
        ),
    )
    return cleaned, log
