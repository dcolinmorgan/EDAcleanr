"""EDA tools for exploratory data analysis and visualization.

Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def describe_numeric(df: pd.DataFrame) -> str:
    """Return descriptive statistics for all numeric columns as a formatted string.

    Computes count, mean, std, min, 25%, 50%, 75%, max for each numeric column
    using ``DataFrame.describe()``.

    Args:
        df: Input DataFrame.

    Returns:
        Formatted string of descriptive statistics, or a message if no numeric
        columns exist.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return "No numeric columns found."
    return numeric_df.describe().to_string()


def describe_categorical(df: pd.DataFrame, top_n: int = 10) -> str:
    """Return value counts for the top N categories per categorical column.

    Considers columns with ``object``, ``string``, or ``category`` dtype as
    categorical.

    Args:
        df: Input DataFrame.
        top_n: Maximum number of top categories to show per column.

    Returns:
        Formatted string with value counts per categorical column, or a message
        if no categorical columns exist.
    """
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    if len(cat_cols) == 0:
        return "No categorical columns found."

    parts: list[str] = []
    for col in cat_cols:
        counts = df[col].value_counts().head(top_n)
        parts.append(f"Column: {col}")
        parts.append(counts.to_string())
        parts.append("")  # blank line separator

    return "\n".join(parts).rstrip()


def compute_correlation(df: pd.DataFrame) -> str:
    """Compute and return the correlation matrix for numeric columns.

    Skips computation if fewer than 2 numeric columns exist.

    Args:
        df: Input DataFrame.

    Returns:
        Formatted string of the correlation matrix, or a message if fewer than
        2 numeric columns are present.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return "Fewer than 2 numeric columns — correlation matrix skipped."
    corr = numeric_df.corr()
    return corr.to_string()


def generate_plots(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Generate histograms, box plots, and a correlation heatmap.

    Saves all figures to *output_dir* and returns the list of saved file paths.

    Plots generated:
    - One histogram per numeric column (``hist_<col>.png``)
    - One box plot per numeric column (``box_<col>.png``)
    - One correlation heatmap if ≥ 2 numeric columns (``correlation_heatmap.png``)

    Uses the matplotlib ``Agg`` backend for non-interactive rendering.

    Args:
        df: Input DataFrame.
        output_dir: Directory path where figures will be saved.

    Returns:
        List of file paths for all saved figures.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Histograms
    for col in numeric_cols:
        try:
            fig, ax = plt.subplots()
            df[col].dropna().hist(ax=ax, bins=20)
            ax.set_title(f"Histogram: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            path = os.path.join(output_dir, f"hist_{col}.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(path)
        except Exception:
            plt.close("all")

    # Box plots
    for col in numeric_cols:
        try:
            fig, ax = plt.subplots()
            df[[col]].dropna().boxplot(ax=ax)
            ax.set_title(f"Box Plot: {col}")
            path = os.path.join(output_dir, f"box_{col}.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(path)
        except Exception:
            plt.close("all")

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols))))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            path = os.path.join(output_dir, "correlation_heatmap.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(path)
        except Exception:
            plt.close("all")

    return saved_paths
