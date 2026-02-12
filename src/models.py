"""Core data models for the Autonomous Data Cleaning & EDA Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Central state object shared across all graph nodes."""

    # Input
    file_path: str

    # Data
    df: Optional[pd.DataFrame]
    original_shape: Optional[tuple[int, int]]

    # Inspection
    profile: Optional[str]
    issue_report: Optional[dict]

    # Cleaning
    cleaning_log: list[str]
    cleaning_iteration: int
    max_cleaning_iterations: int  # default: 3
    needs_more_cleaning: bool

    # EDA
    eda_results: Optional[dict]
    insights: list[str]
    figure_paths: list[str]

    # Output
    report_path: Optional[str]

    # Traceability
    errors: list[str]
    reasoning_log: list[dict]


@dataclass
class IssueReport:
    """Structured summary of detected data quality problems."""

    missing_pct: dict[str, float] = field(default_factory=dict)
    duplicate_count: int = 0
    outliers: dict[str, list[int]] = field(default_factory=dict)
    inconsistent_types: list[str] = field(default_factory=list)
    high_cardinality: list[str] = field(default_factory=list)
    zero_variance: list[str] = field(default_factory=list)
    severity_ranking: list[dict] = field(default_factory=list)


@dataclass
class CleaningLogEntry:
    """Record of a single cleaning operation."""

    timestamp: str
    operation: str
    columns_affected: list[str]
    parameters: dict
    rows_before: int
    rows_after: int
    description: str


@dataclass
class EDAResults:
    """Results from exploratory data analysis."""

    numeric_stats: str = ""
    categorical_stats: dict = field(default_factory=dict)
    correlation_matrix: Optional[str] = None
    figure_paths: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
