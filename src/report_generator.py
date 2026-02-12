"""Report generator that compiles cleaning and EDA results into a Markdown report."""

from __future__ import annotations

import os
from pathlib import Path

from src.models import CleaningLogEntry


def _format_issue_report(issue_report: dict | str) -> str:
    """Format the issue report dict into Markdown content."""
    if not isinstance(issue_report, dict):
        return str(issue_report) if issue_report else "No issues detected.\n"

    lines: list[str] = []

    missing = issue_report.get("missing_pct", {})
    if missing:
        lines.append("### Missing Values\n")
        lines.append("| Column | Missing % |")
        lines.append("|--------|-----------|")
        for col, pct in missing.items():
            lines.append(f"| {col} | {pct:.1f}% |")
        lines.append("")

    dup_count = issue_report.get("duplicate_count", 0)
    lines.append(f"### Duplicate Rows\n")
    lines.append(f"- **{dup_count}** duplicate rows detected\n")

    outliers = issue_report.get("outliers", {})
    if outliers:
        lines.append("### Outliers\n")
        lines.append("| Column | Outlier Count |")
        lines.append("|--------|---------------|")
        for col, indices in outliers.items():
            lines.append(f"| {col} | {len(indices)} |")
        lines.append("")

    inconsistent = issue_report.get("inconsistent_types", [])
    if inconsistent:
        lines.append("### Inconsistent Types\n")
        for col in inconsistent:
            lines.append(f"- {col}")
        lines.append("")

    high_card = issue_report.get("high_cardinality", [])
    if high_card:
        lines.append("### High Cardinality Columns\n")
        for col in high_card:
            lines.append(f"- {col}")
        lines.append("")

    zero_var = issue_report.get("zero_variance", [])
    if zero_var:
        lines.append("### Zero Variance Columns\n")
        for col in zero_var:
            lines.append(f"- {col}")
        lines.append("")

    return "\n".join(lines)


def _format_cleaning_log_entry(entry: str | CleaningLogEntry) -> str:
    """Format a single cleaning log entry as a Markdown list item."""
    if isinstance(entry, CleaningLogEntry):
        parts = [f"**{entry.operation}**"]
        if entry.columns_affected:
            parts.append(f"columns: {', '.join(entry.columns_affected)}")
        if entry.parameters:
            params_str = ", ".join(f"{k}={v}" for k, v in entry.parameters.items())
            parts.append(f"params: {{{params_str}}}")
        parts.append(f"rows: {entry.rows_before} → {entry.rows_after}")
        if entry.description:
            parts.append(entry.description)
        return "- " + " | ".join(parts)
    return f"- {entry}"


def _format_eda_results(eda_results: dict | str) -> str:
    """Format EDA results dict into Markdown content."""
    if not isinstance(eda_results, dict):
        return str(eda_results) if eda_results else ""

    lines: list[str] = []

    numeric_stats = eda_results.get("numeric_stats", "")
    if numeric_stats:
        lines.append("### Numeric Statistics\n")
        lines.append("```")
        lines.append(str(numeric_stats))
        lines.append("```\n")

    categorical_stats = eda_results.get("categorical_stats", {})
    if categorical_stats:
        lines.append("### Categorical Statistics\n")
        if isinstance(categorical_stats, dict):
            for col, counts in categorical_stats.items():
                lines.append(f"**{col}**:\n")
                if isinstance(counts, dict):
                    for val, count in counts.items():
                        lines.append(f"- {val}: {count}")
                else:
                    lines.append(f"- {counts}")
                lines.append("")
        else:
            lines.append("```")
            lines.append(str(categorical_stats))
            lines.append("```\n")

    correlation = eda_results.get("correlation_matrix")
    if correlation:
        lines.append("### Correlation Matrix\n")
        lines.append("```")
        lines.append(str(correlation))
        lines.append("```\n")

    return "\n".join(lines)


def generate_report(
    original_shape: tuple[int, int],
    cleaned_shape: tuple[int, int],
    issue_report: dict | str,
    cleaning_log: list[str | CleaningLogEntry],
    eda_results: dict | str,
    insights: list[str],
    figure_paths: list[str],
    output_dir: str,
) -> str:
    """Generate a Markdown report and save to output_dir/report.md.

    Args:
        original_shape: (rows, cols) of the original dataset.
        cleaned_shape: (rows, cols) of the cleaned dataset.
        issue_report: Dict of detected issues (missing_pct, duplicate_count, etc.).
        cleaning_log: List of cleaning log entries (strings or CleaningLogEntry objects).
        eda_results: Dict with numeric_stats, categorical_stats, correlation_matrix.
        insights: List of insight strings from EDA.
        figure_paths: List of paths to generated figure files.
        output_dir: Directory to save the report.

    Returns:
        The path to the saved report file.
    """
    os.makedirs(output_dir, exist_ok=True)

    sections: list[str] = []

    # Title
    sections.append("# Data Cleaning & EDA Report\n")

    # Dataset Overview
    sections.append("## Dataset Overview\n")
    sections.append(f"- **Original shape**: {original_shape[0]} rows × {original_shape[1]} columns")
    sections.append(f"- **Cleaned shape**: {cleaned_shape[0]} rows × {cleaned_shape[1]} columns")
    rows_diff = original_shape[0] - cleaned_shape[0]
    cols_diff = original_shape[1] - cleaned_shape[1]
    sections.append(f"- **Rows removed**: {rows_diff}")
    sections.append(f"- **Columns removed**: {cols_diff}\n")

    # Issues Found
    sections.append("## Issues Found\n")
    sections.append(_format_issue_report(issue_report))

    # Cleaning Actions
    sections.append("## Cleaning Actions\n")
    if cleaning_log:
        for entry in cleaning_log:
            sections.append(_format_cleaning_log_entry(entry))
    else:
        sections.append("No cleaning actions were performed.\n")
    sections.append("")

    # Key Statistics
    sections.append("## Key Statistics\n")
    sections.append(_format_eda_results(eda_results))

    # Insights
    sections.append("## Insights\n")
    if insights:
        for i, insight in enumerate(insights, 1):
            sections.append(f"{i}. {insight}")
    else:
        sections.append("No insights generated.\n")
    sections.append("")

    # Figures
    if figure_paths:
        sections.append("## Figures\n")
        report_dir = Path(output_dir)
        for fig_path in figure_paths:
            fig = Path(fig_path)
            try:
                rel_path = fig.relative_to(report_dir)
            except ValueError:
                rel_path = fig
            fig_name = fig.stem.replace("_", " ").title()
            sections.append(f"![{fig_name}]({rel_path})\n")

    report_content = "\n".join(sections)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_path
