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


def format_reasoning_log(reasoning_log: list[dict]) -> str:
    """Format the agent's reasoning log into Markdown content."""
    if not reasoning_log:
        return "No reasoning log available.\n"

    lines: list[str] = []
    lines.append("## Agent Reasoning Log\n")
    lines.append("This section shows the step-by-step reasoning process of the AI agent.\n")

    for i, entry in enumerate(reasoning_log, 1):
        agent = entry.get("agent", "unknown")
        reasoning = entry.get("reasoning", "")
        lines.append(f"### Step {i}: {agent}\n")
        lines.append(reasoning)
        lines.append("")

    return "\n".join(lines)


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


_NODE_LABELS = {
    "load_csv_node": "Load CSV",
    "inspect_node": "Inspect Data",
    "clean_decision_node": "Cleaning Decision",
    "clean_node": "Clean Data",
    "eda_node": "Exploratory Data Analysis",
    "report_node": "Generate Report",
}


def _format_reasoning_text(raw: str) -> str:
    """Extract human-readable text from raw LLM reasoning output.

    Handles three formats:
    - Plain strings (returned as-is)
    - Python repr of list-of-dicts with 'type'/'text' keys (Bedrock/Anthropic)
    - Strings containing <thinking> XML tags
    - Prefixed strings like "Iteration 1: [...]"
    """
    import ast
    import re

    text = str(raw).strip()
    if not text:
        return ""

    # Handle "Iteration N: <content>" prefix from clean_node
    iter_match = re.match(r"^(Iteration \d+):\s*(.*)$", text, re.DOTALL)
    if iter_match:
        prefix = iter_match.group(1)
        remainder = iter_match.group(2).strip()
        formatted_remainder = _format_reasoning_text(remainder)
        if formatted_remainder:
            return f"**{prefix}**\n\n{formatted_remainder}"
        return f"**{prefix}**"

    # Try to parse as a Python literal (list of dicts from Bedrock responses)
    parsed_items: list | None = None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed_items = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass

    parts: list[str] = []
    tool_calls: list[str] = []

    if isinstance(parsed_items, list):
        for item in parsed_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                t = item.get("text", "").strip()
                if t:
                    parts.append(t)
            elif item.get("type") == "tool_use":
                name = item.get("name", "unknown")
                args = item.get("input", {})
                if args:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    tool_calls.append(f"`{name}({args_str})`")
                else:
                    tool_calls.append(f"`{name}()`")
    else:
        parts.append(text)

    # Clean up <thinking> tags and escaped newlines
    cleaned: list[str] = []
    for part in parts:
        # Replace literal \n with actual newlines
        part = part.replace("\\n", "\n")
        # Extract content from <thinking> tags
        thinking_matches = re.findall(r"<thinking>(.*?)</thinking>", part, re.DOTALL)
        if thinking_matches:
            for match in thinking_matches:
                cleaned.append(match.strip())
            remainder = re.sub(r"<thinking>.*?</thinking>", "", part, flags=re.DOTALL).strip()
            if remainder:
                cleaned.append(remainder)
        else:
            # Try to extract plan from JSON decision blocks
            # Strip markdown code fences first
            stripped = re.sub(r"```(?:json)?\s*", "", part).strip()
            stripped = re.sub(r"```\s*$", "", stripped).strip()
            json_match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if json_match:
                try:
                    import json as _json
                    decision = _json.loads(json_match.group(), strict=False)
                    plan = decision.get("plan", "")
                    needs = decision.get("needs_cleaning")
                    decision_parts: list[str] = []
                    if needs is not None:
                        decision_parts.append(
                            "**Decision:** Cleaning needed"
                            if needs
                            else "**Decision:** No cleaning needed"
                        )
                    if plan:
                        decision_parts.append(f"\n{plan}")
                    if decision_parts:
                        cleaned.append("\n".join(decision_parts))
                    else:
                        cleaned.append(part)
                except (ValueError, KeyError):
                    cleaned.append(part)
            else:
                cleaned.append(part)

    output: list[str] = []
    reasoning_text = "\n\n".join(c for c in cleaned if c)
    if reasoning_text:
        output.append(reasoning_text)

    if tool_calls:
        output.append("\n**Tools called:**\n")
        for tc in tool_calls:
            output.append(f"- {tc}")

    return "\n".join(output)


def generate_reasoning_report(
    reasoning_log: list[dict],
    errors: list[str],
    output_dir: str,
) -> str:
    """Generate a Markdown report of the agent's reasoning steps.

    Args:
        reasoning_log: List of dicts with 'timestamp', 'agent', 'reasoning'.
        errors: List of error strings encountered during the run.
        output_dir: Directory to save the report.

    Returns:
        The path to the saved reasoning report file.
    """
    os.makedirs(output_dir, exist_ok=True)

    sections: list[str] = []
    sections.append("# Agent Reasoning Log\n")

    if not reasoning_log:
        sections.append("No reasoning entries recorded.\n")
    else:
        for i, entry in enumerate(reasoning_log, 1):
            agent = entry.get("agent", "unknown")
            label = _NODE_LABELS.get(agent, agent)
            timestamp = entry.get("timestamp", "")
            reasoning = entry.get("reasoning", "")

            sections.append(f"## Step {i}: {label}\n")
            if timestamp:
                # Format ISO timestamp to something friendlier
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp)
                    friendly = dt.strftime("%H:%M:%S")
                    sections.append(f"*{friendly} UTC*\n")
                except (ValueError, TypeError):
                    sections.append(f"*{timestamp}*\n")

            formatted = _format_reasoning_text(reasoning)
            if formatted:
                sections.append(f"{formatted}\n")

    if errors:
        sections.append("---\n")
        sections.append("## Errors\n")
        for err in errors:
            sections.append(f"- {err}")
        sections.append("")

    content = "\n".join(sections)
    report_path = os.path.join(output_dir, "reasoning.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    return report_path
