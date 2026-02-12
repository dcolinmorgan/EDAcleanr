"""LangGraph agent orchestrator — graph nodes and workflow builder.

Each node function accepts an AgentState dict (and optionally an LLM) and
returns an updated state dict.  All nodes wrap their logic in try/except,
log errors to state["errors"], and append reasoning entries with timestamps
to state["reasoning_log"].

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 7.2, 7.3, 7.4, 8.3
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.csv_loader import load_csv
from src.models import AgentState
from src.report_generator import generate_report, generate_reasoning_report
from src.tools.cleaning import (
    convert_dtypes,
    drop_duplicates,
    drop_useless_columns,
    fill_missing,
    normalize_columns,
    remove_outliers,
    strip_string_values,
)
from src.tools.eda import (
    compute_correlation,
    describe_categorical,
    describe_numeric,
    generate_plots,
)
from src.tools.inspection import detect_issues, get_df_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _append_reasoning(state: dict, agent: str, reasoning: str) -> None:
    """Append a reasoning log entry to state."""
    if "reasoning_log" not in state or state["reasoning_log"] is None:
        state["reasoning_log"] = []
    state["reasoning_log"].append(
        {"timestamp": _timestamp(), "agent": agent, "reasoning": reasoning}
    )


def _append_error(state: dict, error_msg: str) -> None:
    """Append an error message to state."""
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []
    state["errors"].append(error_msg)


def _ensure_list(state: dict, key: str) -> None:
    """Ensure *key* exists in state as a list."""
    if key not in state or state[key] is None:
        state[key] = []


def _invoke_llm_with_tools(llm: Any, tools: list, messages: list) -> AIMessage:
    """Bind tools to the LLM and invoke with the given messages.

    Handles up to 3 retries on tool-call errors.
    """
    llm_with_tools = llm.bind_tools(tools)
    max_retries = 3
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)
            return response
        except Exception as exc:
            last_error = exc
            messages.append(
                HumanMessage(
                    content=f"Tool call error (attempt {attempt + 1}): {exc}. Please retry."
                )
            )
    raise last_error  # type: ignore[misc]


def _execute_tool_calls(response: AIMessage, tool_map: dict) -> list[str]:
    """Execute tool calls from an AIMessage and return result strings."""
    results: list[str] = []
    for tc in getattr(response, "tool_calls", []) or []:
        fn_name = tc["name"]
        fn_args = tc["args"]
        if fn_name in tool_map:
            try:
                result = tool_map[fn_name](**fn_args)
                results.append(str(result))
            except Exception as exc:
                results.append(f"Error calling {fn_name}: {exc}")
    return results


# ---------------------------------------------------------------------------
# Node: load_csv_node
# ---------------------------------------------------------------------------


def load_csv_node(state: AgentState) -> AgentState:
    """Load a CSV file into the state DataFrame.

    Calls ``csv_loader.load_csv`` and stores the DataFrame and original shape.
    On error the DataFrame is set to ``None`` and the error is logged.
    """
    _ensure_list(state, "errors")
    _ensure_list(state, "reasoning_log")

    try:
        file_path = state.get("file_path", "")
        result = load_csv(file_path)

        if result["error"]:
            _append_error(state, f"CSV load error: {result['error']}")
            _append_reasoning(
                state,
                "load_csv_node",
                f"Failed to load CSV: {result['error']}",
            )
            state["df"] = None
            state["original_shape"] = None
            return state

        df: pd.DataFrame = result["df"]
        state["df"] = df
        state["original_shape"] = (df.shape[0], df.shape[1])
        _append_reasoning(
            state,
            "load_csv_node",
            f"Successfully loaded CSV with shape {df.shape}.",
        )
    except Exception as exc:
        _append_error(state, f"load_csv_node exception: {exc}")
        _append_reasoning(state, "load_csv_node", f"Exception: {exc}")
        state["df"] = None
        state["original_shape"] = None

    return state


# ---------------------------------------------------------------------------
# Node: inspect_node
# ---------------------------------------------------------------------------


def inspect_node(state: AgentState, llm: Any) -> AgentState:
    """Use the LLM with inspection tools to profile data and detect issues.

    Stores ``profile`` and ``issue_report`` in state.
    """
    _ensure_list(state, "errors")
    _ensure_list(state, "reasoning_log")

    try:
        df = state.get("df")
        if df is None:
            _append_error(state, "inspect_node: No DataFrame available.")
            _append_reasoning(state, "inspect_node", "Skipped — no DataFrame loaded.")
            state["profile"] = None
            state["issue_report"] = None
            return state

        # Run inspection tools directly to get data for the LLM
        profile = get_df_info(df)
        issues = detect_issues(df)

        state["profile"] = profile
        state["issue_report"] = issues

        # Build LangChain tools for the LLM to reason about
        @tool
        def inspect_profile() -> str:
            """Return the DataFrame profile (shape, dtypes, head, tail, sample)."""
            return profile

        @tool
        def inspect_issues() -> str:
            """Return detected data quality issues as JSON."""
            return json.dumps(issues, default=str)

        tools = [inspect_profile, inspect_issues]
        tool_map = {t.name: t.invoke for t in tools}

        messages = [
            SystemMessage(
                content=(
                    "You are a data inspection agent. Use the provided tools to "
                    "profile the dataset and identify data quality issues. "
                    "Summarize your findings including severity and priority."
                )
            ),
            HumanMessage(
                content="Please inspect the loaded dataset and report all issues."
            ),
        ]

        response = _invoke_llm_with_tools(llm, tools, messages)

        # Execute any tool calls the LLM made
        _execute_tool_calls(response, tool_map)

        reasoning = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        _append_reasoning(state, "inspect_node", reasoning)

    except Exception as exc:
        _append_error(state, f"inspect_node exception: {exc}")
        _append_reasoning(state, "inspect_node", f"Exception: {exc}")
        # Degraded output — run tools directly without LLM
        try:
            df = state.get("df")
            if df is not None:
                if state.get("profile") is None:
                    state["profile"] = get_df_info(df)
                if state.get("issue_report") is None:
                    state["issue_report"] = detect_issues(df)
        except Exception:
            state["profile"] = None
            state["issue_report"] = None

    return state


# ---------------------------------------------------------------------------
# Node: clean_decision_node
# ---------------------------------------------------------------------------


def clean_decision_node(state: AgentState, llm: Any) -> AgentState:
    """LLM reviews the issue report and decides a cleaning plan.

    Sets ``needs_more_cleaning`` in state.
    """
    _ensure_list(state, "errors")
    _ensure_list(state, "reasoning_log")
    _ensure_list(state, "cleaning_log")

    try:
        issue_report = state.get("issue_report")
        if issue_report is None:
            _append_reasoning(
                state,
                "clean_decision_node",
                "No issue report available — skipping cleaning.",
            )
            state["needs_more_cleaning"] = False
            return state

        issues_json = json.dumps(issue_report, default=str)

        messages = [
            SystemMessage(
                content=(
                    "You are a data cleaning decision agent. Review the issue "
                    "report and decide whether cleaning is needed. Respond with "
                    'a JSON object: {"needs_cleaning": true/false, "plan": '
                    '"description of cleaning steps"}.'
                )
            ),
            HumanMessage(
                content=f"Issue report:\n{issues_json}\n\nShould we clean this data?"
            ),
        ]

        response = llm.invoke(messages)
        reasoning = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # Try to parse the LLM decision
        needs_cleaning = True  # default to cleaning
        try:
            json_match = re.search(r"\{.*\}", reasoning, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                needs_cleaning = decision.get("needs_cleaning", True)
        except (json.JSONDecodeError, AttributeError):
            pass

        state["needs_more_cleaning"] = needs_cleaning
        if "cleaning_iteration" not in state:
            state["cleaning_iteration"] = 0

        _append_reasoning(state, "clean_decision_node", reasoning)

    except Exception as exc:
        _append_error(state, f"clean_decision_node exception: {exc}")
        _append_reasoning(state, "clean_decision_node", f"Exception: {exc}")
        # Default: attempt cleaning
        state["needs_more_cleaning"] = True
        if "cleaning_iteration" not in state:
            state["cleaning_iteration"] = 0

    return state


# ---------------------------------------------------------------------------
# Node: clean_node
# ---------------------------------------------------------------------------


def clean_node(state: AgentState, llm: Any) -> AgentState:
    """Use the LLM with cleaning tools to apply fixes.

    Increments ``cleaning_iteration`` and appends entries to ``cleaning_log``.
    """
    _ensure_list(state, "errors")
    _ensure_list(state, "reasoning_log")
    _ensure_list(state, "cleaning_log")

    try:
        df = state.get("df")
        if df is None:
            _append_error(state, "clean_node: No DataFrame available.")
            _append_reasoning(state, "clean_node", "Skipped — no DataFrame.")
            state["needs_more_cleaning"] = False
            return state

        # Increment iteration counter
        iteration = state.get("cleaning_iteration", 0) + 1
        state["cleaning_iteration"] = iteration
        max_iterations = state.get("max_cleaning_iterations", 3)

        # Mutable container so tool closures can update the df
        df_holder = {"df": df.copy()}
        log_entries: list[str] = []

        @tool
        def tool_drop_duplicates() -> str:
            """Remove duplicate rows from the DataFrame."""
            cleaned, log_entry = drop_duplicates(df_holder["df"])
            df_holder["df"] = cleaned
            log_entries.append(log_entry.description)
            return log_entry.description

        @tool
        def tool_fill_missing(column: str, strategy: str) -> str:
            """Fill missing values in a column. strategy: mean|median|mode|ffill|knn."""
            cleaned, log_entry = fill_missing(df_holder["df"], column, strategy)
            df_holder["df"] = cleaned
            log_entries.append(log_entry.description)
            return log_entry.description

        @tool
        def tool_convert_dtypes(column: str, target_type: str) -> str:
            """Convert a column to a target type. target_type: int|float|str|datetime|bool|category."""
            cleaned, log_entry = convert_dtypes(df_holder["df"], column, target_type)
            df_holder["df"] = cleaned
            log_entries.append(log_entry.description)
            return log_entry.description

        @tool
        def tool_remove_outliers(column: str, method: str = "iqr") -> str:
            """Remove/cap outliers. method: iqr|zscore."""
            cleaned, log_entry = remove_outliers(df_holder["df"], column, method)
            df_holder["df"] = cleaned
            log_entries.append(log_entry.description)
            return log_entry.description

        @tool
        def tool_normalize_columns() -> str:
            """Strip whitespace and lowercase all column names."""
            cleaned, log_entry = normalize_columns(df_holder["df"])
            df_holder["df"] = cleaned
            log_entries.append(log_entry.description)
            return log_entry.description

        @tool
        def tool_strip_string_values() -> str:
            """Strip leading/trailing whitespace from all string columns."""
            cleaned, log_entry = strip_string_values(df_holder["df"])
            df_holder["df"] = cleaned
            log_entries.append(log_entry.description)
            return log_entry.description

        @tool
        def tool_drop_useless_columns(threshold: float = 0.9) -> str:
            """Drop columns with >threshold missing or zero variance."""
            cleaned, log_entry = drop_useless_columns(df_holder["df"], threshold)
            df_holder["df"] = cleaned
            log_entries.append(log_entry.description)
            return log_entry.description

        tools = [
            tool_drop_duplicates,
            tool_fill_missing,
            tool_convert_dtypes,
            tool_remove_outliers,
            tool_normalize_columns,
            tool_strip_string_values,
            tool_drop_useless_columns,
        ]
        tool_map = {t.name: t.invoke for t in tools}

        issue_summary = json.dumps(state.get("issue_report", {}), default=str)
        messages = [
            SystemMessage(
                content=(
                    "You are a data cleaning agent. Use the provided tools to "
                    "clean the dataset based on the detected issues. "
                    f"This is cleaning iteration {iteration} of {max_iterations}. "
                    "Apply appropriate cleaning operations."
                )
            ),
            HumanMessage(
                content=f"Issues detected:\n{issue_summary}\n\nPlease clean the data."
            ),
        ]

        response = _invoke_llm_with_tools(llm, tools, messages)

        # Execute tool calls
        _execute_tool_calls(response, tool_map)

        # Update state with cleaned df
        state["df"] = df_holder["df"]
        state["cleaning_log"].extend(log_entries)

        # Decide if more cleaning is needed
        if iteration >= max_iterations:
            state["needs_more_cleaning"] = False
        else:
            state["needs_more_cleaning"] = False

        reasoning = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        _append_reasoning(
            state,
            "clean_node",
            f"Iteration {iteration}: {reasoning}",
        )

    except Exception as exc:
        _append_error(state, f"clean_node exception: {exc}")
        _append_reasoning(state, "clean_node", f"Exception: {exc}")
        state["needs_more_cleaning"] = False

    return state


# ---------------------------------------------------------------------------
# Node: eda_node
# ---------------------------------------------------------------------------


def eda_node(state: AgentState, llm: Any) -> AgentState:
    """Use the LLM with EDA tools to analyze the cleaned data.

    Stores ``eda_results``, ``insights``, and ``figure_paths`` in state.
    """
    _ensure_list(state, "errors")
    _ensure_list(state, "reasoning_log")
    _ensure_list(state, "insights")
    _ensure_list(state, "figure_paths")

    try:
        df = state.get("df")
        if df is None:
            _append_error(state, "eda_node: No DataFrame available.")
            _append_reasoning(state, "eda_node", "Skipped — no DataFrame.")
            state["eda_results"] = None
            return state

        output_dir = "output/figures"

        # Run EDA tools directly to collect results
        numeric_stats = describe_numeric(df)
        categorical_stats = describe_categorical(df)
        correlation = compute_correlation(df)
        fig_paths = generate_plots(df, output_dir)

        eda_results = {
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "correlation_matrix": correlation,
        }
        state["eda_results"] = eda_results
        state["figure_paths"] = fig_paths

        # Build tools for LLM to reason about the analysis
        @tool
        def eda_numeric_stats() -> str:
            """Return descriptive statistics for numeric columns."""
            return numeric_stats

        @tool
        def eda_categorical_stats() -> str:
            """Return value counts for categorical columns."""
            return categorical_stats

        @tool
        def eda_correlation() -> str:
            """Return the correlation matrix."""
            return correlation

        @tool
        def eda_figures() -> str:
            """Return the list of generated figure paths."""
            return json.dumps(fig_paths)

        tools = [eda_numeric_stats, eda_categorical_stats, eda_correlation, eda_figures]
        tool_map = {t.name: t.invoke for t in tools}

        messages = [
            SystemMessage(
                content=(
                    "You are an EDA agent. Analyze the dataset statistics and "
                    "generate 3 to 5 key insights including trends, anomalies, "
                    "and recommendations. Use the provided tools to review the "
                    "analysis results."
                )
            ),
            HumanMessage(
                content="Please analyze the cleaned dataset and provide insights."
            ),
        ]

        response = _invoke_llm_with_tools(llm, tools, messages)

        # Execute any tool calls
        _execute_tool_calls(response, tool_map)

        reasoning = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # Extract insights from LLM response
        if reasoning:
            lines = [
                line.strip()
                for line in reasoning.split("\n")
                if line.strip()
            ]
            insights = [ln for ln in lines if ln] or [reasoning]
            state["insights"] = insights

        _append_reasoning(state, "eda_node", reasoning)

    except Exception as exc:
        _append_error(state, f"eda_node exception: {exc}")
        _append_reasoning(state, "eda_node", f"Exception: {exc}")
        # Degraded output — run tools directly without LLM
        try:
            df = state.get("df")
            if df is not None:
                if state.get("eda_results") is None:
                    state["eda_results"] = {
                        "numeric_stats": describe_numeric(df),
                        "categorical_stats": describe_categorical(df),
                        "correlation_matrix": compute_correlation(df),
                    }
                if not state.get("figure_paths"):
                    state["figure_paths"] = generate_plots(df, "output/figures")
                if not state.get("insights"):
                    state["insights"] = [
                        "EDA completed with degraded output due to LLM error."
                    ]
        except Exception:
            state["eda_results"] = None
            state["insights"] = []
            state["figure_paths"] = []

    return state


# ---------------------------------------------------------------------------
# Node: report_node
# ---------------------------------------------------------------------------


def report_node(state: AgentState) -> AgentState:
    """Generate the final Markdown report from accumulated state data."""
    _ensure_list(state, "errors")
    _ensure_list(state, "reasoning_log")

    try:
        df = state.get("df")
        original_shape = state.get("original_shape") or (0, 0)
        cleaned_shape = (df.shape[0], df.shape[1]) if df is not None else (0, 0)
        issue_report = state.get("issue_report") or {}
        cleaning_log = state.get("cleaning_log") or []
        eda_results = state.get("eda_results") or {}
        insights = state.get("insights") or []
        figure_paths = state.get("figure_paths") or []
        output_dir = "output"

        report_path = generate_report(
            original_shape=original_shape,
            cleaned_shape=cleaned_shape,
            issue_report=issue_report,
            cleaning_log=cleaning_log,
            eda_results=eda_results,
            insights=insights,
            figure_paths=figure_paths,
            output_dir=output_dir,
        )

        state["report_path"] = report_path
        _append_reasoning(
            state,
            "report_node",
            f"Report generated at {report_path}.",
        )

        # Also generate the reasoning log report
        generate_reasoning_report(
            reasoning_log=state.get("reasoning_log") or [],
            errors=state.get("errors") or [],
            output_dir=output_dir,
        )

    except Exception as exc:
        _append_error(state, f"report_node exception: {exc}")
        _append_reasoning(state, "report_node", f"Exception: {exc}")
        state["report_path"] = None

    return state


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _should_continue_cleaning(state: AgentState) -> str:
    """Conditional edge: loop back to clean or proceed to EDA."""
    if (
        state.get("needs_more_cleaning", False)
        and state.get("cleaning_iteration", 0) < state.get("max_cleaning_iterations", 3)
    ):
        return "clean"
    return "eda"


def build_graph(llm: Any) -> Any:
    """Build and compile the LangGraph workflow.

    Nodes: load_csv_node → inspect_node → clean_decision_node → clean_node
           → (conditional) → eda_node → report_node
    The conditional edge from clean_node loops back to clean_node when
    ``needs_more_cleaning`` is True and ``cleaning_iteration`` is below
    ``max_cleaning_iterations``; otherwise it proceeds to eda_node.

    Args:
        llm: A tool-calling capable LangChain chat model.

    Returns:
        A compiled LangGraph ``StateGraph``.

    Validates: Requirements 6.1, 6.2
    """
    from functools import partial

    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)

    # -- Add nodes (bind llm via partial for nodes that need it) --
    graph.add_node("load", load_csv_node)
    graph.add_node("inspect", partial(inspect_node, llm=llm))
    graph.add_node("clean_decision", partial(clean_decision_node, llm=llm))
    graph.add_node("clean", partial(clean_node, llm=llm))
    graph.add_node("eda", partial(eda_node, llm=llm))
    graph.add_node("report", report_node)

    # -- Wire edges --
    graph.add_edge(START, "load")
    graph.add_edge("load", "inspect")
    graph.add_edge("inspect", "clean_decision")
    graph.add_edge("clean_decision", "clean")
    graph.add_conditional_edges(
        "clean",
        _should_continue_cleaning,
        {"clean": "clean", "eda": "eda"},
    )
    graph.add_edge("eda", "report")
    graph.add_edge("report", END)

    return graph.compile()
