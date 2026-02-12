# Implementation Plan: Autonomous Data Cleaning & EDA Agent

## Overview

Implement a LangGraph-based multi-step agent pipeline that loads messy CSVs, detects issues, cleans data, performs EDA, and generates a Markdown report. Tasks are ordered so each builds on the previous, with testing integrated alongside implementation.

## Tasks

- [x] 1. Set up project structure, dependencies, and core data models
  - [x] 1.1 Create project directory structure (`src/`, `src/tools/`, `tests/`, `output/`, `output/figures/`) and `pyproject.toml` / `requirements.txt` with dependencies: langchain>=0.3, langgraph, pandas, hypothesis, pytest, matplotlib, seaborn, chardet, openai, anthropic, groq
    - _Requirements: 6.1, 7.1_
  - [x] 1.2 Implement core data models in `src/models.py`: `AgentState` TypedDict, `IssueReport`, `CleaningLogEntry`, and `EDAResults` dataclasses
    - _Requirements: 2.7, 3.8, 8.1_
  - [x] 1.3 Implement LLM configuration in `src/llm_config.py`: `get_llm(provider, model)` supporting OpenAI, Anthropic, and Groq
    - _Requirements: 7.1_

- [x] 2. Implement CSV Loader
  - [x] 2.1 Implement `src/csv_loader.py` with `load_csv(file_path)` function: encoding detection via chardet, delimiter inference via csv.Sniffer, fallback logic, empty file validation, error handling returning dict with "df" and "error" keys
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ]* 2.2 Write property test for CSV round-trip loading in `tests/test_csv_loader.py`
    - **Property 1: CSV Load Round-Trip**
    - Generate random tabular data, write to temp CSV with random encoding/delimiter, load via CSV_Loader, verify data matches
    - **Validates: Requirements 1.1, 1.2, 1.3**
  - [ ]* 2.3 Write unit tests for CSV loader edge cases in `tests/test_csv_loader.py`
    - Test non-existent file path returns error
    - Test empty file returns error
    - Test header-only file returns error
    - _Requirements: 1.4, 1.5_

- [x] 3. Implement Inspection Tools
  - [x] 3.1 Implement `src/tools/inspection.py` with `get_df_info(df)` tool returning formatted shape, dtypes, head, tail, sample; and `detect_issues(df)` tool returning dict with missing_
utlier Detection via IQR** — for any numeric column, flagged values are exactly those outside [Q1-1.5*IQR, Q3+1.5*IQR]
    - **Property 5: Inconsistent Type Detection** — for any column with mixed numeric/string values, column appears in inconsistent_types
    - **Property 6: Zero-Variance and High-Cardinality Detection** — columns with one unique value appear in zero_variance; high unique ratio categoricals appear in high_cardinality
    - **Property 7: Profile Completeness** — for any DataFrame, profile contains shape, dtypes, head, tail, sample
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**

- [x] 4. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Cleaning Tools
  - [x] 5.1 Implement `src/tools/cleaning.py` with all cleaning tool functions: `drop_duplicates`, `fill_missing` (mean/median/mode/ffill/knn), `convert_dtypes`, `remove_outliers` (IQR/z-score), `normalize_columns`, `strip_string_values`, `drop_useless_columns`. Each returns (cleaned_df, CleaningLogEntry).
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_
  - [ ]* 5.2 Write property tests for cleaning invariants in `tests/test_cleaning_tools.py`
    - **Property 8: Duplicate Removal Invariant** — after drop_duplicates, zero duplicate rows remain
    - **Property 9: Missing Value Fill Invariant** — after fill_missing with any strategy, target column has zero missing values
    - **Property 10: Type Conversion Consistency** — after convert_dtypes, column has single consistent dtype
    - **Property 11: Outlier Removal Invariant (IQR)** — after remove_outliers, no values outside IQR bounds
    - **Property 12: String Normalization Invariant** — after normalize_columns + strip_string_values, all column names lowercase/trimmed and all string values trimmed
    - **Property 13: Useless Column Removal Invariant** — after drop_useless_columns, no column has >90% missing or zero variance
    - **Property 14: Cleaning Log Completeness** — each operation produces a log entry with operation name, affected columns, parameters, and row counts
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 8.1**

- [x] 6. Implement EDA Tools
  - [x] 6.1 Implement `src/tools/eda.py` with `describe_numeric(df)`, `describe_categorical(df, top_n)`, `compute_correlation(df)`, and `generate_plots(df, output_dir)` saving histograms, box plots, and correlation heatmap via matplotlib/seaborn
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [ ]* 6.2 Write property tests for EDA tools in `tests/test_eda_tools.py`
    - **Property 15: Numeric Descriptive Statistics Coverage** — for any DataFrame with numeric columns, output covers every numeric column
    - **Property 16: Categorical Value Counts Coverage** — for any DataFrame with categorical columns, output covers every categorical column
    - **Property 17: Correlation Matrix Invariants** — matrix is symmetric with 1.0 on diagonal
    - **Validates: Requirements 4.1, 4.2, 4.3**

- [x] 7. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Report Generator
  - [x] 8.1 Implement `src/report_generator.py` with `generate_report(...)` that compiles Markdown report with sections: Dataset Overview (original/cleaned shape), Issues Found, Cleaning Actions (from log), Key Statistics, Insights, and embedded figure references. Save to `output/report.md`.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 8.2_
  - [ ]* 8.2 Write property tests for report generator in `tests/test_report_generator.py`
    - **Property 19: Report Section Completeness** — generated report contains all required sections
    - **Property 20: Report Figure References** — each figure path appears as relative path in report
    - **Property 23: Cleaning Log in Report** — every cleaning log entry appears in the report
    - **Validates: Requirements 5.1, 5.4, 8.2**

- [x] 9. Implement LangGraph Agent Orchestrator
  - [x] 9.1 Implement graph nodes in `src/graph.py`: `load_csv_node` (calls csv_loader), `inspect_node` (binds inspection tools to LLM agent), `clean_decision_node` (LLM decides cleaning plan), `clean_node` (binds cleaning tools to LLM agent with iteration tracking), `eda_node` (binds EDA tools to LLM agent), `report_node` (calls report_generator). Each node wraps logic in try/except, logs errors to state, and appends reasoning to reasoning_log with timestamps.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 7.2, 7.3, 7.4, 8.3_
  - [x] 9.2 Implement `build_graph(llm)` in `src/graph.py` that constructs the StateGraph with nodes wired in order: load → inspect → clean_decision → clean → (conditional loop back to clean or forward to) eda → report. Add conditional edge from clean_node based on `needs_more_cleaning` and `cleaning_iteration < max_cleaning_iterations`.
    - _Requirements: 6.1, 6.2_
  - [ ]* 9.3 Write property tests for orchestrator in `tests/test_graph.py`
    - **Property 21: Error Resilience** — for any stage that raises an exception, orchestrator catches it, appends to errors list, and continues
    - **Property 22: Reasoning Log Traceability** — each reasoning log entry contains timestamp, agent name, and reasoning text
    - **Validates: Requirements 6.3, 6.4, 8.3**
  - [ ]* 9.4 Write unit test for cleaning loop conditional edge
    - Test that when needs_more_cleaning=True and iteration < max, graph loops back to clean_node
    - Test that when needs_more_cleaning=False, graph proceeds to eda_node
    - _Requirements: 6.2_

- [x] 10. Implement CLI Entry Point and Wire Everything Together
  - [x] 10.1 Implement `src/main.py` with CLI entry point that accepts CSV file path, optional LLM provider/model config, optional output directory, builds the graph via `build_graph`, invokes it with initial AgentState, and prints the report path on completion. Use argparse for CLI arguments.
    - _Requirements: 1.1, 6.1, 7.1_
  - [x] 10.2 Create `conftest.py` in `tests/` with shared Hypothesis strategies: `messy_dataframes` (generates DataFrames with controlled missing values, duplicates, mixed types, outliers) and `csv_file_params` (generates encoding/delimiter/data combinations)
    - _Requirements: All testing properties_

- [x] 11. Final Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests use the `hypothesis` library with minimum 100 examples per test
- Checkpoints ensure incremental validation
- LLM-dependent tests (agent reasoning, tool selection) should use mocked LLM responses to keep tests deterministic
