# Requirements Document

## Introduction

This document specifies the requirements for an Autonomous Data Cleaning & EDA Agent built with LangChain/LangGraph. The agent autonomously loads messy CSV files, detects data quality issues, applies intelligent cleaning fixes, performs exploratory data analysis, and produces a concise summary report with optional visualizations. The system uses a multi-step agentic workflow powered by a tool-calling LLM.

## Glossary

- **Agent**: A LangGraph-based autonomous component that uses an LLM to reason about data and invoke tools to accomplish a specific stage of the workflow.
- **CSV_Loader**: The component responsible for reading CSV files into a Pandas DataFrame with encoding and delimiter detection.
- **Inspection_Agent**: The agent responsible for profiling the loaded DataFrame and detecting data quality issues.
- **Cleaning_Agent**: The agent responsible for applying data cleaning operations based on detected issues.
- **EDA_Agent**: The agent responsible for performing exploratory data analysis and summarizing insights.
- **Report_Generator**: The component that compiles cleaning actions, statistics, and insights into a Markdown report.
- **DataFrame**: A Pandas DataFrame holding the tabular data being processed.
- **Issue_Report**: A structured summary of detected data quality problems including missing values, duplicates, outliers, type inconsistencies, high-cardinality categoricals, and zero-variance columns.
- **Cleaning_Plan**: An ordered list of cleaning operations the Cleaning_Agent intends to apply, with severity and priority.
- **LLM**: A tool-calling capable large language model (OpenAI, Anthropic, or Groq) used for reasoning and decision-making.
- **IQR**: Interquartile Range, a method for detecting statistical outliers.
- **EDA**: Exploratory Data Analysis.

## Requirements

### Requirement 1: CSV Loading and Input Handling

**User Story:** As a data analyst, I want to load messy CSV files with automatic encoding and delimiter detection, so that I can begin cleaning without manual preprocessing.

#### Acceptance Criteria

1. WHEN a valid CSV file path is provided, THE CSV_Loader SHALL read the file into a DataFrame and return the loaded DataFrame.
2. WHEN the CSV file uses a non-UTF-8 encoding, THE CSV_Loader SHALL detect the encoding and decode the file correctly.
3. WHEN the CSV file uses a non-comma delimiter, THE CSV_Loader SHALL infer the delimiter and parse the file correctly.
4. IF the CSV file path does not exist or the file is unreadable, THEN THE CSV_Loader SHALL return a descriptive error message indicating the cause of failure.
5. IF the CSV file is empty or contains only headers, THEN THE CSV_Loader SHALL return a descriptive error message indicating the file has no data rows.

### Requirement 2: Data Inspection and Issue Detection

**User Story:** As a data analyst, I want the agent to automatically profile my dataset and detect common data quality issues, so that I understand the state of the data before cleaning.

#### Acceptance Criteria

1. WHEN a DataFrame is loaded, THE Inspection_Agent SHALL produce a profile containing shape, column data types, head(10), tail(10), and a random sample of 5 rows.
2. WHEN the Inspection_Agent analyzes the DataFrame, THE Inspection_Agent SHALL detect and report the percentage of missing values per column.
3. WHEN the Inspection_Agent analyzes the DataFrame, THE Inspection_Agent SHALL detect and report the number of duplicate rows.
4. WHEN the Inspection_Agent analyzes the DataFrame, THE Inspection_Agent SHALL detect outliers in numeric columns using the IQR method.
5. WHEN the Inspection_Agent analyzes the DataFrame, THE Inspection_Agent SHALL detect columns with inconsistent data types (mixed numeric and string values).
6. WHEN the Inspection_Agent analyzes the DataFrame, THE Inspection_Agent SHALL detect high-cardinality categorical columns and zero-variance columns.
7. WHEN all issues are detected, THE Inspection_Agent SHALL produce an Issue_Report with severity and priority rankings determined by the LLM.

### Requirement 3: Autonomous Data Cleaning

**User Story:** As a data analyst, I want the agent to autonomously fix common data quality issues based on the inspection results, so that I get a clean dataset without manual intervention.

#### Acceptance Criteria

1. WHEN the Issue_Report contains duplicate rows, THE Cleaning_Agent SHALL remove duplicate rows from the DataFrame.
2. WHEN the Issue_Report contains columns with missing values, THE Cleaning_Agent SHALL fill missing values using an appropriate strategy (mean, median, mode, forward-fill, or KNN) selected by the LLM based on column characteristics.
3. WHEN the Issue_Report contains columns with inconsistent types, THE Cleaning_Agent SHALL convert columns to appropriate data types using coercion where necessary.
4. WHEN the Issue_Report contains outliers, THE Cleaning_Agent SHALL remove or cap outliers using the IQR or z-score method as selected by the LLM.
5. THE Cleaning_Agent SHALL normalize column names by stripping whitespace and converting to lowercase.
6. WHEN a column has greater than 90 percent missing values or zero variance, THE Cleaning_Agent SHALL drop that column from the DataFrame.
7. THE Cleaning_Agent SHALL strip leading and trailing whitespace from all string values in the DataFrame.
8. WHEN cleaning operations are applied, THE Cleaning_Agent SHALL record each operation performed with a description of the action and the affected columns.
9. WHEN the Cleaning_Agent completes all operations, THE Cleaning_Agent SHALL return the cleaned DataFrame and a log of all cleaning actions taken.

### Requirement 4: Exploratory Data Analysis

**User Story:** As a data analyst, I want the agent to perform automated EDA on the cleaned dataset, so that I receive key statistical insights and visualizations without writing analysis code.

#### Acceptance Criteria

1. WHEN the cleaned DataFrame is provided, THE EDA_Agent SHALL compute descriptive statistics for all numeric columns.
2. WHEN the cleaned DataFrame is provided, THE EDA_Agent SHALL compute value counts for the top N categories in each categorical column.
3. WHEN the cleaned DataFrame contains two or more numeric columns, THE EDA_Agent SHALL compute a correlation matrix.
4. WHEN the EDA_Agent performs analysis, THE EDA_Agent SHALL generate and save visualizations including histograms, box plots, and a correlation heatmap to a figures directory.
5. WHEN the EDA_Agent completes analysis, THE EDA_Agent SHALL produce a text summary containing 3 to 5 key insights including trends, anomalies, and recommendations.

### Requirement 5: Report Generation

**User Story:** As a data analyst, I want a concise summary report of the entire cleaning and analysis process, so that I can review what was done and share findings.

#### Acceptance Criteria

1. WHEN the cleaning and EDA stages are complete, THE Report_Generator SHALL compile a Markdown report containing the cleaned dataset shape, issues found, fixes applied, key statistics, and 3 to 5 insights.
2. THE Report_Generator SHALL save the report as report.md in the output directory.
3. THE Report_Generator SHALL save all generated figures in a figures subdirectory within the output directory.
4. WHEN figures are generated, THE Report_Generator SHALL embed figure references in the Markdown report using relative paths.

### Requirement 6: Agent Workflow Orchestration

**User Story:** As a developer, I want the agent workflow to follow a structured multi-step graph (inspect → decide → clean loop → EDA → report), so that each stage executes in the correct order with conditional transitions.

#### Acceptance Criteria

1. THE Agent_Orchestrator SHALL execute the workflow stages in the order: inspection, cleaning decision, cleaning loop, EDA, report generation.
2. WHEN the Cleaning_Agent determines additional cleaning passes are needed, THE Agent_Orchestrator SHALL loop back to the cleaning stage until the Cleaning_Agent signals completion.
3. WHEN any agent stage encounters an unrecoverable error, THE Agent_Orchestrator SHALL log the error and proceed to the next stage with a degraded result rather than crashing.
4. THE Agent_Orchestrator SHALL expose agent reasoning steps for each stage so that the decision-making process is visible to the user.

### Requirement 7: LLM Integration and Tool Calling

**User Story:** As a developer, I want the agents to use tool-calling capable LLMs to reason about data and invoke cleaning/analysis tools, so that the system makes intelligent autonomous decisions.

#### Acceptance Criteria

1. THE Agent_Orchestrator SHALL support configuration of OpenAI, Anthropic, or Groq as the LLM provider.
2. WHEN an agent invokes a tool, THE LLM SHALL use the tool-calling interface to select and parameterize the appropriate tool based on the current data state.
3. WHEN the LLM generates a code snippet for cleaning, THE Cleaning_Agent SHALL execute the snippet in a sandboxed Python REPL environment.
4. IF the LLM produces an invalid tool call or code snippet, THEN THE Agent_Orchestrator SHALL catch the error, log it, and allow the LLM to retry with corrected parameters.

### Requirement 8: Cleaning Action Logging and Traceability

**User Story:** As a data analyst, I want a complete log of all cleaning actions and agent reasoning, so that I can audit what changes were made to my data.

#### Acceptance Criteria

1. WHEN any cleaning operation is performed, THE Cleaning_Agent SHALL append an entry to the cleaning log containing the operation name, affected columns, parameters used, and row/value counts before and after.
2. WHEN the workflow completes, THE Agent_Orchestrator SHALL include the full cleaning log in the final report.
3. THE Agent_Orchestrator SHALL log all LLM reasoning steps with timestamps so that the decision chain is traceable.
