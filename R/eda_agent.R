# ============================================================================
# EDA Agent — R Wrapper
# ============================================================================
# Lightweight R interface to the Python data-cleaning & EDA agent.
# Uses {reticulate} to call the Python modules directly.
#
# Setup (run once):
#   install.packages("reticulate")
#   reticulate::use_virtualenv("path/to/EDAcleanr/.venv", required = TRUE)
#
# Usage:
#   source("R/eda_agent.R")
#   eda_setup(provider = "bedrock", region = "eu-north-1")
#   df <- eda_load_csv("data.csv")
#   issues <- eda_inspect(df)
#   cleaned <- eda_clean(df)
#   eda_run("data.csv")
# ============================================================================

library(reticulate)

# ---------------------------------------------------------------------------
# Internal: import Python modules lazily
# ---------------------------------------------------------------------------
.eda_env <- new.env(parent = emptyenv())

.ensure_python <- function() {
  if (is.null(.eda_env$loaded)) {
    .eda_env$csv_loader  <- import("src.csv_loader")
    .eda_env$inspection  <- import("src.tools.inspection")
    .eda_env$cleaning    <- import("src.tools.cleaning")
    .eda_env$eda_tools   <- import("src.tools.eda")
    .eda_env$llm_config  <- import("src.llm_config")
    .eda_env$graph       <- import("src.graph")
    .eda_env$report_gen  <- import("src.report_generator")
    .eda_env$loaded      <- TRUE
  }
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

#' Configure the LLM provider for agent nodes.
#'
#' @param provider Character. One of "openai", "anthropic", "groq", "bedrock".
#' @param model    Character or NULL. Model name override.
#' @param region   Character or NULL. AWS region (Bedrock only).
#' @param ...      Extra keyword arguments passed to the LLM constructor.
#' @return Invisible LLM object (also stored internally).
eda_setup <- function(provider = "openai", model = NULL, region = NULL, ...) {
  .ensure_python()
  kwargs <- list(...)
  if (!is.null(region) && provider == "bedrock") {
    kwargs$region_name <- region
  }
  llm <- do.call(.eda_env$llm_config$get_llm,
                  c(list(provider = provider, model = model), kwargs))
  .eda_env$llm <- llm
  message(sprintf("LLM ready: provider=%s", provider))
  invisible(llm)
}

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------

#' Load a CSV file with automatic encoding/delimiter detection.
#'
#' @param path Character. Path to the CSV file.
#' @return A data.frame (converted from pandas DataFrame).
eda_load_csv <- function(path) {
  .ensure_python()
  result <- .eda_env$csv_loader$load_csv(path)
  if (!is.null(result$error)) {
    stop(result$error)
  }
  py_to_r(result$df)
}

# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

#' Profile a data.frame and detect quality issues.
#'
#' @param df A data.frame (will be converted to pandas internally).
#' @return A list with elements: profile (character), issues (list).
eda_inspect <- function(df) {
  .ensure_python()
  pd_df <- r_to_py(df)
  profile <- .eda_env$inspection$get_df_info(pd_df)
  issues  <- .eda_env$inspection$detect_issues(pd_df)
  list(profile = profile, issues = py_to_r(issues))
}

# ---------------------------------------------------------------------------
# Cleaning tools
# ---------------------------------------------------------------------------

#' Run all basic cleaning steps on a data.frame.
#'
#' Applies: normalize columns, strip strings, drop duplicates,
#' drop useless columns, fill missing numerics with median.
#'
#' @param df A data.frame.
#' @param threshold Numeric. Missing-value fraction above which a column is
#'   dropped (default 0.9).
#' @return A list with elements: df (cleaned data.frame), log (character vector).
eda_clean <- function(df, threshold = 0.9) {
  .ensure_python()
  np <- import("numpy")
  pd_df <- r_to_py(df)
  log_msgs <- character()

  # Normalize column names
  result <- .eda_env$cleaning$normalize_columns(pd_df)
  pd_df <- result[[1]]
  log_msgs <- c(log_msgs, result[[2]]$description)

  # Strip string whitespace
  result <- .eda_env$cleaning$strip_string_values(pd_df)
  pd_df <- result[[1]]
  log_msgs <- c(log_msgs, result[[2]]$description)

  # Drop duplicates
  result <- .eda_env$cleaning$drop_duplicates(pd_df)
  pd_df <- result[[1]]
  log_msgs <- c(log_msgs, result[[2]]$description)

  # Drop useless columns
  result <- .eda_env$cleaning$drop_useless_columns(pd_df, threshold)
  pd_df <- result[[1]]
  log_msgs <- c(log_msgs, result[[2]]$description)

  # Fill missing numeric columns with median
  numeric_cols <- pd_df$select_dtypes(include = list(np$number))$columns$tolist()
  for (col in numeric_cols) {
    if (pd_df[[col]]$isna()$sum() > 0L) {
      result <- .eda_env$cleaning$fill_missing(pd_df, col, "median")
      pd_df <- result[[1]]
      log_msgs <- c(log_msgs, result[[2]]$description)
    }
  }

  list(df = py_to_r(pd_df), log = log_msgs)
}

# ---------------------------------------------------------------------------
# EDA tools
# ---------------------------------------------------------------------------

#' Run exploratory data analysis on a data.frame.
#'
#' @param df A data.frame.
#' @param output_dir Character. Directory for saving figures (default "output/figures").
#' @return A list with elements: numeric_stats, categorical_stats,
#'   correlation, figure_paths.
eda_analyze <- function(df, output_dir = "output/figures") {
  .ensure_python()
  pd_df <- r_to_py(df)
  list(
    numeric_stats     = .eda_env$eda_tools$describe_numeric(pd_df),
    categorical_stats = .eda_env$eda_tools$describe_categorical(pd_df),
    correlation       = .eda_env$eda_tools$compute_correlation(pd_df),
    figure_paths      = py_to_r(.eda_env$eda_tools$generate_plots(pd_df, output_dir))
  )
}

# ---------------------------------------------------------------------------
# Full pipeline (LLM-driven)
# ---------------------------------------------------------------------------

#' Run the full autonomous agent pipeline on a CSV file.
#'
#' Requires eda_setup() to be called first.
#'
#' @param csv_path       Character. Path to the CSV file.
#' @param output_dir     Character. Output directory (default "output").
#' @param max_iterations Integer. Max cleaning loop iterations (default 3).
#' @return A list with report_path, errors, reasoning_log, and other state.
eda_run <- function(csv_path, output_dir = "output", max_iterations = 3L) {
  .ensure_python()
  if (is.null(.eda_env$llm)) {
    stop("Call eda_setup() first to configure the LLM provider.")
  }

  dir.create(file.path(output_dir, "figures"), recursive = TRUE, showWarnings = FALSE)

  graph <- .eda_env$graph$build_graph(.eda_env$llm)

  initial_state <- dict(
    file_path              = csv_path,
    cleaning_iteration     = 0L,
    max_cleaning_iterations = as.integer(max_iterations),
    cleaning_log           = list(),
    errors                 = list(),
    reasoning_log          = list(),
    insights               = list(),
    figure_paths           = list(),
    needs_more_cleaning    = FALSE,
    df                     = NULL,
    original_shape         = NULL,
    profile                = NULL,
    issue_report           = NULL,
    eda_results            = NULL,
    report_path            = NULL
  )

  result <- graph$invoke(initial_state)

  report_path <- result$report_path
  if (!is.null(report_path)) {
    message(sprintf("Report saved to: %s", report_path))
  } else {
    warning("Report was not generated.")
    if (length(result$errors) > 0) {
      for (err in result$errors) message(sprintf("  - %s", err))
    }
  }

  invisible(py_to_r(result))
}
