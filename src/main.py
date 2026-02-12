"""CLI entry point for the Autonomous Data Cleaning & EDA Agent.

Validates: Requirements 1.1, 6.1, 7.1
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed namespace with csv_file, provider, model, output_dir,
        and max_iterations.
    """
    parser = argparse.ArgumentParser(
        description="Autonomous Data Cleaning & EDA Agent — "
        "load a messy CSV, clean it, run EDA, and generate a Markdown report.",
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file to process.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "groq", "bedrock"],
        help="LLM provider (default: openai).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (uses provider default when omitted).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for report and figures (default: output).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum cleaning iterations (default: 3).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region for Bedrock provider (e.g., us-east-1). Uses default from AWS config if omitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the data-cleaning & EDA agent pipeline.

    Args:
        argv: Optional argument list for testing; uses sys.argv when None.
    """
    args = parse_args(argv)

    # Validate that the CSV file exists early, before heavy imports.
    if not os.path.isfile(args.csv_file):
        print(f"Error: file not found — {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    try:
        from src.graph import build_graph
        from src.llm_config import get_llm
        from src.models import AgentState

        # 1. Create the LLM
        llm_kwargs = {}
        if args.provider == "bedrock" and args.region:
            llm_kwargs["region_name"] = args.region
        llm = get_llm(provider=args.provider, model=args.model, **llm_kwargs)

        # 2. Build the LangGraph workflow
        graph = build_graph(llm)

        # 3. Prepare initial state
        initial_state: AgentState = {
            "file_path": args.csv_file,
            "cleaning_iteration": 0,
            "max_cleaning_iterations": args.max_iterations,
            "cleaning_log": [],
            "errors": [],
            "reasoning_log": [],
            "insights": [],
            "figure_paths": [],
            "needs_more_cleaning": False,
            "df": None,
            "original_shape": None,
            "profile": None,
            "issue_report": None,
            "eda_results": None,
            "report_path": None,
        }

        # 4. Ensure output directories exist
        os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

        # 5. Run the graph
        result = graph.invoke(initial_state)

        # 6. Print the report path
        report_path = result.get("report_path")
        if report_path:
            print(f"Report saved to: {report_path}")
        else:
            print("Warning: report was not generated.", file=sys.stderr)
            if result.get("errors"):
                for err in result["errors"]:
                    print(f"  - {err}", file=sys.stderr)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
