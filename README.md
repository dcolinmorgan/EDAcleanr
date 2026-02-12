# Autonomous Data Cleaning & EDA Agent

An AI agent that takes a messy CSV file, automatically cleans it, runs exploratory data analysis (EDA), and produces a Markdown report with charts — all powered by a Large Language Model (LLM).

Built with [LangGraph](https://github.com/langchain-ai/langgraph) to demonstrate how AI agents work: an LLM that reasons step-by-step, uses tools, and loops until the job is done.

## How It Works

The agent follows this pipeline:

```
Load CSV → Inspect Data → Decide Cleaning Plan → Clean (loop) → EDA → Report
```

Each step is a **graph node** orchestrated by LangGraph. The LLM decides which tools to call at each stage:

1. **Load** — reads the CSV with automatic encoding and delimiter detection
2. **Inspect** — profiles the data (shape, types, missing values, duplicates, outliers, etc.)
3. **Clean Decision** — LLM reviews issues and decides what to fix
4. **Clean** — applies fixes using 7 cleaning tools (dedup, fill missing, convert types, remove outliers, normalize columns, strip strings, drop useless columns). Can loop up to N iterations.
5. **EDA** — computes statistics, generates histograms, box plots, and a correlation heatmap
6. **Report** — compiles everything into `output/report.md`

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- An API key for at least one LLM provider

### Install

```bash
git clone <this-repo>
cd EDAcleanr
uv sync
```

### Set your API key

Pick one provider and export its key:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Groq
export GROQ_API_KEY="gsk_..."

# AWS Bedrock — uses your AWS credentials (no extra key needed)
# Make sure `aws configure` is set up
```

### Run

```bash
# Default (OpenAI gpt-4o-mini)
uv run eda-agent path/to/your-data.csv

# With Anthropic
uv run eda-agent data.csv --provider anthropic

# With Groq
uv run eda-agent data.csv --provider groq

# With AWS Bedrock
uv run eda-agent data.csv --provider bedrock --region eu-north-1

# Custom model
uv run eda-agent data.csv --provider openai --model gpt-4o
```

### Output

After running, check the `output/` folder:

```
output/
├── report.md              # Full Markdown report
└── figures/
    ├── hist_age.png        # Histogram per numeric column
    ├── box_age.png         # Box plot per numeric column
    └── correlation_heatmap.png
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `csv_file` | (required) | Path to the CSV file |
| `--provider` | `openai` | LLM provider: `openai`, `anthropic`, `groq`, `bedrock` |
| `--model` | auto | Model name override (uses provider default if omitted) |
| `--output-dir` | `output` | Where to save the report and figures |
| `--max-iterations` | `3` | Max cleaning loop iterations |
| `--region` | auto | AWS region for Bedrock (e.g. `eu-north-1`) |

## Default Models

| Provider | Default Model |
|----------|--------------|
| OpenAI | `gpt-4o-mini` |
| Anthropic | `claude-3-5-sonnet-20241022` |
| Groq | `llama-3.1-70b-versatile` |
| Bedrock | `eu.amazon.nova-pro-v1:0` |

## Project Structure

```
src/
├── main.py              # CLI entry point
├── graph.py             # LangGraph workflow (nodes + edges)
├── models.py            # Data models (AgentState, IssueReport, etc.)
├── llm_config.py        # LLM provider setup
├── csv_loader.py        # CSV loading with encoding detection
├── report_generator.py  # Markdown report builder
└── tools/
    ├── inspection.py    # Data profiling & issue detection
    ├── cleaning.py      # 7 cleaning operations
    └── eda.py           # Statistics & plot generation
```

## Running Tests

```bash
uv run pytest
```

## License

MIT
