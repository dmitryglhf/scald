# Installation

## Requirements

Python 3.11+, [uv](https://github.com/astral-sh/uv) package manager, and an OpenRouter API key (or compatible LLM provider).

## Setup

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone and install Scald:

```bash
git clone https://github.com/yourusername/scald.git
cd scald
uv sync
```

## Configuration

Copy the environment template and add your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Verification

Confirm installation:

```bash
scald --help
```

You should see the CLI help output with available commands and options.

## Optional Dependencies

For documentation:

```bash
uv sync --group docs
mkdocs serve  # Available at http://localhost:8000
```

For development:

```bash
uv sync --group dev
```

Continue to [Quick Start](quickstart.md) to run your first AutoML task.
