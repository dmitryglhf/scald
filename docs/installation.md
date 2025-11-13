# Installation

## Requirements

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenRouter API key (or compatible LLM API)

## Install uv

If you don't have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install Scald

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/scald.git
cd scald
uv sync
```

## Configuration

Create an environment file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your API credentials:

```bash
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Verify Installation

Test that Scald is installed correctly:

```bash
scald --help
```

You should see the CLI help message with available options.

## Optional: Documentation Dependencies

To build and serve documentation locally:

```bash
uv sync --group docs
mkdocs serve
```

Documentation will be available at http://localhost:8000

## Optional: Development Dependencies

For development work (testing, linting):

```bash
uv sync --group dev
```

## Next Steps

Continue to [Quick Start](quickstart.md) to run your first AutoML task.
