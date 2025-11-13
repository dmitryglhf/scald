<div align="center">

<img src="./assets/logo.svg" alt="logo" width="200"/>

# SCALD

### Scalable Collaborative Agents for Data Science

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-white.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-white.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-white.svg)](https://dmitryglhf.github.io/scald/)
[![Coverage](./.github/badges/coverage.svg)](htmlcov/index.html)

</div>

## Overview

Scald automates machine learning workflows using Actor-Critic agents and MCP servers.

**Key features:**
- Agent-driven EDA, preprocessing, and model training
- Boosting algorithms: CatBoost, LightGBM, XGBoost
- MCP server integration for data operations
- Iterative refinement via Actor-Critic feedback loop

## Installation

Install Python dependencies:
```bash
uv sync
```

Configure environment variables:
```bash
cp .env.example .env  # Add your api_key and base_url to .env
```

## Usage

### CLI

```bash
scald --train data/train.csv --test data/test.csv --target price --task-type regression
```

### Python API

```python
from scald import Scald

scald = Scald(max_iterations=5)
predictions = await scald.run(
    train_path="data/train.csv",
    test_path="data/test.csv",
    target="target_column",
    task_type="classification",
)
```

## Architecture

- Actor: Analyzes data and trains models using MCP tools
- Critic: Evaluates solutions, provides feedback, decides acceptance
- MCP Servers: data-analysis, data-preview, data-processing, machine-learning, file-operations, sequential-thinking

<img src="./assets/arch.svg" alt="arch"/>

## Benchmarks

WIP...


## Documentation

Serve documentation locally:

1. Install documentation dependencies:

```bash
uv sync --group docs
```

2. Serve documentation:

```bash
mkdocs serve
```

Documentation will be available at http://localhost:8000

## Development

```bash
make test      # Run tests
make lint      # Check code quality
make format    # Format code
make help      # Show all commands
```

## Requirements

- Python 3.11+
- uv
- API key for LLM
