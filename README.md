<div align="center">

<img src="./assets/logo.svg" alt="logo" width="200"/>

# SCALD

### Scalable Collaborative Agents for Data Science

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-white.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-white.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-white.svg)](https://dmitryglhf.github.io/scald/)

</div>

## Overview

Scald automates machine learning workflows through collaborative AI agents using the Actor-Critic pattern. The Actor agent explores data, engineers features, and trains models using six specialized MCP servers. The Critic agent evaluates solutions and provides targeted feedback for iterative refinement. This approach combines LLM-powered reasoning with gradient boosting algorithms (CatBoost, LightGBM, XGBoost) for both classification and regression tasks.

The system learns from past experiences through ChromaDB-based memory, enabling transfer learning across datasets. Each iteration produces executable code artifacts, comprehensive logs, and cost tracking for full reproducibility.

## Installation

Install from PyPI:

```bash
pip install scald
```

Configure API credentials:

```bash
cp .env.example .env  # Add your OpenRouter API key
```

For development work, clone the repository and install with all dependencies:

```bash
git clone https://github.com/dmitryglhf/scald.git
cd scald
uv sync
```

## Usage

Run AutoML from the command line:

```bash
scald --train data/train.csv --test data/test.csv --target price --task-type regression
```

Or use the Python API:

```python
from scald import Scald
import polars as pl

scald = Scald(max_iterations=5)

# Option 1: Using CSV file paths
predictions = await scald.run(
    train="data/train.csv",
    test="data/test.csv",
    target="target_column",
    task_type="classification",
)

# Option 2: Using DataFrames (Polars or Pandas)
train_df = pl.read_csv("data/train.csv")
test_df = pl.read_csv("data/test.csv")

predictions = await scald.run(
    train=train_df,
    test=test_df,
    target="target_column",
    task_type="classification",
)
```

The Actor-Critic loop executes for the specified iterations (default: 5), producing predictions and saving all artifacts to a timestamped session directory.

## Architecture

<img src="./assets/arch.svg" alt="arch"/>

The Actor agent has access to specialized MCP servers for data preview, statistical analysis, preprocessing, model training, file operations, and structured reasoning. The Critic agent reviews solutions without tool access to maintain evaluation objectivity. This separation enables independent verification while the memory system accumulates experience for improved performance on similar tasks.

## Documentation

Full documentation available at [dmitryglhf.github.io/scald](https://dmitryglhf.github.io/scald/)

Serve locally:

```bash
uv sync --group docs
mkdocs serve
```

## Development

Run tests and code quality checks:

```bash
make test      # Run tests with
make lint      # Check code quality
make format    # Format code
make help      # Show all commands
```

## Requirements

Python 3.11+, uv package manager, and an API key from OpenRouter or compatible LLM provider.
