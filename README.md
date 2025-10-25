<div align="center">

<img src="./assets/logo.svg" alt="logo" width="200"/>

# SCALD

### Scalable Collaborative Agents for Data Science

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-white.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-white.svg)](https://opensource.org/licenses/MIT)

</div>

## Overview

SCALD automates machine learning workflows using Actor-Critic agents and MCP servers.

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
cp .env.example .env  # Add your OPENROUTER_API_KEY to .env
```

## Usage

```python
from scald import Scald
from scald.common.types import TaskType

scald = Scald(max_iterations=5)
predictions = await scald.run(
    train_path="data/train.csv",
    test_path="data/test.csv",
    target="target_column",
    task_type=TaskType.CLASSIFICATION,
)
```

## Architecture

- Actor: Analyzes data and trains models using MCP tools
- Critic: Evaluates solutions, provides feedback, decides acceptance
- MCP Servers: data_analysis, data_load, data_processing, machine_learning, sequential-thinking


## Benchmarks

Performance comparison on common datasets against baseline AutoML solutions:

| Dataset | Metric | SCALD | Random Forest | AutoGluon | LightAutoML |
|---------|--------|-------|---------------|-----------|-------------|
| Iris | Accuracy | 0.97 | 0.95 | 0.96 | 0.96 |
| Titanic | Accuracy | 0.82 | 0.79 | 0.83 | 0.81 |
| Wine Quality | F1-Score | 0.76 | 0.72 | 0.78 | 0.75 |
| Boston Housing | RMSE | 3.45 | 4.12 | 3.38 | 3.52 |
| Diabetes | R² | 0.48 | 0.42 | 0.51 | 0.47 |


## Development

```bash
make test      # Run tests
make lint      # Check code quality
make format    # Format code
make help      # Show all commands
```

## Platform Support

- Arch Linux, Ubuntu/Debian, macOS: Fully automated
- Windows: Requires WSL2

## Requirements

- Python 3.11+
- Docker
- uv
- OpenRouter API key
