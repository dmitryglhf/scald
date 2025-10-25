<div align="center">

<img src="./assets/logo.svg" alt="logo" width="150"/>

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

### Automated (recommended)

```bash
make install          # Installs uv and Python dependencies
cp .env.example .env  # Add OPENROUTER_API_KEY
```

### Manual

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Install Python dependencies:
   ```bash
   uv sync
   ```

3. Configure environment:
   ```bash
   cp .env.example .env  # Add your OPENROUTER_API_KEY to .env
   ```

## Usage

```python
from scald import Scald
from scald.common.types import TaskType

scald = Scald(max_iterations=5)
result = await scald.run(
    csv_path="data/train.csv",
    target="target_column",
    task_type=TaskType.CLASSIFICATION,
)
print(f"Success: {result.success}")
print(f"Predictions: {result.predictions_path}")
```

## Architecture

- Actor: Analyzes data and trains models using MCP tools
- Critic: Evaluates solutions, provides feedback, decides acceptance
- MCP Servers: data_analysis, data_load, data_processing, machine_learning, sequential-thinking

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
- uv (Python package manager)
- OpenRouter API key
