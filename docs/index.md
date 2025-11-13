# Scald

**Scalable Collaborative Agents for Data Science**

<div align="center">
  <img src="assets/logo.svg" alt="Scald Logo" width="200"/>
</div>

Scald automates machine learning workflows using collaborative AI agents and the Model Context Protocol. Unlike traditional AutoML frameworks that rely on exhaustive search or rigid pipelines, Scald employs two specialized agents—Actor and Critic—that iteratively refine solutions through feedback loops.

## Core Approach

The Actor agent analyzes data, engineers features, and trains models using six specialized MCP servers as tools. The Critic agent evaluates each solution and provides targeted feedback. Through iterative refinement (typically 5 cycles), this collaboration produces optimized models while learning from past experiences via ChromaDB-based memory.

Scald supports classification and regression tasks using gradient boosting algorithms (CatBoost, LightGBM, XGBoost), with automatic EDA, preprocessing, and hyperparameter tuning.

## Quick Start

```python
from scald import Scald

scald = Scald(max_iterations=5)
predictions = await scald.run(
    train_path="train.csv",
    test_path="test.csv",
    target="price",
    task_type="regression"
)
```

## Why Scald?

Traditional AutoML performs exhaustive grid searches or follows predefined strategies. Scald's agents reason about data characteristics, adapt strategies dynamically, and transfer knowledge between tasks. This results in higher quality solutions with fewer wasted iterations and transparent, interpretable decision-making throughout the process.

## Architecture

<div align="center">
  <img src="assets/arch.svg" alt="Scald Architecture" width="600"/>
</div>

The system orchestrates Actor-Critic loops with workspace isolation, comprehensive logging, and cost tracking. Each session produces artifacts, predictions, and detailed execution logs for full reproducibility.

## Navigation

- [Installation](installation.md) - Setup in minutes
- [Quick Start](quickstart.md) - First AutoML task
- [Architecture](architecture.md) - System design
- [Actor-Critic Pattern](actor-critic.md) - Agent collaboration
- [MCP Servers](mcp-servers.md) - Available tools
