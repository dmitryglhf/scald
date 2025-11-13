# Scald

**Scalable Collaborative Agents for Data Science**

Scald is an automated machine learning framework that uses collaborative AI agents to handle the complete ML workflow—from exploratory data analysis to model training and evaluation.

## What is Scald?

Scald combines the Actor-Critic pattern with Model Context Protocol (MCP) servers to create an intelligent AutoML system. Instead of rigid pipelines, Scald uses two specialized agents that work together:

- **Actor**: A data scientist agent that explores data, engineers features, and trains models
- **Critic**: A reviewer agent that evaluates solutions and provides feedback

Through iterative refinement (typically 5 iterations), these agents converge on optimal solutions while learning from past experiences via long-term memory.

## Key Features

- **Intelligent Automation**: Agents make decisions based on data characteristics, not hardcoded rules
- **Iterative Refinement**: Solutions improve through Actor-Critic feedback loops
- **Learning System**: ChromaDB-based memory learns from previous tasks
- **Production Ready**: Comprehensive logging, cost tracking, and artifact preservation
- **Flexible Integration**: Use via CLI or Python API

## Supported Tasks

- Binary and multiclass classification
- Regression
- Boosting algorithms: CatBoost, LightGBM, XGBoost

## Quick Example

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

Traditional AutoML frameworks rely on exhaustive search or predefined strategies. Scald takes a different approach—using LLM-powered agents that reason about your data, learn from experience, and adapt their strategies accordingly.

The result is a system that's both more flexible and more intelligent, capable of handling novel data patterns without manual intervention.

## Get Started

- [Installation](installation.md) - Set up Scald in minutes
- [Quick Start](quickstart.md) - Run your first AutoML task
- [Architecture](architecture.md) - Understand how Scald works
