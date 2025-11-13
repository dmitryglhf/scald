# MCP Servers

Scald uses the Model Context Protocol to provide specialized tools to the Actor agent. Each server exposes domain-specific operations for data science tasks.

## Available Servers

**data-preview** enables quick inspection of data structure, column types, dimensions, and sample rows. The Actor uses this for initial exploration and schema verification.

**data-analysis** performs statistical computations including descriptive statistics, correlation matrices, distribution analysis, missing value detection, and outlier identification. This supports exploratory data analysis and pattern discovery.

**data-processing** handles transformations: categorical encoding (one-hot, label, target), feature scaling (standard, minmax, robust), missing value imputation, and feature engineering (polynomial features, interactions). The Actor builds preprocessing pipelines using these operations.

**machine-learning** provides model training with CatBoost, LightGBM, and XGBoost. It includes hyperparameter tuning via Optuna, cross-validation, performance evaluation using task-appropriate metrics (accuracy, F1, RMSE, R²), and prediction generation.

**file-operations** manages I/O for CSV data, Python code artifacts, serialized models, and intermediate results. This enables data loading, artifact persistence, and prediction export.

**sequential-thinking** supports structured problem decomposition and multi-step reasoning. The Actor uses this for complex workflows requiring careful planning across multiple operations.

## Server Architecture

Each MCP server runs in isolation, communicating via the standard protocol. The Actor selects appropriate tools based on current task needs, while the Critic has no server access to maintain evaluation objectivity.

## Typical Usage Pattern

Actor workflows generally follow this sequence: preview data structure, analyze statistics and patterns, process features and clean data, train models with machine-learning server, and save artifacts via file-operations. The sequential-thinking server supports complex reasoning throughout.

## Server Benefits

Modularity keeps responsibilities clear and components independently testable. Servers provide validated interfaces with error handling and resource limits for safe execution. The standard MCP protocol enables reusability across different agents and projects. Extending Scald requires only implementing new MCP servers with defined tool interfaces.

## Limitations

Only the Actor accesses MCP servers—the Critic reviews without tools to ensure objective evaluation. Servers operate statelessly, requiring explicit context in each call. This design enforces clear boundaries and prevents hidden dependencies.

Continue to [CLI Usage](usage/cli.md) for command-line interface reference or [Python API](usage/api.md) for programmatic usage.
