# MCP Servers

Scald uses the Model Context Protocol (MCP) to provide specialized tools to the Actor agent. Each server exposes domain-specific operations.

## What is MCP?

Model Context Protocol is a standard for connecting LLMs to external tools and data sources. In Scald, MCP servers act as the Actor's toolkit for data science tasks.

## Available Servers

### 1. data-preview

Quick inspection of data structure and contents.

**Capabilities:**
- View column names and types
- Sample rows from datasets
- Check data dimensions
- Inspect schemas

**Use Cases:**
- Initial data exploration
- Verify data loaded correctly
- Understand feature types

### 2. data-analysis

Statistical analysis and pattern discovery.

**Capabilities:**
- Descriptive statistics (mean, median, std, etc.)
- Correlation matrices
- Distribution analysis
- Missing value detection
- Outlier identification

**Use Cases:**
- Exploratory Data Analysis (EDA)
- Feature relationship discovery
- Data quality assessment

### 3. data-processing

Data transformation and feature engineering.

**Capabilities:**
- Categorical encoding (one-hot, label, target)
- Feature scaling (standard, minmax, robust)
- Missing value imputation
- Feature engineering (polynomial, interactions)
- Column transformations

**Use Cases:**
- Preprocessing pipelines
- Feature engineering
- Data cleaning

### 4. machine-learning

Model training, prediction, and evaluation.

**Capabilities:**
- Train models: CatBoost, LightGBM, XGBoost
- Hyperparameter tuning with Optuna
- Cross-validation
- Model evaluation (accuracy, F1, RMSE, R², etc.)
- Prediction generation

**Use Cases:**
- Model training
- Performance evaluation
- Generating predictions

### 5. file-operations

File system operations for data and artifacts.

**Capabilities:**
- Read CSV files
- Write CSV files
- Save Python code
- Load/save serialized models
- Manage artifacts

**Use Cases:**
- Loading training/test data
- Saving preprocessing code
- Persisting trained models
- Writing predictions

### 6. sequential-thinking

Structured problem decomposition and reasoning.

**Capabilities:**
- Break down complex tasks
- Track reasoning steps
- Maintain context across operations
- Plan multi-step workflows

**Use Cases:**
- Complex feature engineering
- Multi-stage preprocessing
- Strategic planning

## Server Architecture

Each MCP server runs in isolation:

```
┌─────────────┐
│    Actor    │
└──────┬──────┘
       │ MCP Protocol
       │
       ├──▶ [data-preview server]
       ├──▶ [data-analysis server]
       ├──▶ [data-processing server]
       ├──▶ [machine-learning server]
       ├──▶ [file-operations server]
       └──▶ [sequential-thinking server]
```

## Typical Workflow

A typical Actor iteration uses servers in sequence:

1. **Preview** data structure
2. **Analyze** statistics and patterns
3. **Process** features and clean data
4. **Train** models using machine-learning
5. **Save** artifacts with file-operations

## Example: Actor Using MCP Servers

```python
# Conceptual flow (internal to Actor)

# Step 1: Preview data
data_info = call_mcp("data-preview", "inspect", path="train.csv")

# Step 2: Analyze
stats = call_mcp("data-analysis", "describe", data=train_df)
correlations = call_mcp("data-analysis", "correlation", data=train_df)

# Step 3: Process
encoded_data = call_mcp("data-processing", "encode_categorical", 
                        data=train_df, method="onehot")
scaled_data = call_mcp("data-processing", "scale_features",
                       data=encoded_data, method="standard")

# Step 4: Train
model = call_mcp("machine-learning", "train_model",
                 data=scaled_data, algorithm="catboost")
score = call_mcp("machine-learning", "evaluate",
                 model=model, test_data=val_data)

# Step 5: Predict & Save
predictions = call_mcp("machine-learning", "predict",
                       model=model, test_data=test_df)
call_mcp("file-operations", "write_csv",
         data=predictions, path="predictions.csv")
```

## Server Benefits

### Modularity

Each server handles one domain—easy to:
- Understand responsibilities
- Debug issues
- Extend functionality
- Replace implementations

### Safety

Servers provide controlled interfaces:
- Validated inputs
- Error handling
- Resource limits
- Sandboxed execution

### Reusability

MCP servers can be:
- Used by other agents
- Tested independently
- Versioned separately
- Shared across projects

## Extending Scald

To add new capabilities:

1. Create new MCP server
2. Define tools/operations
3. Register with Actor
4. Document usage

MCP's standard protocol makes integration straightforward.

## Server Configuration

MCP servers are configured in Scald's initialization:

```python
# Server paths and settings are internal to Scald
# Users don't typically need to configure servers directly
```

## Limitations

### Actor-Only Access

Only the Actor uses MCP servers. The Critic intentionally has no tool access to maintain objectivity in reviews.

### Stateless Operations

MCP servers don't maintain state between calls. All context must be passed explicitly.

## Next Steps

- [Actor-Critic Pattern](actor-critic.md) - How agents use these tools
- [Python API](usage/api.md) - Running Scald programmatically
- [CLI Usage](usage/cli.md) - Command-line interface
