# Configuration

Control Scald behavior through environment variables and initialization parameters.

## Environment Variables

Configure API access in `.env`:

```bash
# API credentials (required)
OPENROUTER_API_KEY=your_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Model selection (optional)
MODEL_NAME=anthropic/claude-3.5-sonnet

# Logging verbosity (optional)
LOG_LEVEL=INFO
```

Model name depends on your API provider. OpenRouter supports many LLMs. Log level options: DEBUG, INFO, WARNING, ERROR.

## Initialization Parameters

`max_iterations` controls Actor-Critic refinement cycles:

```python
scald = Scald(max_iterations=10)
```

Higher values increase solution quality and cost. Lower values reduce both. Default of 5 balances quality and efficiency.

**Recommendations:** Use 3 for simple tasks, 5 for standard problems, 7-10 for complex datasets.

## Runtime Parameters

`task_type` specifies classification or regression. Classification handles binary or multiclass problems with discrete labels, using accuracy, F1, precision, and recall metrics. Regression predicts continuous numeric values, using RMSE, MAE, and R² metrics.

## Data Requirements

Training and test CSV files must share the same feature columns. The target column must exist in training data and may optionally appear in test data. Both files should use standard CSV format with headers.

## Session Management

Sessions automatically create timestamped directories in `./sessions`. Each contains execution logs, generated artifacts, and predictions. Sessions run independently and support concurrent execution without conflicts.

## Memory Storage

ChromaDB stores long-term memory in `.chroma/` directory. Default settings typically work well. The memory system accumulates experiences across sessions, enabling transfer learning.

## Cost Optimization

Reduce API costs by lowering `max_iterations`, using cheaper models (configured in `.env`), or working with smaller datasets during prototyping. Scald tracks token usage and costs per session in logs.

## Performance Tuning

For faster execution, use `max_iterations=3`, smaller training sets, or simpler models. For higher quality, increase iterations to 7+, provide more training data, or allow more refinement cycles.

Large datasets require sufficient RAM. Consider sampling for initial exploration, then scaling to full data for final training.

## Custom Models

Use different LLM providers by editing `.env`:

```bash
MODEL_NAME=local/llama-3
OPENROUTER_BASE_URL=http://localhost:8000/v1
OPENROUTER_API_KEY=dummy
```

## Example Configurations

**Development (fast, cheap):**

```python
# .env: MODEL_NAME=anthropic/claude-3-haiku, LOG_LEVEL=DEBUG
scald = Scald(max_iterations=3)
```

**Production (high quality):**

```python
# .env: MODEL_NAME=anthropic/claude-3.5-sonnet, LOG_LEVEL=INFO
scald = Scald(max_iterations=7)
```

**Standard (balanced):**

```python
# .env: MODEL_NAME=anthropic/claude-3.5-sonnet, LOG_LEVEL=INFO
scald = Scald(max_iterations=5)  # Default
```

Continue to [API Reference](../api.md) for detailed class documentation.
