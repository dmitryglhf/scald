# Configuration

Configure Scald behavior through environment variables and initialization parameters.

## Environment Variables

Settings in `.env` file:

### API Configuration

```bash
# OpenRouter API credentials
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Required for LLM access.

### Model Selection

```bash
# Default model for Actor and Critic
MODEL_NAME=anthropic/claude-3.5-sonnet
```

Options depend on your API provider. OpenRouter supports many models.

### Logging

```bash
# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```

Controls verbosity of session logs.

## Initialization Parameters

### `max_iterations`

Number of Actor-Critic refinement cycles.

```python
scald = Scald(max_iterations=10)  # More refinement
```

**Trade-offs:**
- Higher = better quality, higher cost
- Lower = faster, cheaper, potentially lower quality
- Default: 5 (good balance)

**Recommendation:**
- Simple tasks: 3
- Standard tasks: 5
- Complex tasks: 7-10

## Runtime Parameters

### `task_type`

Classification or regression.

```python
predictions = await scald.run(
    task_type="classification"  # or "regression"
)
```

**Classification:**
- Binary or multiclass
- Predicts discrete labels
- Metrics: accuracy, F1, precision, recall

**Regression:**
- Continuous values
- Predicts numeric targets
- Metrics: RMSE, MAE, R²

## File Paths

### Data Paths

```python
predictions = await scald.run(
    train_path="path/to/train.csv",
    test_path="path/to/test.csv",
    target="column_name"
)
```

**Requirements:**
- Both must be CSV files
- Same feature columns
- Target column in training data
- Test data may optionally include target

## Session Configuration

### Session Directory

Sessions are automatically created:

```
sessions/session_YYYYMMDD_HHMMSS/
```

Location: `./sessions` (current directory)

### Artifacts

Generated code and models saved to:

```
sessions/<session_id>/artifacts/
```

Includes:
- Preprocessing code
- Training code
- Model checkpoints (if saved)

## Memory Configuration

### ChromaDB Settings

Long-term memory is stored in:

```
.chroma/
```

Default settings usually work well. For custom configuration, modify `MemoryManager` initialization (advanced).

## Cost Tracking

### Token Usage

Scald automatically tracks:
- Tokens per iteration
- Total tokens per session
- Estimated costs (based on model pricing)

View in session logs.

### Cost Optimization

To reduce costs:

1. Lower `max_iterations`
2. Use cheaper models (in `.env`)
3. Smaller datasets for prototyping

## Performance Tuning

### Speed vs Quality

**Faster:**
- `max_iterations=3`
- Smaller training sets
- Simpler models (configure via Actor prompts)

**Higher Quality:**
- `max_iterations=7+`
- More training data
- Ensemble methods

### Memory Usage

For large datasets:
- Ensure sufficient RAM
- Consider sampling for initial iterations
- Use efficient data formats (Polars internally)

## Advanced Configuration

### Custom Models

Edit `.env` to use different LLM providers:

```bash
# Example: Local LLM
MODEL_NAME=local/llama-3
OPENROUTER_BASE_URL=http://localhost:8000/v1
OPENROUTER_API_KEY=dummy
```

### Workspace Isolation

Each session uses isolated workspace:

```python
# Sessions don't interfere with each other
scald1 = Scald()
scald2 = Scald()

# Can run concurrently (separate sessions)
```

## Example Configurations

### Development (Fast & Cheap)

```python
# .env
LOG_LEVEL=DEBUG
MODEL_NAME=anthropic/claude-3-haiku

# code
scald = Scald(max_iterations=3)
```

### Production (High Quality)

```python
# .env
LOG_LEVEL=INFO
MODEL_NAME=anthropic/claude-3.5-sonnet

# code
scald = Scald(max_iterations=7)
```

### Experimentation (Balanced)

```python
# .env
LOG_LEVEL=INFO
MODEL_NAME=anthropic/claude-3.5-sonnet

# code
scald = Scald(max_iterations=5)  # Default
```

## Troubleshooting

### API Errors

Check `.env` has correct credentials:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

### Poor Performance

Try increasing iterations:

```python
scald = Scald(max_iterations=10)
```

### Out of Memory

Reduce dataset size or use machine with more RAM.

## Next Steps

- [CLI Usage](cli.md) - Command-line interface
- [Python API](api.md) - Programmatic usage
