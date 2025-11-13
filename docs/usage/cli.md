# CLI Usage

The Scald command-line interface provides a simple way to run AutoML tasks.

## Basic Usage

```bash
scald --train <train.csv> --test <test.csv> --target <column> --task-type <type>
```

## Required Arguments

### `--train`

Path to training CSV file.

```bash
scald --train data/train.csv ...
```

### `--test`

Path to test CSV file.

```bash
scald --test data/test.csv ...
```

### `--target`

Name of the target column in training data.

```bash
scald --target price ...
```

### `--task-type`

Either `classification` or `regression`.

```bash
scald --task-type regression ...
```

## Optional Arguments

### `--max-iterations`

Number of Actor-Critic refinement iterations (default: 5).

```bash
scald --max-iterations 10 ...
```

More iterations = more refinement, but higher costs.

## Complete Examples

### Classification Task

```bash
scald --train data/titanic_train.csv \
      --test data/titanic_test.csv \
      --target survived \
      --task-type classification \
      --max-iterations 5
```

### Regression Task

```bash
scald --train data/housing_train.csv \
      --test data/housing_test.csv \
      --target price \
      --task-type regression \
      --max-iterations 3
```

## Output

Scald creates a session directory with results:

```
sessions/session_20250113_143022/
├── session.log          # Detailed logs
├── artifacts/           # Generated code
└── predictions.csv      # Final predictions
```

### Console Output

```
Scald AutoML Framework
======================
Train: data/train.csv
Test: data/test.csv
Target: price
Task: regression
Max Iterations: 5

Iteration 1/5... ✓
Iteration 2/5... ✓
Iteration 3/5... ✓
Iteration 4/5... ✓
Iteration 5/5... ✓

Results:
--------
Session: session_20250113_143022
Score: 0.87
Cost: $0.42
Time: 8m 32s

Predictions saved to: predictions.csv
```

## Environment Variables

Ensure `.env` is configured:

```bash
OPENROUTER_API_KEY=your_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Help

View all options:

```bash
scald --help
```

## Exit Codes

- `0`: Success
- `1`: Error (check logs)

## Troubleshooting

### "API key not found"

Set `OPENROUTER_API_KEY` in `.env`.

### "File not found"

Check paths to CSV files are correct.

### "Invalid task type"

Use `classification` or `regression`.

## Next Steps

- [Python API](api.md) - Programmatic usage
- [Configuration](configuration.md) - Advanced settings
