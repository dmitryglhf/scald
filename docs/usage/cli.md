# CLI Usage

Run Scald from the command line for straightforward AutoML tasks.

## Basic Command

```bash
scald --train <train.csv> --test <test.csv> --target <column> --task-type <type>
```

All four parameters are required. Task type must be either `classification` or `regression`.

## Options

`--train` specifies the training CSV file path. `--test` specifies the test CSV file path. `--target` names the target column in training data. `--task-type` defines the problem type. `--max-iterations` controls refinement cycles (default: 5).

## Examples

Classification:

```bash
scald --train data/titanic_train.csv \
      --test data/titanic_test.csv \
      --target survived \
      --task-type classification \
      --max-iterations 5
```

Regression:

```bash
scald --train data/housing_train.csv \
      --test data/housing_test.csv \
      --target price \
      --task-type regression
```

## Output

Scald creates a session directory with logs, artifacts, and predictions:

```
sessions/session_20250113_143022/
├── session.log
├── artifacts/
└── predictions.csv
```

Console output shows iteration progress, final metrics, cost, and execution time.

## Configuration

Ensure `.env` contains API credentials:

```bash
OPENROUTER_API_KEY=your_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Help

View all options:

```bash
scald --help
```

## Troubleshooting

"API key not found" indicates missing `OPENROUTER_API_KEY` in `.env`. "File not found" means incorrect CSV paths. "Invalid task type" requires using `classification` or `regression`.

Continue to [Python API](api.md) for programmatic usage.
