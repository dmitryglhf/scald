# Quick Start

This guide walks through running your first AutoML task with Scald.

## Prepare Your Data

Scald expects CSV files with:

- Training data with features and target column
- Test data with the same features (target optional)

Example structure:

```csv
feature_1,feature_2,feature_3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
...
```

## CLI Usage

The simplest way to use Scald is via the command line:

```bash
scald --train data/train.csv \
      --test data/test.csv \
      --target price \
      --task-type regression
```

### CLI Options

- `--train`: Path to training CSV file (required)
- `--test`: Path to test CSV file (required)
- `--target`: Name of target column (required)
- `--task-type`: Either `classification` or `regression` (required)
- `--max-iterations`: Number of Actor-Critic iterations (default: 5)

## Python API Usage

For more control, use the Python API:

```python
import asyncio
from scald import Scald

async def main():
    # Initialize Scald
    scald = Scald(max_iterations=5)
    
    # Run AutoML workflow
    predictions = await scald.run(
        train_path="data/train.csv",
        test_path="data/test.csv",
        target="target_column",
        task_type="classification"
    )
    
    # predictions is a list of predicted values
    print(f"Generated {len(predictions)} predictions")

if __name__ == "__main__":
    asyncio.run(main())
```

## What Happens During Execution

1. **Data Preview**: Actor examines training and test data
2. **Analysis**: Actor performs EDA, identifies patterns, missing values, outliers
3. **Preprocessing**: Actor applies transformations (encoding, scaling, feature engineering)
4. **Model Training**: Actor trains models (CatBoost, LightGBM, or XGBoost)
5. **Evaluation**: Critic reviews the solution and provides feedback
6. **Refinement**: Based on feedback, Actor improves the solution
7. **Iteration**: Steps 2-6 repeat for max_iterations
8. **Prediction**: Final model generates predictions on test data

## Output

Scald creates a session directory with:

- `session.log`: Detailed execution logs
- `artifacts/`: Generated code and intermediate files
- `predictions.csv`: Final predictions
- Cost and token usage summary

## Example Output

```
Session: session_20250113_143022
Iterations: 5/5
Final Score: 0.87
Cost: $0.42
Predictions saved to: predictions.csv
```

## Troubleshooting

**API Key Issues**: Verify your `.env` file has correct credentials

**Memory Errors**: For large datasets, ensure sufficient RAM

**Poor Performance**: Try increasing `max_iterations` for more refinement

## Next Steps

- [Architecture](architecture.md) - Understand how Scald works
- [Actor-Critic Pattern](actor-critic.md) - Learn about the agent collaboration
- [Configuration](usage/configuration.md) - Customize Scald behavior
