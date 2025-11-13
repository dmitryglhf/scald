# Quick Start

## Prepare Data

Scald expects CSV files with training data (features + target) and test data (same features). The target column should be numeric for regression or categorical for classification.

```csv
feature_1,feature_2,target
1.2,3.4,0
2.3,4.5,1
```

## CLI Usage

Run AutoML with a single command:

```bash
scald --train data/train.csv \
      --test data/test.csv \
      --target price \
      --task-type regression \
      --max-iterations 5
```

Task type must be either `classification` or `regression`. Iterations control Actor-Critic refinement cycles (default: 5).

## Python API

For programmatic control:

```python
import asyncio
from scald import Scald

async def main():
    scald = Scald(max_iterations=5)
    
    predictions = await scald.run(
        train_path="data/train.csv",
        test_path="data/test.csv",
        target="target_column",
        task_type="classification"
    )
    
    print(f"Generated {len(predictions)} predictions")

asyncio.run(main())
```

## Execution Flow

The workflow progresses through data preview, exploratory analysis, preprocessing, model training, and evaluation. The Critic reviews each iteration and provides feedback. The Actor refines the approach based on this feedback, converging on an optimal solution. Each iteration improves upon the previous attempt through targeted adjustments.

## Output

Scald creates a timestamped session directory containing detailed logs, generated code artifacts, and final predictions:

```
sessions/session_20250113_143022/
├── session.log          # Execution logs
├── artifacts/           # Generated code per iteration
└── predictions.csv      # Final predictions
```

Console output shows iteration progress, final metrics, cost, and execution time.

## Troubleshooting

API key errors indicate missing or incorrect credentials in `.env`. Memory errors suggest insufficient RAM for the dataset size. Poor performance can be improved by increasing `max_iterations` for additional refinement cycles.

Continue to [Architecture](architecture.md) to understand how Scald works internally.
