# Python API

Use Scald programmatically for full control over AutoML workflows.

## Basic Usage

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

if __name__ == "__main__":
    asyncio.run(main())
```

## Initialization

### `Scald(max_iterations=5)`

Create a Scald instance.

**Parameters:**

- `max_iterations` (int): Number of Actor-Critic iterations (default: 5)

**Returns:** Scald instance

**Example:**

```python
scald = Scald(max_iterations=10)
```

## Running AutoML

### `scald.run()`

Execute the complete AutoML workflow.

**Parameters:**

- `train_path` (str): Path to training CSV file
- `test_path` (str): Path to test CSV file
- `target` (str): Name of target column
- `task_type` (str): Either "classification" or "regression"

**Returns:** List of predictions for test data

**Example:**

```python
predictions = await scald.run(
    train_path="train.csv",
    test_path="test.csv",
    target="price",
    task_type="regression"
)
```

## Complete Example

### Classification

```python
import asyncio
from scald import Scald

async def classify_customers():
    scald = Scald(max_iterations=5)
    
    predictions = await scald.run(
        train_path="data/customers_train.csv",
        test_path="data/customers_test.csv",
        target="will_purchase",
        task_type="classification"
    )
    
    # predictions is a list of class labels
    print(f"Predicted {sum(predictions)} purchases")
    
    return predictions

if __name__ == "__main__":
    results = asyncio.run(classify_customers())
```

### Regression

```python
import asyncio
from scald import Scald

async def predict_prices():
    scald = Scald(max_iterations=3)
    
    predictions = await scald.run(
        train_path="data/housing_train.csv",
        test_path="data/housing_test.csv",
        target="sale_price",
        task_type="regression"
    )
    
    # predictions is a list of numeric values
    avg_price = sum(predictions) / len(predictions)
    print(f"Average predicted price: ${avg_price:,.2f}")
    
    return predictions

if __name__ == "__main__":
    results = asyncio.run(predict_prices())
```

## Accessing Results

### Predictions

The `run()` method returns predictions as a list:

```python
predictions = await scald.run(...)

# Predictions match test data rows
assert len(predictions) == num_test_rows

# Save to file
import pandas as pd
pd.DataFrame({"prediction": predictions}).to_csv("results.csv")
```

### Session Directory

Each run creates a timestamped session directory:

```python
# sessions/session_20250113_143022/
#   ├── session.log
#   ├── artifacts/
#   └── predictions.csv
```

Access logs and artifacts from this directory.

## Error Handling

```python
import asyncio
from scald import Scald

async def main():
    try:
        scald = Scald(max_iterations=5)
        predictions = await scald.run(
            train_path="train.csv",
            test_path="test.csv",
            target="price",
            task_type="regression"
        )
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
    except ValueError as e:
        print(f"Invalid configuration: {e}")
    except Exception as e:
        print(f"Error during AutoML: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Integration Example

### Batch Processing

```python
import asyncio
from pathlib import Path
from scald import Scald

async def process_datasets(datasets):
    scald = Scald(max_iterations=5)
    results = {}
    
    for name, config in datasets.items():
        print(f"Processing {name}...")
        
        predictions = await scald.run(
            train_path=config["train"],
            test_path=config["test"],
            target=config["target"],
            task_type=config["task_type"]
        )
        
        results[name] = predictions
    
    return results

datasets = {
    "housing": {
        "train": "data/housing_train.csv",
        "test": "data/housing_test.csv",
        "target": "price",
        "task_type": "regression"
    },
    "churn": {
        "train": "data/churn_train.csv",
        "test": "data/churn_test.csv",
        "target": "churned",
        "task_type": "classification"
    }
}

results = asyncio.run(process_datasets(datasets))
```

## Async Context

Scald uses async/await for non-blocking execution:

```python
# Always use asyncio.run() or await
predictions = await scald.run(...)  # ✓ Correct

predictions = scald.run(...)  # ✗ Wrong - returns coroutine
```

## Next Steps

- [Configuration](configuration.md) - Advanced settings
- [CLI Usage](cli.md) - Command-line interface
