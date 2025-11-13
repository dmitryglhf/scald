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

asyncio.run(main())
```

## API Reference

`Scald(max_iterations=5)` creates an instance. The parameter controls Actor-Critic refinement cycles.

`await scald.run(train_path, test_path, target, task_type)` executes the AutoML workflow. Returns a list of predictions matching test data rows. Task type must be "classification" or "regression".

## Examples

Classification:

```python
async def classify():
    scald = Scald(max_iterations=5)
    predictions = await scald.run(
        train_path="customers_train.csv",
        test_path="customers_test.csv",
        target="will_purchase",
        task_type="classification"
    )
    return predictions

results = asyncio.run(classify())
```

Regression:

```python
async def predict_prices():
    scald = Scald(max_iterations=3)
    predictions = await scald.run(
        train_path="housing_train.csv",
        test_path="housing_test.csv",
        target="sale_price",
        task_type="regression"
    )
    return predictions

results = asyncio.run(predict_prices())
```

## Return Values

The `run()` method returns predictions as a list. For classification, elements are class labels (int or str). For regression, elements are numeric values (float). Length matches the number of test data rows.

## Error Handling

```python
try:
    predictions = await scald.run(...)
except FileNotFoundError:
    print("Data file missing")
except ValueError:
    print("Invalid parameters")
except Exception as e:
    print(f"Execution error: {e}")
```

## Batch Processing

Process multiple datasets sequentially:

```python
async def process_batch(datasets):
    scald = Scald(max_iterations=5)
    results = {}
    
    for name, config in datasets.items():
        predictions = await scald.run(**config)
        results[name] = predictions
    
    return results

datasets = {
    "housing": {
        "train_path": "housing_train.csv",
        "test_path": "housing_test.csv",
        "target": "price",
        "task_type": "regression"
    },
    "churn": {
        "train_path": "churn_train.csv",
        "test_path": "churn_test.csv",
        "target": "churned",
        "task_type": "classification"
    }
}

results = asyncio.run(process_batch(datasets))
```

## Async Context

Scald uses async/await for non-blocking execution. Always use `await` with `run()` or wrap in `asyncio.run()` for top-level calls.

Continue to [Configuration](configuration.md) for advanced settings.
