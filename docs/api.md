# API Reference

Complete reference for Scald classes and methods.

## Scald

Main orchestrator for AutoML workflows.

::: scald.Scald
    options:
      show_source: false
      members_order: source
      separate_signature: true
      show_signature_annotations: true

## Usage

```python
import asyncio
from scald import Scald

async def main():
    scald = Scald(max_iterations=5)
    
    predictions = await scald.run(
        train_path="train.csv",
        test_path="test.csv",
        target="price",
        task_type="regression"
    )
    
    return predictions

results = asyncio.run(main())
```

## Type Signatures

```python
from typing import List

class Scald:
    def __init__(self, max_iterations: int = 5) -> None: ...
    
    async def run(
        self,
        train_path: str,
        test_path: str,
        target: str,
        task_type: str
    ) -> List[float | int | str]: ...
```

## Return Values

`run()` returns predictions as a list. For classification, list contains class labels (int or str). For regression, list contains numeric predictions (float). List length matches test data row count.

## Exceptions

Common exceptions: `FileNotFoundError` for missing data files, `ValueError` for invalid task_type or missing target column, `RuntimeError` for API or execution failures.

Error handling:

```python
try:
    predictions = await scald.run(...)
except FileNotFoundError:
    print("Data file missing")
except ValueError:
    print("Invalid parameters")
except Exception as e:
    print(f"Error: {e}")
```

See [Python API Guide](usage/api.md) for practical examples and [Configuration](usage/configuration.md) for settings.
