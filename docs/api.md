# API Reference

Complete API documentation for Scald classes and methods.

## Scald

Main orchestrator class for AutoML workflows.

::: scald.Scald
    options:
      show_source: false
      members_order: source
      separate_signature: true
      show_signature_annotations: true

## Usage Example

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

if __name__ == "__main__":
    results = asyncio.run(main())
```

## Type Hints

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

### `run()` Returns

List of predictions corresponding to test data rows:

- **Classification**: List of class labels (int or str)
- **Regression**: List of numeric predictions (float)

Length matches number of rows in test dataset.

## Exceptions

### Common Exceptions

- `FileNotFoundError`: Training or test file not found
- `ValueError`: Invalid task_type or missing target column
- `RuntimeError`: API errors or execution failures

### Error Handling

```python
try:
    predictions = await scald.run(...)
except FileNotFoundError:
    print("Data file missing")
except ValueError:
    print("Invalid parameters")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## See Also

- [Python API Guide](usage/api.md) - Practical usage examples
- [Configuration](usage/configuration.md) - Settings and options
