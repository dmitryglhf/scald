import asyncio

from scald.common.types import TaskType
from scald.main import Scald


async def main():
    # Test with Docker isolation
    scald = Scald(max_iterations=1, use_docker=True)

    predictions = await scald.run(
        train_path="examples/data/iris_train.csv",
        test_path="examples/data/iris_test.csv",
        target="Species",
        task_type=TaskType.CLASSIFICATION,
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions dtype: {predictions.dtype}")
    print(f"First 10 predictions: {predictions[:10]}")


if __name__ == "__main__":
    asyncio.run(main())
