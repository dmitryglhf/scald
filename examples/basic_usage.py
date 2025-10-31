import asyncio

from scald.main import Scald


async def main():
    scald = Scald(max_iterations=5)

    predictions = await scald.run(
        train_path="examples/data/iris_train.csv",
        test_path="examples/data/iris_test.csv",
        target="Species",
        task_type="classification",
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions dtype: {predictions.dtype}")
    print(f"First 10 predictions: {predictions[:10]}")


if __name__ == "__main__":
    asyncio.run(main())
