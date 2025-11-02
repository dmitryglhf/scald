import asyncio

import polars as pl
from sklearn.metrics import accuracy_score, classification_report

from scald.main import Scald


async def main():
    scald = Scald(max_iterations=5)

    predictions = await scald.run(
        train_path="examples/data/iris_train.csv",
        test_path="examples/data/iris_test.csv",
        target="Species",
        task_type="classification",
    )

    # Load validation data with true labels
    val_df = pl.read_csv("examples/data/iris_val.csv")
    y_true = val_df["Species"].to_numpy()

    # Evaluate predictions
    accuracy = accuracy_score(y_true, predictions)

    print("\nRESULTS")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Correct: {(y_true == predictions).sum()}/{len(y_true)}")
    print(f"\n{classification_report(y_true, predictions, zero_division=0)}")


if __name__ == "__main__":
    asyncio.run(main())
