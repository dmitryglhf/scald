import asyncio
from pathlib import Path

import polars as pl
from sklearn.metrics import accuracy_score, classification_report

from scald.main import Scald


async def main():
    examples_dir = Path(__file__).parent
    data_dir = examples_dir / "data"

    scald = Scald(max_iterations=5)

    train_df = pl.read_csv(data_dir / "iris" / "iris_train.csv")
    test_df = pl.read_csv(data_dir / "iris" / "iris_test.csv")

    y_pred = await scald.run(
        train=train_df,
        test=test_df,
        target="Species",
        task_type="classification",
    )
    print(y_pred)

    val_df = pl.read_csv(data_dir / "iris" / "iris_val.csv")
    y_true = val_df["Species"].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)

    print("\nRESULTS")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Correct: {(y_true == y_pred).sum()}/{len(y_true)}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")


if __name__ == "__main__":
    asyncio.run(main())
