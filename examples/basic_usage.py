import asyncio

import polars as pl
from sklearn.metrics import accuracy_score, classification_report

from scald.main import Scald


async def main():
    scald = Scald(max_iterations=5)

    y_pred = await scald.run(
        train="examples/data/iris_train.csv",
        test="examples/data/iris_test.csv",
        target="Species",
        task_type="classification",
    )
    print(y_pred)

    val_df = pl.read_csv("examples/data/iris_val.csv")
    y_true = val_df["Species"].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)

    print("\nRESULTS")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Correct: {(y_true == y_pred).sum()}/{len(y_true)}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")


async def main_with_dataframe():
    scald = Scald(max_iterations=5)

    train_df = pl.read_csv("examples/data/iris_train.csv")
    test_df = pl.read_csv("examples/data/iris_test.csv")

    y_pred = await scald.run(
        train=train_df,
        test=test_df,
        target="Species",
        task_type="classification",
    )
    print(y_pred)

    val_df = pl.read_csv("examples/data/iris_val.csv")
    y_true = val_df["Species"].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)

    print("\nRESULTS (DataFrame input)")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Correct: {(y_true == y_pred).sum()}/{len(y_true)}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")


if __name__ == "__main__":
    # print("Example 1: Using CSV file paths")
    # asyncio.run(main())

    print("\n" + "=" * 80 + "\n")
    print("Example 2: Using Polars DataFrames")
    asyncio.run(main_with_dataframe())
