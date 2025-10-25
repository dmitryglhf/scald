from pathlib import Path
from typing import Annotated, Any, Optional

import polars as pl
from mcp.server.fastmcp import FastMCP
from pydantic import Field

DESCRIPTION = """
Data processing MCP server.

Available tools:
- Encode categorical features (one-hot, label encoding)
- Handle missing values (drop, fill)
- Remove outliers
- Scale numerical features
- Create train/test splits
- Save processed data

Features:
- Efficient processing with Polars
- Multiple encoding strategies
- Flexible imputation methods
"""

mcp = FastMCP("data-processing", instructions=DESCRIPTION)


@mcp.tool(description="Encode categorical features using one-hot encoding.")
async def encode_categorical_onehot(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    columns: Annotated[list[str], Field(description="Columns to encode")],
    output_path: Annotated[str, Field(description="Path to save encoded data")],
) -> dict:
    """One-hot encode categorical columns."""
    try:
        df = pl.read_csv(Path(file_path))

        for col in columns:
            if col not in df.columns:
                return {"success": False, "error": f"Column {col} not found"}

        df = df.to_dummies(columns=columns)
        df.write_csv(Path(output_path))

        return {
            "success": True,
            "output_path": output_path,
            "new_shape": (df.height, df.width),
            "new_columns": df.columns,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Encode categorical features using label encoding.")
async def encode_categorical_label(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    columns: Annotated[list[str], Field(description="Columns to encode")],
    output_path: Annotated[str, Field(description="Path to save encoded data")],
) -> dict:
    """Label encode categorical columns."""
    try:
        df = pl.read_csv(Path(file_path))

        mappings: dict[str, dict[Any, int]] = {}

        for col in columns:
            if col not in df.columns:
                return {"success": False, "error": f"Column {col} not found"}

            unique_values = df[col].unique().sort().to_list()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            mappings[col] = mapping

            df = df.with_columns(pl.col(col).replace(mapping).alias(col))

        df.write_csv(Path(output_path))

        return {
            "success": True,
            "output_path": output_path,
            "mappings": {k: {str(key): val for key, val in v.items()} for k, v in mappings.items()},
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Handle missing values.")
async def handle_missing_values(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    strategy: Annotated[
        str, Field(description="Strategy: 'drop', 'mean', 'median', 'mode', 'zero'")
    ] = "drop",
    output_path: Annotated[Optional[str], Field(description="Path to save processed data")] = None,
) -> dict:
    """Handle missing values in dataset."""
    try:
        df = pl.read_csv(Path(file_path))
        original_rows = df.height

        if strategy == "drop":
            df = df.drop_nulls()
        elif strategy == "mean":
            for col in df.columns:
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    mean_val = df[col].mean()
                    df = df.with_columns(pl.col(col).fill_null(mean_val))
        elif strategy == "median":
            for col in df.columns:
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    median_val = df[col].median()
                    df = df.with_columns(pl.col(col).fill_null(median_val))
        elif strategy == "mode":
            for col in df.columns:
                mode_val = df[col].mode().first()
                df = df.with_columns(pl.col(col).fill_null(mode_val))
        elif strategy == "zero":
            df = df.fill_null(0)
        else:
            return {"success": False, "error": f"Unknown strategy: {strategy}"}

        if output_path:
            df.write_csv(Path(output_path))

        return {
            "success": True,
            "output_path": output_path,
            "original_rows": original_rows,
            "new_rows": df.height,
            "rows_affected": original_rows - df.height,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Remove outliers using IQR method.")
async def remove_outliers(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    columns: Annotated[list[str], Field(description="Columns to check for outliers")],
    iqr_multiplier: Annotated[float, Field(description="IQR multiplier")] = 1.5,
    output_path: Annotated[Optional[str], Field(description="Path to save cleaned data")] = None,
) -> dict:
    """Remove outliers from specified columns."""
    try:
        df = pl.read_csv(Path(file_path))
        original_rows = df.height

        for col in columns:
            if col not in df.columns:
                return {"success": False, "error": f"Column {col} not found"}

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            df = df.filter((pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound))

        if output_path:
            df.write_csv(Path(output_path))

        return {
            "success": True,
            "output_path": output_path,
            "original_rows": original_rows,
            "new_rows": df.height,
            "rows_removed": original_rows - df.height,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Scale numerical features.")
async def scale_features(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    columns: Annotated[list[str], Field(description="Columns to scale")],
    method: Annotated[str, Field(description="Method: 'minmax' or 'standard'")] = "standard",
    output_path: Annotated[Optional[str], Field(description="Path to save scaled data")] = None,
) -> dict:
    """Scale numerical features."""
    try:
        df = pl.read_csv(Path(file_path))

        for col in columns:
            if col not in df.columns:
                return {"success": False, "error": f"Column {col} not found"}

            if method == "standard":
                mean = df[col].mean()
                std = df[col].std()
                df = df.with_columns(((pl.col(col) - mean) / std).alias(col))
            elif method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                df = df.with_columns(((pl.col(col) - min_val) / (max_val - min_val)).alias(col))
            else:
                return {"success": False, "error": f"Unknown method: {method}"}

        if output_path:
            df.write_csv(Path(output_path))

        return {
            "success": True,
            "output_path": output_path,
            "method": method,
            "scaled_columns": columns,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Create train/test split.")
async def train_test_split(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    test_size: Annotated[float, Field(description="Test set proportion")] = 0.2,
    train_path: Annotated[Optional[str], Field(description="Path to save train set")] = None,
    test_path: Annotated[Optional[str], Field(description="Path to save test set")] = None,
    random_seed: Annotated[int, Field(description="Random seed")] = 42,
) -> dict:
    """Split data into train and test sets."""
    try:
        df = pl.read_csv(Path(file_path))

        df = df.sample(fraction=1.0, seed=random_seed, shuffle=True)

        test_n = int(df.height * test_size)
        train_n = df.height - test_n

        train_df = df.head(train_n)
        test_df = df.tail(test_n)

        if train_path:
            train_df.write_csv(Path(train_path))
        if test_path:
            test_df.write_csv(Path(test_path))

        return {
            "success": True,
            "train_path": train_path,
            "test_path": test_path,
            "train_rows": train_df.height,
            "test_rows": test_df.height,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
