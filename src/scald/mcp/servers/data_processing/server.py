import json
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import polars as pl
from fastmcp import Context, FastMCP
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


@mcp.tool
async def encode_categorical_onehot(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    columns: Annotated[list[str], Field(description="Columns to encode")],
    output_path: Annotated[str, Field(description="Path to save encoded data")],
    ctx: Context,
) -> dict:
    """Encode categorical features using one-hot encoding.

    One-hot encode categorical columns."""
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


@mcp.tool
async def encode_categorical_label(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    columns: Annotated[list[str], Field(description="Columns to encode")],
    output_path: Annotated[str, Field(description="Path to save encoded data")],
    ctx: Context,
    save_mappings: Annotated[bool, Field(description="Save mappings to files")] = True,
    mappings_dir: Annotated[
        Optional[str], Field(description="Directory to save mapping files")
    ] = None,
) -> dict:
    """Encode categorical features using label encoding.

    Label encode categorical columns.

    Automatically saves encoding mappings to JSON files for later decoding.
    Returns both the mappings dict and paths to saved mapping files.
    """
    try:
        df = pl.read_csv(Path(file_path))

        mappings: dict[str, dict[Any, int]] = {}
        mapping_paths: dict[str, str] = {}

        for col in columns:
            if col not in df.columns:
                return {"success": False, "error": f"Column {col} not found"}

            unique_values = df[col].unique().sort().to_list()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            mappings[col] = mapping

            df = df.with_columns(pl.col(col).replace(mapping).cast(pl.Int64).alias(col))

        df.write_csv(Path(output_path))

        # Save mappings to files
        if save_mappings:
            if mappings_dir is None:
                mappings_dir = "/output/encodings"

            mappings_path = Path(mappings_dir)
            mappings_path.mkdir(parents=True, exist_ok=True)

            for col, mapping in mappings.items():
                # Convert all keys to strings for JSON serialization
                json_mapping = {str(k): v for k, v in mapping.items()}

                col_mapping_path = mappings_path / f"{col}_mapping.json"
                with open(col_mapping_path, "w") as f:
                    json.dump(json_mapping, f, indent=2)

                mapping_paths[col] = str(col_mapping_path)
                await ctx.info(f"Saved {col} mapping to {col_mapping_path}")

        return {
            "success": True,
            "output_path": output_path,
            "mappings": {k: {str(key): val for key, val in v.items()} for k, v in mappings.items()},
            "mapping_paths": mapping_paths,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
async def decode_categorical_label(
    file_path: Annotated[str, Field(description="Path to CSV file with encoded predictions")],
    column: Annotated[str, Field(description="Column name to decode")],
    output_path: Annotated[str, Field(description="Path to save decoded data")],
    ctx: Context,
    mapping: Annotated[
        Optional[dict[str, int]],
        Field(description="Encoding mapping dict from encode_categorical_label"),
    ] = None,
    mapping_path: Annotated[Optional[str], Field(description="Path to mapping JSON file")] = None,
) -> dict:
    """Decode label-encoded predictions back to original categories.

    Decode label-encoded column back to original categories.

    Use this after making predictions to convert encoded integers back to original category names.
    Provide either 'mapping' dict or 'mapping_path' to a saved JSON file.
    """
    try:
        # Load mapping from file or use provided dict
        if mapping_path:
            if not Path(mapping_path).exists():
                return {"success": False, "error": f"Mapping file not found: {mapping_path}"}

            with open(Path(mapping_path)) as f:
                mapping = json.load(f)
            await ctx.info(f"Loaded mapping from {mapping_path}")
        elif mapping is None:
            return {
                "success": False,
                "error": "Provide either 'mapping' dict or 'mapping_path' to JSON file",
            }

        df = pl.read_csv(Path(file_path))

        if column not in df.columns:
            return {"success": False, "error": f"Column {column} not found"}

        # Invert the mapping: {0: 'category1', 1: 'category2', ...}
        inverse_mapping = {v: k for k, v in mapping.items()}

        # Convert column to int first in case it's stored as float
        df = df.with_columns(pl.col(column).cast(pl.Int64).replace(inverse_mapping).alias(column))

        df.write_csv(Path(output_path))

        return {
            "success": True,
            "output_path": output_path,
            "decoded_column": column,
            "unique_values": df[column].unique().to_list(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
async def handle_missing_values(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    ctx: Context,
    strategy: Annotated[
        Literal["drop", "mean", "median", "mode", "zero"],
        Field(description="Missing value strategy"),
    ] = "drop",
    output_path: Annotated[Optional[str], Field(description="Path to save processed data")] = None,
) -> dict:
    """Handle missing values.

    Handle missing values in dataset."""
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
                mode_result = df[col].mode()
                if mode_result.height > 0:
                    mode_val = mode_result.first()
                    df = df.with_columns(pl.col(col).fill_null(mode_val))
                else:
                    await ctx.warning(f"Column {col} has no mode (all nulls?), skipping")
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


@mcp.tool
async def remove_outliers(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    columns: Annotated[list[str], Field(description="Columns to check for outliers")],
    ctx: Context,
    iqr_multiplier: Annotated[float, Field(description="IQR multiplier")] = 1.5,
    output_path: Annotated[Optional[str], Field(description="Path to save cleaned data")] = None,
) -> dict:
    """Remove outliers using IQR method.

    Remove outliers from specified columns."""
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


@mcp.tool
async def scale_features(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    columns: Annotated[list[str], Field(description="Columns to scale")],
    ctx: Context,
    method: Annotated[
        Literal["minmax", "standard"], Field(description="Scaling method")
    ] = "standard",
    output_path: Annotated[Optional[str], Field(description="Path to save scaled data")] = None,
) -> dict:
    """Scale numerical features using standard or minmax scaling.

    Scale numerical features."""
    try:
        df = pl.read_csv(Path(file_path))

        for col in columns:
            if col not in df.columns:
                return {"success": False, "error": f"Column {col} not found"}

            if method == "standard":
                mean = df[col].mean()
                std = df[col].std()
                if std == 0 or std is None:
                    await ctx.warning(f"Column {col} has zero std, skipping scaling")
                    continue
                df = df.with_columns(((pl.col(col) - mean) / std).alias(col))
            elif method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val == max_val:
                    await ctx.warning(f"Column {col} has constant value, skipping scaling")
                    continue
                df = df.with_columns(((pl.col(col) - min_val) / (max_val - min_val)).alias(col))
            else:
                return {"success": False, "error": f"Unknown method: {method}"}

        if output_path:
            df.write_csv(Path(output_path))
            await ctx.info(f"Saved scaled data to {output_path}")

        return {
            "success": True,
            "output_path": output_path,
            "method": method,
            "scaled_columns": columns,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
async def train_test_split(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    ctx: Context,
    test_size: Annotated[float, Field(description="Test set proportion (0.0-1.0)")] = 0.2,
    train_path: Annotated[Optional[str], Field(description="Path to save train set")] = None,
    test_path: Annotated[Optional[str], Field(description="Path to save test set")] = None,
    random_seed: Annotated[int, Field(description="Random seed for reproducibility")] = 42,
) -> dict:
    """Split dataset into train and test sets with stratified sampling.

    Split data into train and test sets."""
    try:
        # Validate test_size
        if not 0 < test_size < 1:
            return {
                "success": False,
                "error": f"test_size must be between 0 and 1, got {test_size}",
            }

        df = pl.read_csv(Path(file_path))

        if df.height == 0:
            return {"success": False, "error": "Dataset is empty"}

        df = df.sample(fraction=1.0, seed=random_seed, shuffle=True)

        test_n = int(df.height * test_size)
        train_n = df.height - test_n

        train_df = df.head(train_n)
        test_df = df.tail(test_n)

        if train_path:
            train_df.write_csv(Path(train_path))
            await ctx.info(f"Saved train set to {train_path}")
        if test_path:
            test_df.write_csv(Path(test_path))
            await ctx.info(f"Saved test set to {test_path}")

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
    mcp.run(transport="stdio", show_banner=False)
