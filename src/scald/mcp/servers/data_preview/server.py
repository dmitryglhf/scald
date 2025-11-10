from pathlib import Path
from typing import Annotated, Optional

import polars as pl
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

DESCRIPTION = """
Data preview MCP server for quick CSV file inspection.

Available tools:
- Get CSV metadata (shape, columns, dtypes, missing values, memory usage)

Use cases:
- Quick data exploration before processing
- Validate file structure and schema
- Check data quality issues (missing values, column types)
- Decide which processing steps to apply

Note: This server only inspects files. Other servers (data_analysis,
data_processing, machine_learning) will read CSV files independently.
"""

mcp = FastMCP("data-preview", instructions=DESCRIPTION)


class DataFrameInfo(BaseModel):
    """DataFrame metadata information."""

    shape: tuple[int, int] = Field(description="(rows, columns)")
    columns: list[str] = Field(description="Column names")
    dtypes: dict[str, str] = Field(description="Column data types")
    missing_counts: dict[str, int] = Field(description="Missing values per column")
    memory_usage_mb: float = Field(description="Memory usage in MB")


class InspectResult(BaseModel):
    """Result of CSV inspection."""

    success: bool = Field(description="Inspection succeeded")
    info: Optional[DataFrameInfo] = Field(default=None, description="DataFrame metadata")
    error: Optional[str] = Field(default=None, description="Error if failed")


@mcp.tool
async def inspect_csv(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    ctx: Context,
    infer_schema_length: Annotated[
        int, Field(description="Number of rows to scan for schema inference (default: 1000)")
    ] = 1000,
) -> InspectResult:
    """Inspect CSV file and return metadata: shape, column names, data types, missing value counts, and memory usage. Does not load full dataset.

    Inspect CSV file and return comprehensive metadata without loading full dataset."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pl.read_csv(path, infer_schema_length=infer_schema_length)

        info = DataFrameInfo(
            shape=(df.height, df.width),
            columns=df.columns,
            dtypes={col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            missing_counts={col: df[col].null_count() for col in df.columns},
            memory_usage_mb=df.estimated_size("mb"),
        )

        await ctx.info(
            f"Inspected CSV: {df.height} rows, {df.width} columns, "
            f"{sum(info.missing_counts.values())} missing values"
        )
        return InspectResult(success=True, info=info, error=None)

    except Exception as e:
        await ctx.error(f"Failed to inspect CSV: {e}")
        return InspectResult(success=False, info=None, error=str(e))


@mcp.tool
async def preview_csv(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    ctx: Context,
    n_rows: Annotated[
        int, Field(description="Number of rows to preview (must be > 0, default: 5)")
    ] = 5,
) -> dict:
    """Preview first N rows of CSV file as list of dictionaries. Useful for examining actual data values and patterns.

    Preview first N rows of CSV file to examine actual data."""
    try:
        if n_rows <= 0:
            return {"success": False, "error": f"n_rows must be > 0, got {n_rows}"}

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pl.read_csv(path, n_rows=n_rows)

        await ctx.info(f"Previewed {df.height} rows from CSV")
        return {
            "success": True,
            "preview": df.to_dicts(),
            "columns": df.columns,
            "n_rows": df.height,
        }

    except Exception as e:
        await ctx.error(f"Failed to preview CSV: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
