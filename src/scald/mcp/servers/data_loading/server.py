from pathlib import Path
from typing import Annotated, Optional

import polars as pl
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from scald.common.logger import get_logger

logger = get_logger()

DESCRIPTION = """
Data loading MCP server.

Available tools:
- Load CSV files into Polars DataFrames
- Get basic dataset information
- Save DataFrames to CSV

Features:
- Automatic type inference
- Handle missing values
- Memory-efficient loading with Polars
"""

mcp = FastMCP("data-load", instructions=DESCRIPTION)


class DataFrameInfo(BaseModel):
    """DataFrame information."""

    shape: tuple[int, int] = Field(description="(rows, columns)")
    columns: list[str] = Field(description="Column names")
    dtypes: dict[str, str] = Field(description="Column data types")
    missing_counts: dict[str, int] = Field(description="Missing values per column")
    memory_usage_mb: float = Field(description="Memory usage in MB")


class LoadResult(BaseModel):
    """Result of loading CSV."""

    success: bool = Field(description="Load succeeded")
    info: Optional[DataFrameInfo] = Field(default=None, description="DataFrame info")
    error: Optional[str] = Field(default=None, description="Error if failed")


@mcp.tool(
    description="Load CSV file and return dataset information (shape, dtypes, missing values)."
)
async def load_csv(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    infer_schema_length: Annotated[
        int, Field(description="Number of rows to scan for schema inference")
    ] = 1000,
) -> LoadResult:
    """Load CSV file and return comprehensive dataset information."""
    logger.info(f"[MCP:data_loading] load_csv: {file_path}")
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

        logger.info(f"[MCP:data_loading] Loaded CSV: {df.height} rows, {df.width} columns")
        return LoadResult(success=True, info=info, error=None)

    except Exception as e:
        logger.error(f"[MCP:data_loading] Failed to load CSV: {e}")
        return LoadResult(success=False, info=None, error=str(e))


@mcp.tool(description="Preview first N rows of CSV file as list of dictionaries.")
async def preview_csv(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    n_rows: Annotated[int, Field(description="Number of rows to preview (must be > 0)")] = 5,
) -> dict:
    """Preview first N rows of CSV file."""
    logger.info(f"[MCP:data_loading] preview_csv: {file_path}, n_rows={n_rows}")
    try:
        if n_rows <= 0:
            return {"success": False, "error": f"n_rows must be > 0, got {n_rows}"}

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pl.read_csv(path, n_rows=n_rows)

        logger.info(f"[MCP:data_loading] Previewed {df.height} rows from CSV")
        return {
            "success": True,
            "preview": df.to_dicts(),
            "columns": df.columns,
            "n_rows": df.height,
        }

    except Exception as e:
        logger.error(f"[MCP:data_loading] Failed to preview CSV: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool(
    description="Get descriptive statistics (mean, std, min, max, quartiles) for all columns."
)
async def describe_csv(
    file_path: Annotated[str, Field(description="Path to CSV file")],
) -> dict:
    """Get descriptive statistics for CSV file."""
    logger.info(f"[MCP:data_loading] describe_csv: {file_path}")
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pl.read_csv(path)
        stats = df.describe().to_dicts()

        logger.info(f"[MCP:data_loading] Generated statistics for {df.width} columns")
        return {"success": True, "statistics": stats}

    except Exception as e:
        logger.error(f"[MCP:data_loading] Failed to describe CSV: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
