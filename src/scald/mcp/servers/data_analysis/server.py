from pathlib import Path
from typing import Annotated, Any, Literal

import polars as pl
from fastmcp import Context, FastMCP
from pydantic import Field

DESCRIPTION = """
Exploratory Data Analysis MCP server.

Available tools:
- Get feature distributions
- Analyze correlations
- Detect outliers
- Identify categorical vs numerical features
- Check data quality issues

Features:
- Statistical analysis with Polars
- Distribution analysis
- Missing value patterns
"""

mcp = FastMCP("data-analysis", instructions=DESCRIPTION)


@mcp.tool
async def get_feature_distributions(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    ctx: Context,
) -> dict:
    """Get distribution statistics for all features."""
    try:
        df = pl.read_csv(Path(file_path))

        if df.height == 0:
            return {"success": False, "error": "Dataset is empty"}

        distributions = {}
        for col in df.columns:
            dtype = str(df[col].dtype)

            if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                distributions[col] = {
                    "type": "numerical",
                    "dtype": dtype,
                    "count": df[col].count(),
                    "null_count": df[col].null_count(),
                    "mean": float(df[col].mean()) if df[col].count() > 0 else None,
                    "std": float(df[col].std()) if df[col].count() > 0 else None,
                    "min": float(df[col].min()) if df[col].count() > 0 else None,
                    "max": float(df[col].max()) if df[col].count() > 0 else None,
                    "median": float(df[col].median()) if df[col].count() > 0 else None,
                    "unique": df[col].n_unique(),
                }
            else:
                value_counts = df[col].value_counts().limit(10).to_dicts()
                distributions[col] = {
                    "type": "categorical",
                    "dtype": dtype,
                    "count": df[col].count(),
                    "null_count": df[col].null_count(),
                    "unique": df[col].n_unique(),
                    "top_values": value_counts,
                }

        await ctx.info(f"Analyzed {len(distributions)} features")
        return {"success": True, "distributions": distributions}

    except Exception as e:
        await ctx.error(f"Failed to get feature distributions: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool
async def get_correlations(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    ctx: Context,
    method: Annotated[Literal["pearson"], Field(description="Correlation method")] = "pearson",
) -> dict:
    """Calculate correlation matrix."""
    try:
        df = pl.read_csv(Path(file_path))

        numeric_cols = [
            col
            for col in df.columns
            if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]
        ]

        if not numeric_cols:
            return {"success": False, "error": "No numerical columns found"}

        numeric_df = df.select(numeric_cols)
        corr_matrix = {}

        for col1 in numeric_cols:
            corr_matrix[col1] = {}
            for col2 in numeric_cols:
                if col1 == col2:
                    corr_matrix[col1][col2] = 1.0
                else:
                    corr = numeric_df.select(pl.corr(col1, col2)).item()
                    corr_matrix[col1][col2] = float(corr) if corr is not None else None

        await ctx.info(f"Computed correlation matrix for {len(numeric_cols)} columns")
        return {"success": True, "correlations": corr_matrix, "columns": numeric_cols}

    except Exception as e:
        await ctx.error(f"Failed to get correlations: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool
async def detect_outliers(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    ctx: Context,
    iqr_multiplier: Annotated[
        float, Field(description="IQR multiplier for outlier detection (typically 1.5)")
    ] = 1.5,
) -> dict:
    """Detect outliers in numerical columns."""
    try:
        df = pl.read_csv(Path(file_path))

        if df.height == 0:
            return {"success": False, "error": "Dataset is empty"}

        numeric_cols = [
            col
            for col in df.columns
            if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]
        ]

        if not numeric_cols:
            return {"success": False, "error": "No numerical columns found"}

        outliers = {}
        total_outliers = 0
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            outlier_count = df.filter((df[col] < lower_bound) | (df[col] > upper_bound)).height
            total_outliers += outlier_count

            outliers[col] = {
                "count": outlier_count,
                "percentage": (outlier_count / df.height * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }

        await ctx.info(f"Found {total_outliers} outliers across {len(numeric_cols)} columns")
        return {"success": True, "outliers": outliers}

    except Exception as e:
        await ctx.error(f"Failed to detect outliers: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool
async def identify_feature_types(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    ctx: Context,
    categorical_threshold: Annotated[
        int, Field(description="Max unique values to treat as categorical (must be > 0)")
    ] = 20,
) -> dict:
    """Identify feature types automatically."""
    try:
        if categorical_threshold <= 0:
            return {
                "success": False,
                "error": f"categorical_threshold must be > 0, got {categorical_threshold}",
            }

        df = pl.read_csv(Path(file_path))

        categorical = []
        numerical = []
        binary = []

        for col in df.columns:
            n_unique = df[col].n_unique()

            if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                if n_unique == 2:
                    binary.append(col)
                elif n_unique <= categorical_threshold:
                    categorical.append(col)
                else:
                    numerical.append(col)
            else:
                categorical.append(col)

        await ctx.info(
            f"Identified {len(categorical)} categorical, {len(numerical)} numerical, {len(binary)} binary features"
        )
        return {
            "success": True,
            "categorical": categorical,
            "numerical": numerical,
            "binary": binary,
        }

    except Exception as e:
        await ctx.error(f"Failed to identify feature types: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool
async def check_data_quality(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    ctx: Context,
) -> dict:
    """Check for data quality issues."""
    try:
        df = pl.read_csv(Path(file_path))

        if df.height == 0:
            return {"success": False, "error": "Dataset is empty"}

        missing_data: dict[str, Any] = {}
        constant_columns: list[str] = []

        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                missing_data[col] = {
                    "count": null_count,
                    "percentage": (null_count / df.height * 100),
                }

            if df[col].n_unique() == 1:
                constant_columns.append(col)

        duplicate_rows = df.height - df.unique().height

        quality_report = {
            "total_rows": df.height,
            "total_columns": df.width,
            "missing_data": missing_data,
            "duplicate_rows": duplicate_rows,
            "constant_columns": constant_columns,
        }

        issues_count = len(missing_data) + len(constant_columns) + (1 if duplicate_rows > 0 else 0)
        await ctx.info(f"Quality check: {issues_count} issues found")
        return {"success": True, "quality": quality_report}

    except Exception as e:
        await ctx.error(f"Failed to check data quality: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
