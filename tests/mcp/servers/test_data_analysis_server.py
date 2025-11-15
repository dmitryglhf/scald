import tempfile
from pathlib import Path

import polars as pl
import pytest
from fastmcp.client import Client

from scald.mcp.servers.data_analysis.server import mcp


@pytest.fixture
def analysis_server():
    """Fixture that provides the data_analysis server."""
    return mcp


class TestFeatureDistributions:
    """Test suite for get_feature_distributions tool."""

    @pytest.mark.asyncio
    async def test_mixed_types(self, analysis_server):
        """Test feature distributions with mixed data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            df = pl.DataFrame(
                {
                    "numeric_int": [1, 2, 3, 4, 5],
                    "categorical": ["A", "B", "A", "C", "B"],
                }
            )
            df.write_csv(csv_path)

            async with Client(analysis_server) as client:
                result = await client.call_tool(
                    name="get_feature_distributions",
                    arguments={"file_path": str(csv_path)},
                )

                assert result.data["success"] is True
                assert "distributions" in result.data


class TestCorrelations:
    """Test suite for get_correlations tool."""

    @pytest.mark.asyncio
    async def test_numeric_correlations(self, analysis_server):
        """Test correlation matrix calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "numeric.csv"
            df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
            df.write_csv(csv_path)

            async with Client(analysis_server) as client:
                result = await client.call_tool(
                    name="get_correlations",
                    arguments={"file_path": str(csv_path)},
                )

                assert result.data["success"] is True
                assert "correlations" in result.data
