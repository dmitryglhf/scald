import tempfile
from pathlib import Path

import polars as pl
import pytest
from fastmcp.client import Client

from scald.mcp.servers.data_processing.server import mcp


@pytest.fixture
def processing_server():
    """Fixture that provides the data_processing server."""
    return mcp


class TestCategoricalEncoding:
    """Test suite for categorical encoding tools."""

    @pytest.mark.asyncio
    async def test_onehot_encoding(self, processing_server):
        """Test one-hot encoding of categorical features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.csv"

            df = pl.DataFrame({"cat": ["A", "B", "A", "C"], "num": [1, 2, 3, 4]})
            df.write_csv(input_path)

            async with Client(processing_server) as client:
                result = await client.call_tool(
                    name="encode_categorical_onehot",
                    arguments={
                        "file_path": str(input_path),
                        "columns": ["cat"],
                        "output_path": str(output_path),
                    },
                )

                assert result.data["success"] is True
                assert output_path.exists()


class TestMissingValues:
    """Test suite for missing value handling."""

    @pytest.mark.asyncio
    async def test_drop_missing(self, processing_server):
        """Test dropping rows with missing values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.csv"

            df = pl.DataFrame({"col1": [1.0, None, 3.0], "col2": [10, 20, None]})
            df.write_csv(input_path)

            async with Client(processing_server) as client:
                result = await client.call_tool(
                    name="handle_missing_values",
                    arguments={
                        "file_path": str(input_path),
                        "strategy": "drop",
                        "output_path": str(output_path),
                    },
                )

                assert result.data["success"] is True
