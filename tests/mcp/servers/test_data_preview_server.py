import tempfile
from pathlib import Path

import polars as pl
import pytest
from fastmcp.client import Client

from scald.mcp.servers.data_preview.server import mcp


@pytest.fixture
def preview_server():
    """Fixture that provides the data_preview server."""
    return mcp


class TestInspectCSV:
    """Test suite for inspect_csv tool."""

    @pytest.mark.asyncio
    async def test_basic_inspection(self, preview_server):
        """Test basic CSV inspection with metadata retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            df = pl.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
            df.write_csv(csv_path)

            async with Client(preview_server) as client:
                result = await client.call_tool(
                    name="inspect_csv",
                    arguments={"file_path": str(csv_path)},
                )

                assert result.data.success is True
                assert result.data.info.shape == [3, 2]


class TestPreviewCSV:
    """Test suite for preview_csv tool."""

    @pytest.mark.asyncio
    async def test_default_preview(self, preview_server):
        """Test default preview."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
            df.write_csv(csv_path)

            async with Client(preview_server) as client:
                result = await client.call_tool(
                    name="preview_csv",
                    arguments={"file_path": str(csv_path)},
                )

                assert result.data["success"] is True
                assert len(result.data["preview"]) == 5
