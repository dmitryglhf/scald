import tempfile
from pathlib import Path

import pytest
from fastmcp.client import Client

from scald.mcp.servers.file_operations.server import mcp


@pytest.fixture
def fileops_server():
    """Fixture that provides the file_operations server."""
    return mcp


class TestListFiles:
    """Test suite for list_files tool."""

    @pytest.mark.asyncio
    async def test_list_csv_files(self, fileops_server):
        """Test listing CSV files in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.csv").write_text("a,b,c\n1,2,3")
            (Path(tmpdir) / "file2.csv").write_text("x,y,z\n4,5,6")
            (Path(tmpdir) / "readme.txt").write_text("not a csv")

            async with Client(fileops_server) as client:
                result = await client.call_tool(
                    name="list_files",
                    arguments={"directory": tmpdir, "pattern": "*.csv"},
                )

                assert result.data["success"] is True
                assert "files" in result.data
                assert len(result.data["files"]) == 2


class TestCopyFile:
    """Test suite for copy_file tool."""

    @pytest.mark.asyncio
    async def test_copy_file_basic(self, fileops_server):
        """Test basic file copy operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.csv"
            dest = Path(tmpdir) / "dest.csv"

            source.write_text("a,b,c\n1,2,3")

            async with Client(fileops_server) as client:
                result = await client.call_tool(
                    name="copy_file",
                    arguments={"source": str(source), "destination": str(dest)},
                )

                assert result.data["success"] is True
                assert dest.exists()


class TestFileExists:
    """Test suite for file_exists tool."""

    @pytest.mark.asyncio
    async def test_file_exists_true(self, fileops_server):
        """Test checking existence of existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "exists.csv"
            file_path.write_text("data")

            async with Client(fileops_server) as client:
                result = await client.call_tool(
                    name="file_exists",
                    arguments={"path": str(file_path)},
                )

                assert result.data["success"] is True
                assert result.data["exists"] is True
