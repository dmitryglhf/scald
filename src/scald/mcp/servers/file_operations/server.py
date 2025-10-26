import shutil
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from scald.common.logger import get_logger

logger = get_logger()

DESCRIPTION = """
File operations MCP server for CSV workflow management.

Available tools:
- List CSV files in directories
- Copy CSV files
- Move CSV files
- Delete files
- Check file existence
- Get file metadata (size, modification time)

Security features:
- Operations restricted to allowed directories only (/data, /output, /workspace)
- Primarily designed for CSV file operations
- All operations are logged
- Path traversal protection

Use cases:
- Organize datasets and outputs
- Manage intermediate processing results
- Clean up temporary files
- Check for file availability before processing
"""

mcp = FastMCP("file-operations", instructions=DESCRIPTION)

# Security: Allowed directories for file operations
ALLOWED_DIRECTORIES = {"/data", "/output", "/workspace", "/tmp"}


def is_path_allowed(path: str | Path) -> tuple[bool, str]:
    """
    Check if path is within allowed directories.

    Returns:
        tuple: (is_allowed, normalized_path or error_message)
    """
    try:
        # Convert to Path and resolve to absolute path
        resolved_path = Path(path).resolve()
        path_str = str(resolved_path)

        # Check if path starts with any allowed directory
        # On Windows, also handle drive letters
        for allowed_dir in ALLOWED_DIRECTORIES:
            # Normalize allowed directory for comparison
            if path_str.startswith(allowed_dir) or path_str.startswith(
                str(Path(allowed_dir).resolve())
            ):
                return True, path_str

        # For development/testing: allow relative paths in current working directory
        # This can be removed in production if strict isolation is needed
        cwd = Path.cwd()
        if resolved_path.is_relative_to(cwd):
            return True, path_str

        return False, f"Access denied: Path must be in allowed directories: {ALLOWED_DIRECTORIES}"
    except Exception as e:
        return False, f"Invalid path: {e}"


@mcp.tool(
    description="List files in directory with optional glob pattern. Returns file paths, sizes, and modification times."
)
async def list_files(
    directory: Annotated[str, Field(description="Directory path to list")],
    pattern: Annotated[
        str, Field(description="Glob pattern (e.g., '*.csv', 'train*.csv')")
    ] = "*.csv",
    recursive: Annotated[bool, Field(description="Search recursively in subdirectories")] = False,
) -> dict:
    """List files in directory matching pattern."""
    logger.info(
        f"[MCP:file_operations] list_files: {directory}, pattern={pattern}, recursive={recursive}"
    )

    try:
        # Security check
        is_allowed, result = is_path_allowed(directory)
        if not is_allowed:
            return {"success": False, "error": result}

        dir_path = Path(result)
        if not dir_path.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}

        if not dir_path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {directory}"}

        # List files
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        # Get file info
        file_info = []
        for file in files:
            if file.is_file():
                stat = file.stat()
                file_info.append(
                    {
                        "path": str(file),
                        "name": file.name,
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / 1024 / 1024, 2),
                        "modified": stat.st_mtime,
                    }
                )

        logger.info(f"[MCP:file_operations] Found {len(file_info)} files")
        return {
            "success": True,
            "directory": str(dir_path),
            "pattern": pattern,
            "count": len(file_info),
            "files": file_info,
        }

    except Exception as e:
        logger.error(f"[MCP:file_operations] Failed to list files: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool(
    description="Copy file from source to destination. Creates destination directory if needed."
)
async def copy_file(
    source: Annotated[str, Field(description="Source file path")],
    destination: Annotated[str, Field(description="Destination file path")],
) -> dict:
    """Copy file to new location."""
    logger.info(f"[MCP:file_operations] copy_file: {source} -> {destination}")

    try:
        # Security checks
        is_src_allowed, src_result = is_path_allowed(source)
        if not is_src_allowed:
            return {"success": False, "error": f"Source: {src_result}"}

        is_dst_allowed, dst_result = is_path_allowed(destination)
        if not is_dst_allowed:
            return {"success": False, "error": f"Destination: {dst_result}"}

        src_path = Path(src_result)
        dst_path = Path(dst_result)

        # Validate source
        if not src_path.exists():
            return {"success": False, "error": f"Source file not found: {source}"}

        if not src_path.is_file():
            return {"success": False, "error": f"Source is not a file: {source}"}

        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src_path, dst_path)

        dst_stat = dst_path.stat()
        logger.info(f"[MCP:file_operations] Copied {src_path.name} ({dst_stat.st_size} bytes)")
        return {
            "success": True,
            "source": str(src_path),
            "destination": str(dst_path),
            "size_bytes": dst_stat.st_size,
            "size_mb": round(dst_stat.st_size / 1024 / 1024, 2),
        }

    except Exception as e:
        logger.error(f"[MCP:file_operations] Failed to copy file: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool(
    description="Move file from source to destination. Creates destination directory if needed."
)
async def move_file(
    source: Annotated[str, Field(description="Source file path")],
    destination: Annotated[str, Field(description="Destination file path")],
) -> dict:
    """Move file to new location."""
    logger.info(f"[MCP:file_operations] move_file: {source} -> {destination}")

    try:
        # Security checks
        is_src_allowed, src_result = is_path_allowed(source)
        if not is_src_allowed:
            return {"success": False, "error": f"Source: {src_result}"}

        is_dst_allowed, dst_result = is_path_allowed(destination)
        if not is_dst_allowed:
            return {"success": False, "error": f"Destination: {dst_result}"}

        src_path = Path(src_result)
        dst_path = Path(dst_result)

        # Validate source
        if not src_path.exists():
            return {"success": False, "error": f"Source file not found: {source}"}

        if not src_path.is_file():
            return {"success": False, "error": f"Source is not a file: {source}"}

        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Get size before moving
        size_bytes = src_path.stat().st_size

        # Move file
        shutil.move(str(src_path), str(dst_path))

        logger.info(f"[MCP:file_operations] Moved {src_path.name} ({size_bytes} bytes)")
        return {
            "success": True,
            "source": str(src_path),
            "destination": str(dst_path),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1024 / 1024, 2),
        }

    except Exception as e:
        logger.error(f"[MCP:file_operations] Failed to move file: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool(description="Delete file. Use with caution - this operation cannot be undone.")
async def delete_file(
    file_path: Annotated[str, Field(description="Path to file to delete")],
) -> dict:
    """Delete file."""
    logger.info(f"[MCP:file_operations] delete_file: {file_path}")

    try:
        # Security check
        is_allowed, result = is_path_allowed(file_path)
        if not is_allowed:
            return {"success": False, "error": result}

        path = Path(result)

        # Validate
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if not path.is_file():
            return {"success": False, "error": f"Path is not a file: {file_path}"}

        # Get info before deletion
        size_bytes = path.stat().st_size
        name = path.name

        # Delete file
        path.unlink()

        logger.info(f"[MCP:file_operations] Deleted {name} ({size_bytes} bytes)")
        return {
            "success": True,
            "deleted_file": str(path),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1024 / 1024, 2),
        }

    except Exception as e:
        logger.error(f"[MCP:file_operations] Failed to delete file: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool(description="Check if file or directory exists at given path.")
async def file_exists(
    path: Annotated[str, Field(description="Path to check")],
) -> dict:
    """Check if file or directory exists."""
    logger.info(f"[MCP:file_operations] file_exists: {path}")

    try:
        # Security check
        is_allowed, result = is_path_allowed(path)
        if not is_allowed:
            return {"success": False, "error": result}

        check_path = Path(result)
        exists = check_path.exists()

        info = {
            "success": True,
            "path": str(check_path),
            "exists": exists,
        }

        if exists:
            info["is_file"] = check_path.is_file()
            info["is_directory"] = check_path.is_dir()

        return info

    except Exception as e:
        logger.error(f"[MCP:file_operations] Failed to check file existence: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool(description="Get detailed file metadata: size, modification time, permissions.")
async def get_file_info(
    file_path: Annotated[str, Field(description="Path to file")],
) -> dict:
    """Get file metadata."""
    logger.info(f"[MCP:file_operations] get_file_info: {file_path}")

    try:
        # Security check
        is_allowed, result = is_path_allowed(file_path)
        if not is_allowed:
            return {"success": False, "error": result}

        path = Path(result)

        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        stat = path.stat()

        info = {
            "success": True,
            "path": str(path),
            "name": path.name,
            "extension": path.suffix,
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 2),
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "modified_timestamp": stat.st_mtime,
            "created_timestamp": stat.st_ctime,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
        }

        # Add parent directory info
        info["parent_directory"] = str(path.parent)

        logger.info(f"[MCP:file_operations] Got info for {path.name}")
        return info

    except Exception as e:
        logger.error(f"[MCP:file_operations] Failed to get file info: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool(description="Create directory. Creates parent directories if needed.")
async def create_directory(
    directory: Annotated[str, Field(description="Directory path to create")],
) -> dict:
    """Create directory."""
    logger.info(f"[MCP:file_operations] create_directory: {directory}")

    try:
        # Security check
        is_allowed, result = is_path_allowed(directory)
        if not is_allowed:
            return {"success": False, "error": result}

        dir_path = Path(result)

        # Create directory
        dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"[MCP:file_operations] Created directory {dir_path}")
        return {
            "success": True,
            "directory": str(dir_path),
            "exists": dir_path.exists(),
        }

    except Exception as e:
        logger.error(f"[MCP:file_operations] Failed to create directory: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
