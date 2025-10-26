from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerStdio

from ..common.logger import get_logger
from .utils import (
    MCPServerConfig,
    create_mcp_server_stdio,
    get_server_description,
    npx_server,
    python_server,
    validate_server_config,
)

load_dotenv()

logger = get_logger()

# MCP Servers Registry
# Add new servers using helper functions:
# python_server(), npx_server(), uvx_server(), npx_remote_server(), external_server()
MCP_SERVERS: dict[str, MCPServerConfig] = {
    # Reasoning & Analysis
    "sequential-thinking": npx_server(
        "@modelcontextprotocol/server-sequential-thinking", timeout=10
    ),
    # System Tools
    "file_operations": python_server("file_operations/server.py", timeout=30, retries=3),
    # Data Science Tools
    "data_preview": python_server("data_preview/server.py", timeout=30, retries=3),
    "data_analysis": python_server("data_analysis/server.py", timeout=30, retries=3),
    "data_processing": python_server("data_processing/server.py", timeout=30, retries=3),
    "machine_learning": python_server("machine_learning/server.py", timeout=60, retries=3),
}


def _create_single_toolset(name: str) -> MCPServerStdio:
    """Create a single MCP toolset from server name."""
    if name not in MCP_SERVERS:
        logger.error(f"Unknown MCP server requested: {name}")
        raise ValueError(f"Unknown MCP server: {name}")

    config = MCP_SERVERS[name]

    try:
        validate_server_config(name, config)
        return create_mcp_server_stdio(name, config)
    except ValueError as e:
        logger.error(f"Failed to create server '{name}': {e}")
        raise


def get_mcp_toolsets(tool_names: list[str]) -> list[MCPServerStdio]:
    """Create MCP toolsets from server names."""
    return [_create_single_toolset(name) for name in tool_names]


def get_server_descriptions(servers: dict[str, MCPServerConfig] | None = None) -> dict[str, str]:
    """Get server descriptions by importing DESCRIPTION from each server module."""
    if servers is None:
        servers = MCP_SERVERS

    descriptions = {}
    for name, config in servers.items():
        if config.module_path:
            descriptions[name] = get_server_description(config.module_path, name)
        else:
            descriptions[name] = "No description available"

    return descriptions
