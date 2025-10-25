import os

from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerStdio

from ..common.logger import get_logger
from .utils import (
    MCPServerConfig,
    create_mcp_server_stdio,
    get_server_description,
    npx_remote_server,
    npx_server,
    python_server,
    uvx_server,
    validate_server_config,
)

load_dotenv()

logger = get_logger()

# MCP Servers Registry
# Add new servers using helper functions: python_server(), npx_server(), uvx_server(), npx_remote_server()
MCP_SERVERS: dict[str, MCPServerConfig] = {
    # Web Search & Research
    "browseruse-search": python_server("web_search/browseruse_search.py", timeout=10),
    "tavily-search": npx_remote_server(
        f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY', '')}",
        timeout=15,
        module_path="web_search/tavily_search.py",
    ),
    # "tavily-search": python_server("web_search/tavily_search.py", timeout=10),  # Old local implementation
    "duckduckgo-search": uvx_server("duckduckgo-mcp-server", timeout=15),
    # "duckduckgo-search": python_server("web_search/duckduckgo_search.py", timeout=10),  # Old local implementation
    "jina-search": npx_remote_server(
        "https://mcp.jina.ai/sse",
        timeout=15,
        headers={"Authorization": f"Bearer {os.getenv('JINA_API_KEY', '')}"},
        module_path="web_search/jina_search.py",
    ),
    "wayback-search": python_server("web_search/wayback_search.py", timeout=30),
    # Code & Repository Analysis
    "github-analysis": npx_remote_server("https://gitmcp.io/docs", timeout=30),
    # File Operations
    "download-server": python_server("download/url_download.py", timeout=30),
    "document-server": python_server("document/server.py", timeout=15),
    # Media Processing
    "audio-server": python_server("audio/server.py", timeout=60),
    "image-server": python_server("image/server.py", timeout=60),
    "youtube-transcript-server": python_server("video/yt_transcript_server.py", timeout=60),
    # Development & Execution
    "e2b-sandbox": python_server("sandbox/e2b_sandbox.py", timeout=30),
    # Reasoning & Analysis
    "sequential-thinking": npx_server(
        "@modelcontextprotocol/server-sequential-thinking", timeout=10
    ),
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
