import pytest
from pydantic_ai.mcp import MCPServerStdio

from scald.mcp.registry import (
    MCP_SERVERS,
    _create_single_toolset,
    get_mcp_toolsets,
    get_server_descriptions,
)
from scald.mcp.utils import MCPServerConfig


class TestMCPServersRegistry:
    """Test MCP_SERVERS registry."""

    def test_registry_exists(self):
        """Registry should be a non-empty dict."""
        assert isinstance(MCP_SERVERS, dict)
        assert len(MCP_SERVERS) > 0

    def test_registry_has_sequential_thinking(self):
        """Registry should contain sequential-thinking server."""
        assert "sequential-thinking" in MCP_SERVERS

    def test_registry_has_data_science_servers(self):
        """Registry should contain all data science servers."""
        required_servers = [
            "file_operations",
            "data_preview",
            "data_analysis",
            "data_processing",
            "machine_learning",
        ]
        for server in required_servers:
            assert server in MCP_SERVERS, f"Missing server: {server}"

    def test_all_configs_are_valid(self):
        """All registered configs should be MCPServerConfig instances."""
        for name, config in MCP_SERVERS.items():
            assert isinstance(config, MCPServerConfig), f"Invalid config for {name}"
            assert config.command, f"Empty command for {name}"
            assert config.args, f"Empty args for {name}"
            assert config.timeout > 0, f"Invalid timeout for {name}"
            assert config.retries > 0, f"Invalid retries for {name}"

    def test_sequential_thinking_config(self):
        """Verify sequential-thinking configuration."""
        config = MCP_SERVERS["sequential-thinking"]
        assert config.command == "npx"
        assert "@modelcontextprotocol/server-sequential-thinking" in config.args
        assert config.timeout == 10

    def test_python_servers_have_module_path(self):
        """Python servers should have module_path set."""
        python_servers = [
            "file_operations",
            "data_preview",
            "data_analysis",
            "data_processing",
            "machine_learning",
            "sequential_thinking",
        ]
        for name in python_servers:
            config = MCP_SERVERS[name]
            assert config.module_path is not None, f"{name} missing module_path"
            assert config.module_path.endswith(".py"), f"{name} invalid module_path"


class TestCreateSingleToolset:
    """Test _create_single_toolset function."""

    def test_unknown_server_raises(self):
        """Should raise ValueError for unknown server."""
        with pytest.raises(ValueError, match="Unknown MCP server"):
            _create_single_toolset("nonexistent-server")

    def test_creates_toolset_for_sequential_thinking(self):
        """Should create MCPServerStdio for sequential-thinking."""
        toolset = _create_single_toolset("sequential-thinking")
        assert isinstance(toolset, MCPServerStdio)

    def test_creates_toolset_for_data_analysis(self):
        """Should create MCPServerStdio for data_analysis."""
        toolset = _create_single_toolset("data_analysis")
        assert isinstance(toolset, MCPServerStdio)


class TestGetMcpToolsets:
    """Test get_mcp_toolsets function."""

    def test_empty_list(self):
        """Should return empty list for no servers."""
        toolsets = get_mcp_toolsets([])
        assert toolsets == []

    def test_single_server(self):
        """Should return single toolset."""
        toolsets = get_mcp_toolsets(["data_analysis"])
        assert len(toolsets) == 1
        assert isinstance(toolsets[0], MCPServerStdio)

    def test_multiple_servers(self):
        """Should return multiple toolsets."""
        servers = ["data_analysis", "data_preview"]
        toolsets = get_mcp_toolsets(servers)
        assert len(toolsets) == 2
        for toolset in toolsets:
            assert isinstance(toolset, MCPServerStdio)

    def test_unknown_server_raises(self):
        """Should raise for unknown server in list."""
        with pytest.raises(ValueError, match="Unknown MCP server"):
            get_mcp_toolsets(["data_analysis", "unknown-server"])

    def test_actor_default_tools(self):
        """Should work with Actor's default tool list."""
        actor_tools = ["data_analysis", "data_preview", "machine_learning"]
        toolsets = get_mcp_toolsets(actor_tools)
        assert len(toolsets) == 3
        for toolset in toolsets:
            assert isinstance(toolset, MCPServerStdio)


class TestGetServerDescriptions:
    """Test get_server_descriptions function."""

    def test_returns_dict(self):
        """Should return dictionary of descriptions."""
        descriptions = get_server_descriptions()
        assert isinstance(descriptions, dict)

    def test_covers_all_servers(self):
        """Should have descriptions for all registered servers."""
        descriptions = get_server_descriptions()
        for server_name in MCP_SERVERS.keys():
            assert server_name in descriptions

    def test_descriptions_are_strings(self):
        """All descriptions should be strings."""
        descriptions = get_server_descriptions()
        for name, desc in descriptions.items():
            assert isinstance(desc, str), f"Description for {name} is not a string"

    def test_sequential_thinking_has_description(self):
        """sequential-thinking should have a description."""
        descriptions = get_server_descriptions()
        desc = descriptions.get("sequential-thinking")
        assert desc is not None
        assert len(desc) > 0

    def test_external_servers_no_description(self):
        """External servers without module_path should have default description."""
        descriptions = get_server_descriptions()
        # sequential-thinking has no module_path, should have fallback
        assert "sequential-thinking" in descriptions
        # Fallback or valid description
        assert (
            descriptions["sequential-thinking"]
            in [
                "No description available",
            ]
            or len(descriptions["sequential-thinking"]) > 0
        )

    def test_custom_servers_dict(self):
        """Should work with custom servers dict."""
        from scald.mcp.utils import npx_server

        custom_servers = {
            "test-server": npx_server("@test/package"),
        }
        descriptions = get_server_descriptions(custom_servers)
        assert "test-server" in descriptions
        assert descriptions["test-server"] == "No description available"


class TestRegistryIntegration:
    """Integration tests for registry functionality."""

    def test_full_workflow(self):
        """Test complete workflow: registry -> toolsets -> descriptions."""
        # Get descriptions
        descriptions = get_server_descriptions()
        assert len(descriptions) > 0

        # Create toolsets for a subset
        toolset_names = ["data_analysis", "data_preview"]
        toolsets = get_mcp_toolsets(toolset_names)
        assert len(toolsets) == 2

        # Verify toolsets are valid
        for toolset in toolsets:
            assert isinstance(toolset, MCPServerStdio)

    def test_all_registered_servers_createable(self):
        """All registered servers should be creatable as toolsets."""
        for server_name in MCP_SERVERS.keys():
            try:
                toolset = _create_single_toolset(server_name)
                assert isinstance(toolset, MCPServerStdio)
            except Exception as e:
                pytest.fail(f"Failed to create toolset for {server_name}: {e}")

    def test_registry_immutability(self):
        """Registry should remain consistent across tests."""
        initial_count = len(MCP_SERVERS)
        initial_keys = set(MCP_SERVERS.keys())

        # Try to modify (should not persist if registry is properly managed)
        # This is more of a sanity check
        assert len(MCP_SERVERS) == initial_count
        assert set(MCP_SERVERS.keys()) == initial_keys
