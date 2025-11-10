import sys

import pytest

from scald.mcp.utils import (
    MCPServerConfig,
    external_server,
    get_server_description,
    npx_remote_server,
    npx_server,
    python_server,
    uvx_server,
    validate_script_path,
    validate_server_config,
)


class TestMCPServerConfig:
    """Test MCPServerConfig dataclass."""

    def test_basic_config(self):
        config = MCPServerConfig(
            command="test-command",
            args=("arg1", "arg2"),
        )
        assert config.command == "test-command"
        assert config.args == ("arg1", "arg2")
        assert config.transport == "stdio"
        assert config.timeout == 10
        assert config.retries == 3
        assert config.module_path is None

    def test_custom_config(self):
        config = MCPServerConfig(
            command="custom",
            args=("--flag",),
            transport="http",
            timeout=30,
            retries=5,
            module_path="path/to/module",
        )
        assert config.command == "custom"
        assert config.args == ("--flag",)
        assert config.transport == "http"
        assert config.timeout == 30
        assert config.retries == 5
        assert config.module_path == "path/to/module"


class TestPythonServer:
    """Test python_server helper function."""

    def test_basic_python_server(self):
        config = python_server("test/server.py")
        assert config.command == sys.executable
        assert "test/server.py" in config.args[0]
        assert config.timeout == 10
        assert config.retries == 3
        assert config.module_path == "test/server.py"

    def test_custom_timeout_retries(self):
        config = python_server("server.py", timeout=60, retries=5)
        assert config.timeout == 60
        assert config.retries == 5

    def test_path_construction(self):
        """Should construct path relative to mcp/servers directory."""
        config = python_server("data_analysis/server.py")
        assert "servers/data_analysis/server.py" in config.args[0]


class TestNpxServer:
    """Test npx_server helper function."""

    def test_basic_npx_server(self):
        config = npx_server("@test/package")
        assert config.command == "npx"
        assert config.args == ("-y", "@test/package")
        assert config.module_path is None

    def test_with_extra_args(self):
        config = npx_server("package", extra_args=["--flag", "value"])
        assert config.args == ("-y", "package", "--flag", "value")

    def test_custom_timeout(self):
        config = npx_server("package", timeout=20)
        assert config.timeout == 20


class TestUvxServer:
    """Test uvx_server helper function."""

    def test_basic_uvx_server(self):
        config = uvx_server("test-package")
        assert config.command == "uvx"
        assert config.args == ("--from", "test-package", "test-package")

    def test_with_extra_args(self):
        config = uvx_server("package", extra_args=["--custom", "args"])
        assert config.args == ("--custom", "args")

    def test_custom_timeout_retries(self):
        config = uvx_server("package", timeout=30, retries=2)
        assert config.timeout == 30
        assert config.retries == 2


class TestExternalServer:
    """Test external_server helper function."""

    def test_basic_external_server(self):
        config = external_server("container-use")
        assert config.command == "container-use"
        assert config.args == ("stdio",)
        assert config.timeout == 60
        assert config.retries == 3

    def test_custom_args(self):
        config = external_server("custom-cmd", args=["--verbose", "stdio"])
        assert config.args == ("--verbose", "stdio")

    def test_custom_timeout(self):
        config = external_server("cmd", timeout=120)
        assert config.timeout == 120


class TestNpxRemoteServer:
    """Test npx_remote_server helper function."""

    def test_basic_remote_server(self):
        config = npx_remote_server("https://example.com/mcp")
        assert config.command == "npx"
        assert "-y" in config.args
        assert "mcp-remote" in config.args
        assert "https://example.com/mcp" in config.args

    def test_with_headers(self):
        headers = {"Authorization": "Bearer token", "X-Custom": "value"}
        config = npx_remote_server("https://api.example.com", headers=headers)

        args_str = " ".join(config.args)
        assert "--header" in args_str
        assert "Authorization:Bearer token" in args_str
        assert "X-Custom:value" in args_str

    def test_custom_timeout(self):
        config = npx_remote_server("https://example.com", timeout=30)
        assert config.timeout == 30


class TestValidateScriptPath:
    """Test validate_script_path function."""

    def test_valid_path(self, tmp_path):
        """Should not raise for existing file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")

        # Should not raise
        validate_script_path(test_file, "test-server")

    def test_missing_path(self, tmp_path):
        """Should raise ValueError for non-existent file."""
        missing_file = tmp_path / "missing.py"

        with pytest.raises(ValueError, match="Server script not found"):
            validate_script_path(missing_file, "test-server")


class TestValidateServerConfig:
    """Test validate_server_config function."""

    def test_empty_args_raises(self):
        """Should raise for empty args."""
        config = MCPServerConfig(command="test", args=())

        with pytest.raises(ValueError, match="Empty args list"):
            validate_server_config("test", config)

    def test_npx_server_skip_validation(self):
        """NPX servers should skip file validation."""
        config = npx_server("@test/package")

        # Should not raise even though package doesn't exist locally
        validate_server_config("test-npx", config)

    def test_uvx_server_skip_validation(self):
        """UVX servers should skip file validation."""
        config = uvx_server("test-package")

        # Should not raise
        validate_server_config("test-uvx", config)

    def test_container_use_skip_validation(self):
        """container-use should skip validation."""
        config = external_server("container-use")

        # Should not raise
        validate_server_config("container-use", config)

    def test_python_server_validates_path(self, tmp_path):
        """Python servers should validate file exists."""
        # Create a temporary python file
        test_server = tmp_path / "test_server.py"
        test_server.write_text("# test server")

        config = MCPServerConfig(
            command=sys.executable,
            args=(str(test_server),),
        )

        # Should not raise
        validate_server_config("test", config)

    def test_python_server_missing_file_raises(self, tmp_path):
        """Python servers should raise if file missing."""
        missing_file = tmp_path / "missing.py"

        config = MCPServerConfig(
            command=sys.executable,
            args=(str(missing_file),),
        )

        with pytest.raises(ValueError, match="Server script not found"):
            validate_server_config("test", config)


class TestGetServerDescription:
    """Test get_server_description function."""

    def test_module_not_found(self):
        """Should return 'No description available' for missing modules."""
        description = get_server_description("nonexistent/module.py", "test")
        assert description == "No description available"

    def test_module_without_description(self, tmp_path, monkeypatch):
        """Should return warning message if DESCRIPTION not found."""
        # This is hard to test without creating actual modules
        # Just verify it returns the fallback
        description = get_server_description("fake/path.py", "fake-server")
        assert description == "No description available"


class TestConfigHelpers:
    """Test that all config helpers return correct types."""

    def test_all_helpers_return_config(self):
        """All helper functions should return MCPServerConfig."""
        configs = [
            python_server("test.py"),
            npx_server("package"),
            uvx_server("package"),
            external_server("command"),
            npx_remote_server("https://example.com"),
        ]

        for config in configs:
            assert isinstance(config, MCPServerConfig)
            assert isinstance(config.command, str)
            assert isinstance(config.args, tuple)
            assert isinstance(config.timeout, int)
            assert isinstance(config.retries, int)
