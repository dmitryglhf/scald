import json
import tempfile
from pathlib import Path

import pytest

from scald.common.logger import (
    get_artifact_path,
    get_logger,
    get_session_dir,
    reset_logging,
    save_json,
    save_text,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset logger state before and after each test."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestLoggingSetup:
    """Tests for logging setup and configuration."""

    def test_setup_logging_creates_session_dir(self, temp_log_dir):
        """Should create session directory."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test_session")

        session_dir = get_session_dir()
        assert session_dir.exists()
        assert session_dir.is_dir()
        assert "test_session" in str(session_dir)

    def test_setup_logging_with_auto_session_name(self, temp_log_dir):
        """Should create session with timestamp when name not provided."""
        setup_logging(base_log_dir=temp_log_dir)

        session_dir = get_session_dir()
        assert session_dir.exists()
        assert "session_" in session_dir.name

    def test_setup_logging_creates_log_file(self, temp_log_dir):
        """Should create log file when file logging enabled."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test", enable_file=True)

        log_file = get_session_dir() / "scald.log"
        # Log file might not exist until first log message
        logger = get_logger()
        logger.info("Test message")
        assert log_file.exists()

    def test_setup_logging_no_reinit_by_default(self, temp_log_dir):
        """Should not reinitialize if already initialized."""
        setup_logging(base_log_dir=temp_log_dir, session_name="first")
        first_dir = get_session_dir()

        setup_logging(base_log_dir=temp_log_dir, session_name="second")
        second_dir = get_session_dir()

        assert first_dir == second_dir

    def test_setup_logging_force_reinit(self, temp_log_dir):
        """Should reinitialize when force_reinit=True."""
        setup_logging(base_log_dir=temp_log_dir, session_name="first")
        first_dir = get_session_dir()

        setup_logging(base_log_dir=temp_log_dir, session_name="second", force_reinit=True)
        second_dir = get_session_dir()

        assert first_dir != second_dir
        assert "second" in str(second_dir)

    def test_reset_logging(self, temp_log_dir):
        """Should reset logging state."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")
        reset_logging()

        # After reset, setup should work again
        setup_logging(base_log_dir=temp_log_dir, session_name="new")
        session_dir = get_session_dir()
        assert "new" in str(session_dir)


class TestLoggerAccess:
    """Tests for getting logger instances."""

    def test_get_logger_returns_logger(self):
        """Should return a logger instance."""
        logger = get_logger()
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")

    def test_get_logger_with_name(self):
        """Should accept name parameter for API compatibility."""
        logger = get_logger("test_module")
        assert logger is not None

    def test_get_logger_auto_initializes(self, temp_log_dir):
        """Should auto-initialize if not initialized."""
        reset_logging()
        logger = get_logger()
        assert logger is not None
        assert get_session_dir().exists()


class TestArtifactManagement:
    """Tests for artifact path and file saving."""

    def test_get_artifact_path(self, temp_log_dir):
        """Should return path in session directory."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        artifact_path = get_artifact_path("test.txt")
        assert artifact_path.parent == get_session_dir()
        assert artifact_path.name == "test.txt"

    def test_save_json_creates_file(self, temp_log_dir):
        """Should save JSON data to file."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        data = {"key": "value", "number": 42}
        filepath = save_json(data, "test_data.json")

        assert filepath.exists()
        assert filepath.suffix == ".json"

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_json_adds_extension(self, temp_log_dir):
        """Should add .json extension if missing."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        filepath = save_json({"test": "data"}, "no_extension")
        assert filepath.suffix == ".json"
        assert filepath.exists()

    def test_save_json_handles_datetime(self, temp_log_dir):
        """Should serialize datetime objects."""
        from datetime import datetime

        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        data = {"timestamp": datetime.now()}
        filepath = save_json(data, "datetime_test.json")

        assert filepath.exists()
        with open(filepath) as f:
            loaded = json.load(f)
        assert "timestamp" in loaded

    def test_save_json_error_handling(self, temp_log_dir):
        """Should raise error on save failure."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        # Make directory read-only to trigger save error
        session_dir = get_session_dir()
        session_dir.chmod(0o444)

        try:
            with pytest.raises(Exception):
                save_json({"test": "data"}, "test.json")
        finally:
            session_dir.chmod(0o755)

    def test_save_text_creates_file(self, temp_log_dir):
        """Should save text content to file."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        content = "Hello, World!\nThis is a test."
        filepath = save_text(content, "test.txt")

        assert filepath.exists()
        with open(filepath) as f:
            loaded = f.read()
        assert loaded == content

    def test_save_text_error_handling(self, temp_log_dir):
        """Should raise error on save failure."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        # Try to save to invalid path
        with pytest.raises(Exception):
            # Create read-only directory to trigger error
            session_dir = get_session_dir()
            session_dir.chmod(0o444)
            try:
                save_text("content", "test.txt")
            finally:
                session_dir.chmod(0o755)


class TestSessionDirectory:
    """Tests for session directory management."""

    def test_get_session_dir_auto_initializes(self, temp_log_dir):
        """Should auto-initialize if not initialized."""
        reset_logging()
        session_dir = get_session_dir()
        assert session_dir.exists()

    def test_get_session_dir_returns_path(self, temp_log_dir):
        """Should return Path object."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")
        session_dir = get_session_dir()
        assert isinstance(session_dir, Path)
        assert session_dir.exists()
