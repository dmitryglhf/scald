import tempfile
from pathlib import Path

import pytest

from scald.common.logger import get_logger, reset_logging, setup_logging
from scald.common.session import get_session_dir


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

        setup_logging(
            base_log_dir=temp_log_dir, session_name="second", force_reinit=True
        )
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

    def test_get_logger_auto_initializes(self, temp_log_dir):
        """Should auto-initialize if not initialized."""
        reset_logging()
        logger = get_logger()
        assert logger is not None
        assert get_session_dir().exists()
