import tempfile
from pathlib import Path

import pytest

from scald.common.logger import reset_logging, setup_logging
from scald.common.session import get_artifact_path, get_session_dir, save_text


@pytest.fixture(autouse=True)
def reset_session_state():
    """Reset session/logger state before and after each test."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for session artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestArtifactManagement:
    """Tests for artifact path and file saving."""

    def test_get_artifact_path(self, temp_log_dir):
        """Should return path in session directory."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        artifact_path = get_artifact_path("test.txt")
        assert artifact_path.parent == get_session_dir()
        assert artifact_path.name == "test.txt"

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

        session_dir = get_session_dir()
        session_dir.chmod(0o444)
        try:
            with pytest.raises(Exception):
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
