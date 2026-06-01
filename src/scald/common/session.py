from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from logly import logger
from platformdirs import user_log_dir

_session_dir: Optional[Path] = None


def default_session_root() -> Path:
    """OS-correct base directory for session output (replaces CWD ``scald_logs``)."""
    return Path(user_log_dir("scald"))


def init_session(
    base_dir: Optional[Path] = None,
    session_name: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Create (or return the existing) session directory.

    Idempotent unless ``force`` is set, in which case a fresh session directory
    is created and becomes the new current session.
    """
    global _session_dir

    if _session_dir is not None and not force:
        return _session_dir

    if base_dir is None:
        base_dir = default_session_root()

    if session_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_name = f"session_{timestamp}"

    _session_dir = base_dir / session_name
    _session_dir.mkdir(parents=True, exist_ok=True)
    return _session_dir


def get_session_dir() -> Path:
    """Return the current session directory, initializing a default one if needed."""
    if _session_dir is None:
        return init_session()
    return _session_dir


def reset_session() -> None:
    """Forget the current session so the next access creates a new one."""
    global _session_dir
    _session_dir = None


def get_artifact_path(filename: str) -> Path:
    """Return a path for ``filename`` inside the current session directory."""
    return get_session_dir() / filename


def save_text(content: str, filename: str) -> Path:
    """Save text content in the session directory."""
    filepath = get_artifact_path(filename)

    content_length = len(content)
    logger.debug(
        f"Attempting to save text | filename={filename} | length={content_length}"
    )

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        file_size = filepath.stat().st_size
        logger.info(
            f"Saved text artifact | path={filepath} | size_bytes={file_size} | "
            f"size_kb={file_size / 1024:.2f} | lines={content.count(chr(10)) + 1}"
        )
    except (IOError, OSError) as e:
        logger.error(
            f"Failed to save text | path={filepath} | error_type={type(e).__name__} | error={e}"
        )
        raise

    return filepath
