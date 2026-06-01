from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from logly import logger

from scald.common.session import init_session, reset_session

# Global state
_initialized = False


def setup_logging(
    base_log_dir: Optional[Path] = None,
    session_name: Optional[str] = None,
    enable_file: bool = True,
    force_reinit: bool = False,
) -> None:
    global _initialized

    if _initialized and not force_reinit:
        return

    session_dir = init_session(base_log_dir, session_name, force=force_reinit)

    if enable_file:
        logger.add(
            str(session_dir / "scald.log"),
            size_limit="10MB",
            retention=5,
        )

    _initialized = True


def reset_logging() -> None:
    global _initialized
    _initialized = False
    reset_session()


def get_logger() -> Any:
    if not _initialized:
        setup_logging()
    return logger
