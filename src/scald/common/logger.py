import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Global state
_session_dir: Optional[Path] = None
_initialized = False


def setup_logging(
    base_log_dir: Path = Path("scald_logs"),
    session_name: Optional[str] = None,
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    force_reinit: bool = False,
) -> None:
    global _session_dir, _initialized

    if _initialized and not force_reinit:
        return

    if enable_file:
        if session_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_name = f"session_{timestamp}"

        _session_dir = base_log_dir / session_name
        _session_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default handler

    if enable_console:
        logger.add(
            sys.stderr,
            format="<dim>{time:YYYY-MM-DD HH:mm:ss}</dim> | <level>{level: <8}</level> | <blue>{name}</blue>:<cyan>{function}</cyan>:<magenta>{line}</magenta> - {message}",
            level=log_level,
            colorize=True,
        )

    if enable_file:
        logger.add(
            _session_dir / "scald.log",  # type: ignore
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention=5,
            compression="zip",
            encoding="utf-8",
        )

    _initialized = True


def reset_logging() -> None:
    global _session_dir, _initialized

    logger.remove()
    _session_dir = None
    _initialized = False


def get_logger(enable_file: bool = True) -> Any:
    if not _initialized:
        setup_logging(enable_file=enable_file)
    return logger


def get_session_dir() -> Path:
    assert _session_dir is not None
    return _session_dir


def get_artifact_path(filename: str) -> Path:
    return get_session_dir() / filename


def save_json(data: Any, filename: str) -> Path:
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = get_artifact_path(filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise

    return filepath


def save_text(content: str, filename: str) -> Path:
    filepath = get_artifact_path(filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved text to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save text to {filepath}: {e}")
        raise

    return filepath
