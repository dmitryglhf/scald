from pathlib import Path

from .logger import get_logger

# Re-export Path for convenience
__all__ = [
    "Path",
    "resolve_csv_path",
    "find_csv_in_project",
    "validate_file_access",
    "ensure_output_dir",
    "get_project_root",
]

logger = get_logger()


def resolve_csv_path(path: Path | str, base_dir: Path | None = None) -> Path:
    """Resolve and validate CSV file path."""
    if isinstance(path, str):
        path = Path(path)

    path = path.expanduser()

    if not path.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()

    if not path.exists():
        logger.error(f"CSV file not found: {path}")
        raise FileNotFoundError(f"CSV file not found: {path}")

    if not path.is_file():
        logger.error(f"Path is not a file: {path}")
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in [".csv"]:
        logger.warning(f"File does not have .csv extension: {path}")

    if not path.stat().st_mode & 0o400:
        logger.error(f"File is not readable: {path}")
        raise PermissionError(f"File is not readable: {path}")

    logger.debug(f"Resolved CSV path: {path}")
    return path


def find_csv_in_project(filename: str, search_dirs: list[str] | None = None) -> Path | None:
    """Search for CSV file in common project locations."""
    default_dirs = [
        "data",
        "examples/data",
        "datasets",
        "examples/datasets",
        ".",
    ]

    if search_dirs:
        default_dirs.extend(search_dirs)

    base_dir = Path.cwd()

    for search_dir in default_dirs:
        potential_path = base_dir / search_dir / filename
        if potential_path.exists() and potential_path.is_file():
            logger.info(f"Found CSV file: {potential_path}")
            return potential_path.resolve()

    logger.warning(f"CSV file '{filename}' not found in project directories")
    return None


def validate_file_access(path: Path, check_read: bool = True, check_write: bool = False) -> None:
    """Validate file exists and check permissions."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    file_mode = path.stat().st_mode

    if check_read and not (file_mode & 0o400):
        raise PermissionError(f"File is not readable: {path}")

    if check_write and not (file_mode & 0o200):
        raise PermissionError(f"File is not writable: {path}")


def ensure_output_dir(path: Path) -> Path:
    """Ensure output directory exists for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get project root directory by searching for markers like .git, pyproject.toml."""
    current = Path.cwd()
    markers = [".git", "pyproject.toml", ".env", "setup.py", "README.md"]

    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                logger.debug(f"Found project root: {current}")
                return current
        current = current.parent

    logger.warning("Could not determine project root, using current directory")
    return Path.cwd()
