import shutil
from pathlib import Path
from typing import Optional

from scald.agents.actor import ActorSolution
from scald.common.logger import get_logger, get_session_dir, save_text

logger = get_logger()

ACTOR_WORKSPACE = Path.home() / ".scald" / "actor"


def create_workspace_directories() -> tuple[Path, Path, Path]:
    """Create isolated workspace directories."""
    data_dir = ACTOR_WORKSPACE / "data"
    output_dir = ACTOR_WORKSPACE / "output"
    workspace_dir = ACTOR_WORKSPACE / "workspace"

    for directory in [data_dir, output_dir, workspace_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

    return data_dir, output_dir, workspace_dir


def copy_datasets_to_workspace(train_path: Path, test_path: Path) -> tuple[Path, Path]:
    """Copy user datasets to workspace data directory."""
    data_dir, _, _ = create_workspace_directories()

    workspace_train = data_dir / train_path.name
    workspace_test = data_dir / test_path.name

    shutil.copy2(train_path, workspace_train)
    shutil.copy2(test_path, workspace_test)

    logger.info("Copied datasets to workspace:")
    logger.info(f"  Train: {workspace_train}")
    logger.info(f"  Test: {workspace_test}")

    return workspace_train, workspace_test


def save_workspace_artifacts(solution: ActorSolution) -> Optional[Path]:
    """Save workspace artifacts to session log directory."""
    session_dir = get_session_dir()
    output_dir = ACTOR_WORKSPACE / "output"

    saved_predictions_path = None

    # Save predictions CSV
    if solution.predictions_path and solution.predictions_path.exists():
        predictions_filename = solution.predictions_path.name
        dest_path = session_dir / predictions_filename
        shutil.copy2(solution.predictions_path, dest_path)
        saved_predictions_path = dest_path
        logger.info(f"Saved predictions to: {dest_path}")
    elif output_dir.exists():
        for csv_file in output_dir.glob("*.csv"):
            dest_path = session_dir / csv_file.name
            shutil.copy2(csv_file, dest_path)
            saved_predictions_path = dest_path
            logger.info(f"Saved {csv_file.name} to: {dest_path}")

    # Save actor report
    if solution.report:
        report_path = save_text(solution.report, "actor_report.md")
        logger.info(f"Saved actor report to: {report_path}")

    return saved_predictions_path


def cleanup_workspace():
    """Clean up workspace directory."""
    if ACTOR_WORKSPACE.exists():
        shutil.rmtree(ACTOR_WORKSPACE)
        logger.info(f"Cleaned up workspace: {ACTOR_WORKSPACE}")


def get_workspace_path() -> Path:
    """Get the actor workspace root directory path."""
    return ACTOR_WORKSPACE
