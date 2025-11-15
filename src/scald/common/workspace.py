import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import polars as pl

from scald.agents.actor import ActorSolution
from scald.common.logger import get_logger, get_session_dir, save_text

if TYPE_CHECKING:
    import pandas as pd

    DatasetInput = Union[str, Path, pl.DataFrame, pd.DataFrame]
else:
    DatasetInput = Union[str, Path, pl.DataFrame]

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


def prepare_datasets_for_workspace(
    train: DatasetInput,
    test: DatasetInput,
) -> tuple[Path, Path]:
    data_dir, _, _ = create_workspace_directories()

    workspace_train = _prepare_dataset(train, data_dir, "train.csv")
    workspace_test = _prepare_dataset(test, data_dir, "test.csv")

    logger.info("Prepared datasets in workspace:")
    logger.info(f"  Train: {workspace_train}")
    logger.info(f"  Test: {workspace_test}")

    return workspace_train, workspace_test


def _prepare_dataset(data: DatasetInput, data_dir: Path, default_name: str) -> Path:
    if isinstance(data, pl.DataFrame):
        dest_path = data_dir / default_name
        data.write_csv(dest_path)
        logger.debug(f"Converted Polars DataFrame to CSV: {dest_path}")
        return dest_path
    elif hasattr(data, "to_csv") and hasattr(data, "columns"):
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            dest_path = data_dir / default_name
            data.to_csv(dest_path, index=False)
            logger.debug(f"Converted Pandas DataFrame to CSV: {dest_path}")
            return dest_path

    source_path = Path(data).expanduser().resolve()
    dest_path = data_dir / source_path.name
    shutil.copy2(source_path, dest_path)
    logger.debug(f"Copied CSV file: {dest_path}")
    return dest_path


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

    # Save actor report sections
    report_text = "\n\n".join(
        [
            f"# Data Analysis\n{solution.data_analysis}",
            f"# Preprocessing\n{solution.preprocessing}",
            f"# Model Training\n{solution.model_training}",
            f"# Results\n{solution.results}",
        ]
    )
    if report_text.strip():
        report_path = save_text(report_text, "actor_report.md")
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
