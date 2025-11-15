import shutil
import time
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
    logger.debug(f"Creating workspace directories | root={ACTOR_WORKSPACE}")

    data_dir = ACTOR_WORKSPACE / "data"
    output_dir = ACTOR_WORKSPACE / "output"
    workspace_dir = ACTOR_WORKSPACE / "workspace"

    try:
        for directory in [data_dir, output_dir, workspace_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory | path={directory}")

        logger.info(
            f"Workspace directories ready | data={data_dir} | output={output_dir} | workspace={workspace_dir}"
        )
    except (OSError, PermissionError) as e:
        logger.error(
            f"Failed to create workspace directories | root={ACTOR_WORKSPACE} | "
            f"error_type={type(e).__name__} | error={e}"
        )
        raise

    return data_dir, output_dir, workspace_dir


def prepare_datasets_for_workspace(
    train: DatasetInput,
    test: DatasetInput,
) -> tuple[Path, Path]:
    logger.debug(
        f"Preparing datasets for workspace | train_type={type(train).__name__} | "
        f"test_type={type(test).__name__}"
    )

    prep_start = time.time()
    data_dir, _, _ = create_workspace_directories()

    workspace_train = _prepare_dataset(train, data_dir, "train.csv")
    workspace_test = _prepare_dataset(test, data_dir, "test.csv")

    prep_duration = time.time() - prep_start

    train_size = workspace_train.stat().st_size
    test_size = workspace_test.stat().st_size

    logger.info(
        f"Prepared datasets in workspace | train={workspace_train} | test={workspace_test} | "
        f"train_size_kb={train_size / 1024:.2f} | test_size_kb={test_size / 1024:.2f} | "
        f"duration_sec={prep_duration:.3f}"
    )

    return workspace_train, workspace_test


def _prepare_dataset(data: DatasetInput, data_dir: Path, default_name: str) -> Path:
    logger.debug(f"Preparing dataset | type={type(data).__name__} | default_name={default_name}")

    prep_start = time.time()

    if isinstance(data, pl.DataFrame):
        dest_path = data_dir / default_name
        try:
            data.write_csv(dest_path)
            prep_duration = time.time() - prep_start
            file_size = dest_path.stat().st_size
            logger.debug(
                f"Converted Polars DataFrame to CSV | path={dest_path} | "
                f"shape={data.shape} | size_kb={file_size / 1024:.2f} | duration_sec={prep_duration:.3f}"
            )
            return dest_path
        except (IOError, OSError) as e:
            logger.error(
                f"Failed to write Polars DataFrame | path={dest_path} | "
                f"error_type={type(e).__name__} | error={e}"
            )
            raise
    elif hasattr(data, "to_csv") and hasattr(data, "columns"):
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            dest_path = data_dir / default_name
            try:
                data.to_csv(dest_path, index=False)
                prep_duration = time.time() - prep_start
                file_size = dest_path.stat().st_size
                logger.debug(
                    f"Converted Pandas DataFrame to CSV | path={dest_path} | "
                    f"shape={data.shape} | size_kb={file_size / 1024:.2f} | duration_sec={prep_duration:.3f}"
                )
                return dest_path
            except (IOError, OSError) as e:
                logger.error(
                    f"Failed to write Pandas DataFrame | path={dest_path} | "
                    f"error_type={type(e).__name__} | error={e}"
                )
                raise

    try:
        source_path = Path(data).expanduser().resolve()
        if not source_path.exists():
            logger.error(f"Dataset file not found | path={source_path}")
            raise FileNotFoundError(f"Dataset file not found: {source_path}")

        dest_path = data_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        prep_duration = time.time() - prep_start
        file_size = dest_path.stat().st_size
        logger.debug(
            f"Copied CSV file | source={source_path} | dest={dest_path} | "
            f"size_kb={file_size / 1024:.2f} | duration_sec={prep_duration:.3f}"
        )
        return dest_path
    except (IOError, OSError, shutil.Error) as e:
        logger.error(
            f"Failed to copy dataset file | source={source_path} | dest={dest_path} | "
            f"error_type={type(e).__name__} | error={e}"
        )
        raise


def save_workspace_artifacts(solution: ActorSolution) -> Optional[Path]:
    """Save workspace artifacts to session log directory."""
    logger.debug("Saving workspace artifacts to session directory")

    save_start = time.time()
    session_dir = get_session_dir()
    output_dir = ACTOR_WORKSPACE / "output"

    saved_predictions_path = None
    artifacts_saved = 0

    # Save predictions CSV
    try:
        if solution.predictions_path and solution.predictions_path.exists():
            predictions_filename = solution.predictions_path.name
            dest_path = session_dir / predictions_filename
            shutil.copy2(solution.predictions_path, dest_path)
            saved_predictions_path = dest_path
            file_size = dest_path.stat().st_size
            artifacts_saved += 1
            logger.info(f"Saved predictions | path={dest_path} | size_kb={file_size / 1024:.2f}")
        elif output_dir.exists():
            for csv_file in output_dir.glob("*.csv"):
                dest_path = session_dir / csv_file.name
                shutil.copy2(csv_file, dest_path)
                saved_predictions_path = dest_path
                file_size = dest_path.stat().st_size
                artifacts_saved += 1
                logger.info(
                    f"Saved output CSV | filename={csv_file.name} | path={dest_path} | "
                    f"size_kb={file_size / 1024:.2f}"
                )
    except (IOError, OSError, shutil.Error) as e:
        logger.error(f"Failed to save predictions | error_type={type(e).__name__} | error={e}")

    # Save actor report sections
    try:
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
            report_size = report_path.stat().st_size
            artifacts_saved += 1
            logger.info(
                f"Saved actor report | path={report_path} | size_kb={report_size / 1024:.2f}"
            )
    except Exception as e:
        logger.error(f"Failed to save actor report | error_type={type(e).__name__} | error={e}")

    save_duration = time.time() - save_start
    logger.info(
        f"Workspace artifacts saved | count={artifacts_saved} | "
        f"session_dir={session_dir} | duration_sec={save_duration:.3f}"
    )

    return saved_predictions_path


def cleanup_workspace():
    """Clean up workspace directory."""
    if ACTOR_WORKSPACE.exists():
        logger.debug(f"Cleaning up workspace | path={ACTOR_WORKSPACE}")
        cleanup_start = time.time()
        try:
            shutil.rmtree(ACTOR_WORKSPACE)
            cleanup_duration = time.time() - cleanup_start
            logger.info(
                f"Cleaned up workspace | path={ACTOR_WORKSPACE} | duration_sec={cleanup_duration:.3f}"
            )
        except (OSError, PermissionError) as e:
            logger.error(
                f"Failed to cleanup workspace | path={ACTOR_WORKSPACE} | "
                f"error_type={type(e).__name__} | error={e}"
            )
            raise
    else:
        logger.debug(f"Workspace does not exist, skipping cleanup | path={ACTOR_WORKSPACE}")


def get_workspace_path() -> Path:
    """Get the actor workspace root directory path."""
    return ACTOR_WORKSPACE
