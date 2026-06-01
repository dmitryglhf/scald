import tempfile
from pathlib import Path

import pytest

from scald.agents.actor import ActorSolution
from scald.common.logger import reset_logging, setup_logging
from scald.common.session import get_session_dir
from scald.common.workspace import (
    WORKSPACE_ENV_VAR,
    cleanup_workspace,
    create_workspace_directories,
    default_workspace_root,
    prepare_datasets_for_workspace,
    resolve_actor_workspace,
    save_workspace_artifacts,
)


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset logger state before and after each test."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def workspace(tmp_path):
    """A hermetic per-run workspace under pytest's tmp_path."""
    ws = tmp_path / "run_workspace"
    yield ws
    cleanup_workspace(ws)


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_files(tmp_path):
    """Create sample CSV files for testing."""
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    train_csv.write_text("feature1,feature2,target\n1,2,0\n3,4,1\n")
    test_csv.write_text("feature1,feature2\n5,6\n7,8\n")

    return train_csv, test_csv


class TestWorkspaceResolution:
    """Tests for workspace root resolution."""

    def test_default_workspace_root_is_under_cache(self):
        root = default_workspace_root()
        assert isinstance(root, Path)
        assert "scald" in str(root).lower()

    def test_resolve_uses_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path / "from_env"))
        assert resolve_actor_workspace() == tmp_path / "from_env"

    def test_resolve_falls_back_without_env(self, monkeypatch):
        monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
        assert resolve_actor_workspace() == default_workspace_root() / "actor"


class TestWorkspaceDirectories:
    """Tests for workspace directory management."""

    def test_create_workspace_directories(self, workspace):
        data_dir, output_dir, workspace_dir = create_workspace_directories(workspace)

        assert data_dir.exists()
        assert output_dir.exists()
        assert workspace_dir.exists()

        assert data_dir == workspace / "data"
        assert output_dir == workspace / "output"
        assert workspace_dir == workspace / "workspace"

    def test_create_workspace_directories_idempotent(self, workspace):
        create_workspace_directories(workspace)
        data_dir, output_dir, workspace_dir = create_workspace_directories(workspace)

        assert data_dir.exists()
        assert output_dir.exists()
        assert workspace_dir.exists()

    def test_cleanup_workspace(self, workspace):
        create_workspace_directories(workspace)
        assert workspace.exists()

        cleanup_workspace(workspace)
        assert not workspace.exists()

    def test_cleanup_workspace_when_not_exists(self, workspace):
        cleanup_workspace(workspace)
        cleanup_workspace(workspace)


class TestDatasetCopying:
    def test_prepare_datasets_to_workspace(self, sample_csv_files, workspace):
        train_csv, test_csv = sample_csv_files

        workspace_train, workspace_test = prepare_datasets_for_workspace(
            train_csv, test_csv, workspace
        )

        assert workspace_train.exists()
        assert workspace_test.exists()
        assert workspace_train.parent == workspace / "data"
        assert workspace_test.parent == workspace / "data"

        assert workspace_train.name == train_csv.name
        assert workspace_test.name == test_csv.name

    def test_prepare_datasets_preserves_content(self, sample_csv_files, workspace):
        train_csv, test_csv = sample_csv_files
        original_train_content = train_csv.read_text()
        original_test_content = test_csv.read_text()

        workspace_train, workspace_test = prepare_datasets_for_workspace(
            train_csv, test_csv, workspace
        )

        assert workspace_train.read_text() == original_train_content
        assert workspace_test.read_text() == original_test_content

    def test_prepare_datasets_creates_directories(self, sample_csv_files, workspace):
        train_csv, test_csv = sample_csv_files

        workspace_train, workspace_test = prepare_datasets_for_workspace(
            train_csv, test_csv, workspace
        )

        assert workspace_train.exists()
        assert workspace_test.exists()


class TestArtifactSaving:
    """Tests for saving workspace artifacts."""

    def test_save_workspace_artifacts_with_predictions_path(
        self, temp_log_dir, workspace
    ):
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        create_workspace_directories(workspace)
        predictions_file = workspace / "output" / "predictions.csv"
        predictions_file.write_text("prediction\n1\n2\n3\n")

        solution = ActorSolution(
            predictions_path=predictions_file,
            data_analysis="Test data analysis",
            preprocessing="Test preprocessing",
            model_training="Test model",
            results="Test results",
        )

        saved_path = save_workspace_artifacts(solution, workspace)

        assert saved_path is not None
        assert saved_path.exists()
        assert saved_path.parent == get_session_dir()
        assert saved_path.name == "predictions.csv"

    def test_save_workspace_artifacts_saves_report(self, temp_log_dir, workspace):
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        create_workspace_directories(workspace)
        predictions_file = workspace / "output" / "predictions.csv"
        predictions_file.write_text("prediction\n1\n2\n3\n")

        solution = ActorSolution(
            predictions_path=predictions_file,
            data_analysis="Detailed analysis",
            preprocessing="Multiple preprocessing steps",
            model_training="Model details",
            results="Results summary",
        )

        save_workspace_artifacts(solution, workspace)

        report_path = get_session_dir() / "actor_report.md"
        assert report_path.exists()
        assert "Detailed analysis" in report_path.read_text()

    def test_save_workspace_artifacts_no_metrics_file(self, temp_log_dir, workspace):
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        create_workspace_directories(workspace)
        predictions_file = workspace / "output" / "predictions.csv"
        predictions_file.write_text("prediction\n1\n2\n3\n")

        solution = ActorSolution(predictions_path=predictions_file)

        save_workspace_artifacts(solution, workspace)

        metrics_path = get_session_dir() / "metrics.json"
        assert not metrics_path.exists()

    def test_save_workspace_artifacts_normalizes_predictions_name(
        self, temp_log_dir, workspace
    ):
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        create_workspace_directories(workspace)
        # Source can be named anything; it is copied to the canonical predictions.csv.
        predictions_file = workspace / "output" / "results.csv"
        predictions_file.write_text("prediction\n0\n1\n")

        solution = ActorSolution(predictions_path=predictions_file)

        saved_path = save_workspace_artifacts(solution, workspace)

        assert saved_path is not None
        assert saved_path.exists()
        assert saved_path.name == "predictions.csv"

    def test_save_workspace_artifacts_without_section_text(
        self, temp_log_dir, workspace
    ):
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        create_workspace_directories(workspace)
        predictions_file = workspace / "output" / "predictions.csv"
        predictions_file.write_text("prediction\n1\n2\n3\n")

        solution = ActorSolution(
            predictions_path=predictions_file,
            data_analysis="Data analysis",
        )

        saved_path = save_workspace_artifacts(solution, workspace)

        assert saved_path is not None
        report_path = get_session_dir() / "actor_report.md"
        assert report_path.exists()

    def test_save_workspace_artifacts_empty_solution(self, temp_log_dir, workspace):
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        create_workspace_directories(workspace)
        predictions_file = workspace / "output" / "predictions.csv"
        predictions_file.write_text("prediction\n")

        solution = ActorSolution(predictions_path=predictions_file)

        saved_path = save_workspace_artifacts(solution, workspace)
        assert saved_path is not None
