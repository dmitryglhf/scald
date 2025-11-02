import json
import tempfile
from pathlib import Path

import pytest

from scald.agents.actor import ActorSolution
from scald.common.logger import get_session_dir, reset_logging, setup_logging
from scald.common.workspace import (
    ACTOR_WORKSPACE,
    cleanup_workspace,
    copy_datasets_to_workspace,
    create_workspace_directories,
    get_workspace_path,
    save_workspace_artifacts,
)


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset logger state before and after each test."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture(autouse=True)
def clean_workspace():
    """Ensure workspace is clean before and after each test."""
    cleanup_workspace()
    yield
    cleanup_workspace()


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


class TestWorkspaceDirectories:
    """Tests for workspace directory management."""

    def test_get_workspace_path(self):
        """Should return correct workspace path."""
        path = get_workspace_path()
        assert path == Path.home() / ".scald" / "actor"
        assert path == ACTOR_WORKSPACE

    def test_create_workspace_directories(self):
        """Should create all required directories."""
        data_dir, output_dir, workspace_dir = create_workspace_directories()

        assert data_dir.exists()
        assert output_dir.exists()
        assert workspace_dir.exists()

        assert data_dir == ACTOR_WORKSPACE / "data"
        assert output_dir == ACTOR_WORKSPACE / "output"
        assert workspace_dir == ACTOR_WORKSPACE / "workspace"

    def test_create_workspace_directories_idempotent(self):
        """Should be safe to call multiple times."""
        create_workspace_directories()
        data_dir, output_dir, workspace_dir = create_workspace_directories()

        assert data_dir.exists()
        assert output_dir.exists()
        assert workspace_dir.exists()

    def test_cleanup_workspace(self):
        """Should remove workspace directory."""
        create_workspace_directories()
        assert ACTOR_WORKSPACE.exists()

        cleanup_workspace()
        assert not ACTOR_WORKSPACE.exists()

    def test_cleanup_workspace_when_not_exists(self):
        """Should not raise error if workspace doesn't exist."""
        cleanup_workspace()
        cleanup_workspace()


class TestDatasetCopying:
    """Tests for copying datasets to workspace."""

    def test_copy_datasets_to_workspace(self, sample_csv_files):
        """Should copy both train and test files."""
        train_csv, test_csv = sample_csv_files

        workspace_train, workspace_test = copy_datasets_to_workspace(train_csv, test_csv)

        assert workspace_train.exists()
        assert workspace_test.exists()
        assert workspace_train.parent == ACTOR_WORKSPACE / "data"
        assert workspace_test.parent == ACTOR_WORKSPACE / "data"

        assert workspace_train.name == train_csv.name
        assert workspace_test.name == test_csv.name

    def test_copy_datasets_preserves_content(self, sample_csv_files):
        """Should preserve file content when copying."""
        train_csv, test_csv = sample_csv_files
        original_train_content = train_csv.read_text()
        original_test_content = test_csv.read_text()

        workspace_train, workspace_test = copy_datasets_to_workspace(train_csv, test_csv)

        assert workspace_train.read_text() == original_train_content
        assert workspace_test.read_text() == original_test_content

    def test_copy_datasets_creates_directories(self, sample_csv_files):
        """Should create workspace directories if they don't exist."""
        cleanup_workspace()
        train_csv, test_csv = sample_csv_files

        workspace_train, workspace_test = copy_datasets_to_workspace(train_csv, test_csv)

        assert workspace_train.exists()
        assert workspace_test.exists()


class TestArtifactSaving:
    """Tests for saving workspace artifacts."""

    def test_save_workspace_artifacts_with_predictions_path(self, temp_log_dir):
        """Should save predictions CSV to session directory."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        create_workspace_directories()
        predictions_file = ACTOR_WORKSPACE / "output" / "predictions.csv"
        predictions_file.write_text("prediction\n1\n2\n3\n")

        solution = ActorSolution(
            predictions_path=predictions_file,
            predictions=[1, 2, 3],
            metrics={"accuracy": 0.95},
            report="Test report",
        )

        saved_path = save_workspace_artifacts(solution)

        assert saved_path is not None
        assert saved_path.exists()
        assert saved_path.parent == get_session_dir()
        assert saved_path.name == "predictions.csv"

    def test_save_workspace_artifacts_saves_report(self, temp_log_dir):
        """Should save actor report to session directory."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        solution = ActorSolution(
            predictions=[1, 2, 3],
            metrics={},
            report="Detailed analysis report with multiple lines.",
        )

        save_workspace_artifacts(solution)

        report_path = get_session_dir() / "actor_report.md"
        assert report_path.exists()
        assert "Detailed analysis" in report_path.read_text()

    def test_save_workspace_artifacts_saves_metrics(self, temp_log_dir):
        """Should save metrics as JSON."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        solution = ActorSolution(
            predictions=[1, 2, 3],
            metrics={"accuracy": 0.95, "f1": 0.93, "precision": 0.94},
            report="Test",
        )

        save_workspace_artifacts(solution)

        metrics_path = get_session_dir() / "metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert metrics["accuracy"] == 0.95
        assert metrics["f1"] == 0.93
        assert metrics["precision"] == 0.94

    def test_save_workspace_artifacts_finds_csv_in_output(self, temp_log_dir):
        """Should find CSV files in output directory."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        create_workspace_directories()
        predictions_file = ACTOR_WORKSPACE / "output" / "results.csv"
        predictions_file.write_text("prediction\n0\n1\n")

        solution = ActorSolution(
            predictions=[0, 1],
            metrics={},
            report="",
        )

        saved_path = save_workspace_artifacts(solution)

        assert saved_path is not None
        assert saved_path.exists()
        assert saved_path.name == "results.csv"

    def test_save_workspace_artifacts_without_predictions(self, temp_log_dir):
        """Should handle solution without predictions file."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        solution = ActorSolution(
            predictions=[1, 2, 3],
            metrics={"accuracy": 0.95},
            report="Report without predictions file",
        )

        saved_path = save_workspace_artifacts(solution)

        assert saved_path is None
        report_path = get_session_dir() / "actor_report.md"
        assert report_path.exists()

    def test_save_workspace_artifacts_empty_solution(self, temp_log_dir):
        """Should handle empty solution gracefully."""
        setup_logging(base_log_dir=temp_log_dir, session_name="test")

        solution = ActorSolution(
            predictions=[],
            metrics={},
            report="",
        )

        saved_path = save_workspace_artifacts(solution)
        assert saved_path is None
