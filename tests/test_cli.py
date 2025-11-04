import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scald.cli import main


@pytest.fixture
def sample_csv_files(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    train_csv.write_text("feature1,feature2,target\n1,2,0\n3,4,1\n")
    test_csv.write_text("feature1,feature2,target\n5,6,0\n7,8,1\n")

    return train_csv, test_csv


class TestCLIArguments:
    def test_missing_required_arguments(self):
        with patch.object(sys, "argv", ["scald"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_missing_train_argument(self):
        with patch.object(
            sys,
            "argv",
            ["scald", "--test", "test.csv", "--target", "y", "--task-type", "classification"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_missing_test_argument(self):
        with patch.object(
            sys,
            "argv",
            ["scald", "--train", "train.csv", "--target", "y", "--task-type", "classification"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_missing_target_argument(self):
        with patch.object(
            sys,
            "argv",
            [
                "scald",
                "--train",
                "train.csv",
                "--test",
                "test.csv",
                "--task-type",
                "classification",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_missing_task_type_argument(self):
        with patch.object(
            sys, "argv", ["scald", "--train", "train.csv", "--test", "test.csv", "--target", "y"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_invalid_task_type(self):
        with patch.object(
            sys,
            "argv",
            [
                "scald",
                "--train",
                "train.csv",
                "--test",
                "test.csv",
                "--target",
                "y",
                "--task-type",
                "invalid",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_train_file_not_found(self, sample_csv_files):
        _, test_csv = sample_csv_files
        with patch.object(
            sys,
            "argv",
            [
                "scald",
                "--train",
                "nonexistent.csv",
                "--test",
                str(test_csv),
                "--target",
                "y",
                "--task-type",
                "classification",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_test_file_not_found(self, sample_csv_files):
        train_csv, _ = sample_csv_files
        with patch.object(
            sys,
            "argv",
            [
                "scald",
                "--train",
                str(train_csv),
                "--test",
                "nonexistent.csv",
                "--target",
                "y",
                "--task-type",
                "classification",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestCLIExecution:
    @patch("scald.cli.Scald")
    @patch("scald.cli.asyncio.run")
    def test_successful_execution_classification(
        self, mock_asyncio_run, mock_scald_class, sample_csv_files
    ):
        train_csv, test_csv = sample_csv_files

        mock_scald = MagicMock()
        mock_scald_class.return_value = mock_scald

        predictions = np.array([0, 1, 0, 1])
        mock_asyncio_run.return_value = predictions

        with patch.object(
            sys,
            "argv",
            [
                "scald",
                "--train",
                str(train_csv),
                "--test",
                str(test_csv),
                "--target",
                "target",
                "--task-type",
                "classification",
            ],
        ):
            main()

        mock_scald_class.assert_called_once_with(max_iterations=5)
        mock_asyncio_run.assert_called_once()

    @patch("scald.cli.Scald")
    @patch("scald.cli.asyncio.run")
    def test_successful_execution_regression(
        self, mock_asyncio_run, mock_scald_class, sample_csv_files
    ):
        train_csv, test_csv = sample_csv_files

        mock_scald = MagicMock()
        mock_scald_class.return_value = mock_scald

        predictions = np.array([1.2, 3.4, 5.6])
        mock_asyncio_run.return_value = predictions

        with patch.object(
            sys,
            "argv",
            [
                "scald",
                "--train",
                str(train_csv),
                "--test",
                str(test_csv),
                "--target",
                "price",
                "--task-type",
                "regression",
            ],
        ):
            main()

        mock_scald_class.assert_called_once_with(max_iterations=5)
        mock_asyncio_run.assert_called_once()

    @patch("scald.cli.Scald")
    @patch("scald.cli.asyncio.run")
    def test_custom_max_iterations(self, mock_asyncio_run, mock_scald_class, sample_csv_files):
        train_csv, test_csv = sample_csv_files

        mock_scald = MagicMock()
        mock_scald_class.return_value = mock_scald

        predictions = np.array([0, 1])
        mock_asyncio_run.return_value = predictions

        with patch.object(
            sys,
            "argv",
            [
                "scald",
                "--train",
                str(train_csv),
                "--test",
                str(test_csv),
                "--target",
                "target",
                "--task-type",
                "classification",
                "--max-iterations",
                "10",
            ],
        ):
            main()

        mock_scald_class.assert_called_once_with(max_iterations=10)

    @patch("scald.cli.Scald")
    @patch("scald.cli.asyncio.run")
    def test_execution_failure(self, mock_asyncio_run, mock_scald_class, sample_csv_files):
        train_csv, test_csv = sample_csv_files

        mock_scald = MagicMock()
        mock_scald_class.return_value = mock_scald

        mock_asyncio_run.side_effect = Exception("Execution failed")

        with patch.object(
            sys,
            "argv",
            [
                "scald",
                "--train",
                str(train_csv),
                "--test",
                str(test_csv),
                "--target",
                "target",
                "--task-type",
                "classification",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
