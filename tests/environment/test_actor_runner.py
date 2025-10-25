import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scald.common.types import TaskType


class TestActorRunner:
    @pytest.mark.asyncio
    @patch("scald.environment.actor_runner.Actor")
    @patch.dict(
        os.environ,
        {
            "CSV_PATH": "/data/test.csv",
            "TARGET": "species",
            "TASK_TYPE": "classification",
            "OUTPUT_DIR": "/tmp/output",
        },
    )
    async def test_actor_runner_success(self, mock_actor_class):
        mock_actor = MagicMock()
        mock_actor_class.return_value = mock_actor

        mock_solution = MagicMock()
        mock_solution.predictions_path = "/output/predictions.csv"
        mock_solution.metrics = {"accuracy": 0.95}

        mock_actor.solve_task = AsyncMock(return_value=mock_solution)

        output_dir = Path("/tmp/output")
        output_dir.mkdir(exist_ok=True)

        from scald.environment.actor_runner import main

        with pytest.raises(SystemExit) as exc_info:
            await main()

        assert exc_info.value.code == 0  # type: ignore

        solution_file = output_dir / "solution.json"
        assert solution_file.exists()

        with open(solution_file) as f:
            data = json.load(f)

        assert data["predictions_path"] == "/output/predictions.csv"
        assert data["metrics"] == {"accuracy": 0.95}

        solution_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    @patch("scald.environment.actor_runner.Actor")
    @patch.dict(
        os.environ,
        {
            "CSV_PATH": "/data/test.csv",
            "TARGET": "label",
            "TASK_TYPE": "regression",
            "OUTPUT_DIR": "/tmp/output",
            "FEEDBACK": "reduce overfitting",
        },
    )
    async def test_actor_runner_with_feedback(self, mock_actor_class):
        mock_actor = MagicMock()
        mock_actor_class.return_value = mock_actor

        mock_solution = MagicMock()
        mock_solution.predictions_path = "/output/predictions.csv"
        mock_solution.metrics = {"mse": 0.01}

        mock_actor.solve_task = AsyncMock(return_value=mock_solution)

        output_dir = Path("/tmp/output")
        output_dir.mkdir(exist_ok=True)

        from scald.environment.actor_runner import main

        with pytest.raises(SystemExit) as exc_info:
            await main()

        assert exc_info.value.code == 0  # type: ignore

        mock_actor.solve_task.assert_called_once()
        call_kwargs = mock_actor.solve_task.call_args[1]
        assert call_kwargs["feedback"] == "reduce overfitting"
        assert call_kwargs["task_type"] == TaskType.REGRESSION

        solution_file = output_dir / "solution.json"
        solution_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    @patch("scald.environment.actor_runner.Actor")
    @patch.dict(
        os.environ,
        {
            "CSV_PATH": "/data/test.csv",
            "TARGET": "target",
            "TASK_TYPE": "classification",
            "OUTPUT_DIR": "/tmp/output",
        },
    )
    async def test_actor_runner_failure(self, mock_actor_class):
        mock_actor = MagicMock()
        mock_actor_class.return_value = mock_actor
        mock_actor.solve_task = AsyncMock(side_effect=Exception("Model training failed"))

        from scald.environment.actor_runner import main

        with pytest.raises(SystemExit) as exc_info:
            await main()

        assert exc_info.value.code == 1  # type: ignore

    @pytest.mark.asyncio
    @patch("scald.environment.actor_runner.Actor")
    @patch.dict(
        os.environ,
        {
            "CSV_PATH": "/data/test.csv",
            "TARGET": "target",
            "TASK_TYPE": "classification",
            "OUTPUT_DIR": "/tmp/output",
        },
    )
    async def test_actor_runner_defaults(self, mock_actor_class):
        mock_actor = MagicMock()
        mock_actor_class.return_value = mock_actor

        mock_solution = MagicMock()
        mock_solution.predictions_path = "/output/predictions.csv"
        mock_solution.metrics = {}

        mock_actor.solve_task = AsyncMock(return_value=mock_solution)

        output_dir = Path("/tmp/output")
        output_dir.mkdir(exist_ok=True)

        from scald.environment.actor_runner import main

        with pytest.raises(SystemExit):
            await main()

        call_kwargs = mock_actor.solve_task.call_args[1]
        assert call_kwargs["csv_path"] == Path("/data/test.csv")
        assert call_kwargs["target"] == "target"
        assert call_kwargs["task_type"] == TaskType.CLASSIFICATION
        assert call_kwargs["feedback"] is None

        solution_file = output_dir / "solution.json"
        solution_file.unlink(missing_ok=True)
