from pathlib import Path
from typing import Type

from pydantic import BaseModel

from scald.agents.base import BaseAgent
from scald.common.paths import resolve_csv_path
from scald.common.types import ActorSolution, TaskType


class Actor(BaseAgent):
    """Data scientist agent."""

    def _get_system_prompt(self) -> str:
        return """You are an expert data scientist solving ML tasks.

WORKFLOW:
1. Load and analyze the dataset using data_loading and data_analysis tools
2. Preprocess data: handle missing values, encode categorical features, split train/test using data_processing tools
3. Train models using machine_learning tools (train_catboost, train_lightgbm, or train_xgboost)
   - Pass test_path parameter to evaluate on test set
   - Pass predictions_path="/output/predictions.csv" to save predictions
4. Return the metrics from training and the predictions_path

IMPORTANT REQUIREMENTS:
- You MUST use predictions_path parameter when calling train functions
- You MUST save predictions to /output/predictions.csv (this is required!)
- You MUST return test_metrics from the training function
- Use train_test_split from data_processing to create train/test sets first

EXAMPLE WORKFLOW:
1. Use train_test_split to split data into /output/train.csv and /output/test.csv
2. Call train_catboost(train_path="/output/train.csv", test_path="/output/test.csv", predictions_path="/output/predictions.csv", ...)
3. Return ActorSolution with predictions_path="/output/predictions.csv" and metrics from test_metrics

OUTPUT FORMAT:
Return ActorSolution with:
- predictions_path: "/output/predictions.csv" (required)
- metrics: test_metrics from training function (required, e.g., {"accuracy": 0.95, "f1": 0.94})
"""

    def _get_output_type(self) -> Type[BaseModel]:
        return ActorSolution

    def _get_mcp_tools(self) -> list[str]:
        return [
            "terminal-controller",
            "data_analysis",
            "data_loading",
            "machine_learning",
            "data_processing",
        ]

    async def solve_task(
        self, csv_path: str | Path, target: str, task_type: TaskType, feedback: str | None = None
    ) -> ActorSolution:
        """Solve data science task."""
        resolved_path = resolve_csv_path(csv_path)

        prompt = f"""Solve {task_type.value} task:
- CSV: {resolved_path}
- Target: {target}
{f"- Previous feedback: {feedback}" if feedback else ""}
"""

        return await self._run_agent(prompt)
