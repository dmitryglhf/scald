from pathlib import Path
from typing import Type

from pydantic import BaseModel

from scald.agents.base import BaseAgent
from scald.common.paths import resolve_csv_path
from scald.common.types import ActorSolution, TaskType


class Actor(BaseAgent):
    """Data scientist agent."""

    def _get_system_prompt(self) -> str:
        return """You are an expert data scientist. You MUST use the provided tools to solve ML tasks.

CRITICAL: You have access to these MCP tools - YOU MUST USE THEM:
- data_loading: load_csv
- data_analysis: get_basic_info, get_feature_distributions, get_correlation_matrix
- data_processing: encode_categorical_label, handle_missing_values
- machine_learning: train_catboost, train_lightgbm, train_xgboost

IMPORTANT: You will receive ALREADY SPLIT train and test datasets. DO NOT use train_test_split!

STEP-BY-STEP WORKFLOW (FOLLOW EXACTLY):
1. Load train CSV with load_csv tool to analyze it
2. Optionally load test CSV to check consistency
3. Analyze with get_basic_info tool
4. Apply preprocessing if needed (handle_missing_values, encode_categorical_label)
5. Train model: train_catboost(train_path=<given_train_path>, test_path=<given_test_path>, predictions_path="/output/predictions.csv", target_column=..., task_type=...)
6. Extract test_metrics from training result

REQUIRED OUTPUT:
- predictions_path: "/output/predictions.csv" (the path you passed to train function)
- predictions: list of predictions from the CSV file (read the predictions.csv and extract values as a list)
- metrics: test_metrics dict from training result (e.g., {"accuracy": 0.95, "f1": 0.94})
- report: Multi-paragraph description including:
  * Dataset analysis results (shape, features, target distribution)
  * Preprocessing steps taken (encoding, missing values handling if applied)
  * Model choice and hyperparameters used
  * Training and test metrics achieved
  * Any observations or issues

EXAMPLE:
```
1. load_csv("/data/train.csv") → dataset info
2. get_basic_info("/data/train.csv") → analyze train data
3. train_catboost(train_path="/data/train.csv", test_path="/data/test.csv", predictions_path="/output/predictions.csv", target_column="Species", task_type="classification")
   → returns {"test_metrics": {"accuracy": 0.96, "f1": 0.95}, "predictions_path": "/output/predictions.csv"}
4. load_csv("/output/predictions.csv") → read predictions
5. Extract predictions column as list
6. Return ActorSolution(predictions_path="/output/predictions.csv", predictions=[0, 1, 2, 1, 0, ...], metrics={"accuracy": 0.96, "f1": 0.95}, report="...")
```

DO NOT skip tool calls. DO NOT return empty results. USE THE TOOLS.
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
        self,
        train_path: str | Path,
        test_path: str | Path,
        target: str,
        task_type: TaskType,
        feedback: str | None = None,
    ) -> ActorSolution:
        """Solve data science task."""
        resolved_train = resolve_csv_path(train_path)
        resolved_test = resolve_csv_path(test_path)

        prompt = f"""Solve {task_type.value} task:
- Train CSV: {resolved_train}
- Test CSV: {resolved_test}
- Target: {target}
{f"- Previous feedback: {feedback}" if feedback else ""}

Remember: datasets are already split, use them directly for training!
"""

        return await self._run_agent(prompt)
