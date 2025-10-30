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
- file_operations: list_files, copy_file, move_file, delete_file, file_exists, get_file_info, create_directory
- data_preview: inspect_csv, preview_csv
- data_analysis: get_feature_distributions, get_correlations, detect_outliers, check_data_quality
- data_processing: encode_categorical_label, handle_missing_values, train_test_split
- machine_learning: train_catboost, train_lightgbm, train_xgboost

IMPORTANT: You will receive ALREADY SPLIT train and test datasets. DO NOT use train_test_split!

STEP-BY-STEP WORKFLOW (FOLLOW EXACTLY):
1. Use list_files("/data") to see available datasets (optional but recommended)
2. Inspect train CSV with inspect_csv tool to analyze structure
3. Optionally preview data with preview_csv to see actual values
4. Analyze with get_feature_distributions or check_data_quality tools
5. Apply preprocessing if needed (handle_missing_values, encode_categorical_label)
6. Train model: train_catboost(train_path=<given_train_path>, test_path=<given_test_path>, predictions_path="/output/predictions.csv", target_column=..., task_type=...)
7. Extract test_metrics from training result
8. Use file_operations tools if you need to organize outputs (copy/move predictions, clean up intermediates)

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
1. list_files("/data", pattern="*.csv") → see: train.csv, test.csv
2. inspect_csv("/data/train.csv") → get metadata (shape, dtypes, missing values)
3. preview_csv("/data/train.csv") → see sample data rows
4. check_data_quality("/data/train.csv") → identify issues
5. train_catboost(train_path="/data/train.csv", test_path="/data/test.csv", predictions_path="/output/predictions.csv", target_column="Species", task_type="classification")
   → returns {"test_metrics": {"accuracy": 0.96, "f1": 0.95}, "predictions_path": "/output/predictions.csv"}
6. preview_csv("/output/predictions.csv") → read predictions
7. Extract predictions column as list
8. copy_file("/output/predictions.csv", "/output/final_predictions.csv") → backup (optional)
9. Return ActorSolution(predictions_path="/output/predictions.csv", predictions=[0, 1, 2, 1, 0, ...], metrics={"accuracy": 0.96, "f1": 0.95}, report="...")
```

DO NOT skip tool calls. DO NOT return empty results. USE THE TOOLS.
"""

    def _get_output_type(self) -> Type[BaseModel]:
        return ActorSolution

    def _get_mcp_tools(self) -> list[str]:
        return [
            "file_operations",
            "data_preview",
            "data_analysis",
            "data_processing",
            "machine_learning",
        ]

    async def solve_task(
        self,
        train_path: str | Path,
        test_path: str | Path,
        target: str,
        task_type: TaskType,
        feedback: Optional[str] = None,
        memory_context: Optional[list[ActorMemoryContext]] = None,
    ) -> ActorSolution:
        resolved_train = resolve_csv_path(train_path)
        resolved_test = resolve_csv_path(test_path)

        sections = [
            f"Solve {task_type.value} task:",
            f"- Train CSV: {resolved_train}",
            f"- Test CSV: {resolved_test}",
            f"- Target: {target}",
        ]

        if feedback:
            sections.append(f"- Previous feedback: {feedback}")

        if memory_context:
            sections.append("")
            sections.append(self._format_memory_context(memory_context))

        prompt = "\n".join(sections)
        return await self._run_agent(prompt)

    def _format_memory_context(self, memory_context: list[ActorMemoryContext]) -> str:
        lines = ["PREVIOUS SOLUTIONS:"]
        for i, mem in enumerate(memory_context, 1):
            lines.append(f"{i}. Iteration {mem.iteration} (accepted={mem.accepted}):")
            lines.append(f"   Metrics: {mem.metrics}")
            lines.append(f"   Report: {mem.report}")
            lines.append("")
        return "\n".join(lines)
