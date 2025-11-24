from pathlib import Path
from typing import Literal, cast

from toon import encode

from scald.agents.base import BaseAgent
from scald.agents.context import ActorContext, TaskContext
from scald.models import ActorMemoryContext, ActorSolution

__all__ = ["Actor", "ActorSolution", "TaskType"]

TaskType = Literal["classification", "regression"]


class Actor(BaseAgent[ActorContext]):
    def _get_system_prompt(self) -> str:
        return """You are an expert data scientist solving ML tasks with provided tools for data science.

WORKFLOW:
1. Inspect data: inspect_csv, preview_csv, check_data_quality
2. Preprocess if needed: handle_missing_values, encode_categorical_label
3. Train model(s): train_catboost/lightgbm/xgboost
   - Training tools automatically save models and predictions with unique paths
   - Tool returns: {"model_path": "...", "predictions_path": "...", "train_metrics": {...}}
   - Store predictions_path from your chosen approach for the solution
4. Optional: ensemble_predictions using multiple predictions_paths from step 3
5. If you encoded target: decode final predictions using decode_categorical_label

CRITICAL - Categorical Encoding:
If you encode target column, you MUST DECODE predictions before returning:
- Target in test dataset is ALWAYS empty column
- encode_categorical_label saves mapping to /output/encodings/{column}_mapping.json
- After training, decode predictions: decode_categorical_label(column="prediction", mapping_path="...")
- Return decoded values (original labels, not integers)

CRITICAL - Predictions CSV Format:
The final predictions CSV MUST have a column named "prediction":
- Training tools automatically create this column
- If you manually create predictions CSV, use: pl.DataFrame({"prediction": predictions_array})
- The "prediction" column is mandatory for the framework to extract results

OUTPUT REQUIREMENTS - Return ActorSolution with:
- predictions_path: Use the path returned from your final training/ensemble tool call
- data_analysis: Dataset shape, features, distributions, missing values, quality issues
- preprocessing: Steps taken for cleaning, encoding, feature engineering
- model_training: Model choice, hyperparameters, training strategy, metrics comparison
- results: Performance metrics, validation results, predictions summary

Example multi-model workflow:
result1 = train_catboost(train_path, test_path, target, task_type)
result2 = train_lightgbm(train_path, test_path, target, task_type)
Use predictions_path from best performing model: ActorSolution(predictions_path=result1["predictions_path"], ...)
"""

    def _get_output_type(self) -> type[ActorSolution]:
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
        iteration: int = 1,
        feedback: str | None = None,
        past_experiences: list[ActorMemoryContext] | None = None,
    ) -> ActorSolution:
        ctx = ActorContext(
            task=TaskContext(
                train_path=Path(train_path),
                test_path=Path(test_path),
                target=target,
                task_type=task_type,
                iteration=iteration,
            ),
            feedback=feedback,
            past_experiences=past_experiences if past_experiences is not None else [],
        )

        sections = [
            f"Solve {task_type} task:",
            f"- Train Dataset CSV: {train_path}",
            f"- Test Dataset CSV: {test_path}",
            f"- Target column: {target}",
        ]

        if feedback:
            sections.append(f"- Previous feedback: {feedback}")

        if past_experiences:
            sections.append(
                f"\nPast experiences: {encode([e.model_dump() for e in past_experiences])}"
            )

        prompt = "\n".join(sections)
        result = await self._run_agent(prompt, deps=ctx)
        return cast(ActorSolution, result)
