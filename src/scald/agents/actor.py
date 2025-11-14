from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Type

from pydantic import BaseModel, Field
from toon import encode

from scald.agents.base import BaseAgent
from scald.agents.context import ActorContext

if TYPE_CHECKING:
    from scald.memory.types import ActorMemoryContext

TaskType = Literal["classification", "regression"]


class ActorSolution(BaseModel):
    predictions_path: Path = Field(
        description="Absolute path to predictions CSV file (e.g., /home/user/.scald/actor/output/predictions.csv)"
    )
    data_analysis: str = Field(
        default="",
        description="Data exploration: dataset shape, features, target distribution, missing values, data quality issues",
    )
    preprocessing: str = Field(
        default="",
        description="Preprocessing steps: missing value handling, encoding, feature engineering, scaling",
    )
    model_training: str = Field(
        default="",
        description="Model selection rationale, hyperparameters, training approach, cross-validation strategy",
    )
    results: str = Field(
        default="",
        description="Training metrics, validation results, model performance, final predictions summary",
    )

    @property
    def report(self) -> str:
        return "\n\n".join(
            [
                f"# Data Analysis\n{self.data_analysis}",
                f"# Preprocessing\n{self.preprocessing}",
                f"# Model Training\n{self.model_training}",
                f"# Results\n{self.results}",
            ]
        )


class Actor(BaseAgent[ActorContext]):
    def _get_system_prompt(self) -> str:
        return """You are an expert data scientist solving ML tasks with provided MCP tools.

AVAILABLE TOOLS:
- file_operations: list_files, copy_file, move_file, delete_file, file_exists, get_file_info, create_directory
- data_preview: inspect_csv, preview_csv
- data_analysis: get_feature_distributions, get_correlations, detect_outliers, check_data_quality
- data_processing: encode_categorical_label, decode_categorical_label, handle_missing_values
- machine_learning: train_catboost, train_lightgbm, train_xgboost

WORKFLOW:
1. Inspect data: inspect_csv, preview_csv, check_data_quality
2. Preprocess if needed: handle_missing_values, encode_categorical_label
3. Train model: train_catboost/lightgbm/xgboost (always use predictions_path="/output/predictions.csv")
4. If you encoded target: decode predictions using decode_categorical_label with saved mapping

CRITICAL - Categorical Encoding:
If you encode target column, you MUST decode predictions before returning:
- Target in test dataset is ALWAYS empty column
- You MUST train model/models on train dataset and return predictions on test dataset
- encode_categorical_label saves mapping to /output/encodings/{column}_mapping.json
- After training, decode predictions: decode_categorical_label(column="prediction", mapping_path="...")
- Return decoded values (original labels, not integers)

CRITICAL - Predictions CSV Format:
The final predictions CSV MUST have a column named "prediction":
- Training tools automatically create this column
- If you manually create predictions CSV, use: pl.DataFrame({"prediction": predictions_array})
- The "prediction" column is mandatory for the framework to extract results

OUTPUT REQUIREMENTS - Return ActorSolution with:
- predictions_path: REQUIRED absolute path (e.g., /home/user/.scald/actor/output/predictions.csv)
- data_analysis: Dataset shape, features, distributions, missing values, quality issues
- preprocessing: Steps taken for cleaning, encoding, feature engineering
- model_training: Model choice, hyperparameters, training strategy
- results: Performance metrics, validation results, predictions summary
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
        iteration: int = 1,
        feedback: Optional[str] = None,
        past_experiences: Optional[list["ActorMemoryContext"]] = None,
    ) -> ActorSolution:
        from scald.agents.context import TaskContext

        ctx = ActorContext(
            task=TaskContext(
                train_path=Path(train_path),
                test_path=Path(test_path),
                target=target,
                task_type=task_type,
                iteration=iteration,
            ),
            feedback=feedback,
            past_experiences=past_experiences or [],
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
        return await self._run_agent(prompt, deps=ctx)
