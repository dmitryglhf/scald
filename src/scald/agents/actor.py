from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Type

from pydantic import BaseModel, Field, field_validator
from toon import encode

from scald.agents.base import BaseAgent

if TYPE_CHECKING:
    from scald.memory.types import ActorMemoryContext

TaskType = Literal["classification", "regression"]


class ActorSolution(BaseModel):
    """Solution from Actor."""

    predictions_path: Optional[Path] = Field(default=None, description="Path to predictions CSV")
    predictions: list[Any] = Field(
        default_factory=list, description="List of predictions on test set"
    )
    report: str = Field(
        default="",
        description="Detailed report of all actions taken: data preprocessing, models trained, results achieved",
    )

    @field_validator("predictions_path", mode="before")
    @classmethod
    def validate_predictions_path(cls, v):
        """Handle 'null' string and invalid paths, extract valid path from junk."""
        if v is None:
            return None

        if isinstance(v, str):
            # Handle explicit null/none values
            if v.strip().lower() in ("null", "none", ""):
                return None

            # If string contains XML/reasoning junk, try to extract valid path
            if "<" in v or ">" in v or len(v) > 500:
                # Try to extract /output/predictions.csv pattern
                import re

                match = re.search(r"(/output/[a-zA-Z0-9_\-./]+\.csv)", v)
                if match:
                    return Path(match.group(1))
                # If no valid path found, return None
                return None

            # Valid string path
            return Path(v)

        return v


class Actor(BaseAgent):
    """Data scientist agent."""

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
5. Read predictions CSV to extract list of values
6. Return ActorSolution with all fields filled

CRITICAL - Categorical Encoding:
If you encode target column, you MUST decode predictions before returning:
- encode_categorical_label saves mapping to /output/encodings/{column}_mapping.json
- After training, decode predictions: decode_categorical_label(column="prediction", mapping_path="...")
- Return decoded values (original labels, not integers)

OUTPUT REQUIREMENTS:
- predictions_path: absolute path (e.g., ~/.scald/actor/output/predictions.csv)
- predictions: list of actual prediction values from CSV
- metrics: dict with test metrics from training result
- report: detailed markdown report covering data analysis, preprocessing, model, and results
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
        past_experiences: Optional[list["ActorMemoryContext"]] = None,
    ) -> ActorSolution:
        sections = [
            f"Solve {task_type} task:",
            f"- Train CSV: {train_path}",
            f"- Test CSV: {test_path}",
            f"- Target column: {target}",
        ]

        if feedback:
            sections.append(f"- Previous feedback: {feedback}")

        if past_experiences:
            sections.append(
                f"\nPast experiences: {encode([e.model_dump() for e in past_experiences])}"
            )

        prompt = "\n".join(sections)
        return await self._run_agent(prompt)
