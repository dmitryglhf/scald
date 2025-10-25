from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    """Type of data science task."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class AgentResult(BaseModel):
    """Result from base agent execution."""

    success: bool = Field(description="Execution succeeded")
    output: Any = Field(description="Agent output")
    error: Optional[str] = Field(default=None, description="Error if failed")


class ActorSolution(BaseModel):
    """Solution from Actor."""

    predictions_path: Optional[Path] = Field(default=None, description="Path to predictions CSV")
    predictions: list[Any] = Field(
        default_factory=list, description="List of predictions on test set"
    )
    metrics: dict[str, float] = Field(default_factory=dict, description="Performance metrics")
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


class CriticEvaluation(BaseModel):
    """Evaluation from Critic."""

    score: int = Field(ge=0, le=1, description="0=reject, 1=accept")
    feedback: str = Field(description="Feedback and suggestions")


class FinalResult(BaseModel):
    """Final result from Orchestrator."""

    success: bool = Field(description="Task completed successfully")
    solution: Optional[ActorSolution] = Field(default=None, description="Final solution")
    iterations: int = Field(description="Actor-Critic iterations")
    report_path: Optional[Path] = Field(default=None, description="Path to report")
    predictions_path: Optional[Path] = Field(default=None, description="Path to predictions")
