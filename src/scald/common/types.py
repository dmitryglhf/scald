from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


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
    metrics: dict[str, float] = Field(default_factory=dict, description="Performance metrics")


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
