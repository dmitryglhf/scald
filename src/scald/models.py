from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


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


class CriticEvaluation(BaseModel):
    score: int = Field(ge=0, le=1, description="0=reject, 1=accept")
    feedback: str = Field(description="Feedback and suggestions")


class MemoryEntry(BaseModel):
    entry_id: str
    timestamp: datetime
    task_type: str
    data_analysis: str
    preprocessing: str
    model_training: str
    results: str
    critic_feedback: str
    iteration: int
    accepted: bool


class ActorMemoryContext(BaseModel):
    iteration: int
    accepted: bool
    actions_summary: str
    feedback_received: str


class CriticMemoryContext(BaseModel):
    iteration: int
    score: int
    actions_observed: str
    feedback_given: str
