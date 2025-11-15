from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from scald.models import ActorMemoryContext, CriticMemoryContext

TaskType = Literal["classification", "regression"]


class TaskContext(BaseModel):
    train_path: Path
    test_path: Path
    target: str
    task_type: TaskType
    iteration: int = 1


class ActorContext(BaseModel):
    task: TaskContext
    feedback: str | None = None
    past_experiences: list[ActorMemoryContext] = []


class CriticContext(BaseModel):
    task: TaskContext
    past_evaluations: list[CriticMemoryContext] = []
