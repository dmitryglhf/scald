from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from scald.memory.types import ActorMemoryContext, CriticMemoryContext

TaskType = Literal["classification", "regression"]


class TaskContext(BaseModel):
    train_path: Path
    test_path: Path
    target: str
    task_type: TaskType
    iteration: int = 1


class ActorContext(BaseModel):
    task: TaskContext
    feedback: Optional[str] = None
    past_experiences: list["ActorMemoryContext"] = []


class CriticContext(BaseModel):
    task: TaskContext
    past_evaluations: list["CriticMemoryContext"] = []
