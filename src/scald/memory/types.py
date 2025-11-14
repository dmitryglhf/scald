from datetime import datetime

from pydantic import BaseModel


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
