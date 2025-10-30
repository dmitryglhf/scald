from datetime import datetime

from pydantic import BaseModel


class MemoryEntry(BaseModel):
    """Single memory entry stored in ChromaDB"""

    entry_id: str
    timestamp: datetime

    task_type: str
    actor_report: str

    critic_feedback: str

    iteration: int
    accepted: bool
    metrics: dict[str, float]


class ActorMemoryContext(BaseModel):
    """Memory context passed to Actor agent"""

    iteration: int
    accepted: bool
    actions_summary: str
    feedback_received: str
    metrics: dict[str, float]


class CriticMemoryContext(BaseModel):
    """Memory context passed to Critic agent"""

    iteration: int
    score: int
    actions_observed: str
    feedback_given: str
