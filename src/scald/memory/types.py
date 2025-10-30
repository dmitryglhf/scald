from pydantic import BaseModel, Field


class ActorMemoryContext(BaseModel):
    id: str
    task_type: str
    target: str
    iteration: int
    accepted: bool
    timestamp: str
    metrics: dict[str, float]
    report: str

    class Config:
        frozen = True


class CriticMemoryContext(BaseModel):
    id: str
    task_type: str
    iteration: int
    score: int = Field(ge=0, le=1)
    timestamp: str
    feedback: str

    class Config:
        frozen = True
