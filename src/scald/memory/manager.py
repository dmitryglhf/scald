from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import chromadb
from chromadb.config import Settings

from scald.common.logger import get_logger
from scald.common.types import ActorSolution, CriticEvaluation, TaskType
from scald.memory.actor_memory import ActorMemory
from scald.memory.critic_memory import CriticMemory

if TYPE_CHECKING:
    from scald.memory.types import ActorMemoryContext, CriticMemoryContext

logger = get_logger()


class MemoryManager:
    def __init__(
        self,
        persist_path: Optional[str] = None,
        use_jina: bool = True,
        jina_api_key: Optional[str] = None,
        jina_model: str = "jina-embeddings-v3",
    ):
        if persist_path is None:
            persist_path = str(Path.cwd() / "data" / "chroma")

        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        if use_jina:
            try:
                embedding_function = self._create_jina_embedding_function(
                    api_key=jina_api_key,
                    model_name=jina_model,
                )
                logger.info("Using Jina AI embeddings")
            except ValueError as e:
                logger.warning(f"Jina AI not available: {e}. Falling back to default embeddings")
                embedding_function = self._create_default_embedding_function()
        else:
            embedding_function = self._create_default_embedding_function()

        self.actors = ActorMemory(
            client=self.client,
            collection_name="actor_solutions",
            embedding_function=embedding_function,
        )

        self.critics = CriticMemory(
            client=self.client,
            collection_name="critic_evaluations",
            embedding_function=embedding_function,
        )

        logger.debug(f"MemoryManager initialized at {persist_path}")

    def save_actor_solution(
        self,
        solution: ActorSolution,
        task_type: TaskType,
        target: str,
        iteration: int,
        accepted: bool,
    ) -> str:
        return self.actors.save(solution, task_type, target, iteration, accepted)

    def get_actor_context(
        self,
        task_type: TaskType,
        target: str,
        limit: int = 3,
        query_text: Optional[str] = None,
    ) -> list[ActorMemoryContext]:
        return self.actors.search(task_type, target, query_text, limit)

    def update_actor_solution_status(
        self, task_type: TaskType, target: str, iteration: int, accepted: bool
    ) -> bool:
        return self.actors.update_status(task_type, target, iteration, accepted)

    def save_critic_evaluation(
        self, evaluation: CriticEvaluation, task_type: TaskType, iteration: int
    ) -> str:
        return self.critics.save(evaluation, task_type, iteration)

    def get_critic_context(
        self, task_type: TaskType, limit: int = 5, query_text: Optional[str] = None
    ) -> list[CriticMemoryContext]:
        return self.critics.search(task_type, query_text, limit)

    def clear_all(self) -> None:
        self.actors.clear()
        self.critics.clear()
        logger.debug("Cleared all memory")

    def _create_jina_embedding_function(
        self,
        api_key: Optional[str] = None,
        model_name: str = "jina-embeddings-v3",
    ):
        if api_key is None:
            api_key = os.getenv("JINA_API_KEY")

        if not api_key:
            raise ValueError(
                "Jina AI API key required. Set JINA_API_KEY env var or pass api_key parameter"
            )

        logger.debug(f"Creating Jina embedding function: {model_name}")

        return embedding_functions.JinaEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
        )

    def _create_default_embedding_function(self):
        logger.debug("Creating default embedding function")
        return embedding_functions.DefaultEmbeddingFunction()
