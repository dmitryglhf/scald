import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult
from chromadb.utils.embedding_functions import JinaEmbeddingFunction

from scald.agents.actor import ActorSolution
from scald.agents.critic import CriticEvaluation
from scald.memory.types import ActorMemoryContext, CriticMemoryContext

TaskType = Literal["classification", "regression"]


class MemoryManager:
    """Long-term memory management via ChromaDB with Jina embeddings"""

    COLLECTION_NAME = "scald_memory"
    MEMORY_DIR = Path.home() / ".scald" / "chromadb"

    def __init__(self, memory_dir: Optional[Path] = None):
        if memory_dir is None:
            memory_dir = self.MEMORY_DIR

        memory_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_fn = self._create_embedding_function()
        self.client: ClientAPI = PersistentClient(path=str(memory_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    async def retrieve(
        self, actor_report: str, task_type: TaskType, top_k: int = 5
    ) -> tuple[list[ActorMemoryContext], list[CriticMemoryContext]]:
        # Query ChromaDB with filters
        q_result: QueryResult | None = self.collection.query(
            query_texts=[actor_report],
            n_results=top_k,
            where={"task_type": task_type},
        )

        # Handle empty or None results
        if not q_result or not q_result.get("ids") or not q_result["ids"][0]:
            return [], []

        actor_contexts = []
        critic_contexts = []

        for i in range(len(q_result["ids"][0])):
            document = q_result["documents"][0][i]
            metadata = q_result["metadatas"][0][i]

            # Deserialize critic_evaluation from JSON
            critic_eval_data = json.loads(metadata["critic_evaluation"])
            critic_evaluation = CriticEvaluation(**critic_eval_data)

            # Extract known fields and reconstruct metrics
            known_fields = {"task_type", "iteration", "critic_evaluation", "timestamp"}
            metrics = {k: v for k, v in metadata.items() if k not in known_fields}

            # Create ActorMemoryContext
            actor_ctx = ActorMemoryContext(
                iteration=metadata["iteration"],
                accepted=critic_evaluation.score == 1,
                actions_summary=document,
                feedback_received=critic_evaluation.feedback,
                metrics=metrics,
            )
            actor_contexts.append(actor_ctx)

            # Create CriticMemoryContext
            critic_ctx = CriticMemoryContext(
                iteration=metadata["iteration"],
                score=critic_evaluation.score,
                actions_observed=document,
                feedback_given=critic_evaluation.feedback,
            )
            critic_contexts.append(critic_ctx)

        return actor_contexts, critic_contexts

    async def save(
        self,
        actor_solution: ActorSolution,
        critic_evaluation: CriticEvaluation,
        task_type: TaskType,
        iteration: int,
    ) -> str:
        entry_id = str(uuid.uuid4())

        # Create metadata with flattened metrics
        metadata = {
            "task_type": task_type,
            "iteration": iteration,
            "critic_evaluation": critic_evaluation.model_dump_json(),
            "timestamp": datetime.now().isoformat(),
        }

        self.collection.add(
            ids=[entry_id],
            documents=[actor_solution.report],
            metadatas=[metadata],
        )

        return entry_id

    def clear(self) -> None:
        """Remove all entries from collection"""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)

    def _create_embedding_function(self) -> JinaEmbeddingFunction:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError("JINA_API_KEY environment variable not set")
        return JinaEmbeddingFunction(api_key=api_key, model_name="jina-embeddings-v3")
