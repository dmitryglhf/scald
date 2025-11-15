import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import JinaEmbeddingFunction

from scald.common.logger import get_logger
from scald.models import ActorMemoryContext, ActorSolution, CriticEvaluation, CriticMemoryContext

TaskType = Literal["classification", "regression"]

logger = get_logger()


class MemoryManager:
    COLLECTION_NAME = "scald_memory"
    MEMORY_DIR = Path.home() / ".scald" / "chromadb"

    def __init__(self, memory_dir: Optional[Path] = None):
        if memory_dir is None:
            memory_dir = self.MEMORY_DIR

        memory_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_fn = self._create_embedding_function()
        self.client = PersistentClient(path=str(memory_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(
        self, actor_report: str, task_type: TaskType, top_k: int = 5
    ) -> tuple[list[ActorMemoryContext], list[CriticMemoryContext]]:
        try:
            q_result = self.collection.query(
                query_texts=[actor_report],
                n_results=top_k,
                where={"task_type": task_type},
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return [], []

        if not q_result or not q_result.get("ids") or not q_result["ids"][0]:
            return [], []

        actor_contexts = []
        critic_contexts = []

        for i in range(len(q_result["ids"][0])):
            document = q_result["documents"][0][i]
            metadata = q_result["metadatas"][0][i]

            critic_eval_data = json.loads(metadata["critic_evaluation"])
            critic_evaluation = CriticEvaluation(**critic_eval_data)

            critic_score = metadata.get("critic_score", critic_evaluation.score)

            actor_ctx = ActorMemoryContext(
                iteration=metadata["iteration"],
                accepted=critic_score == 1,
                actions_summary=document,
                feedback_received=critic_evaluation.feedback,
            )
            actor_contexts.append(actor_ctx)

            critic_ctx = CriticMemoryContext(
                iteration=metadata["iteration"],
                score=critic_score,
                actions_observed=document,
                feedback_given=critic_evaluation.feedback,
            )
            critic_contexts.append(critic_ctx)

        return actor_contexts, critic_contexts

    def save(
        self,
        actor_solution: ActorSolution,
        critic_evaluation: CriticEvaluation,
        task_type: TaskType,
        iteration: int,
    ) -> str:
        entry_id = str(uuid.uuid4())

        metadata = {
            "task_type": task_type,
            "iteration": iteration,
            "critic_score": critic_evaluation.score,
            "critic_evaluation": critic_evaluation.model_dump_json(),
            "timestamp": datetime.now().isoformat(),
        }

        try:
            self.collection.add(
                ids=[entry_id],
                documents=[actor_solution.report],
                metadatas=[metadata],
            )
        except Exception as e:
            logger.error(f"Failed to save to ChromaDB: {e}")
            raise

        return entry_id

    def clear(self) -> None:
        self.client.delete_collection(name=self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _create_embedding_function(self) -> JinaEmbeddingFunction:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError("JINA_API_KEY environment variable not set")
        return JinaEmbeddingFunction(api_key=api_key, model_name="jina-embeddings-v3")
