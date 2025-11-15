import json
import os
import time
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
        logger.debug(
            f"Retrieving memory contexts | task_type={task_type} | top_k={top_k} | "
            f"report_length={len(actor_report)}"
        )

        query_start = time.time()
        try:
            q_result = self.collection.query(
                query_texts=[actor_report],
                n_results=top_k,
                where={"task_type": task_type},
            )
            query_duration = time.time() - query_start
            logger.debug(f"ChromaDB query completed | duration_sec={query_duration:.3f}")
        except Exception as e:
            logger.error(
                f"ChromaDB query failed | task_type={task_type} | top_k={top_k} | "
                f"error_type={type(e).__name__} | error={e}"
            )
            return [], []

        if not q_result or not q_result.get("ids") or not q_result["ids"][0]:
            logger.debug(f"No memory contexts found | task_type={task_type}")
            return [], []

        results_count = len(q_result["ids"][0])
        logger.debug(f"Processing memory results | count={results_count}")

        actor_contexts = []
        critic_contexts = []

        for i in range(results_count):
            document = q_result["documents"][0][i]
            metadata = q_result["metadatas"][0][i]

            try:
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
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    f"Failed to parse memory context | index={i} | "
                    f"error_type={type(e).__name__} | error={e}"
                )
                continue

        total_duration = time.time() - query_start
        logger.info(
            f"Retrieved memory contexts | task_type={task_type} | "
            f"actor_contexts={len(actor_contexts)} | critic_contexts={len(critic_contexts)} | "
            f"duration_sec={total_duration:.3f}"
        )

        return actor_contexts, critic_contexts

    def save(
        self,
        actor_solution: ActorSolution,
        critic_evaluation: CriticEvaluation,
        task_type: TaskType,
        iteration: int,
    ) -> str:
        logger.debug(
            f"Saving memory entry | task_type={task_type} | iteration={iteration} | "
            f"score={critic_evaluation.score:.3f} | report_length={len(actor_solution.report)}"
        )

        entry_id = str(uuid.uuid4())

        metadata = {
            "task_type": task_type,
            "iteration": iteration,
            "critic_score": critic_evaluation.score,
            "critic_evaluation": critic_evaluation.model_dump_json(),
            "timestamp": datetime.now().isoformat(),
        }

        save_start = time.time()
        try:
            self.collection.add(
                ids=[entry_id],
                documents=[actor_solution.report],
                metadatas=[metadata],
            )
            save_duration = time.time() - save_start
            logger.info(
                f"Saved memory entry | entry_id={entry_id} | task_type={task_type} | "
                f"iteration={iteration} | score={critic_evaluation.score:.3f} | "
                f"duration_sec={save_duration:.3f}"
            )
        except Exception as e:
            logger.error(
                f"Failed to save to ChromaDB | task_type={task_type} | iteration={iteration} | "
                f"error_type={type(e).__name__} | error={e}"
            )
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
