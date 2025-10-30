from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from scald.common.logger import get_logger
from scald.common.types import CriticEvaluation, TaskType
from scald.memory.base import BaseMemory

if TYPE_CHECKING:
    from scald.memory.types import CriticMemoryContext

logger = get_logger()


class CriticMemory(BaseMemory):
    def _get_collection_description(self) -> str:
        return "Critic evaluations with embeddings for semantic search"

    def save(
        self,
        evaluation: CriticEvaluation,
        task_type: TaskType,
        iteration: int,
    ) -> str:
        timestamp = datetime.now().isoformat()
        doc_id = f"{task_type.value}_{iteration}_{timestamp}"

        document_text = self._create_document_text(evaluation, task_type)

        metadata = {
            "task_type": task_type.value,
            "iteration": iteration,
            "score": evaluation.score,
            "timestamp": timestamp,
        }

        self.collection.add(
            ids=[doc_id],
            documents=[document_text],
            metadatas=[metadata],
        )

        logger.debug(f"Saved critic evaluation: {doc_id} (score={evaluation.score})")
        return doc_id

    def search(
        self,
        task_type: TaskType,
        query_text: Optional[str] = None,
        limit: int = 5,
    ) -> list[CriticMemoryContext]:
        if query_text:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=limit,
                where={"task_type": task_type.value},
            )
            ids = results["ids"][0] if results["ids"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            documents = results["documents"][0] if results["documents"] else []
        else:
            results = self.collection.get(
                where={"task_type": task_type.value},
                limit=limit,
            )
            ids = results["ids"]
            metadatas = results["metadatas"]
            documents = results["documents"]

        from scald.memory.types import CriticMemoryContext

        contexts = []
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i]
            document = documents[i]

            context = CriticMemoryContext(
                id=doc_id,
                task_type=metadata["task_type"],
                iteration=metadata["iteration"],
                score=metadata["score"],
                timestamp=metadata["timestamp"],
                feedback=document,
            )
            contexts.append(context)

        logger.debug(f"Retrieved {len(contexts)} critic contexts")
        return contexts

    def _create_document_text(self, evaluation: CriticEvaluation, task_type: TaskType) -> str:
        return f"""Task: {task_type.value}
Score: {evaluation.score}
Feedback: {evaluation.feedback}"""
