from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from scald.common.logger import get_logger
from scald.common.types import ActorSolution, TaskType
from scald.memory.base import BaseMemory

if TYPE_CHECKING:
    from scald.memory.types import ActorMemoryContext

logger = get_logger()


class ActorMemory(BaseMemory):
    def _get_collection_description(self) -> str:
        return "Actor solutions with embeddings for semantic search"

    def save(
        self,
        solution: ActorSolution,
        task_type: TaskType,
        target: str,
        iteration: int,
        accepted: bool,
    ) -> str:
        timestamp = datetime.now().isoformat()
        doc_id = f"{task_type.value}_{target}_{iteration}_{timestamp}"

        document_text = self._create_document_text(solution, task_type, target)

        metadata = {
            "task_type": task_type.value,
            "target": target,
            "iteration": iteration,
            "accepted": accepted,
            "timestamp": timestamp,
            "metrics": json.dumps(solution.metrics),
        }

        self.collection.add(
            ids=[doc_id],
            documents=[document_text],
            metadatas=[metadata],
        )

        logger.debug(f"Saved actor solution: {doc_id} (accepted={accepted})")
        return doc_id

    def update_status(
        self, task_type: TaskType, target: str, iteration: int, accepted: bool
    ) -> bool:
        results = self.collection.get(
            where={
                "$and": [
                    {"task_type": task_type.value},
                    {"target": target},
                    {"iteration": iteration},
                ]
            }
        )

        if not results["ids"]:
            logger.warning(f"Actor solution not found: {task_type.value}/{target}/{iteration}")
            return False

        doc_id = results["ids"][0]
        current_metadata = results["metadatas"][0]
        current_metadata["accepted"] = accepted

        self.collection.update(ids=[doc_id], metadatas=[current_metadata])

        logger.debug(f"Updated actor solution {doc_id} status: accepted={accepted}")
        return True

    def search(
        self,
        task_type: TaskType,
        target: str,
        query_text: Optional[str] = None,
        limit: int = 3,
    ) -> list[ActorMemoryContext]:
        if query_text:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=limit,
                where={
                    "$and": [
                        {"task_type": task_type.value},
                        {"target": target},
                    ]
                },
            )
            ids = results["ids"][0] if results["ids"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            documents = results["documents"][0] if results["documents"] else []
        else:
            results = self.collection.get(
                where={
                    "$and": [
                        {"task_type": task_type.value},
                        {"target": target},
                    ]
                },
                limit=limit * 10,
            )
            ids = results["ids"]
            metadatas = results["metadatas"]
            documents = results["documents"]

        from scald.memory.types import ActorMemoryContext

        contexts = []
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i]
            document = documents[i]

            context = ActorMemoryContext(
                id=doc_id,
                task_type=metadata["task_type"],
                target=metadata["target"],
                iteration=metadata["iteration"],
                accepted=metadata["accepted"],
                timestamp=metadata["timestamp"],
                metrics=json.loads(metadata["metrics"]),
                report=document,
            )
            contexts.append(context)

        contexts = self._sort_by_relevance(contexts, limit)

        logger.debug(f"Retrieved {len(contexts)} actor contexts")
        return contexts

    def _create_document_text(
        self, solution: ActorSolution, task_type: TaskType, target: str
    ) -> str:
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in solution.metrics.items()])
        return f"""Task: {task_type.value}
Target: {target}
Metrics: {metrics_str}
Report: {solution.report}"""

    def _sort_by_relevance(
        self, contexts: list[ActorMemoryContext], limit: int
    ) -> list[ActorMemoryContext]:
        sorted_contexts = sorted(
            contexts,
            key=lambda x: (x.accepted, x.iteration),
            reverse=True,
        )
        return sorted_contexts[:limit]
