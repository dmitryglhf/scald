import uuid
from pathlib import Path
from typing import Optional

from tinydb import Query, TinyDB
from tinydb.table import Table

from scald.common.logger import get_logger
from scald.common.types import ActorSolution, CriticEvaluation, TaskType

logger = get_logger()


class MemoryManager:
    def __init__(
        self,
        persist_path: str = "./scald_memory.json",
    ):
        self.persist_path = Path(persist_path)
        self.db: TinyDB
        self.actors: Table
        self.critics: Table
        self.generic: Table

        self._initialize_db()

    def _initialize_db(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = TinyDB(str(self.persist_path))
        self.actors = self.db.table("actors")
        self.critics = self.db.table("critics")
        self.generic = self.db.table("generic")

        total_memories = len(self.actors) + len(self.critics) + len(self.generic)
        logger.info(f"MemoryManager initialized: {total_memories} memories loaded")

    def save_actor_solution(
        self,
        solution: ActorSolution,
        task_type: TaskType,
        target: str,
        iteration: int,
        accepted: bool,
    ) -> None:
        text = self._format_actor_text(solution, task_type, target)

        memory = {
            "id": str(uuid.uuid4()),
            "agent": "actor",
            "task_type": task_type.value,
            "target": target,
            "iteration": iteration,
            "accepted": accepted,
            "text": text,
            "metrics": solution.metrics,
        }

        self.actors.insert(memory)
        logger.debug(
            f"Saved actor solution: task={task_type.value}, target={target}, iteration={iteration}"
        )

    def get_actor_context(
        self,
        task_type: TaskType,
        target: str,
        limit: int = 3,
    ) -> list[dict]:
        Q = Query()
        results = self.actors.search((Q.task_type == task_type.value) & (Q.target == target))

        # Sort by: accepted first, then by iteration (most recent first)
        sorted_results = sorted(
            results,
            key=lambda x: (x.get("accepted", False), x.get("iteration", 0)),
            reverse=True,
        )[:limit]

        # Sandwich pattern: best at start and end to avoid "lost in the middle"
        if len(sorted_results) > 2:
            reordered = [
                sorted_results[0],  # Best - at the beginning
                *sorted_results[2:],  # Middle ones
                sorted_results[1],  # Second best - at the end
            ]
            memories = [{"id": r["id"], "text": r["text"], "metadata": r} for r in reordered]
        else:
            memories = [{"id": r["id"], "text": r["text"], "metadata": r} for r in sorted_results]

        logger.debug(f"Retrieved {len(memories)} actor memories for {task_type.value}")
        return memories

    def update_actor_solution_status(
        self,
        task_type: TaskType,
        target: str,
        iteration: int,
        accepted: bool,
    ) -> bool:
        """Update accepted status for a specific actor solution."""
        Q = Query()
        updated = self.actors.update(
            {"accepted": accepted},
            (Q.task_type == task_type.value) & (Q.target == target) & (Q.iteration == iteration),
        )

        if updated:
            logger.debug(
                f"Updated actor solution: task={task_type.value}, target={target}, iteration={iteration}, accepted={accepted}"
            )
            return True
        return False

    def save_critic_evaluation(
        self,
        evaluation: CriticEvaluation,
        solution: ActorSolution,
        task_type: TaskType,
        iteration: int,
    ) -> None:
        text = self._format_critic_text(evaluation, task_type)

        memory = {
            "id": str(uuid.uuid4()),
            "agent": "critic",
            "task_type": task_type.value,
            "iteration": iteration,
            "score": evaluation.score,
            "feedback": evaluation.feedback[:200],
            "text": text,
            "metrics": solution.metrics,
        }

        self.critics.insert(memory)
        logger.debug(
            f"Saved critic evaluation: task={task_type.value}, score={evaluation.score}, iteration={iteration}"
        )

    def get_critic_context(
        self,
        task_type: TaskType,
        limit: int = 3,
    ) -> list[dict]:
        Q = Query()
        results = self.critics.search(Q.task_type == task_type.value)

        # Sort by iteration (most recent first)
        sorted_results = sorted(results, key=lambda x: x.get("iteration", 0), reverse=True)[:limit]

        memories = [{"id": r["id"], "text": r["text"], "metadata": r} for r in sorted_results]

        logger.debug(f"Retrieved {len(memories)} critic memories for {task_type.value}")
        return memories

    def search(
        self,
        filter_metadata: dict,
        limit: int = 5,
    ) -> list[dict]:
        Q = Query()

        # Build query from filter metadata
        query = None
        for key, value in filter_metadata.items():
            condition = Q[key] == value
            query = condition if query is None else query & condition

        # Search across all tables
        all_results = []
        if query:
            all_results.extend(self.actors.search(query))
            all_results.extend(self.critics.search(query))
            all_results.extend(self.generic.search(query))

        memories = [
            {"id": r.get("id"), "text": r.get("text"), "metadata": r} for r in all_results[:limit]
        ]
        return memories

    def add(
        self,
        text: str,
        metadata: dict,
        memory_id: Optional[str] = None,
    ) -> str:
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        memory = {
            "id": memory_id,
            "text": text,
            **metadata,
        }
        self.generic.insert(memory)
        logger.debug(f"Added memory: {memory_id}")
        return memory_id

    def get_all(
        self,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        all_results = []

        if filter_metadata:
            Q = Query()
            query = None
            for key, value in filter_metadata.items():
                condition = Q[key] == value
                query = condition if query is None else query & condition

            if query:
                all_results.extend(self.actors.search(query))
                all_results.extend(self.critics.search(query))
                all_results.extend(self.generic.search(query))
        else:
            all_results.extend(self.actors.all())
            all_results.extend(self.critics.all())
            all_results.extend(self.generic.all())

        memories = [{"id": r.get("id"), "text": r.get("text"), "metadata": r} for r in all_results]
        return memories

    def delete(
        self,
        memory_id: str,
    ) -> bool:
        Q = Query()

        # Try to delete from all tables
        removed_count = 0
        removed_count += len(self.actors.remove(Q.id == memory_id))
        removed_count += len(self.critics.remove(Q.id == memory_id))
        removed_count += len(self.generic.remove(Q.id == memory_id))

        if removed_count > 0:
            logger.debug(f"Deleted memory: {memory_id}")
            return True
        return False

    def clear(
        self,
        filter_metadata: Optional[dict] = None,
    ) -> None:
        if filter_metadata:
            Q = Query()
            query = None
            for key, value in filter_metadata.items():
                condition = Q[key] == value
                query = condition if query is None else query & condition

            if query:
                removed_count = 0
                removed_count += len(self.actors.remove(query))
                removed_count += len(self.critics.remove(query))
                removed_count += len(self.generic.remove(query))
                logger.info(f"Cleared {removed_count} memories with filter")
        else:
            self.actors.truncate()
            self.critics.truncate()
            self.generic.truncate()
            logger.info("Cleared all memories")

    def _format_actor_text(self, solution: ActorSolution, task_type: TaskType, target: str) -> str:
        report_snippet = solution.report[:300] if solution.report else "No report"
        metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in solution.metrics.items()])

        return (
            f"{task_type.value} task on {target} target: {report_snippet}. Metrics: {metrics_str}"
        )

    def _format_critic_text(self, evaluation: CriticEvaluation, task_type: TaskType) -> str:
        status = "Accepted" if evaluation.score == 1 else "Rejected"
        feedback_snippet = evaluation.feedback[:200] if evaluation.feedback else "No feedback"

        return f"{task_type.value} evaluation: {status}. Feedback: {feedback_snippet}"
