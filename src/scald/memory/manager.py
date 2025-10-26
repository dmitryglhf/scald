import uuid
from pathlib import Path

from tinydb import Query, TinyDB
from tinydb.table import Document, Table

from scald.common.logger import get_logger
from scald.common.types import ActorSolution, CriticEvaluation, TaskType

logger = get_logger()


class MemoryManager:
    def __init__(self, persist_path: str = "./scald_memory.json"):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self.db: TinyDB = TinyDB(str(self.persist_path))
        self.actors: Table = self.db.table("actors")
        self.critics: Table = self.db.table("critics")

        logger.info(
            f"MemoryManager initialized: {len(self.actors)} actors, {len(self.critics)} critics"
        )

    def save_actor_solution(
        self,
        solution: ActorSolution,
        task_type: TaskType,
        target: str,
        iteration: int,
        accepted: bool = False,
    ) -> str:
        """Save actor solution and return memory ID."""
        memory_id = str(uuid.uuid4())

        self.actors.insert(
            {
                "id": memory_id,
                "task_type": task_type.value,
                "target": target,
                "iteration": iteration,
                "accepted": accepted,
                "text": self._format_actor_text(solution, task_type, target),
                "metrics": solution.metrics,
            }
        )

        logger.debug(f"Saved actor solution: {task_type.value}/{target}, iter={iteration}")
        return memory_id

    def get_actor_context(self, task_type: TaskType, target: str, limit: int = 3) -> list[Document]:
        """Get top actor solutions with sandwich pattern."""
        Q = Query()
        results = self.actors.search((Q.task_type == task_type.value) & (Q.target == target))

        # Sort: accepted first, then most recent
        sorted_results = sorted(
            results,
            key=lambda x: (x.get("accepted", False), x.get("iteration", 0)),
            reverse=True,
        )[:limit]

        # Sandwich pattern: best at start and end to avoid "lost in the middle"
        if len(sorted_results) >= 3:
            sorted_results = [sorted_results[0], *sorted_results[2:], sorted_results[1]]

        logger.debug(f"Retrieved {len(sorted_results)} actor memories for {task_type.value}")
        return sorted_results

    def update_actor_solution_status(
        self, task_type: TaskType, target: str, iteration: int, accepted: bool
    ) -> bool:
        """Update accepted status for a specific actor solution."""
        Q = Query()
        updated = self.actors.update(
            {"accepted": accepted},
            (Q.task_type == task_type.value) & (Q.target == target) & (Q.iteration == iteration),
        )

        if updated:
            logger.debug(f"Updated actor solution: {task_type.value}/{target}, iter={iteration}")
            return True
        return False

    def save_critic_evaluation(
        self, evaluation: CriticEvaluation, task_type: TaskType, iteration: int
    ) -> str:
        """Save critic evaluation and return memory ID."""
        memory_id = str(uuid.uuid4())

        self.critics.insert(
            {
                "id": memory_id,
                "task_type": task_type.value,
                "iteration": iteration,
                "score": evaluation.score,
                "text": self._format_critic_text(evaluation, task_type),
            }
        )

        logger.debug(f"Saved critic evaluation: {task_type.value}, score={evaluation.score}")
        return memory_id

    def get_critic_context(self, task_type: TaskType, limit: int = 3) -> list[Document]:
        """Get recent critic evaluations for task type."""
        Q = Query()
        results = self.critics.search(Q.task_type == task_type.value)

        # Sort by most recent
        sorted_results = sorted(results, key=lambda x: x.get("iteration", 0), reverse=True)[:limit]

        logger.debug(f"Retrieved {len(sorted_results)} critic memories for {task_type.value}")
        return sorted_results

    def clear_all(self) -> None:
        """Clear all memories (for testing/reset)."""
        self.actors.truncate()
        self.critics.truncate()
        logger.info("Cleared all memories")

    def _format_actor_text(self, solution: ActorSolution, task_type: TaskType, target: str) -> str:
        """Format actor solution as text."""
        report_snippet = solution.report[:300] if solution.report else "No report"
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in solution.metrics.items())
        return f"{task_type.value} on {target}: {report_snippet}. Metrics: {metrics_str}"

    def _format_critic_text(self, evaluation: CriticEvaluation, task_type: TaskType) -> str:
        """Format critic evaluation as text."""
        status = "Accepted" if evaluation.score == 1 else "Rejected"
        feedback_snippet = evaluation.feedback[:200] if evaluation.feedback else "No feedback"
        return f"{task_type.value}: {status}. {feedback_snippet}"
