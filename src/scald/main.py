from pathlib import Path
from typing import Literal

import numpy as np

from scald.agents.actor import Actor, ActorSolution
from scald.agents.critic import Critic
from scald.common.logger import get_logger
from scald.common.workspace import (
    cleanup_workspace,
    copy_datasets_to_workspace,
    save_workspace_artifacts,
)
from scald.memory import MemoryManager

logger = get_logger()

TaskType = Literal["classification", "regression"]


class Scald:
    """Main orchestrator for Actor-Critic ML automation."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.actor = Actor()
        self.critic = Critic()
        self.mm: MemoryManager = MemoryManager()

    async def run(
        self,
        train_path: str | Path,
        test_path: str | Path,
        target: str,
        task_type: TaskType,
    ) -> np.ndarray:
        """Execute Actor-Critic loop with long-term memory."""
        train_path = Path(train_path).expanduser().resolve()
        test_path = Path(test_path).expanduser().resolve()

        workspace_train, workspace_test = copy_datasets_to_workspace(train_path, test_path)

        # Retrieve relevant past experiences from similar tasks
        actor_memory, critic_memory = await self.mm.retrieve(
            actor_report="",  # Empty query - filter only by task_type
            task_type=task_type,
            top_k=5,
        )
        logger.info(f"Retrieved {len(actor_memory)} relevant past experiences")

        feedback = None

        try:
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"Iteration {iteration}/{self.max_iterations}")

                actor_solution = await self.actor.solve_task(
                    train_path=workspace_train,
                    test_path=workspace_test,
                    target=target,
                    task_type=task_type,
                    feedback=feedback,
                    past_experiences=actor_memory,
                )

                critic_evaluation = await self.critic.evaluate(
                    actor_solution,
                    past_evaluations=critic_memory,
                )

                # Save iteration to long-term memory
                entry_id = await self.mm.save(
                    actor_solution=actor_solution,
                    critic_evaluation=critic_evaluation,
                    task_type=task_type,
                    iteration=iteration,
                )
                logger.info(f"Saved iteration {iteration} to memory: {entry_id}")

                if critic_evaluation.score == 1:
                    logger.info(f"Solution accepted on iteration {iteration}")
                    save_workspace_artifacts(actor_solution)
                    return self._extract_predictions(actor_solution)

                feedback = critic_evaluation.feedback

            logger.warning(
                f"Max iterations ({self.max_iterations}) reached without acceptance, returning last solution"
            )
            save_workspace_artifacts(actor_solution)
            return self._extract_predictions(actor_solution)

        finally:
            cleanup_workspace()

    def _extract_predictions(self, solution: ActorSolution) -> np.ndarray:
        """Extract predictions as numpy array"""
        try:
            return np.array(solution.predictions)
        except Exception as e:
            raise ValueError("No predictions available in ActorSolution") from e
