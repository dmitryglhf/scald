from pathlib import Path

import numpy as np
import polars as pl

from scald.agents.actor import Actor
from scald.agents.critic import Critic
from scald.common.logger import get_logger
from scald.common.types import ActorSolution, CriticEvaluation, TaskType
from scald.memory import MemoryManager

logger = get_logger(__name__)


class Scald:
    """Main orchestrator for Actor-Critic ML automation with long-term memory"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.actor = Actor()
        self.critic = Critic()
        self.memory_manager = MemoryManager()

    async def run(
        self,
        train_path: str | Path,
        test_path: str | Path,
        target: str,
        task_type: TaskType,
    ) -> np.ndarray:
        """Execute Actor-Critic loop with long-term memory"""

        # First iteration without memory
        actor_solution = await self.actor.solve_task(
            train_path=train_path,
            test_path=test_path,
            target=target,
            task_type=task_type,
        )

        critic_evaluation = await self.critic.evaluate(actor_solution)

        # Retrieve and populate memory
        await self._retrieve_and_populate_memory(actor_solution, task_type)

        # Save first iteration
        entry_id = await self.memory_manager.save_iteration(
            actor_solution=actor_solution,
            critic_evaluation=critic_evaluation,
            task_type=task_type.value,
            iteration=1,
        )
        logger.info(f"Saved iteration 1 to memory: {entry_id}")

        self._update_agents_memory(actor_solution, critic_evaluation, iteration=1)

        # Check if first iteration was accepted
        if critic_evaluation.score == 1:
            logger.info("Solution accepted on iteration 1")
            return self._extract_predictions(actor_solution)

        feedback = critic_evaluation.feedback

        for iteration in range(2, self.max_iterations + 1):
            logger.info(f"Iteration {iteration}/{self.max_iterations}")

            actor_solution = await self.actor.solve_task(
                train_path=train_path,
                test_path=test_path,
                target=target,
                task_type=task_type,
                feedback=feedback,
            )

            critic_evaluation = await self.critic.evaluate(actor_solution)

            # Save iteration
            entry_id = await self.memory_manager.save_iteration(
                actor_solution=actor_solution,
                critic_evaluation=critic_evaluation,
                task_type=task_type.value,
                iteration=iteration,
            )
            logger.info(f"Saved to memory: {entry_id}")

            self._update_agents_memory(actor_solution, critic_evaluation, iteration)

            if critic_evaluation.score == 1:
                logger.info(f"Solution accepted on iteration {iteration}")
                return self._extract_predictions(actor_solution)

            feedback = critic_evaluation.feedback

        logger.warning(
            f"Max iterations ({self.max_iterations}) reached without acceptance, returning last solution"
        )
        return self._extract_predictions(actor_solution)

    async def _retrieve_and_populate_memory(
        self, actor_solution: ActorSolution, task_type: TaskType
    ) -> None:
        """Retrieve relevant memory and populate agents' memory contexts"""
        actor_memory, critic_memory = await self.memory_manager.retrieve_relevant_context(
            actor_report=actor_solution.report,
            task_type=task_type.value,
            top_k=5,
        )
        self.actor.memory_context = actor_memory
        self.critic.memory_context = critic_memory
        logger.info(f"Retrieved {len(actor_memory)} memory entries")

    def _update_agents_memory(
        self,
        actor_solution: ActorSolution,
        critic_evaluation: CriticEvaluation,
        iteration: int,
    ) -> None:
        """Update agents' memory contexts with current iteration results"""
        raise NotImplementedError

    def _extract_predictions(self, solution: ActorSolution) -> np.ndarray:
        """Extract predictions as numpy array"""
        if solution.predictions:
            return np.array(solution.predictions)

        if solution.predictions_path and solution.predictions_path.exists():
            df = pl.read_csv(solution.predictions_path)
            if "prediction" in df.columns:
                return df["prediction"].to_numpy()

        raise ValueError("No predictions available in ActorSolution")
