from pathlib import Path
from typing import Literal, Optional

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from scald.agents.actor import Actor, ActorSolution
from scald.agents.critic import Critic
from scald.common.logger import get_logger
from scald.memory import MemoryManager

logger = get_logger(__name__)

TaskType = Literal["classification", "regression"]


class FinalResult(BaseModel):
    """Final result from Orchestrator."""

    success: bool = Field(description="Task completed successfully")
    solution: Optional[ActorSolution] = Field(default=None, description="Final solution")
    iterations: int = Field(description="Actor-Critic iterations")
    report_path: Optional[Path] = Field(default=None, description="Path to report")
    predictions_path: Optional[Path] = Field(default=None, description="Path to predictions")


class Scald:
    """Main orchestrator for Actor-Critic ML automation with long-term memory"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.actor = Actor()
        self.critic = Critic()
        self.memory_manager: MemoryManager = MemoryManager()

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

        # Save first iteration to long-term memory
        entry_id = await self.memory_manager.save(
            actor_solution=actor_solution,
            critic_evaluation=critic_evaluation,
            task_type=task_type,
            iteration=1,
        )
        logger.info(f"Saved iteration 1 to memory: {entry_id}")

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
            entry_id = await self.memory_manager.save(
                actor_solution=actor_solution,
                critic_evaluation=critic_evaluation,
                task_type=task_type,
                iteration=iteration,
            )
            logger.info(f"Saved to memory: {entry_id}")

            if critic_evaluation.score == 1:
                logger.info(f"Solution accepted on iteration {iteration}")
                return self._extract_predictions(actor_solution)

            feedback = critic_evaluation.feedback

        logger.warning(
            f"Max iterations ({self.max_iterations}) reached without acceptance, returning last solution"
        )
        return self._extract_predictions(actor_solution)

    def _extract_predictions(self, solution: ActorSolution) -> np.ndarray:
        """Extract predictions as numpy array"""
        if solution.predictions:
            return np.array(solution.predictions)

        if solution.predictions_path and solution.predictions_path.exists():
            df = pl.read_csv(solution.predictions_path)
            if "prediction" in df.columns:
                return df["prediction"].to_numpy()

        raise ValueError("No predictions available in ActorSolution")
