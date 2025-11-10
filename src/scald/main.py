from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from scald.agents.actor import Actor
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
                    saved_pred_path = save_workspace_artifacts(actor_solution)
                    return self._extract_predictions(saved_pred_path)

                feedback = critic_evaluation.feedback

            logger.warning(
                f"Max iterations ({self.max_iterations}) reached without acceptance, returning last solution"
            )
            saved_pred_path = save_workspace_artifacts(actor_solution)
            return self._extract_predictions(saved_pred_path)

        finally:
            cleanup_workspace()

    def _extract_predictions(self, saved_pred_path: Path | None) -> np.ndarray:
        """Extract predictions as numpy array from saved CSV file"""
        try:
            if saved_pred_path and saved_pred_path.exists():
                logger.info(f"Reading predictions from saved CSV: {saved_pred_path}")
                pred_df = pl.read_csv(saved_pred_path)

                if "prediction" in pred_df.columns:
                    predictions_array = pred_df["prediction"].to_numpy()
                else:
                    predictions_array = pred_df[:, 0].to_numpy()

                logger.info(f"Extracted {len(predictions_array)} predictions from CSV file")
                return predictions_array

            raise ValueError(
                "predictions_path not available in saved artifacts. "
                "Actor must return valid predictions_path."
            )

        except Exception as e:
            raise ValueError(f"Failed to extract predictions: {e}") from e
