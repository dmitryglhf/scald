import time
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from scald.agents.actor import Actor
from scald.agents.critic import Critic
from scald.common.logger import get_logger
from scald.common.workspace import (
    DatasetInput,
    cleanup_workspace,
    prepare_datasets_for_workspace,
    save_workspace_artifacts,
)
from scald.memory import MemoryManager

logger = get_logger()

TaskType = Literal["classification", "regression"]


class Scald:
    def __init__(self, max_iterations: int = 5, acceptance_threshold: float = 0.75):
        self.max_iterations = max_iterations
        self.acceptance_threshold = acceptance_threshold
        self.actor = Actor()
        self.critic = Critic(acceptance_threshold=acceptance_threshold)
        self.mm: MemoryManager = MemoryManager()

        logger.debug(
            f"Scald initialized | max_iterations={max_iterations} | "
            f"acceptance_threshold={acceptance_threshold}"
        )

    async def run(
        self,
        train: DatasetInput,
        test: DatasetInput,
        target: str,
        task_type: TaskType,
    ) -> np.ndarray:
        run_start_time = time.time()
        logger.info(
            f"Starting Scald run | task_type={task_type} | target={target} | "
            f"max_iterations={self.max_iterations} | acceptance_threshold={self.acceptance_threshold}"
        )

        workspace_train, workspace_test = prepare_datasets_for_workspace(train, test)

        actor_memory: list = []
        critic_memory: list = []
        feedback: str | None = None
        actor_solution = None

        try:
            for iteration in range(1, self.max_iterations + 1):
                iter_start_time = time.time()
                logger.info(
                    f"Iteration {iteration}/{self.max_iterations} started | task_type={task_type}"
                )

                logger.debug(
                    f"Actor solving | iteration={iteration} | has_feedback={feedback is not None} | "
                    f"past_experiences={len(actor_memory)}"
                )
                actor_start = time.time()
                actor_solution = await self.actor.solve_task(
                    train_path=workspace_train,
                    test_path=workspace_test,
                    target=target,
                    task_type=task_type,
                    iteration=iteration,
                    feedback=feedback,
                    past_experiences=actor_memory,
                )
                actor_duration = time.time() - actor_start
                logger.info(
                    f"Actor completed | iteration={iteration} | duration_sec={actor_duration:.2f} | "
                    f"has_predictions={actor_solution.predictions_path is not None}"
                )

                logger.debug(f"Critic evaluating | iteration={iteration}")
                critic_start = time.time()
                critic_evaluation = await self.critic.evaluate(
                    solution=actor_solution,
                    train_path=workspace_train,
                    test_path=workspace_test,
                    target=target,
                    task_type=task_type,
                    past_evaluations=critic_memory,
                )
                critic_duration = time.time() - critic_start
                logger.info(
                    f"Critic completed | iteration={iteration} | duration_sec={critic_duration:.2f} | "
                    f"score={critic_evaluation.score:.3f} | threshold={self.acceptance_threshold}"
                )

                entry_id = self.mm.save(
                    actor_solution=actor_solution,
                    critic_evaluation=critic_evaluation,
                    task_type=task_type,
                    iteration=iteration,
                )
                logger.info(f"Memory saved | iteration={iteration} | entry_id={entry_id}")

                actor_memory, critic_memory = self.mm.retrieve(
                    actor_report=actor_solution.report,
                    task_type=task_type,
                    top_k=5,
                )
                logger.info(
                    f"Memory retrieved | iteration={iteration} | actor_entries={len(actor_memory)} | "
                    f"critic_entries={len(critic_memory)}"
                )

                iter_duration = time.time() - iter_start_time
                accepted = critic_evaluation.score >= self.acceptance_threshold
                logger.info(
                    f"=== Iteration {iteration} completed | duration_sec={iter_duration:.2f} | "
                    f"score={critic_evaluation.score:.3f} | accepted={accepted} ==="
                )

                if accepted:
                    total_duration = time.time() - run_start_time
                    logger.info(
                        f"Solution ACCEPTED | iteration={iteration} | score={critic_evaluation.score:.3f} | "
                        f"threshold={self.acceptance_threshold} | total_duration_sec={total_duration:.2f}"
                    )
                    saved_pred_path = save_workspace_artifacts(actor_solution)
                    return self._extract_predictions(saved_pred_path)

                feedback = critic_evaluation.feedback
                logger.debug(
                    f"Feedback prepared for next iteration | iteration={iteration} | "
                    f"feedback_length={len(feedback)}"
                )

            if actor_solution is None:
                raise ValueError("No iterations executed - this should never happen")

            total_duration = time.time() - run_start_time
            logger.warning(
                f"Max iterations reached | iterations={self.max_iterations} | "
                f"final_score={critic_evaluation.score:.3f} | threshold={self.acceptance_threshold} | "
                f"total_duration_sec={total_duration:.2f} | status=returning_last_solution"
            )
            saved_pred_path = save_workspace_artifacts(actor_solution)
            return self._extract_predictions(saved_pred_path)

        except Exception as e:
            logger.exception(
                f"Scald run failed | task_type={task_type} | target={target} | "
                f"error_type={type(e).__name__}"
            )
            raise
        finally:
            cleanup_start = time.time()
            cleanup_workspace()
            logger.debug(
                f"Workspace cleanup completed | duration_sec={time.time() - cleanup_start:.2f}"
            )

    def _extract_predictions(self, saved_pred_path: Path | None) -> np.ndarray:
        try:
            if saved_pred_path and saved_pred_path.exists():
                logger.info(f"Reading predictions from saved CSV: {saved_pred_path}")
                pred_df = pl.read_csv(saved_pred_path)

                if "prediction" not in pred_df.columns:
                    raise ValueError(
                        f"Predictions CSV must have 'prediction' column, "
                        f"found columns: {pred_df.columns}"
                    )

                predictions_array = pred_df["prediction"].to_numpy()
                logger.info(f"Extracted {len(predictions_array)} predictions from CSV file")
                return predictions_array

            raise ValueError(
                "predictions_path not available in saved artifacts. "
                "Actor must return valid predictions_path."
            )

        except Exception as e:
            raise ValueError(f"Failed to extract predictions: {e}") from e
