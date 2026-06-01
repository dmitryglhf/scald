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
from scald.graph import ActorNode, GraphDeps, RunState, solution_graph
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

        deps = GraphDeps(
            actor=self.actor,
            critic=self.critic,
            memory=self.mm,
            train_path=workspace_train,
            test_path=workspace_test,
            target=target,
            task_type=task_type,
            max_iterations=self.max_iterations,
            acceptance_threshold=self.acceptance_threshold,
        )
        state = RunState()

        try:
            result = await solution_graph.run(ActorNode(), state=state, deps=deps)
            actor_solution = result.output

            logger.info(
                f"Scald run finished | iterations={state.iteration} | "
                f"score_history={[round(s, 3) for s in state.score_history]} | "
                f"total_duration_sec={time.time() - run_start_time:.2f}"
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
                logger.info(
                    f"Extracted {len(predictions_array)} predictions from CSV file"
                )
                return predictions_array

            raise ValueError(
                "predictions_path not available in saved artifacts. "
                "Actor must return valid predictions_path."
            )

        except Exception as e:
            raise ValueError(f"Failed to extract predictions: {e}") from e
