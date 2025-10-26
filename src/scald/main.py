import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from scald.agents.critic import Critic
from scald.common.logger import get_logger, save_text
from scald.common.np_parsers import predictions_to_numpy
from scald.common.paths import resolve_csv_path
from scald.common.report import generate_report
from scald.common.types import ActorSolution, CriticEvaluation, TaskType
from scald.memory import MemoryManager

if TYPE_CHECKING:
    from tinydb.table import Document

logger = get_logger()


class Scald:
    def __init__(
        self,
        max_iterations: int = 5,
        use_docker: bool = True,
        rebuild_docker: bool = True,
    ):
        self.max_iterations = max_iterations
        self.use_docker = use_docker
        self.critic = Critic()
        self.memory = MemoryManager()

        if self.use_docker:
            from scald.environment.docker_runner import DockerRunner

            self.docker_runner = DockerRunner(rebuild=rebuild_docker)
        else:
            self.docker_runner = None

    async def run(
        self, train_path: str | Path, test_path: str | Path, target: str, task_type: TaskType
    ) -> np.ndarray:
        train_path = resolve_csv_path(train_path)
        test_path = resolve_csv_path(test_path)

        logger.info(f"Starting for {task_type.value} task")
        logger.info(f"Train: {train_path}, Test: {test_path}")
        logger.info(f"Target column: {target}, Max iterations: {self.max_iterations}")

        try:
            start_time = time.time()
            solution, evaluations, iterations = await self._run_actor_critic_loop(
                train_path, test_path, target, task_type
            )
            total_time = time.time() - start_time

            self._save_report(
                train_path,
                test_path,
                target,
                task_type,
                iterations,
                evaluations,
                solution,
                total_time,
            )

            return self._extract_predictions(solution)

        except Exception as e:
            logger.error(f"Run failed: {e}", exc_info=True)
            return np.array([])

    async def _run_actor_critic_loop(
        self, train_path: Path, test_path: Path, target: str, task_type: TaskType
    ) -> tuple[Optional[ActorSolution], list[CriticEvaluation], int]:
        evaluations: list[CriticEvaluation] = []
        solution: Optional[ActorSolution] = None
        feedback = None

        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Get actor memory context
            try:
                actor_context = self.memory.get_actor_context(task_type=task_type, target=target)
                logger.debug(f"Retrieved {len(actor_context)} actor memories")
            except Exception as e:
                logger.warning(f"Failed to retrieve actor memory: {e}")
                actor_context = []

            # Run actor with memory context
            solution = await self._run_actor(
                train_path, test_path, target, task_type, feedback, actor_context
            )

            # Save actor solution (initially not accepted)
            try:
                self.memory.save_actor_solution(
                    solution=solution,
                    task_type=task_type,
                    target=target,
                    iteration=iteration + 1,
                    accepted=False,
                )
            except Exception as e:
                logger.warning(f"Failed to save actor solution to memory: {e}")

            # Get critic memory context
            try:
                critic_context = self.memory.get_critic_context(task_type=task_type)
                logger.debug(f"Retrieved {len(critic_context)} critic memories")
            except Exception as e:
                logger.warning(f"Failed to retrieve critic memory: {e}")
                critic_context = []

            # Run critic with memory context
            evaluation = await self._run_critic(solution, critic_context)

            # Save critic evaluation
            try:
                self.memory.save_critic_evaluation(
                    evaluation=evaluation, task_type=task_type, iteration=iteration + 1
                )
            except Exception as e:
                logger.warning(f"Failed to save critic evaluation to memory: {e}")

            evaluations.append(evaluation)

            if evaluation.score == 1:
                logger.info("Critic accepted solution")

                # Update actor solution status to accepted
                try:
                    self.memory.update_actor_solution_status(
                        task_type=task_type, target=target, iteration=iteration + 1, accepted=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to update actor solution status: {e}")
                break

            feedback = evaluation.feedback

        return solution, evaluations, iteration + 1

    async def _run_actor(
        self,
        train_path: Path,
        test_path: Path,
        target: str,
        task_type: TaskType,
        feedback: Optional[str],
        memory_context: list[Document],
    ) -> ActorSolution:
        logger.info("Actor solving task...")

        if self.docker_runner:
            solution = self.docker_runner.run_actor(
                train_path=train_path,
                test_path=test_path,
                target=target,
                task_type=task_type,
                feedback=feedback,
                memory_context=memory_context,
            )
        else:
            from scald.agents.actor import Actor

            actor = Actor()
            solution: ActorSolution = await actor.solve_task(
                train_path=train_path,
                test_path=test_path,
                target=target,
                task_type=task_type,
                feedback=feedback,
                memory_context=memory_context,
            )

        logger.info(f"Actor completed: {solution.metrics}")
        return solution

    async def _run_critic(
        self, solution: ActorSolution, memory_context: list[Document]
    ) -> CriticEvaluation:
        logger.info("Critic evaluating solution...")
        evaluation = await self.critic.evaluate(solution, memory_context=memory_context)
        logger.info(f"Critic evaluation: score={evaluation.score}")
        return evaluation

    def _save_report(
        self,
        train_path: Path,
        test_path: Path,
        target: str,
        task_type: TaskType,
        iterations: int,
        evaluations: list[CriticEvaluation],
        solution: Optional[ActorSolution],
        total_time: float,
    ) -> None:
        report = generate_report(
            train_path,
            test_path,
            target,
            task_type,
            self.max_iterations,
            iterations,
            evaluations,
            solution,
            total_time,
        )
        report_path = save_text(report, "scald_report.md")
        logger.info(f"Report saved to: {report_path}")

    def _extract_predictions(self, solution: Optional[ActorSolution]) -> np.ndarray:
        if solution and solution.predictions:
            predictions_array = predictions_to_numpy(solution.predictions)
            logger.info(
                f"Returning predictions array: shape={predictions_array.shape}, dtype={predictions_array.dtype}"
            )
            return predictions_array
        else:
            logger.warning("No predictions found in solution, returning empty array")
            return np.array([])
