import time
from pathlib import Path
from typing import Optional

import numpy as np

from scald.agents.critic import Critic
from scald.common.logger import get_logger, save_text
from scald.common.np_parsers import predictions_to_numpy
from scald.common.paths import resolve_csv_path
from scald.common.report import generate_report
from scald.common.types import ActorSolution, CriticEvaluation, TaskType

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
        self.rebuild_docker = rebuild_docker
        self.critic = Critic()

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

            solution = await self._run_actor(train_path, test_path, target, task_type, feedback)
            evaluation = await self._run_critic(solution)
            evaluations.append(evaluation)

            if evaluation.score == 1:
                logger.info("Critic accepted solution!")
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
    ) -> ActorSolution:
        logger.info("Actor solving task...")

        if self.use_docker:
            from scald.environment.docker_runner import DockerRunner

            runner = DockerRunner(rebuild=self.rebuild_docker)
            solution = runner.run_actor(
                train_path=train_path,
                test_path=test_path,
                target=target,
                task_type=task_type,
                feedback=feedback,
            )
        else:
            from scald.agents.actor import Actor

            actor = Actor()
            solution = await actor.solve_task(
                train_path=train_path,
                test_path=test_path,
                target=target,
                task_type=task_type,
                feedback=feedback,
            )

        logger.info(f"Actor completed: {solution}")
        return solution

    async def _run_critic(self, solution: ActorSolution) -> CriticEvaluation:
        logger.info("Critic evaluating solution...")
        evaluation = await self.critic.evaluate(solution)
        logger.info(f"Critic evaluation: score={evaluation.score}, feedback={evaluation.feedback}")
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
