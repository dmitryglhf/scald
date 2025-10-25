import time
from pathlib import Path
from typing import Optional

from scald.agents.critic import Critic
from scald.common.logger import get_logger, save_text
from scald.common.paths import resolve_csv_path
from scald.common.types import ActorSolution, CriticEvaluation, FinalResult, TaskType
from scald.environment import run_actor_in_docker

logger = get_logger()


class Scald:
    def __init__(
        self,
        max_iterations: int = 5,
        use_docker: bool = False,
    ):
        self.max_iterations = max_iterations
        self.use_docker = use_docker
        self.critic = Critic()

    async def run(self, csv_path: str | Path, target: str, task_type: TaskType) -> FinalResult:
        """Run Actor-Critic loop to solve data science task."""
        csv_path = resolve_csv_path(csv_path)

        start_time = time.time()
        iterations = 0
        evaluations: list[CriticEvaluation] = []
        solution: Optional[ActorSolution] = None

        logger.info(f"Starting for {task_type.value} task on {csv_path}")
        logger.info(f"Target column: {target}, Max iterations: {self.max_iterations}")

        try:
            feedback = None

            for iteration in range(self.max_iterations):
                iterations += 1
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

                logger.info("Actor solving task...")
                if self.use_docker:
                    solution = run_actor_in_docker(
                        csv_path=csv_path,
                        target=target,
                        task_type=task_type,
                        feedback=feedback,
                    )
                else:
                    from scald.agents.actor import Actor

                    actor = Actor()
                    solution = await actor.solve_task(
                        csv_path=csv_path,
                        target=target,
                        task_type=task_type,
                        feedback=feedback,
                    )
                logger.info(f"Actor completed: {solution}")

                logger.info("Critic evaluating solution...")
                evaluation = await self.critic.evaluate(solution)
                evaluations.append(evaluation)
                logger.info(
                    f"Critic evaluation: score={evaluation.score}, feedback={evaluation.feedback}"
                )

                if evaluation.score == 1:
                    logger.info("Critic accepted solution!")
                    break

                feedback = evaluation.feedback
                logger.info(f"Refining based on feedback: {feedback}")

            total_time = time.time() - start_time

            report = self._generate_report(
                csv_path=csv_path,
                target=target,
                task_type=task_type,
                iterations=iterations,
                evaluations=evaluations,
                solution=solution,
                total_time=total_time,
            )
            report_path = save_text(report, "scald_report.md")

            return FinalResult(
                success=evaluations[-1].score == 1 if evaluations else False,
                solution=solution,
                iterations=iterations,
                report_path=report_path,
                predictions_path=solution.predictions_path if solution else None,
            )

        except Exception as e:
            logger.error(f"SCALD failed: {e}", exc_info=True)

            return FinalResult(
                success=False,
                solution=None,
                iterations=iterations,
                report_path=None,
                predictions_path=None,
            )

    def _generate_report(
        self,
        csv_path: Path,
        target: str,
        task_type: TaskType,
        iterations: int,
        evaluations: list[CriticEvaluation],
        solution: Optional[ActorSolution],
        total_time: float,
    ) -> str:
        """Generate final report."""
        report = f"""# SCALD Report

## Task Configuration
- **Dataset**: {csv_path}
- **Target**: {target}
- **Task Type**: {task_type.value}
- **Max Iterations**: {self.max_iterations}

## Execution Summary
- **Total Iterations**: {iterations}
- **Total Time**: {total_time:.2f}s
- **Final Status**: {"✅ Accepted" if evaluations and evaluations[-1].score == 1 else "❌ Rejected"}

## Iterations

"""
        for i, eval in enumerate(evaluations, 1):
            report += f"""### Iteration {i}
- **Score**: {eval.score}
- **Feedback**: {eval.feedback}

"""

        if solution:
            report += f"""## Final Solution
- **Predictions Path**: {solution.predictions_path}
- **Metrics**: {solution.metrics}
"""

        return report
