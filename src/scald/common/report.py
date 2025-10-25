from pathlib import Path
from typing import Optional

from scald.common.types import ActorSolution, CriticEvaluation, TaskType


def generate_report(
    train_path: Path,
    test_path: Path,
    target: str,
    task_type: TaskType,
    max_iterations: int,
    iterations: int,
    evaluations: list[CriticEvaluation],
    solution: Optional[ActorSolution],
    total_time: float,
) -> str:
    config_section = _generate_config_section(
        train_path, test_path, target, task_type, max_iterations
    )
    summary_section = _generate_summary_section(iterations, total_time, evaluations)
    iterations_section = _generate_iterations_section(evaluations)
    solution_section = _generate_solution_section(solution)

    return f"{config_section}\n{summary_section}\n{iterations_section}\n{solution_section}"


def _generate_config_section(
    train_path: Path, test_path: Path, target: str, task_type: TaskType, max_iterations: int
) -> str:
    return f"""# SCALD Report

## Task Configuration
- **Train Dataset**: {train_path}
- **Test Dataset**: {test_path}
- **Target**: {target}
- **Task Type**: {task_type.value}
- **Max Iterations**: {max_iterations}"""


def _generate_summary_section(
    iterations: int, total_time: float, evaluations: list[CriticEvaluation]
) -> str:
    final_status = "✅ Accepted" if evaluations and evaluations[-1].score == 1 else "❌ Rejected"
    return f"""
## Execution Summary
- **Total Iterations**: {iterations}
- **Total Time**: {total_time:.2f}s
- **Final Status**: {final_status}"""


def _generate_iterations_section(evaluations: list[CriticEvaluation]) -> str:
    section = "\n## Iterations\n\n"
    for i, eval in enumerate(evaluations, 1):
        section += f"""### Iteration {i}
- **Score**: {eval.score}
- **Feedback**: {eval.feedback}

"""
    return section


def _generate_solution_section(solution: Optional[ActorSolution]) -> str:
    if not solution:
        return ""

    return f"""## Final Solution
- **Predictions Path**: {solution.predictions_path}
- **Metrics**: {solution.metrics}

### Actor's Report
{solution.report if solution.report else "No report provided"}
"""
