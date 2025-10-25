import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from scald.agents.actor import Actor
from scald.common.logger import get_logger
from scald.common.types import ActorSolution, TaskType

logger = get_logger()


def copy_predictions_file(solution: ActorSolution, output_dir: Path) -> Optional[Path]:
    """Copy predictions file to output directory if it exists."""
    if not solution.predictions_path:
        return None

    predictions_source = Path(solution.predictions_path)
    if not predictions_source.exists():
        logger.warning(f"Predictions file not found: {predictions_source}")
        return None

    predictions_output_path = output_dir / predictions_source.name
    shutil.copy2(predictions_source, predictions_output_path)
    logger.info(f"Copied predictions to {predictions_output_path}")
    return predictions_output_path


def save_solution(solution: ActorSolution, output_dir: Path) -> None:
    """Save solution to JSON file in output directory."""
    solution_file = output_dir / "solution.json"
    try:
        solution_data = solution.model_dump(mode="json")
        with open(solution_file, "w", encoding="utf-8") as f:
            json.dump(solution_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved solution to {solution_file}")
    except Exception as e:
        logger.error(f"Failed to save solution: {e}")
        raise


async def run_actor_task(
    csv_path: Path,
    target: str,
    task_type: TaskType,
    output_dir: Path,
    feedback: Optional[str] = None,
) -> ActorSolution:
    """Run Actor to solve task and save results."""
    actor = Actor()
    solution = await actor.solve_task(
        csv_path=csv_path,
        target=target,
        task_type=task_type,
        feedback=feedback,
    )

    # Copy predictions file and update path
    predictions_output_path = copy_predictions_file(solution, output_dir)
    if predictions_output_path:
        solution.predictions_path = predictions_output_path

    # Save solution
    save_solution(solution, output_dir)

    return solution


async def main():
    """Main entrypoint for Actor runner in Docker container."""
    # Read environment variables
    csv_path = Path(os.getenv("CSV_PATH", "/data/train.csv"))
    target = os.getenv("TARGET", "target")
    task_type_str = os.getenv("TASK_TYPE", "classification")
    output_dir = Path(os.getenv("OUTPUT_DIR", "/output"))
    feedback = os.getenv("FEEDBACK")

    task_type = TaskType(task_type_str)

    try:
        await run_actor_task(csv_path, target, task_type, output_dir, feedback)
        logger.info("Actor completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Actor failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
