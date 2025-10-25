import asyncio
import json
import os
import sys
from pathlib import Path

from scald.agents.actor import Actor
from scald.common.logger import get_logger
from scald.common.types import TaskType

logger = get_logger()


async def main():
    # Read environment variables
    csv_path = Path(os.getenv("CSV_PATH", "/data/train.csv"))
    target = os.getenv("TARGET", "target")
    task_type_str = os.getenv("TASK_TYPE", "classification")
    output_dir = Path(os.getenv("OUTPUT_DIR", "/output"))
    feedback = os.getenv("FEEDBACK")

    task_type = TaskType(task_type_str)

    try:
        actor = Actor()
        solution = await actor.solve_task(
            csv_path=csv_path,
            target=target,
            task_type=task_type,
            feedback=feedback,
        )

        # Copy predictions file to output directory if it exists
        predictions_output_path = None
        if solution.predictions_path:
            predictions_source = Path(solution.predictions_path)
            if predictions_source.exists():
                predictions_output_path = output_dir / predictions_source.name
                import shutil

                shutil.copy2(predictions_source, predictions_output_path)
                logger.info(f"Copied predictions to {predictions_output_path}")
            else:
                logger.warning(f"Predictions file not found: {predictions_source}")

        # Save solution to output directory
        solution_file = output_dir / "solution.json"
        with open(solution_file, "w") as f:
            json.dump(
                {
                    "predictions_path": str(predictions_output_path)
                    if predictions_output_path
                    else None,
                    "metrics": solution.metrics,
                    "report": solution.report,
                },
                f,
                indent=2,
            )

        logger.info("Actor completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Actor failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
