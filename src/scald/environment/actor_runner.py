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

        # Save solution to output directory
        solution_file = output_dir / "solution.json"
        with open(solution_file, "w") as f:
            json.dump(
                {
                    "predictions_path": str(solution.predictions_path),
                    "metrics": solution.metrics,
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
