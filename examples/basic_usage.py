import asyncio

from scald.common.types import TaskType
from scald.main import Scald


async def main():
    """Example usage of SCALD framework."""

    csv_path = "examples/data/iris.csv"
    target = "Species"
    task_type = TaskType.CLASSIFICATION

    scald = Scald(
        max_iterations=5,
    )

    result = await scald.run(
        csv_path=csv_path,
        target=target,
        task_type=task_type,
    )

    print(f"{'=' * 50}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Report: {result.report_path}")
    print(f"Predictions: {result.predictions_path}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    asyncio.run(main())
