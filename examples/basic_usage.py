import asyncio

from scald.common.types import TaskType
from scald.main import Scald


async def main():
    scald = Scald(
        max_iterations=5,
    )

    result = await scald.run(
        csv_path="examples/data/iris.csv",
        target="Species",
        task_type=TaskType.CLASSIFICATION,
    )

    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Report: {result.report_path}")
    print(f"Predictions: {result.predictions_path}")


if __name__ == "__main__":
    asyncio.run(main())
