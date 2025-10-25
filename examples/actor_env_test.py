import asyncio

from scald.common.types import TaskType
from scald.main import Scald


async def main():
    # Test with Docker isolation
    scald = Scald(max_iterations=1, use_docker=True)

    result = await scald.run(
        csv_path="examples/data/iris.csv",
        target="Species",
        task_type=TaskType.CLASSIFICATION,
    )

    print(f"\n{'=' * 60}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Report: {result.report_path}")
    print(f"Predictions: {result.predictions_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
