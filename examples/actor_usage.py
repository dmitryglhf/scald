import asyncio

from scald.agents.actor import Actor
from scald.common.types import TaskType


async def main():
    actor = Actor()

    # Run Actor to solve the task
    solution = await actor.solve_task(
        csv_path="examples/data/iris.csv",
        target="Species",
        task_type=TaskType.CLASSIFICATION,
    )

    print(f"Predictions: {solution.predictions_path}")
    print(f"Metrics: {solution.metrics}")


if __name__ == "__main__":
    asyncio.run(main())
