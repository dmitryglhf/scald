import asyncio

from scald.agents.actor import Actor
from scald.common.types import TaskType


async def main():
    actor = Actor()

    csv_path = "examples/data/iris.csv"
    target = "Species"
    task_type = TaskType.CLASSIFICATION

    # Run Actor to solve the task
    solution = await actor.solve_task(
        csv_path=csv_path,
        target=target,
        task_type=task_type,
        feedback=None,
    )

    print(f"Predictions: {solution.predictions_path}")
    print(f"Metrics: {solution.metrics}")


if __name__ == "__main__":
    asyncio.run(main())
