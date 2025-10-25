import asyncio

from scald.agents.actor import Actor
from scald.common.types import TaskType


async def main():
    actor = Actor()

    # Run Actor to solve the task
    solution = await actor.solve_task(
        train_path="examples/data/iris_train.csv",
        test_path="examples/data/iris_test.csv",
        target="Species",
        task_type=TaskType.CLASSIFICATION,
    )

    print(f"Predictions path: {solution.predictions_path}")
    print(f"Predictions count: {len(solution.predictions)}")
    print(f"Metrics: {solution.metrics}")
    print(f"First 10 predictions: {solution.predictions[:10]}")


if __name__ == "__main__":
    asyncio.run(main())
