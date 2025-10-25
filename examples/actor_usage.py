import asyncio
from pathlib import Path

from scald.agents.actor import Actor
from scald.common.types import TaskType


async def main():
    """Example of using Actor agent with container-use MCP server."""

    # Initialize Actor agent
    # Actor has access to:
    # - container-use (for isolated environments)
    # - data_analysis (for EDA)
    # - data_load (for reading CSV files)
    # - machine_learning (for training models)
    actor = Actor()

    csv_path = Path("data/train.csv")
    target = "target"
    task_type = TaskType.CLASSIFICATION

    # Run Actor to solve the task
    solution = await actor.solve_task(
        csv_path=csv_path,
        target=target,
        task_type=task_type,
        feedback=None,
    )

    print("\n" + "=" * 60)
    print("Actor Solution")
    print("=" * 60)
    print(f"Predictions: {solution.predictions_path}")
    print(f"Metrics: {solution.metrics}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
