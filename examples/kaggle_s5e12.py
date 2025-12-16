import asyncio
from pathlib import Path

from scald.main import Scald


async def main():
    examples_dir = Path(__file__).parent
    data_dir = examples_dir / "data"

    scald = Scald(max_iterations=5)

    await scald.run(
        train=str(data_dir / "s5e12" / "train.csv"),
        test=str(data_dir / "s5e12" / "test.csv"),
        target="diagnosed_diabetes",
        task_type="classification",
    )


if __name__ == "__main__":
    asyncio.run(main())
