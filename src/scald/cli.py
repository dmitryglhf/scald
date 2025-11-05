import argparse
import asyncio
import sys
from pathlib import Path

from scald.common.logger import get_logger
from scald.main import Scald

logger = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scald",
        description="Scald: Scalable Collaborative Agents for Data Science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scald --train data/train.csv --test data/test.csv --target price --task-type regression
  scald --train iris_train.csv --test iris_test.csv --target Species --task-type classification --max-iterations 3
        """,
    )

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Path to training dataset (CSV)",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help="Path to test dataset (CSV)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="ML task type",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum Actor-Critic iterations (default: 5)",
    )

    args = parser.parse_args()

    if not args.train.exists():
        logger.error(f"Train file not found: {args.train}")
        sys.exit(1)
    if not args.test.exists():
        logger.error(f"Test file not found: {args.test}")
        sys.exit(1)

    logger.info("Starting Scald...")
    logger.info(f"Train: {args.train}")
    logger.info(f"Test: {args.test}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Task type: {args.task_type}")
    logger.info(f"Max iterations: {args.max_iterations}")

    try:
        scald = Scald(max_iterations=args.max_iterations)
        predictions = asyncio.run(
            scald.run(
                train_path=args.train,
                test_path=args.test,
                target=args.target,
                task_type=args.task_type,
            )
        )
        logger.info(f"Completed successfully. Predictions shape: {predictions.shape}")
        logger.info("Check workspace/artifacts/ for results")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
