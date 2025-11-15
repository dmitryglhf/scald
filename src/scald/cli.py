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

    logger.debug(f"CLI arguments parsed | args={vars(args)}")

    if not args.train.exists():
        logger.error(f"Input validation failed | file=train | path={args.train} | error=not_found")
        sys.exit(2)
    if not args.test.exists():
        logger.error(f"Input validation failed | file=test | path={args.test} | error=not_found")
        sys.exit(2)

    logger.info(
        f"Scald CLI started | train={args.train} | test={args.test} | target={args.target} | "
        f"task_type={args.task_type} | max_iterations={args.max_iterations}"
    )

    try:
        scald = Scald(max_iterations=args.max_iterations)
        predictions = asyncio.run(
            scald.run(
                train=args.train,
                test=args.test,
                target=args.target,
                task_type=args.task_type,
            )
        )

        logger.info(
            f"Scald completed successfully | predictions_shape={predictions.shape} | "
            f"predictions_count={len(predictions)} | predictions_dtype={predictions.dtype}"
        )
        logger.info("Results saved to workspace/artifacts/")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user | signal=SIGINT")
        sys.exit(130)
    except Exception as e:
        logger.exception(
            f"Scald execution failed | error_type={type(e).__name__} | "
            f"task_type={args.task_type} | target={args.target}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
