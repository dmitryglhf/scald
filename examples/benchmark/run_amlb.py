import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

import openml
import polars as pl
import yaml

from scald.common.logger import get_logger
from scald.main import Scald

logger = get_logger(__name__)


class BenchmarkRunner:
    """Runner for OpenML benchmarks."""

    def __init__(
        self,
        max_iterations: int = 5,
        output_dir: Path = Path("benchmark_results"),
        config_path: Path = Path("examples/benchmark/openml_datasets.yaml"),
    ):
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.config_path = config_path
        self.results: list[dict[str, Any]] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def load_datasets_config(self) -> list[dict[str, Any]]:
        """Load dataset configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config["datasets"]

    async def run_single_task(self, task_name: str, task_id: int) -> dict[str, Any]:
        """Run benchmark on a single OpenML task."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting task: {task_name} (OpenML Task ID: {task_id})")
        logger.info(f"{'=' * 60}")

        result = {
            "task_name": task_name,
            "openml_task_id": task_id,
            "status": "failed",
            "error": None,
            "duration_seconds": 0.0,
        }

        start_time = time.time()

        try:
            # Download task from OpenML
            logger.info(f"Downloading OpenML task {task_id}...")
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()

            # Get train/test split
            X, y, _, _ = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

            # Get train/test indices from OpenML task
            train_indices, test_indices = task.get_train_test_split_indices()
            logger.info(f"Split: {len(train_indices)} train, {len(test_indices)} test")

            # Create temporary directory for this task
            task_dir = self.output_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            # Save train and test data
            train_path = task_dir / "train.csv"
            test_path = task_dir / "test.csv"

            X_train = X.iloc[train_indices].copy()
            y_train = y.iloc[train_indices].copy()
            X_test = X.iloc[test_indices].copy()
            y_test = y.iloc[test_indices].copy()

            # Combine features and target
            train_data = X_train.copy()
            train_data[dataset.default_target_attribute] = y_train
            test_data = X_test.copy()
            test_data[dataset.default_target_attribute] = y_test

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

            logger.info(f"Saved train data to {train_path}")
            logger.info(f"Saved test data to {test_path}")

            # Determine task type (classification or regression)
            task_type = "classification" if task.task_type_id == 1 else "regression"
            logger.info(f"Task type: {task_type}")

            # Run Scald
            scald = Scald(max_iterations=self.max_iterations)

            predictions = await scald.run(
                train_path=train_path,
                test_path=test_path,
                target=dataset.default_target_attribute,
                task_type=task_type,
            )

            # Save predictions
            predictions_path = task_dir / "predictions.csv"
            pred_df = pl.DataFrame({"prediction": predictions})
            pred_df.write_csv(predictions_path)
            logger.info(f"Saved predictions to {predictions_path}")

            # Calculate metrics (basic accuracy for classification)
            if task_type == "classification":
                from sklearn.metrics import accuracy_score

                accuracy = accuracy_score(y_test, predictions)
                result["accuracy"] = float(accuracy)
                logger.info(f"Accuracy: {accuracy:.4f}")

            # Update result
            result["status"] = "success"
            result["predictions_path"] = str(predictions_path)
            result["task_type"] = task_type
            result["n_samples"] = len(y_test)
            result["n_features"] = X.shape[1]

        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            result["error"] = str(e)

        finally:
            result["duration_seconds"] = time.time() - start_time
            logger.info(f"Task {task_name} completed in {result['duration_seconds']:.2f}s")

        return result

    async def run_benchmark(self, task_filter: list[str] | None = None):
        """Run benchmark on all or selected tasks."""
        datasets = self.load_datasets_config()

        # Filter datasets if specified
        if task_filter:
            datasets = [d for d in datasets if d["name"] in task_filter]
            logger.info(f"Running benchmark on {len(datasets)} selected tasks")
        else:
            logger.info(f"Running benchmark on all {len(datasets)} tasks")

        # Run all tasks
        for dataset in datasets:
            result = await self.run_single_task(dataset["name"], dataset["openml_task_id"])
            self.results.append(result)

        # Save summary
        self.save_summary()

    def save_summary(self):
        """Save benchmark summary to CSV."""
        if not self.results:
            logger.warning("No results to save")
            return

        summary_path = self.output_dir / "benchmark_summary.csv"
        df = pl.DataFrame(self.results)
        df.write_csv(summary_path)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Benchmark Summary saved to {summary_path}")
        logger.info(f"{'=' * 60}")

        # Print summary statistics
        successful = df.filter(pl.col("status") == "success")
        failed = df.filter(pl.col("status") == "failed")

        logger.info(f"Total tasks: {len(df)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")

        if len(successful) > 0:
            logger.info(f"Average duration: {successful['duration_seconds'].mean():.2f}s")
            if "accuracy" in successful.columns:
                logger.info(f"Average accuracy: {successful['accuracy'].mean():.4f}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run OpenML benchmark for Scald framework")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific task names to run (default: all tasks)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum Actor-Critic iterations (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/benchmark/amlb_small_set.yaml"),
        help="Path to datasets config file",
    )

    args = parser.parse_args()

    # Create and run benchmark
    runner = BenchmarkRunner(
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        config_path=args.config,
    )

    await runner.run_benchmark(task_filter=args.tasks)


if __name__ == "__main__":
    asyncio.run(main())
