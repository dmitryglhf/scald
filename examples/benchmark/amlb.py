import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import openml
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scald.common.logger import get_logger
from scald.main import Scald

logger = get_logger()


@dataclass
class BenchmarkResult:
    """Results for a single benchmark task."""

    task_name: str
    openml_task_id: int
    dataset_name: str
    n_samples: int
    n_features: int
    n_classes: int

    # Performance metrics
    accuracy: float
    f1_score: float
    roc_auc: float | None

    # Iterations info
    iterations_run: int
    max_iterations: int
    accepted_iteration: int | None

    # Token usage
    actor_input_tokens: int
    actor_output_tokens: int
    actor_total_tokens: int
    critic_input_tokens: int
    critic_output_tokens: int
    critic_total_tokens: int
    total_tokens: int

    # Cost breakdown
    actor_input_cost: float
    actor_output_cost: float
    actor_total_cost: float
    critic_input_cost: float
    critic_output_cost: float
    critic_total_cost: float
    total_cost: float

    # Timing
    runtime_seconds: float

    # Status
    status: Literal["success", "error"]
    error_message: str | None = None


class AutoMLBenchmark:
    """Benchmark runner for AutoML evaluation on OpenML tasks."""

    def __init__(
        self,
        config_path: Path,
        max_iterations: int = 5,
        random_state: int = 42,
    ):
        self.config_path = config_path
        self.max_iterations = max_iterations
        self.random_state = random_state

        # Create timestamped output directory in script location
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_dir = Path(__file__).parent
        self.output_dir = script_dir / "benchmark_results" / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Results will be saved to: {self.output_dir}")

        # Load task configs
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.tasks = config["datasets"]

        logger.info(f"Loaded {len(self.tasks)} tasks from {config_path}")

    async def run_single_task(self, task_config: dict) -> BenchmarkResult:
        """Run benchmark on a single OpenML task."""
        task_name = task_config["name"]
        task_id = task_config["openml_task_id"]

        logger.info(f"Starting task: {task_name} (OpenML task {task_id})")
        start_time = time.time()

        try:
            # Create task-specific directory
            task_dir = self.output_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            # Download and prepare data
            logger.info(f"Downloading OpenML task {task_id}")
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()

            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe",
                target=dataset.default_target_attribute,
            )

            # Get dataset info
            n_samples, n_features = X.shape
            n_classes = len(y.unique()) if hasattr(y, "unique") else len(set(y))

            logger.info(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")

            # Split data: 70% train, 30% test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state, stratify=y
            )

            # Get target column name
            target_col = dataset.default_target_attribute

            # Save to CSV in task directory
            train_df = X_train.copy()
            train_df[target_col] = y_train
            test_df = X_test.copy()  # No target in test (sklearn-compatible)
            val_df = X_test.copy()
            val_df[target_col] = y_test  # For validation

            train_path = task_dir / "train.csv"
            test_path = task_dir / "test.csv"
            val_path = task_dir / "validation.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            val_df.to_csv(val_path, index=False)

            logger.info(f"Saved datasets to {task_dir}")

            scald = Scald(max_iterations=self.max_iterations)

            y_pred = await scald.run(
                train_path=train_path,
                test_path=test_path,
                target=target_col,
                task_type="classification",
            )

            pred_df = pd.DataFrame({"prediction": y_pred, "true_label": y_test})
            pred_df.to_csv(task_dir / "predictions.csv", index=False)

            y_test_str = y_test.astype(str)
            y_pred_str = pd.Series(y_pred).astype(str).values

            accuracy = accuracy_score(y_test_str, y_pred_str)
            f1 = f1_score(y_test_str, y_pred_str, average="weighted", zero_division=0)

            roc_auc = None
            if n_classes == 2:
                try:
                    le = LabelEncoder()
                    # Fit on both to ensure all labels are seen
                    le.fit(list(y_test_str) + list(y_pred_str))
                    y_test_binary = le.transform(y_test_str)
                    y_pred_binary = le.transform(y_pred_str)
                    roc_auc = roc_auc_score(y_test_binary, y_pred_binary)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")

            actor_cost = scald.actor.cost
            critic_cost = scald.critic.cost

            total_tokens = scald.actor.total_tokens + scald.critic.total_tokens
            total_cost = actor_cost.total_price + critic_cost.total_price

            runtime = time.time() - start_time

            iterations_run = self.max_iterations
            accepted_iteration = self.max_iterations

            result = BenchmarkResult(
                task_name=task_name,
                openml_task_id=task_id,
                dataset_name=dataset.name,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                accuracy=accuracy,
                f1_score=f1,
                roc_auc=roc_auc,
                iterations_run=iterations_run,
                max_iterations=self.max_iterations,
                accepted_iteration=accepted_iteration,
                actor_input_tokens=scald.actor.input_tokens,
                actor_output_tokens=scald.actor.output_tokens,
                actor_total_tokens=scald.actor.total_tokens,
                critic_input_tokens=scald.critic.input_tokens,
                critic_output_tokens=scald.critic.output_tokens,
                critic_total_tokens=scald.critic.total_tokens,
                total_tokens=total_tokens,
                actor_input_cost=actor_cost.input_price,
                actor_output_cost=actor_cost.output_price,
                actor_total_cost=actor_cost.total_price,
                critic_input_cost=critic_cost.input_price,
                critic_output_cost=critic_cost.output_price,
                critic_total_cost=critic_cost.total_price,
                total_cost=total_cost,
                runtime_seconds=runtime,
                status="success",
            )

            # Save detailed results to task directory
            result_json = task_dir / "result.json"
            with open(result_json, "w") as f:
                json.dump(result.__dict__, f, indent=2, default=str)

            # Save metrics summary
            metrics_path = task_dir / "metrics.txt"
            with open(metrics_path, "w") as f:
                f.write(f"Task: {task_name}\n")
                f.write(f"Dataset: {dataset.name}\n")
                f.write(f"Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}\n\n")
                f.write("Performance:\n")
                f.write(f"  Accuracy:  {accuracy:.4f}\n")
                f.write(f"  F1-Score:  {f1:.4f}\n")
                if roc_auc:
                    f.write(f"  ROC AUC:   {roc_auc:.4f}\n")
                f.write("\nTokens:\n")
                f.write(
                    f"  Actor:  {scald.actor.total_tokens:,} (input: {scald.actor.input_tokens:,}, output: {scald.actor.output_tokens:,})\n"
                )
                f.write(
                    f"  Critic: {scald.critic.total_tokens:,} (input: {scald.critic.input_tokens:,}, output: {scald.critic.output_tokens:,})\n"
                )
                f.write(f"  Total:  {total_tokens:,}\n")
                f.write("\nCost:\n")
                f.write(
                    f"  Actor:  ${actor_cost.total_price:.6f} (input: ${actor_cost.input_price:.6f}, output: ${actor_cost.output_price:.6f})\n"
                )
                f.write(
                    f"  Critic: ${critic_cost.total_price:.6f} (input: ${critic_cost.input_price:.6f}, output: ${critic_cost.output_price:.6f})\n"
                )
                f.write(f"  Total:  ${total_cost:.6f}\n")
                f.write(f"\nRuntime: {runtime:.2f}s\n")

            logger.info(
                f"✓ {task_name}: Acc={accuracy:.4f}, F1={f1:.4f}, "
                f"Cost=${total_cost:.4f}, Time={runtime:.1f}s"
            )

            return result

        except Exception as e:
            runtime = time.time() - start_time
            logger.error(f"✗ {task_name}: {e}", exc_info=True)

            # Save error info
            task_dir = self.output_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            error_path = task_dir / "error.txt"
            with open(error_path, "w") as f:
                f.write(f"Error running task {task_name}:\n\n")
                f.write(str(e))

            return BenchmarkResult(
                task_name=task_name,
                openml_task_id=task_id,
                dataset_name="",
                n_samples=0,
                n_features=0,
                n_classes=0,
                accuracy=0.0,
                f1_score=0.0,
                roc_auc=None,
                iterations_run=0,
                max_iterations=self.max_iterations,
                accepted_iteration=None,
                actor_input_tokens=0,
                actor_output_tokens=0,
                actor_total_tokens=0,
                critic_input_tokens=0,
                critic_output_tokens=0,
                critic_total_tokens=0,
                total_tokens=0,
                actor_input_cost=0.0,
                actor_output_cost=0.0,
                actor_total_cost=0.0,
                critic_input_cost=0.0,
                critic_output_cost=0.0,
                critic_total_cost=0.0,
                total_cost=0.0,
                runtime_seconds=runtime,
                status="error",
                error_message=str(e),
            )

    async def run_benchmark(self, task_names: list[str] | None = None) -> list[BenchmarkResult]:
        """Run benchmark on selected tasks."""
        tasks_to_run = self.tasks

        if task_names:
            tasks_to_run = [t for t in self.tasks if t["name"] in task_names]
            if not tasks_to_run:
                raise ValueError(f"No tasks found matching: {task_names}")

        logger.info(f"Running benchmark on {len(tasks_to_run)} tasks")
        logger.info(f"Max iterations per task: {self.max_iterations}")

        results = []
        for idx, task in enumerate(tasks_to_run, 1):
            logger.info(f"Task {idx}/{len(tasks_to_run)}: {task['name']}")
            result = await self.run_single_task(task)
            results.append(result)

        return results

    def save_summary(self, results: list[BenchmarkResult]):
        """Save benchmark summary to CSV and generate report."""
        # Save CSV
        df = pd.DataFrame([r.__dict__ for r in results])
        summary_csv = self.output_dir / "benchmark_summary.csv"
        df.to_csv(summary_csv, index=False)
        logger.info(f"Saved summary CSV to {summary_csv}")

        # Generate detailed text report
        report_path = self.output_dir / "benchmark_report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 100 + "\n")
            f.write("SCALD AUTOML BENCHMARK REPORT\n")
            f.write("=" * 100 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max Iterations: {self.max_iterations}\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")

            successful = [r for r in results if r.status == "success"]
            failed = [r for r in results if r.status == "error"]

            f.write(f"Tasks Completed: {len(successful)}/{len(results)}\n")
            f.write(f"Tasks Failed: {len(failed)}\n\n")

            if successful:
                f.write("-" * 100 + "\n")
                f.write("PERFORMANCE METRICS (Successful Tasks)\n")
                f.write("-" * 100 + "\n")
                f.write(
                    f"Mean Accuracy:  {sum(r.accuracy for r in successful) / len(successful):.4f}\n"
                )
                f.write(
                    f"Mean F1-Score:  {sum(r.f1_score for r in successful) / len(successful):.4f}\n"
                )
                f.write(f"Min Accuracy:   {min(r.accuracy for r in successful):.4f}\n")
                f.write(f"Max Accuracy:   {max(r.accuracy for r in successful):.4f}\n\n")

                f.write("-" * 100 + "\n")
                f.write("TOKEN USAGE (Successful Tasks)\n")
                f.write("-" * 100 + "\n")
                f.write(f"Total Tokens:    {sum(r.total_tokens for r in successful):,}\n")
                f.write(f"Actor Tokens:    {sum(r.actor_total_tokens for r in successful):,}\n")
                f.write(f"Critic Tokens:   {sum(r.critic_total_tokens for r in successful):,}\n")
                f.write(
                    f"Avg Tokens/Task: {sum(r.total_tokens for r in successful) / len(successful):,.0f}\n\n"
                )

                f.write("-" * 100 + "\n")
                f.write("COST BREAKDOWN (Successful Tasks)\n")
                f.write("-" * 100 + "\n")
                f.write(f"Total Cost:      ${sum(r.total_cost for r in successful):.6f}\n")
                f.write(f"Actor Cost:      ${sum(r.actor_total_cost for r in successful):.6f}\n")
                f.write(f"Critic Cost:     ${sum(r.critic_total_cost for r in successful):.6f}\n")
                f.write(
                    f"Avg Cost/Task:   ${sum(r.total_cost for r in successful) / len(successful):.6f}\n"
                )
                f.write(f"Min Cost/Task:   ${min(r.total_cost for r in successful):.6f}\n")
                f.write(f"Max Cost/Task:   ${max(r.total_cost for r in successful):.6f}\n\n")

                f.write("-" * 100 + "\n")
                f.write("TIMING (Successful Tasks)\n")
                f.write("-" * 100 + "\n")
                total_time = sum(r.runtime_seconds for r in successful)
                f.write(f"Total Runtime:    {total_time:.1f}s ({total_time / 60:.1f}m)\n")
                f.write(f"Avg Runtime/Task: {total_time / len(successful):.1f}s\n")
                f.write(f"Min Runtime/Task: {min(r.runtime_seconds for r in successful):.1f}s\n")
                f.write(f"Max Runtime/Task: {max(r.runtime_seconds for r in successful):.1f}s\n\n")

                f.write("-" * 100 + "\n")
                f.write("PER-TASK RESULTS\n")
                f.write("-" * 100 + "\n")
                f.write(
                    f"{'Task':<25} {'Accuracy':>10} {'F1-Score':>10} {'Cost ($)':>12} {'Time (s)':>10}\n"
                )
                f.write("-" * 100 + "\n")
                for r in successful:
                    f.write(
                        f"{r.task_name:<25} {r.accuracy:>10.4f} {r.f1_score:>10.4f} "
                        f"{r.total_cost:>12.6f} {r.runtime_seconds:>10.1f}\n"
                    )

            if failed:
                f.write("\n" + "-" * 100 + "\n")
                f.write("FAILED TASKS\n")
                f.write("-" * 100 + "\n")
                for r in failed:
                    f.write(f"- {r.task_name}: {r.error_message}\n")

            f.write("\n" + "=" * 100 + "\n")

        logger.info(f"Saved detailed report to {report_path}")

        # Print summary to console
        print("\n" + "=" * 100)
        print("AUTOML BENCHMARK SUMMARY")
        print("=" * 100)

        if successful:
            print(f"\n✓ Completed: {len(successful)}/{len(results)} tasks")
            print("\nPerformance:")
            print(f"  Mean Accuracy: {sum(r.accuracy for r in successful) / len(successful):.4f}")
            print(f"  Mean F1-Score: {sum(r.f1_score for r in successful) / len(successful):.4f}")

            print("\nTokens:")
            print(f"  Total: {sum(r.total_tokens for r in successful):,}")
            print(f"  Avg/Task: {sum(r.total_tokens for r in successful) / len(successful):,.0f}")

            print("\nCost:")
            print(f"  Total: ${sum(r.total_cost for r in successful):.4f}")
            print(f"  Avg/Task: ${sum(r.total_cost for r in successful) / len(successful):.4f}")

            print("\nTiming:")
            total_time = sum(r.runtime_seconds for r in successful)
            print(f"  Total: {total_time:.1f}s ({total_time / 60:.1f}m)")
            print(f"  Avg/Task: {total_time / len(successful):.1f}s")

        if failed:
            print(f"\n✗ Failed: {len(failed)} tasks")
            print(f"  {', '.join(r.task_name for r in failed)}")

        print(f"\nResults saved to: {self.output_dir}")
        print("=" * 100 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Run SCALD AutoML benchmark on OpenML tasks with detailed cost tracking"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "amlb_small_set.yaml",
        help="Path to benchmark config YAML (default: amlb_small_set.yaml in script directory)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum iterations for SCALD (default: 5)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific tasks to run (default: all tasks from config)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    print("\nStarting SCALD AutoML Benchmark")
    print(f"Config: {config_path}")
    print(f"Max Iterations: {args.max_iterations}")
    if args.tasks:
        print(f"Tasks: {', '.join(args.tasks)}")
    else:
        print("Tasks: All tasks from config")
    print()

    benchmark = AutoMLBenchmark(
        config_path=config_path,
        max_iterations=args.max_iterations,
        random_state=args.random_state,
    )

    results = await benchmark.run_benchmark(task_names=args.tasks)
    benchmark.save_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
