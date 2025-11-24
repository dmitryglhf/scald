import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import numpy as np
import optuna
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from fastmcp import Context, FastMCP
from lightgbm import LGBMClassifier, LGBMRegressor
from pydantic import Field
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from scald.common.workspace import ACTOR_WORKSPACE

DESCRIPTION = """
Machine learning MCP server.

Available tools:
- Train CatBoost model
- Train LightGBM model
- Train XGBoost model
- Ensemble predictions with Optuna optimization

Features:
- Support for classification and regression
- Automatic metric calculation
- Automatic model and prediction saving
- Optuna-based ensemble weighting
"""

mcp = FastMCP("machine-learning", instructions=DESCRIPTION)


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict[str, float]:
    metrics = {}

    if task_type == "classification":
        y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred_binary))
        metrics["f1"] = float(f1_score(y_true, y_pred_binary, average="weighted"))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        except Exception:
            pass
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    return metrics


def _generate_model_path(model_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]

    extensions = {
        "catboost": ".cbm",
        "lightgbm": ".pkl",
        "xgboost": ".pkl",
    }
    ext = extensions.get(model_type, ".pkl")

    workspace_dir = ACTOR_WORKSPACE / "workspace"
    filename = f"model_{model_type}_{timestamp}_{unique_id}{ext}"
    return str(workspace_dir / filename)


def _generate_predictions_path(model_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]

    workspace_dir = ACTOR_WORKSPACE / "workspace"
    filename = f"predictions_{model_type}_{timestamp}_{unique_id}.csv"
    return str(workspace_dir / filename)


async def _train_model_generic(
    model_type: str,
    model_factory: Callable[[], Any],
    model_saver: Callable[[Any, str], None],
    train_path: str,
    test_path: str,
    target_column: str,
    task_type: str,
    ctx: Context,
    hyperparams: dict[str, Any],
) -> dict[str, Any]:
    try:
        train_df = pl.read_csv(Path(train_path))
        X_train = train_df.drop(target_column).to_numpy()
        y_train = train_df[target_column].to_numpy()

        await ctx.info(
            f"Loaded training data: {X_train.shape[0]} rows, {X_train.shape[1]} features"
        )

        model = model_factory()

        hyperparam_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
        await ctx.info(f"Training {model_type} {task_type} model ({hyperparam_str})...")
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_metrics = _calculate_metrics(y_train, train_pred, task_type)

        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
        await ctx.info(f"Training complete. Train metrics: {metrics_str}")

        model_path = _generate_model_path(model_type)
        model_saver(model, model_path)
        await ctx.info(f"Saved model to {model_path}")

        test_df = pl.read_csv(Path(test_path))
        has_target = target_column in test_df.columns

        result: dict[str, Any] = {
            "success": True,
            "model_type": model_type,
            "model_path": model_path,
            "train_metrics": train_metrics,
        }

        if has_target:
            await ctx.info(
                f"Loaded test data with labels: {test_df.shape[0]} rows (validation mode)"
            )
            X_test = test_df.drop(target_column).to_numpy()
            y_test = test_df[target_column].to_numpy()
            test_pred = model.predict(X_test)
            test_metrics = _calculate_metrics(y_test, test_pred, task_type)
            result["test_metrics"] = test_metrics

            test_metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
            await ctx.info(f"Test metrics: {test_metrics_str}")
        else:
            await ctx.info(
                f"Loaded test data without labels: {test_df.shape[0]} rows (production mode)"
            )
            X_test = test_df.to_numpy()
            test_pred = model.predict(X_test)

        predictions_path = _generate_predictions_path(model_type)
        pred_df = pl.DataFrame({"prediction": test_pred.flatten()})
        pred_df.write_csv(Path(predictions_path))
        result["predictions_path"] = predictions_path
        await ctx.info(f"Saved predictions to {predictions_path}")

        return result

    except Exception as e:
        await ctx.error(f"{model_type} training failed: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool
async def train_catboost(
    train_path: Annotated[str, Field(description="Path to train CSV file")],
    test_path: Annotated[str, Field(description="Path to test CSV file")],
    target_column: Annotated[str, Field(description="Name of the target column in CSV")],
    task_type: Annotated[Literal["classification", "regression"], Field(description="Task type")],
    ctx: Context,
    iterations: Annotated[int, Field(description="Number of boosting iterations")] = 100,
    learning_rate: Annotated[float, Field(description="Learning rate (0.0-1.0)")] = 0.1,
    depth: Annotated[int | None, Field(description="Tree depth (1-16)")] = None,
    l2_leaf_reg: Annotated[float | None, Field(description="L2 regularization coefficient")] = None,
    subsample: Annotated[float | None, Field(description="Row sampling rate (0.0-1.0)")] = None,
    colsample_bylevel: Annotated[
        float | None, Field(description="Column sampling rate per level (0.0-1.0)")
    ] = None,
) -> dict[str, Any]:
    """Train CatBoost model and generate predictions. Always returns model_path and predictions_path."""

    def model_factory():
        params: dict[str, Any] = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "verbose": False,
            "random_seed": 42,
        }

        if depth is not None:
            params["depth"] = depth
        if l2_leaf_reg is not None:
            params["l2_leaf_reg"] = l2_leaf_reg
        if subsample is not None:
            params["subsample"] = subsample
            params["bootstrap_type"] = "Bernoulli"
        if colsample_bylevel is not None:
            params["colsample_bylevel"] = colsample_bylevel

        return (
            CatBoostClassifier(**params)
            if task_type == "classification"
            else CatBoostRegressor(**params)
        )

    def model_saver(model, path):
        model.save_model(path)

    return await _train_model_generic(
        model_type="catboost",
        model_factory=model_factory,
        model_saver=model_saver,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        task_type=task_type,
        ctx=ctx,
        hyperparams={"iterations": iterations, "lr": learning_rate},
    )


@mcp.tool
async def train_lightgbm(
    train_path: Annotated[str, Field(description="Path to train CSV file")],
    test_path: Annotated[str, Field(description="Path to test CSV file")],
    target_column: Annotated[str, Field(description="Name of the target column in CSV")],
    task_type: Annotated[Literal["classification", "regression"], Field(description="Task type")],
    ctx: Context,
    num_iterations: Annotated[int, Field(description="Number of boosting iterations")] = 100,
    learning_rate: Annotated[float, Field(description="Learning rate (0.0-1.0)")] = 0.1,
    max_depth: Annotated[int | None, Field(description="Maximum tree depth")] = None,
    reg_alpha: Annotated[float | None, Field(description="L1 regularization coefficient")] = None,
    reg_lambda: Annotated[float | None, Field(description="L2 regularization coefficient")] = None,
    subsample: Annotated[float | None, Field(description="Row sampling rate (0.0-1.0)")] = None,
    colsample_bytree: Annotated[
        float | None, Field(description="Column sampling rate (0.0-1.0)")
    ] = None,
) -> dict[str, Any]:
    """Train LightGBM model and generate predictions. Always returns model_path and predictions_path."""

    def model_factory():
        params: dict[str, Any] = {
            "num_iterations": num_iterations,
            "learning_rate": learning_rate,
            "verbose": -1,
            "random_state": 42,
        }

        if max_depth is not None:
            params["max_depth"] = max_depth
        if reg_alpha is not None:
            params["reg_alpha"] = reg_alpha
        if reg_lambda is not None:
            params["reg_lambda"] = reg_lambda
        if subsample is not None:
            params["subsample"] = subsample
        if colsample_bytree is not None:
            params["colsample_bytree"] = colsample_bytree

        if task_type == "classification":
            return LGBMClassifier(**params)
        else:
            return LGBMRegressor(**params)

    def model_saver(model, path):
        with open(path, "wb") as f:
            pickle.dump(model, f)

    return await _train_model_generic(
        model_type="lightgbm",
        model_factory=model_factory,
        model_saver=model_saver,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        task_type=task_type,
        ctx=ctx,
        hyperparams={"iterations": num_iterations, "lr": learning_rate},
    )


@mcp.tool
async def train_xgboost(
    train_path: Annotated[str, Field(description="Path to train CSV file")],
    test_path: Annotated[str, Field(description="Path to test CSV file")],
    target_column: Annotated[str, Field(description="Name of the target column in CSV")],
    task_type: Annotated[Literal["classification", "regression"], Field(description="Task type")],
    ctx: Context,
    n_estimators: Annotated[int, Field(description="Number of boosting estimators")] = 100,
    learning_rate: Annotated[float, Field(description="Learning rate (0.0-1.0)")] = 0.1,
    max_depth: Annotated[int | None, Field(description="Maximum tree depth")] = None,
    reg_alpha: Annotated[float | None, Field(description="L1 regularization coefficient")] = None,
    reg_lambda: Annotated[float | None, Field(description="L2 regularization coefficient")] = None,
    subsample: Annotated[float | None, Field(description="Row sampling rate (0.0-1.0)")] = None,
    colsample_bytree: Annotated[
        float | None, Field(description="Column sampling rate (0.0-1.0)")
    ] = None,
) -> dict[str, Any]:
    """Train XGBoost model and generate predictions. Always returns model_path and predictions_path."""

    def model_factory():
        params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": 42,
        }

        if max_depth is not None:
            params["max_depth"] = max_depth
        if reg_alpha is not None:
            params["reg_alpha"] = reg_alpha
        if reg_lambda is not None:
            params["reg_lambda"] = reg_lambda
        if subsample is not None:
            params["subsample"] = subsample
        if colsample_bytree is not None:
            params["colsample_bytree"] = colsample_bytree

        if task_type == "classification":
            return XGBClassifier(**params)
        else:
            return XGBRegressor(**params)

    def model_saver(model, path):
        with open(path, "wb") as f:
            pickle.dump(model, f)

    return await _train_model_generic(
        model_type="xgboost",
        model_factory=model_factory,
        model_saver=model_saver,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        task_type=task_type,
        ctx=ctx,
        hyperparams={"estimators": n_estimators, "lr": learning_rate},
    )


@mcp.tool
async def ensemble_predictions(
    predictions_paths: Annotated[list[str], Field(description="Paths to prediction CSVs")],
    true_labels_path: Annotated[str, Field(description="Path to true labels CSV")],
    target_column: Annotated[str, Field(description="Target column name")],
    task_type: Annotated[Literal["classification", "regression"], Field(description="Task type")],
    ctx: Context,
    n_trials: Annotated[int, Field(description="Optuna trials")] = 100,
) -> dict[str, Any]:
    """Ensemble predictions using Optuna for weight optimization. Returns optimized predictions_path."""
    try:
        await ctx.info(f"Loading {len(predictions_paths)} prediction files for ensemble")

        predictions_list = []
        for i, pred_path in enumerate(predictions_paths):
            pred_df = pl.read_csv(Path(pred_path))
            predictions_list.append(pred_df["prediction"].to_numpy())
            await ctx.info(
                f"Loaded prediction {i + 1}: {len(pred_df)} samples from {Path(pred_path).name}"
            )

        y_true = pl.read_csv(Path(true_labels_path))[target_column].to_numpy()
        await ctx.info(f"Loaded true labels: {len(y_true)} samples")

        await ctx.info(
            f"Optimizing ensemble weights with Optuna ({n_trials} trials, task={task_type})..."
        )

        def objective(trial: optuna.Trial) -> float:
            weights = [
                trial.suggest_float(f"weight_{i}", 0.0, 1.0) for i in range(len(predictions_list))
            ]
            total = sum(weights)
            if total == 0:
                return float("inf")

            weights = [w / total for w in weights]
            ensemble_pred = np.sum([w * pred for w, pred in zip(weights, predictions_list)], axis=0)

            if task_type == "classification":
                ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
                return -float(accuracy_score(y_true, ensemble_pred_binary))
            else:
                return float(mean_squared_error(y_true, ensemble_pred))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_weights = [study.best_params[f"weight_{i}"] for i in range(len(predictions_list))]
        total = sum(best_weights)
        best_weights = [w / total for w in best_weights]

        weights_str = ", ".join([f"model_{i + 1}={w:.4f}" for i, w in enumerate(best_weights)])
        await ctx.info(f"Optimized weights: {weights_str}")

        ensemble_pred = np.sum(
            [w * pred for w, pred in zip(best_weights, predictions_list)], axis=0
        )
        metrics = _calculate_metrics(y_true, ensemble_pred, task_type)

        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        await ctx.info(f"Ensemble metrics: {metrics_str}")

        predictions_path = _generate_predictions_path("ensemble")
        ensemble_df = pl.DataFrame({"prediction": ensemble_pred})
        ensemble_df.write_csv(Path(predictions_path))
        await ctx.info(f"Saved ensemble predictions to {predictions_path}")

        return {
            "success": True,
            "predictions_path": predictions_path,
            "weights": best_weights,
            "metrics": metrics,
        }

    except Exception as e:
        await ctx.error(f"Ensemble predictions failed: {str(e)}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
