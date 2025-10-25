import pickle
from pathlib import Path
from typing import Annotated, Any, Optional

import numpy as np
import optuna
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor  # type: ignore

from scald.common.logger import get_logger

logger = get_logger()

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
- Model serialization
- Optuna-based ensemble weighting
"""

mcp = FastMCP("machine-learning", instructions=DESCRIPTION)


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict[str, float]:
    """Calculate metrics based on task type."""
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


@mcp.tool(description="Train CatBoost model.")
async def train_catboost(
    train_path: Annotated[str, Field(description="Path to train CSV")],
    target_column: Annotated[str, Field(description="Target column name")],
    task_type: Annotated[str, Field(description="'classification' or 'regression'")],
    test_path: Annotated[Optional[str], Field(description="Path to test CSV")] = None,
    model_path: Annotated[Optional[str], Field(description="Path to save model")] = None,
    predictions_path: Annotated[
        Optional[str], Field(description="Path to save test predictions CSV")
    ] = None,
    iterations: Annotated[int, Field(description="Number of iterations")] = 100,
    learning_rate: Annotated[float, Field(description="Learning rate")] = 0.1,
) -> dict:
    """Train CatBoost model."""
    logger.info(f"[MCP:machine_learning] train_catboost: {task_type}")
    try:
        train_df = pl.read_csv(Path(train_path))
        X_train = train_df.drop(target_column).to_numpy()
        y_train = train_df[target_column].to_numpy()

        params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "verbose": False,
            "random_seed": 42,
        }

        if task_type == "classification":
            model = CatBoostClassifier(**params)
        else:
            model = CatBoostRegressor(**params)

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_metrics = _calculate_metrics(y_train, train_pred, task_type)

        result: dict[str, Any] = {
            "success": True,
            "model_type": "catboost",
            "train_metrics": train_metrics,
        }

        if test_path:
            test_df = pl.read_csv(Path(test_path))
            X_test = test_df.drop(target_column).to_numpy()
            y_test = test_df[target_column].to_numpy()
            test_pred = model.predict(X_test)
            test_metrics = _calculate_metrics(y_test, test_pred, task_type)
            result["test_metrics"] = test_metrics

            # Save predictions if path provided
            if predictions_path:
                pred_df = pl.DataFrame({"prediction": test_pred})
                pred_df.write_csv(Path(predictions_path))
                result["predictions_path"] = predictions_path

        if model_path:
            model.save_model(model_path)
            result["model_path"] = model_path

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Train LightGBM model.")
async def train_lightgbm(
    train_path: Annotated[str, Field(description="Path to train CSV")],
    target_column: Annotated[str, Field(description="Target column name")],
    task_type: Annotated[str, Field(description="'classification' or 'regression'")],
    test_path: Annotated[Optional[str], Field(description="Path to test CSV")] = None,
    model_path: Annotated[Optional[str], Field(description="Path to save model")] = None,
    predictions_path: Annotated[
        Optional[str], Field(description="Path to save test predictions CSV")
    ] = None,
    num_iterations: Annotated[int, Field(description="Number of iterations")] = 100,
    learning_rate: Annotated[float, Field(description="Learning rate")] = 0.1,
) -> dict:
    """Train LightGBM model."""
    logger.info(f"[MCP:machine_learning] train_lightgbm: {task_type}")
    try:
        train_df = pl.read_csv(Path(train_path))
        X_train = train_df.drop(target_column).to_numpy()
        y_train = train_df[target_column].to_numpy()

        if task_type == "classification":
            model = LGBMClassifier(
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                objective="binary",
                verbose=-1,
                random_state=42,
            )
        else:
            model = LGBMRegressor(
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                objective="regression",
                verbose=-1,
                random_state=42,
            )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_metrics = _calculate_metrics(y_train, train_pred, task_type)

        result: dict[str, Any] = {
            "success": True,
            "model_type": "lightgbm",
            "train_metrics": train_metrics,
        }

        if test_path:
            test_df = pl.read_csv(Path(test_path))
            X_test = test_df.drop(target_column).to_numpy()
            y_test = test_df[target_column].to_numpy()
            test_pred = model.predict(X_test)
            test_metrics = _calculate_metrics(y_test, test_pred, task_type)
            result["test_metrics"] = test_metrics

            # Save predictions if path provided
            if predictions_path:
                pred_df = pl.DataFrame({"prediction": test_pred})
                pred_df.write_csv(Path(predictions_path))
                result["predictions_path"] = predictions_path

        if model_path:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            result["model_path"] = model_path

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Train XGBoost model.")
async def train_xgboost(
    train_path: Annotated[str, Field(description="Path to train CSV")],
    target_column: Annotated[str, Field(description="Target column name")],
    task_type: Annotated[str, Field(description="'classification' or 'regression'")],
    test_path: Annotated[Optional[str], Field(description="Path to test CSV")] = None,
    model_path: Annotated[Optional[str], Field(description="Path to save model")] = None,
    predictions_path: Annotated[
        Optional[str], Field(description="Path to save test predictions CSV")
    ] = None,
    n_estimators: Annotated[int, Field(description="Number of estimators")] = 100,
    learning_rate: Annotated[float, Field(description="Learning rate")] = 0.1,
) -> dict:
    """Train XGBoost model."""
    logger.info(f"[MCP:machine_learning] train_xgboost: {task_type}")
    try:
        train_df = pl.read_csv(Path(train_path))
        X_train = train_df.drop(target_column).to_numpy()
        y_train = train_df[target_column].to_numpy()

        if task_type == "classification":
            model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                objective="binary:logistic",
                random_state=42,
            )
        else:
            model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                objective="reg:squarederror",
                random_state=42,
            )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_metrics = _calculate_metrics(y_train, train_pred, task_type)

        result: dict[str, Any] = {
            "success": True,
            "model_type": "xgboost",
            "train_metrics": train_metrics,
        }

        if test_path:
            test_df = pl.read_csv(Path(test_path))
            X_test = test_df.drop(target_column).to_numpy()
            y_test = test_df[target_column].to_numpy()
            test_pred = model.predict(X_test)
            test_metrics = _calculate_metrics(y_test, test_pred, task_type)
            result["test_metrics"] = test_metrics

            # Save predictions if path provided
            if predictions_path:
                pred_df = pl.DataFrame({"prediction": test_pred})
                pred_df.write_csv(Path(predictions_path))
                result["predictions_path"] = predictions_path

        if model_path:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            result["model_path"] = model_path

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Ensemble predictions using Optuna for weight optimization.")
async def ensemble_predictions(
    predictions_paths: Annotated[list[str], Field(description="Paths to prediction CSVs")],
    true_labels_path: Annotated[str, Field(description="Path to true labels CSV")],
    target_column: Annotated[str, Field(description="Target column name")],
    task_type: Annotated[str, Field(description="'classification' or 'regression'")],
    output_path: Annotated[
        Optional[str], Field(description="Path to save ensemble predictions")
    ] = None,
    n_trials: Annotated[int, Field(description="Optuna trials")] = 100,
) -> dict:
    """Ensemble multiple predictions with Optuna-optimized weights."""
    try:
        predictions_list = []
        for pred_path in predictions_paths:
            pred_df = pl.read_csv(Path(pred_path))
            predictions_list.append(pred_df["prediction"].to_numpy())

        y_true = pl.read_csv(Path(true_labels_path))[target_column].to_numpy()

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

        ensemble_pred = np.sum(
            [w * pred for w, pred in zip(best_weights, predictions_list)], axis=0
        )
        metrics = _calculate_metrics(y_true, ensemble_pred, task_type)

        result: dict[str, Any] = {
            "success": True,
            "weights": best_weights,
            "metrics": metrics,
        }

        if output_path:
            ensemble_df = pl.DataFrame({"prediction": ensemble_pred})
            ensemble_df.write_csv(Path(output_path))
            result["output_path"] = output_path

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
