import tempfile
from pathlib import Path

import polars as pl
import pytest
from fastmcp.client import Client

from scald.mcp.servers.machine_learning.server import mcp


@pytest.fixture
def ml_server():
    return mcp


@pytest.fixture
def sample_classification_data():
    train_df = pl.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
            "feature2": [0.5, 1.5, 2.5, 3.5, 4.5] * 20,
            "target": [0, 0, 0, 1, 1] * 20,
        }
    )
    return train_df


class TestCatBoostTraining:
    @pytest.mark.asyncio
    async def test_classification_basic(self, ml_server, sample_classification_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            sample_classification_data.write_csv(train_path)

            async with Client(ml_server) as client:
                result = await client.call_tool(
                    name="train_catboost",
                    arguments={
                        "train_path": str(train_path),
                        "target_column": "target",
                        "task_type": "classification",
                        "iterations": 10,
                    },
                )

                assert result.data["success"] is True
                assert result.data["model_type"] == "catboost"


class TestLightGBMTraining:
    @pytest.mark.asyncio
    async def test_classification_basic(self, ml_server, sample_classification_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            sample_classification_data.write_csv(train_path)

            async with Client(ml_server) as client:
                result = await client.call_tool(
                    name="train_lightgbm",
                    arguments={
                        "train_path": str(train_path),
                        "target_column": "target",
                        "task_type": "classification",
                        "num_iterations": 10,
                    },
                )

                assert result.data["success"] is True


class TestXGBoostTraining:
    @pytest.mark.asyncio
    async def test_classification_basic(self, ml_server, sample_classification_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            sample_classification_data.write_csv(train_path)

            async with Client(ml_server) as client:
                result = await client.call_tool(
                    name="train_xgboost",
                    arguments={
                        "train_path": str(train_path),
                        "target_column": "target",
                        "task_type": "classification",
                        "n_estimators": 10,
                    },
                )

                assert result.data["success"] is True
