from pathlib import Path
from typing import Type

from pydantic import BaseModel

from scald.agents.base import BaseAgent
from scald.common.types import Solution, TaskType


class Actor(BaseAgent):
    """Data scientist agent."""

    def _get_system_prompt(self) -> str:
        return """You are an expert data scientist.
Your task is to solve ML problems using the provided tools.
Always follow best practices for data preprocessing and model training."""

    def _get_output_type(self) -> Type[BaseModel]:
        return Solution

    def _get_mcp_tools(self) -> list[str]:
        return ["data_analysis", "data_load", "machine_learning"]

    async def solve_task(
        self, csv_path: Path, target: str, task_type: TaskType, feedback: str | None = None
    ) -> Solution:
        """Solve data science task."""
        prompt = f"""Solve {task_type.value} task:
- CSV: {csv_path}
- Target: {target}
{f"- Previous feedback: {feedback}" if feedback else ""}

Use available tools to:
1. Analyze feature distributions
2. Encode categorical features
3. Train boosting models
4. Ensemble predictions with Optuna"""

        return await self._run_agent(prompt)
