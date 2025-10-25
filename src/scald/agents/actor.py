from pathlib import Path
from typing import Type

from pydantic import BaseModel

from scald.agents.base import BaseAgent
from scald.common.types import ActorSolution, TaskType
from scald.mcp.registry import get_server_descriptions


class Actor(BaseAgent):
    """Data scientist agent."""

    def _get_system_prompt(self) -> str:
        mcp_servers_desc = get_server_descriptions()
        return f"""You are an expert data scientist.
Your task is to solve ML problems using the provided tools.
Always follow best practices for data preprocessing and model training.
Available MCP servers: {mcp_servers_desc}

You have access to container-use for creating isolated environments.
Use it to safely experiment with different approaches in parallel.

Workflow:
1. Create container-use environment for your work
2. Use data analysis tools to understand the dataset
3. Encode categorical features and preprocess data
4. Train boosting models (CatBoost, LightGBM, XGBoost)
5. Generate predictions and save results
6. Merge successful work or discard failed experiments
"""

    def _get_output_type(self) -> Type[BaseModel]:
        return ActorSolution

    def _get_mcp_tools(self) -> list[str]:
        return ["container-use", "data_analysis", "data_load", "machine_learning"]

    async def solve_task(
        self, csv_path: Path, target: str, task_type: TaskType, feedback: str | None = None
    ) -> ActorSolution:
        """Solve data science task."""
        prompt = f"""Solve {task_type.value} task:
- CSV: {csv_path}
- Target: {target}
{f"- Previous feedback: {feedback}" if feedback else ""}
"""

        return await self._run_agent(prompt)
