from typing import Type

from pydantic import BaseModel

from scald.agents.base import BaseAgent
from scald.common.types import Evaluation, Solution


class Critic(BaseAgent):
    """Reviewer agent."""

    def _get_system_prompt(self) -> str:
        return """You are an expert ML reviewer.
Evaluate data science solutions critically and provide constructive feedback.
Use sequential thinking to assess quality thoroughly."""

    def _get_output_type(self) -> Type[BaseModel]:
        return Evaluation

    async def evaluate(self, solution: Solution, criteria: dict | None = None) -> Evaluation:
        """Evaluate solution quality."""
        prompt = f"""Evaluate this ML solution using sequential thinking:
- Predictions: {solution.predictions_path}
- Metrics: {solution.metrics}
{f"- Criteria: {criteria}" if criteria else ""}

Assess:
1. Data preprocessing quality
2. Model selection and training
3. Performance metrics
4. Overall approach

Return score: 1 (accept) or 0 (reject with suggestions)"""

        return await self._run_agent(prompt)
