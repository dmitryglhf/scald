from typing import Optional, Type

from pydantic import BaseModel

from scald.agents.base import BaseAgent
from scald.common.types import ActorSolution, CriticEvaluation
from scald.memory.types import CriticMemoryContext


class Critic(BaseAgent):
    """Reviewer agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_context: list = []  # Will be populated by Scald

    def _get_system_prompt(self) -> str:
        return """You are an expert ML reviewer.
Evaluate data science solutions critically and provide constructive feedback.
Use sequential thinking to assess quality thoroughly.

Assess:
1. Data preprocessing quality (based on Actor's report)
2. Model selection and training approach
3. Performance metrics adequacy
4. Overall methodology and reasoning
5. Completeness of the solution

Return score: 1 (accept) or 0 (reject with detailed suggestions for improvement)"""

    def _get_output_type(self) -> Type[BaseModel]:
        return CriticEvaluation

    def _get_mcp_tools(self) -> list[str]:
        return ["sequential-thinking"]

    async def evaluate(
        self,
        solution: ActorSolution,
        criteria: Optional[dict] = None,
    ) -> CriticEvaluation:
        """Evaluate solution quality."""
        sections = [
            "ACTOR'S REPORT:",
            solution.report if solution.report else "No report provided",
            "",
            "RESULTS:",
            f"- Predictions: {solution.predictions_path}",
            f"- Metrics: {solution.metrics}",
        ]

        if criteria:
            sections.append(f"- Criteria: {criteria}")

        if self.memory_context:
            sections.append("")
            sections.append(self._format_memory_context(self.memory_context))

        prompt = "\n".join(sections)
        return await self._run_agent(prompt)

    def _format_memory_context(self, memory_context: list[CriticMemoryContext]) -> str:
        lines = ["EVALUATION STANDARDS (from previous iterations):"]
        for i, mem in enumerate(memory_context, 1):
            lines.append(f"{i}. Iteration {mem.iteration} (score={mem.score}):")
            lines.append(f"   Feedback: {mem.feedback}")
            lines.append("")
        lines.append("Apply consistent standards when evaluating this solution.")
        return "\n".join(lines)
