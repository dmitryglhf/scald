from typing import TYPE_CHECKING, Optional, Type

from pydantic import BaseModel, Field
from toon import encode

from scald.agents.actor import ActorSolution
from scald.agents.base import BaseAgent

if TYPE_CHECKING:
    from scald.memory.types import CriticMemoryContext


class CriticEvaluation(BaseModel):
    """Evaluation from Critic."""

    score: int = Field(ge=0, le=1, description="0=reject, 1=accept")
    feedback: str = Field(description="Feedback and suggestions")


class Critic(BaseAgent):
    """Reviewer agent."""

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
        past_evaluations: Optional[list["CriticMemoryContext"]] = None,
    ) -> CriticEvaluation:
        """Evaluate solution quality."""
        sections = [
            "ACTOR'S REPORT:",
            solution.report if solution.report else "No report provided",
            "",
            "RESULTS:",
            f"- Predictions: {solution.predictions_path}",
        ]

        if past_evaluations:
            sections.append(
                f"\nPast evaluations: {encode([e.model_dump() for e in past_evaluations])}"
            )

        prompt = "\n".join(sections)
        return await self._run_agent(prompt)
