from pathlib import Path
from typing import Any, cast

from pydantic_ai import Agent, RunContext
from toon import encode

from scald.agents.base import BaseAgent
from scald.agents.context import CriticContext, TaskContext
from scald.models import ActorSolution, CriticEvaluation, CriticMemoryContext

__all__ = ["Critic", "CriticEvaluation"]

RUBRICS: tuple[tuple[str, str], ...] = (
    (
        "data_analysis",
        "Is the data exploration thorough, with proper analysis of features, "
        "distributions, and quality issues?",
    ),
    (
        "preprocessing",
        "Are preprocessing steps appropriate and well-documented "
        "(missing values, encoding, feature engineering)?",
    ),
    (
        "model_training",
        "Is model selection appropriate for the task type, with clear rationale "
        "and hyperparameter choices?",
    ),
    (
        "results",
        "Are results clearly reported and does the approach follow ML best practices "
        "without data leakage?",
    ),
)


class Critic(BaseAgent[CriticContext]):
    """LLM reviewer that scores an :class:`ActorSolution` without tool access.

    Withholding tools keeps evaluation objective: the critic judges the
    reported solution, it cannot re-run the work itself.
    """

    def __init__(self, acceptance_threshold: float = 0.75, **kwargs: Any):
        if not 0.0 <= acceptance_threshold <= 1.0:
            raise ValueError(
                f"acceptance_threshold must be in [0.0, 1.0], got {acceptance_threshold}"
            )
        self.acceptance_threshold = acceptance_threshold
        super().__init__(**kwargs)

    def _get_system_prompt(self) -> str:
        rubric_lines = "\n".join(f"- {name}: {desc}" for name, desc in RUBRICS)
        return f"""You are an expert data scientist reviewing another agent's ML solution.

You receive a written report split into four sections. Score each rubric on a
continuous [0.0, 1.0] scale where 0.0 is unacceptable and 1.0 is excellent:

{rubric_lines}

OUTPUT REQUIREMENTS - return a CriticEvaluation with:
- rubric_scores: a dict mapping each rubric name above to its [0.0, 1.0] score
- score: the mean of the rubric scores (it will be recomputed, so be consistent)
- feedback: concise, actionable guidance the actor can use to improve on the next
  iteration. Be specific about what is missing or wrong; do not restate what is fine.

Judge only what the report claims and whether the methodology is sound. You have no
tools and must not request to run anything.
"""

    def _get_output_type(self) -> type[CriticEvaluation]:
        return CriticEvaluation

    def _get_mcp_tools(self) -> list[str]:
        return []

    def _register_dynamic_prompts(self, agent: Agent[CriticContext, Any]) -> None:
        @agent.system_prompt
        def task_prompt(ctx: RunContext[CriticContext]) -> str:
            task = ctx.deps.task
            sections = [
                f"Task under review: {task.task_type} | target column: {task.target} "
                f"| iteration: {task.iteration}",
            ]
            if ctx.deps.past_evaluations:
                sections.append(
                    "\nYour past evaluations of similar tasks:\n"
                    + encode([e.model_dump() for e in ctx.deps.past_evaluations])
                )
            return "\n".join(sections)

    async def evaluate(
        self,
        solution: ActorSolution,
        train_path: Path,
        test_path: Path,
        target: str,
        task_type: str,
        iteration: int = 1,
        past_evaluations: list[CriticMemoryContext] | None = None,
    ) -> CriticEvaluation:
        ctx = CriticContext(
            task=TaskContext(
                train_path=Path(train_path),
                test_path=Path(test_path),
                target=target,
                task_type=cast(Any, task_type),
                iteration=iteration,
            ),
            past_evaluations=past_evaluations if past_evaluations is not None else [],
        )

        result = cast(
            CriticEvaluation,
            await self._run_agent(
                f"Evaluate this solution report:\n\n{solution.report}",
                deps=ctx,
            ),
        )

        # Recompute the overall score deterministically from the rubric scores so it
        # cannot drift from the per-rubric judgement (LLM arithmetic is not trusted).
        if result.rubric_scores:
            result.score = sum(result.rubric_scores.values()) / len(
                result.rubric_scores
            )

        return result

    def is_accepted(self, evaluation: CriticEvaluation) -> bool:
        return evaluation.score >= self.acceptance_threshold
