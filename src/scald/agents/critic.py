from pathlib import Path

from pydantic_evals.evaluators import EvaluatorContext, LLMJudge
from pydantic_evals.otel._errors import SpanTreeRecordingError

from scald.models import ActorSolution, CriticEvaluation, CriticMemoryContext


class Critic:
    def __init__(self, acceptance_threshold: float = 0.75):
        if not 0.0 <= acceptance_threshold <= 1.0:
            raise ValueError(
                f"acceptance_threshold must be in [0.0, 1.0], got {acceptance_threshold}"
            )
        self.acceptance_threshold = acceptance_threshold
        self.judges = self._create_judges()

    def _create_judges(self) -> list[LLMJudge]:
        return [
            LLMJudge(
                rubric="Evaluate the 'data_analysis' section: Is the data exploration thorough with proper analysis of features, distributions, and quality issues?",
                include_input=True,
                score={"include_reason": True},
                assertion=False,
            ),
            LLMJudge(
                rubric="Evaluate the 'preprocessing' section: Are preprocessing steps appropriate and well-documented (missing values, encoding, feature engineering)?",
                include_input=True,
                score={"include_reason": True},
                assertion=False,
            ),
            LLMJudge(
                rubric="Evaluate the 'model_training' section: Is model selection appropriate for the task type with clear rationale and hyperparameter choices?",
                include_input=True,
                score={"include_reason": True},
                assertion=False,
            ),
            LLMJudge(
                rubric="Evaluate the 'results' section and overall methodology: Are results clearly reported and does the approach follow ML best practices without data leakage?",
                include_input=True,
                score={"include_reason": True},
                assertion=False,
            ),
        ]

    async def evaluate(
        self,
        solution: ActorSolution,
        train_path: Path,
        test_path: Path,
        target: str,
        task_type: str,
        past_evaluations: list[CriticMemoryContext] | None = None,
    ) -> CriticEvaluation:
        ctx = EvaluatorContext(
            name="solution_evaluation",
            inputs={
                "train_path": str(train_path),
                "test_path": str(test_path),
                "target": target,
                "task_type": task_type,
            },
            metadata=None,
            expected_output=None,
            output=solution,
            duration=0.0,
            _span_tree=SpanTreeRecordingError(""),
            attributes={},
            metrics={},
        )

        results = {}
        for judge in self.judges:
            result = await judge.evaluate(ctx)
            if isinstance(result, dict):
                results.update(result)
            else:
                judge_name = judge.rubric[:50]
                results[judge_name] = result

        return self._aggregate_results(results)

    def _aggregate_results(self, results: dict) -> CriticEvaluation:
        scores = []
        feedback_parts = []

        for eval_name, result in results.items():
            if hasattr(result, "value") and isinstance(result.value, (int, float)):
                scores.append(float(result.value))
                feedback_parts.append(f"**{eval_name}**: {result.value:.2f} - {result.reason}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        accept = avg_score >= self.acceptance_threshold

        if accept:
            feedback = f"## Solution Accepted\n\nOverall score: {avg_score:.2f}\n\n" + "\n\n".join(
                feedback_parts
            )
        else:
            feedback = (
                f"## Solution Needs Improvement\n\nOverall score: {avg_score:.2f}\n\n"
                + "\n\n".join(feedback_parts)
            )

        return CriticEvaluation(score=1 if accept else 0, feedback=feedback)
