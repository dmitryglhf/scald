"""Actor-Critic control flow as a pydantic-graph finite state machine.

The loop is a textbook FSM: the actor proposes a solution, the critic scores it,
and execution either terminates (accepted, or budget exhausted) or loops back with
feedback. Modelling it as a graph keeps the transition logic explicit and the run
state inspectable/persistable, instead of a hand-rolled ``for`` loop.

    ActorNode --> CriticNode --> (ActorNode | End[ActorSolution])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from scald.agents.actor import Actor
from scald.agents.critic import Critic
from scald.common.logger import get_logger
from scald.memory import MemoryManager
from scald.models import ActorMemoryContext, ActorSolution, CriticMemoryContext

logger = get_logger()

TaskType = Literal["classification", "regression"]


@dataclass
class GraphDeps:
    """Immutable collaborators and task configuration for a single run."""

    actor: Actor
    critic: Critic
    memory: MemoryManager
    train_path: Path
    test_path: Path
    target: str
    task_type: TaskType
    max_iterations: int
    acceptance_threshold: float


@dataclass
class RunState:
    """Mutable state threaded through the Actor-Critic iterations."""

    iteration: int = 1
    feedback: str | None = None
    actor_solution: ActorSolution | None = None
    actor_memory: list[ActorMemoryContext] = field(default_factory=list)
    critic_memory: list[CriticMemoryContext] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


@dataclass
class ActorNode(BaseNode[RunState, GraphDeps, ActorSolution]):
    """Actor proposes a solution for the current iteration."""

    async def run(self, ctx: GraphRunContext[RunState, GraphDeps]) -> CriticNode:
        state, deps = ctx.state, ctx.deps
        logger.info(
            f"Iteration {state.iteration}/{deps.max_iterations} started | "
            f"task_type={deps.task_type} | has_feedback={state.feedback is not None} | "
            f"past_experiences={len(state.actor_memory)}"
        )

        started = time.time()
        state.actor_solution = await deps.actor.solve_task(
            train_path=deps.train_path,
            test_path=deps.test_path,
            target=deps.target,
            task_type=deps.task_type,
            iteration=state.iteration,
            feedback=state.feedback,
            past_experiences=state.actor_memory,
        )
        logger.info(
            f"Actor completed | iteration={state.iteration} | "
            f"duration_sec={time.time() - started:.2f} | "
            f"cost_usd={deps.actor.cost.total_price:.4f}"
        )
        return CriticNode()


@dataclass
class CriticNode(BaseNode[RunState, GraphDeps, ActorSolution]):
    """Critic scores the solution, then the FSM decides: accept, loop, or stop."""

    async def run(
        self, ctx: GraphRunContext[RunState, GraphDeps]
    ) -> ActorNode | End[ActorSolution]:
        state, deps = ctx.state, ctx.deps
        assert state.actor_solution is not None  # set by ActorNode

        started = time.time()
        evaluation = await deps.critic.evaluate(
            solution=state.actor_solution,
            train_path=deps.train_path,
            test_path=deps.test_path,
            target=deps.target,
            task_type=deps.task_type,
            iteration=state.iteration,
            past_evaluations=state.critic_memory,
        )
        accepted = evaluation.score >= deps.acceptance_threshold
        state.score_history.append(evaluation.score)
        logger.info(
            f"Critic completed | iteration={state.iteration} | "
            f"duration_sec={time.time() - started:.2f} | score={evaluation.score:.3f} | "
            f"threshold={deps.acceptance_threshold} | accepted={accepted} | "
            f"cost_usd={deps.critic.cost.total_price:.4f}"
        )

        deps.memory.save(
            actor_solution=state.actor_solution,
            critic_evaluation=evaluation,
            task_type=deps.task_type,
            iteration=state.iteration,
            accepted=accepted,
        )
        state.actor_memory, state.critic_memory = deps.memory.retrieve(
            actor_report=state.actor_solution.report,
            task_type=deps.task_type,
            top_k=5,
        )

        if accepted:
            logger.info(
                f"Solution ACCEPTED | iteration={state.iteration} | "
                f"score={evaluation.score:.3f} | "
                f"total_duration_sec={time.time() - state.started_at:.2f}"
            )
            return End(state.actor_solution)

        if state.iteration >= deps.max_iterations:
            logger.warning(
                f"Max iterations reached | iterations={deps.max_iterations} | "
                f"final_score={evaluation.score:.3f} | "
                f"total_duration_sec={time.time() - state.started_at:.2f} | "
                f"status=returning_last_solution"
            )
            return End(state.actor_solution)

        state.iteration += 1
        state.feedback = evaluation.feedback
        return ActorNode()


solution_graph: Graph[RunState, GraphDeps, ActorSolution] = Graph(
    nodes=(ActorNode, CriticNode),
    state_type=RunState,
    run_end_type=ActorSolution,
    name="scald_actor_critic",
)
