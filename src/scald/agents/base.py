import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings, RunUsage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from scald.common.cost import CostBreakdown
from scald.common.logger import get_logger
from scald.mcp.registry import get_mcp_toolsets

load_dotenv()

logger = get_logger()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-5-mini")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.3))
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", 120))
DEFAULT_RETRIES = int(os.getenv("DEFAULT_RETRIES", 3))

DepsT = TypeVar("DepsT")


class BaseAgent(ABC, Generic[DepsT]):
    """Base class for all agents with common initialization and configuration."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        workspace_dir: "Path | None" = None,
    ):
        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"Temperature must be in [0.0, 1.0], got {temperature}")

        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.workspace_dir = workspace_dir

        self._model = self._create_model()
        self.agent = self._create_agent()
        self._usage: Optional[RunUsage] = None

    def _create_model(self) -> OpenAIChatModel:
        settings = ModelSettings(
            temperature=self.temperature,
            timeout=self.timeout,
        )

        return OpenAIChatModel(
            self.model,
            provider=OpenAIProvider(),
            settings=settings,
        )

    def _get_mcp_tools(self) -> list[str]:
        """Override to specify MCP tools for this agent. Returns empty list by default."""
        return []

    def _create_agent(self) -> Agent[DepsT, Any]:
        system_prompt = self._get_system_prompt()
        output_type = self._get_output_type()
        mcp_tools = self._get_mcp_tools()

        toolsets = (
            get_mcp_toolsets(mcp_tools, workspace_dir=self.workspace_dir)
            if mcp_tools
            else []
        )

        agent: Agent[DepsT, Any] = Agent(
            name=self.__class__.__name__,
            model=self._model,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=self.retries,
            instrument=True,
            toolsets=toolsets,
        )
        self._register_dynamic_prompts(agent)
        return agent

    def _register_dynamic_prompts(self, agent: Agent[DepsT, Any]) -> None:
        """Override to attach deps-aware dynamic system prompts.

        Subclasses register functions taking ``RunContext[DepsT]`` via
        ``agent.system_prompt`` so per-run context (task, feedback, past
        experiences) is injected through dependency injection rather than
        hand-built user-prompt strings.
        """
        return None

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Returns system prompt for the agent."""
        pass

    @abstractmethod
    def _get_output_type(
        self,
    ) -> Type[BaseModel] | Type[dict[str, Any]] | Type[list[Any]]:
        """Returns output type for structured responses."""
        pass

    async def _run_agent(self, prompt: str, deps: DepsT) -> Any:
        result = await self.agent.run(prompt, deps=deps)
        self._usage = result.usage()
        return result.output

    @property
    def input_tokens(self) -> int:
        return self._usage.input_tokens or 0 if self._usage else 0

    @property
    def output_tokens(self) -> int:
        return self._usage.output_tokens or 0 if self._usage else 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> CostBreakdown:
        return CostBreakdown.from_usage(self._usage, self.model)
