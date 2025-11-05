import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings, RunUsage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from scald.common.logger import get_logger
from scald.common.mixins import UsageTrackingMixin
from scald.mcp.registry import get_mcp_toolsets

load_dotenv()

logger = get_logger()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-oss:20b")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.3))
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", 120))
DEFAULT_RETRIES = int(os.getenv("DEFAULT_RETRIES", 3))


class BaseAgent(UsageTrackingMixin, ABC):
    """Base class for all agents with common initialization and configuration."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ):
        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"Temperature must be in [0.0, 1.0], got {temperature}")

        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

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

    def _create_agent(self) -> Agent:
        system_prompt = self._get_system_prompt()
        output_type = self._get_output_type()
        mcp_tools = self._get_mcp_tools()

        # Get toolsets if MCP tools are specified
        toolsets = get_mcp_toolsets(mcp_tools) if mcp_tools else []

        return Agent(
            name=self.__class__.__name__,
            model=self._model,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=self.retries,
            instrument=True,
            toolsets=toolsets,
        )

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Returns system prompt for the agent."""
        pass

    @abstractmethod
    def _get_output_type(self) -> Type[BaseModel] | Type[dict] | Type[list]:
        """Returns output type for structured responses."""
        pass

    async def _run_agent(self, prompt: str) -> Any:
        """Run agent with given prompt and return structured output."""
        result = await self.agent.run(prompt)
        self._usage = result.usage()
        return result.output
