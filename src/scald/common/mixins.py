from genai_prices import Usage, calc_price
from pydantic import BaseModel

from scald.common.logger import get_logger

logger = get_logger()


class CostBreakdown(BaseModel):
    """Price of entire execution"""

    total_price: float
    input_price: float
    output_price: float


class UsageTrackingMixin:
    """Mixin providing token usage and cost tracking properties.

    Requires implementing class to have:
    - self._usage: Optional[RunUsage] - usage data from last run
    - self.model: str - model name (e.g., "anthropic/claude-sonnet-4")

    Provides properties:
    - input_tokens: int - total input tokens from last run
    - output_tokens: int - total output tokens from last run
    - total_tokens: int - sum of input and output tokens
    - cost: CostBreakdown - cost breakdown (input_price, output_price, total_price)
    """

    @property
    def input_tokens(self) -> int:
        """Total input tokens from last run."""
        if not hasattr(self, "_usage") or not self._usage:
            return 0
        return self._usage.input_tokens or 0

    @property
    def output_tokens(self) -> int:
        """Total output tokens from last run."""
        if not hasattr(self, "_usage") or not self._usage:
            return 0
        return self._usage.output_tokens or 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) from last run."""
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> CostBreakdown:
        """Cost breakdown for last run."""
        if not hasattr(self, "_usage") or not self._usage:
            return CostBreakdown(input_price=0.0, output_price=0.0, total_price=0.0)

        if not hasattr(self, "model"):
            logger.warning("CostableMixin requires 'model' attribute")
            return CostBreakdown(input_price=0.0, output_price=0.0, total_price=0.0)

        try:
            usage = Usage(
                input_tokens=self._usage.input_tokens or 0,
                output_tokens=self._usage.output_tokens or 0,
            )

            # TODO: Implement model selection logic
            provider_id = self.model.split("/")[0]
            model = self.model.split("/")[1]

            price_data = calc_price(usage, model_ref=model, provider_id=provider_id)

            return CostBreakdown(
                input_price=float(price_data.input_price),
                output_price=float(price_data.output_price),
                total_price=float(price_data.input_price) + float(price_data.output_price),
            )

        except Exception as e:
            logger.debug(f"Could not calculate price for model {self.model}: {e}")
            return CostBreakdown(input_price=0.0, output_price=0.0, total_price=0.0)
