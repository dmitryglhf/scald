from __future__ import annotations

from typing import TYPE_CHECKING

from genai_prices import Usage, calc_price
from pydantic import BaseModel

from scald.common.logger import get_logger

if TYPE_CHECKING:
    from pydantic_ai import RunUsage

logger = get_logger()


class CostBreakdown(BaseModel):
    """Price of an execution, derived from token usage."""

    total_price: float
    input_price: float
    output_price: float

    @classmethod
    def zero(cls) -> "CostBreakdown":
        return cls(total_price=0.0, input_price=0.0, output_price=0.0)

    @classmethod
    def from_usage(cls, usage: "RunUsage | None", model: str) -> "CostBreakdown":
        """Compute the cost for ``usage`` under a ``provider/model`` reference.

        Returns a zero breakdown when usage is missing, the model string is not in
        ``provider/model`` form, or pricing lookup fails.
        """
        if usage is None:
            return cls.zero()

        provider_id, _, model_ref = model.partition("/")
        if not model_ref:
            logger.warning(
                f"Model {model!r} is not in 'provider/model' form; cost set to 0"
            )
            return cls.zero()

        try:
            price_data = calc_price(
                Usage(
                    input_tokens=usage.input_tokens or 0,
                    output_tokens=usage.output_tokens or 0,
                ),
                model_ref=model_ref,
                provider_id=provider_id,
            )
            input_price = float(price_data.input_price)
            output_price = float(price_data.output_price)
            return cls(
                input_price=input_price,
                output_price=output_price,
                total_price=input_price + output_price,
            )
        except Exception as e:
            logger.debug(f"Could not calculate price for model {model}: {e}")
            return cls.zero()
