from unittest.mock import Mock, patch

from pydantic import BaseModel

from scald.common.cost import CostBreakdown


class MockUsage(BaseModel):
    """Stand-in for pydantic_ai RunUsage."""

    input_tokens: int | None = 0
    output_tokens: int | None = 0


class TestCostBreakdown:
    """Tests for the CostBreakdown model and its constructors."""

    def test_cost_breakdown_creation(self):
        breakdown = CostBreakdown(total_price=0.05, input_price=0.02, output_price=0.03)
        assert breakdown.total_price == 0.05
        assert breakdown.input_price == 0.02
        assert breakdown.output_price == 0.03

    def test_zero(self):
        breakdown = CostBreakdown.zero()
        assert breakdown.total_price == 0.0
        assert breakdown.input_price == 0.0
        assert breakdown.output_price == 0.0


class TestFromUsage:
    """Tests for CostBreakdown.from_usage."""

    def test_with_usage(self):
        usage = MockUsage(input_tokens=1000, output_tokens=500)
        with patch("scald.common.cost.calc_price") as mock_calc:
            mock_calc.return_value = Mock(input_price=0.003, output_price=0.015)
            cost = CostBreakdown.from_usage(usage, "anthropic/claude-sonnet-4")  # type: ignore[arg-type]

        assert cost.input_price == 0.003
        assert cost.output_price == 0.015
        assert cost.total_price == 0.018

    def test_without_usage(self):
        cost = CostBreakdown.from_usage(None, "openai/gpt-4")
        assert cost.total_price == 0.0
        assert cost.input_price == 0.0
        assert cost.output_price == 0.0

    def test_model_without_slash_returns_zero(self):
        usage = MockUsage(input_tokens=100, output_tokens=50)
        cost = CostBreakdown.from_usage(usage, "no-provider-model")  # type: ignore[arg-type]
        assert cost.total_price == 0.0

    def test_price_calculation_error_returns_zero(self):
        usage = MockUsage(input_tokens=100, output_tokens=50)
        with patch(
            "scald.common.cost.calc_price", side_effect=Exception("Price error")
        ):
            cost = CostBreakdown.from_usage(usage, "invalid/model")  # type: ignore[arg-type]
        assert cost.total_price == 0.0

    def test_parses_provider_and_model(self):
        usage = MockUsage(input_tokens=100, output_tokens=50)
        with patch("scald.common.cost.calc_price") as mock_calc:
            mock_calc.return_value = Mock(input_price=0.01, output_price=0.03)
            _ = CostBreakdown.from_usage(usage, "openai/gpt-4")  # type: ignore[arg-type]

        mock_calc.assert_called_once()
        call_kwargs = mock_calc.call_args[1]
        assert call_kwargs["model_ref"] == "gpt-4"
        assert call_kwargs["provider_id"] == "openai"

    def test_none_tokens_handled_as_zero(self):
        usage = MockUsage(input_tokens=None, output_tokens=None)
        with patch("scald.common.cost.calc_price") as mock_calc:
            mock_calc.return_value = Mock(input_price=0.0, output_price=0.0)
            cost = CostBreakdown.from_usage(usage, "test/model")  # type: ignore[arg-type]
            # Usage built with zeros, not None
            passed_usage = mock_calc.call_args[0][0]
            assert passed_usage.input_tokens == 0
            assert passed_usage.output_tokens == 0
        assert cost.total_price == 0.0
