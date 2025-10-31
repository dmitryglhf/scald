from unittest.mock import Mock, patch

from pydantic import BaseModel

from scald.common.mixins import CostBreakdown, UsageTrackingMixin


class MockUsage(BaseModel):
    """Mock usage data."""

    input_tokens: int = 0
    output_tokens: int = 0


class TestClass(UsageTrackingMixin):
    """Test class using the mixin."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4"):
        self.model = model
        self._usage: MockUsage | None = None

    def set_usage(self, input_tokens: int, output_tokens: int):
        """Set usage data for testing."""
        self._usage = MockUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class TestCostBreakdown:
    """Tests for CostBreakdown model."""

    def test_cost_breakdown_creation(self):
        """Should create CostBreakdown with all fields."""
        breakdown = CostBreakdown(total_price=0.05, input_price=0.02, output_price=0.03)

        assert breakdown.total_price == 0.05
        assert breakdown.input_price == 0.02
        assert breakdown.output_price == 0.03

    def test_cost_breakdown_zero_costs(self):
        """Should handle zero costs."""
        breakdown = CostBreakdown(total_price=0.0, input_price=0.0, output_price=0.0)

        assert breakdown.total_price == 0.0


class TestUsageTrackingMixin:
    """Tests for UsageTrackingMixin."""

    def test_input_tokens_with_usage(self):
        """Should return input tokens when usage is set."""
        obj = TestClass()
        obj.set_usage(input_tokens=100, output_tokens=50)

        assert obj.input_tokens == 100

    def test_input_tokens_without_usage(self):
        """Should return 0 when no usage data."""
        obj = TestClass()
        assert obj.input_tokens == 0

    def test_output_tokens_with_usage(self):
        """Should return output tokens when usage is set."""
        obj = TestClass()
        obj.set_usage(input_tokens=100, output_tokens=50)

        assert obj.output_tokens == 50

    def test_output_tokens_without_usage(self):
        """Should return 0 when no usage data."""
        obj = TestClass()
        assert obj.output_tokens == 0

    def test_total_tokens(self):
        """Should return sum of input and output tokens."""
        obj = TestClass()
        obj.set_usage(input_tokens=100, output_tokens=50)

        assert obj.total_tokens == 150

    def test_total_tokens_zero(self):
        """Should return 0 when no usage."""
        obj = TestClass()
        assert obj.total_tokens == 0

    def test_cost_with_usage(self):
        """Should calculate cost when usage is set."""
        obj = TestClass(model="anthropic/claude-sonnet-4")
        obj.set_usage(input_tokens=1000, output_tokens=500)

        with patch("scald.common.mixins.calc_price") as mock_calc:
            mock_price = Mock()
            mock_price.input_price = 0.003
            mock_price.output_price = 0.015
            mock_calc.return_value = mock_price

            cost = obj.cost

            assert isinstance(cost, CostBreakdown)
            assert cost.input_price == 0.003
            assert cost.output_price == 0.015
            assert cost.total_price == 0.018

    def test_cost_without_usage(self):
        """Should return zero cost when no usage data."""
        obj = TestClass()

        cost = obj.cost
        assert cost.total_price == 0.0
        assert cost.input_price == 0.0
        assert cost.output_price == 0.0

    def test_cost_without_model_attribute(self):
        """Should return zero cost when model attribute missing."""

        class NoModelClass(UsageTrackingMixin):
            def __init__(self):
                self._usage = MockUsage(input_tokens=100, output_tokens=50)

        obj = NoModelClass()
        cost = obj.cost

        assert cost.total_price == 0.0

    def test_cost_with_price_calculation_error(self):
        """Should return zero cost on calculation error."""
        obj = TestClass(model="invalid/model")
        obj.set_usage(input_tokens=100, output_tokens=50)

        with patch("scald.common.mixins.calc_price", side_effect=Exception("Price error")):
            cost = obj.cost

            assert cost.total_price == 0.0
            assert cost.input_price == 0.0
            assert cost.output_price == 0.0

    def test_cost_parses_model_string(self):
        """Should correctly parse provider and model from model string."""
        obj = TestClass(model="openai/gpt-4")
        obj.set_usage(input_tokens=100, output_tokens=50)

        with patch("scald.common.mixins.calc_price") as mock_calc:
            mock_price = Mock()
            mock_price.input_price = 0.01
            mock_price.output_price = 0.03
            mock_calc.return_value = mock_price

            cost = obj.cost

            # Verify calc_price was called with correct parameters
            mock_calc.assert_called_once()
            call_kwargs = mock_calc.call_args[1]
            assert call_kwargs["model_ref"] == "gpt-4"
            assert call_kwargs["provider_id"] == "openai"

    def test_none_tokens_handled_as_zero(self):
        """Should handle None tokens as zero."""

        class NullTokensUsage(BaseModel):
            input_tokens: int | None = None
            output_tokens: int | None = None

        class TestClassWithNull(UsageTrackingMixin):
            def __init__(self):
                self.model = "test/model"
                self._usage = NullTokensUsage()

        obj = TestClassWithNull()
        assert obj.input_tokens == 0
        assert obj.output_tokens == 0
        assert obj.total_tokens == 0


class TestUsageTrackingIntegration:
    """Integration tests for UsageTrackingMixin."""

    def test_full_usage_tracking_cycle(self):
        """Should track usage through complete cycle."""
        obj = TestClass(model="anthropic/claude-sonnet-4")

        # Initially no usage
        assert obj.total_tokens == 0

        # Set usage
        obj.set_usage(input_tokens=250, output_tokens=150)

        # Verify all properties
        assert obj.input_tokens == 250
        assert obj.output_tokens == 150
        assert obj.total_tokens == 400

        # Cost should be calculated
        with patch("scald.common.mixins.calc_price") as mock_calc:
            mock_price = Mock()
            mock_price.input_price = 0.001
            mock_price.output_price = 0.002
            mock_calc.return_value = mock_price

            cost = obj.cost
            assert cost.total_price > 0
