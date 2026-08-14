"""Pricing must be exact, positive and derived from the model actually used."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.services.llm.models import (
    CREDIT_PRECISION,
    PROFIT_MARGIN,
    USD_TO_CREDITS_RATE,
    get_spec,
    price_in_credits,
)
from app.services.llm.usage import Usage
from app.services.media.image_openai import UnsupportedImageOptionError, image_service
from app.services.media.tts_google import tts_service


class TestTokenPricing:
    def test_cost_matches_the_published_rate(self):
        usage = Usage(prompt_tokens=1_000_000, completion_tokens=0)
        spec = get_spec("gpt-5.2")
        expected = spec.input_price * PROFIT_MARGIN * USD_TO_CREDITS_RATE
        assert price_in_credits(usage, "gpt-5.2") == expected.quantize(CREDIT_PRECISION)

    def test_output_tokens_are_priced_higher_than_input(self):
        input_only = price_in_credits(Usage(prompt_tokens=10_000), "gpt-5.2")
        output_only = price_in_credits(Usage(completion_tokens=10_000), "gpt-5.2")
        assert output_only > input_only

    def test_zero_usage_costs_nothing(self):
        assert price_in_credits(Usage(), "gpt-5.2") == Decimal("0")

    def test_result_is_decimal_at_ledger_precision(self):
        cost = price_in_credits(Usage(prompt_tokens=1234, completion_tokens=567), "claude-4.5-opus")
        assert isinstance(cost, Decimal)
        assert -cost.as_tuple().exponent <= 6

    def test_each_model_is_priced_with_its_own_rates(self):
        """A cheap model must never be billed at an expensive model's rate."""
        usage = Usage(prompt_tokens=100_000, completion_tokens=100_000)
        assert price_in_credits(usage, "gpt-5-mini") < price_in_credits(usage, "gpt-5.2-pro")
        assert price_in_credits(usage, "claude-4.5-haiku") < price_in_credits(usage, "claude-4.5-opus")


class TestUsageAccounting:
    def test_none_from_a_provider_is_ignored(self):
        """Streaming chunks report None before the final usage chunk arrives."""
        usage = Usage(prompt_tokens=50, completion_tokens=10)
        usage.record(None, None)
        assert usage.total_tokens == 60

    def test_reported_counts_replace_the_estimate(self):
        usage = Usage()
        usage.record(120, 80)
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (120, 80, 200)

    def test_missing_usage_falls_back_to_an_estimate(self):
        usage = Usage()
        usage.ensure_validity(prompt_text="a" * 400, completion_text="b" * 800)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 200

    def test_reported_usage_is_not_overwritten_by_the_estimate(self):
        usage = Usage(prompt_tokens=7, completion_tokens=9)
        usage.ensure_validity(prompt_text="a" * 4000, completion_text="b" * 4000)
        assert usage.total_tokens == 16


class TestMediaPricing:
    def test_speech_cost_scales_with_length(self):
        short = tts_service.calculate_cost("hello")
        long = tts_service.calculate_cost("hello" * 100)
        assert 0 < short < long

    def test_image_quality_tiers_are_ordered(self):
        low = image_service.calculate_cost("gpt-image-1.5", "low", "1024x1024")
        high = image_service.calculate_cost("gpt-image-1.5", "high", "1024x1024")
        assert low < high

    @pytest.mark.parametrize(
        ("model", "quality", "size"),
        [
            ("midjourney", "high", "1024x1024"),
            ("gpt-image-1.5", "ultra", "1024x1024"),
            ("gpt-image-1.5", "high", "4096x4096"),
            # DALL·E 3 does not produce the gpt-image aspect ratios.
            ("dall-e-3", "standard", "1024x1536"),
        ],
    )
    def test_unpriced_combinations_are_rejected(self, model, quality, size):
        """Falling back to an arbitrary price would bill for the wrong thing."""
        with pytest.raises(UnsupportedImageOptionError):
            image_service.calculate_cost(model, quality, size)

    def test_published_options_are_all_priceable(self):
        """Whatever /media/images/options advertises must be chargeable."""
        for model, config in image_service.options().items():
            for quality in config["qualities"]:
                for size in config["sizes"]:
                    # Not every size exists at every quality, but each size must
                    # be valid for at least one quality of that model.
                    assert any(
                        size in image_service.PRICING_TABLE[model][q] for q in config["qualities"]
                    ), f"{model}/{quality}/{size} is advertised but unpriced"
