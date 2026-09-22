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
    usd_to_credits,
)
from app.services.llm.usage import Usage
from app.services.media.images import UnsupportedImageOptionError, image_service
from app.services.media.tts import InvalidVoiceError, tts_service


class TestTokenPricing:
    def test_cost_matches_the_published_rate(self):
        usage = Usage(prompt_tokens=1_000_000, completion_tokens=0)
        spec = get_spec("gpt-5.5")
        expected = spec.input_price * PROFIT_MARGIN * USD_TO_CREDITS_RATE
        assert price_in_credits(usage, "gpt-5.5") == expected.quantize(CREDIT_PRECISION)

    def test_output_tokens_are_priced_higher_than_input(self):
        input_only = price_in_credits(Usage(prompt_tokens=10_000), "gpt-5.5")
        output_only = price_in_credits(Usage(completion_tokens=10_000), "gpt-5.5")
        assert output_only > input_only

    def test_zero_usage_costs_nothing(self):
        assert price_in_credits(Usage(), "gpt-5.5") == Decimal("0")

    def test_result_is_decimal_at_ledger_precision(self):
        cost = price_in_credits(Usage(prompt_tokens=1234, completion_tokens=567), "claude-opus-5")
        assert isinstance(cost, Decimal)
        assert -cost.as_tuple().exponent <= 6

    def test_each_model_is_priced_with_its_own_rates(self):
        """A cheap model must never be billed at an expensive model's rate."""
        usage = Usage(prompt_tokens=100_000, completion_tokens=100_000)
        assert price_in_credits(usage, "gpt-5.4-nano") < price_in_credits(usage, "gpt-5.5-pro")
        assert price_in_credits(usage, "claude-haiku-4-5") < price_in_credits(
            usage, "claude-opus-5"
        )

    def test_legacy_ids_are_billed_at_their_replacement(self):
        usage = Usage(prompt_tokens=1000, completion_tokens=1000)
        assert price_in_credits(usage, "claude-4.5-opus") == price_in_credits(
            usage, "claude-opus-5"
        )

    def test_usd_conversion_applies_margin(self):
        assert usd_to_credits(Decimal("1")) == (PROFIT_MARGIN * USD_TO_CREDITS_RATE).quantize(
            CREDIT_PRECISION
        )


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


class TestSpeechPricing:
    def test_speech_cost_scales_with_length(self):
        short = tts_service.calculate_cost("hello")
        long = tts_service.calculate_cost("hello" * 100)
        assert 0 < short < long

    def test_premium_voices_cost_more(self):
        neural = tts_service.calculate_cost("hello world", "en-US-Neural2-F")
        studio = tts_service.calculate_cost("hello world", "en-US-Chirp3-HD-Aoede")
        assert studio > neural

    def test_openai_voices_are_namespaced(self):
        voice = tts_service.validate_voice("openai/coral")
        assert voice.provider == "openai"

    def test_unknown_voice_is_rejected(self):
        with pytest.raises(InvalidVoiceError):
            tts_service.validate_voice("robot voice; DROP TABLE")

    def test_any_well_formed_google_voice_is_accepted(self):
        assert tts_service.validate_voice("de-DE-Neural2-B").provider == "google"

    def test_catalogue_quotes_prices(self):
        for voice in tts_service.voices():
            assert float(voice["credits_per_1k_chars"]) > 0
            assert "enabled" in voice


class TestImagePricing:
    def test_image_quality_tiers_are_ordered(self):
        low = image_service.calculate_cost("gpt-image-2", "low", "1024x1024")
        high = image_service.calculate_cost("gpt-image-2", "high", "1024x1024")
        assert low < high

    def test_google_resolutions_are_ordered(self):
        one_k = image_service.calculate_cost("gemini-3.1-flash-image", "1K", "1:1")
        four_k = image_service.calculate_cost("gemini-3.1-flash-image", "4K", "1:1")
        assert one_k < four_k

    @pytest.mark.parametrize(
        ("model", "quality", "size"),
        [
            ("midjourney", "high", "1024x1024"),
            ("gpt-image-2", "ultra", "1024x1024"),
            ("gpt-image-2", "high", "4096x4096"),
            ("dall-e-3", "standard", "1024x1024"),  # retired
            ("gemini-3.1-flash-image", "1K", "1024x1024"),  # google takes aspect ratios
        ],
    )
    def test_unpriced_combinations_are_rejected(self, model, quality, size):
        """Falling back to an arbitrary price would bill for the wrong thing."""
        with pytest.raises(UnsupportedImageOptionError):
            image_service.calculate_cost(model, quality, size)

    def test_published_options_are_all_priceable(self):
        """Whatever /media/images/options advertises must be chargeable."""
        for entry in image_service.options():
            for quality, sizes in entry["prices"].items():
                for size, credits in sizes.items():
                    assert image_service.calculate_cost(entry["id"], quality, size) == Decimal(
                        credits
                    )
            assert entry["default_quality"] in entry["prices"]
            assert entry["default_size"] in entry["prices"][entry["default_quality"]]
