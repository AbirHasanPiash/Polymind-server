"""Model routing and the registry that backs it."""

from __future__ import annotations

import pytest

from app.services.llm.factory import LLMFactory
from app.services.llm.models import (
    DEFAULT_CODING_MODEL,
    DEFAULT_FAST_MODEL,
    DEFAULT_LONG_CONTEXT_MODEL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_MODEL,
    MODEL_REGISTRY,
    UnknownModelError,
    get_spec,
)
from app.services.llm.router import ModelRouter


class TestAutoRouting:
    def test_coding_prompt_routes_to_the_coding_model(self):
        model = ModelRouter.determine_model("Refactor this Python function to use async/await")
        assert model == DEFAULT_CODING_MODEL

    def test_math_prompt_routes_to_the_reasoning_model(self):
        model = ModelRouter.determine_model(
            "Solve this calculus problem and prove the theorem step by step"
        )
        assert model == DEFAULT_REASONING_MODEL

    def test_short_smalltalk_routes_to_the_fast_model(self):
        assert ModelRouter.determine_model("hi there") == DEFAULT_FAST_MODEL

    def test_very_long_prompt_routes_to_the_long_context_model(self):
        assert ModelRouter.determine_model("lorem ipsum " * 500) == DEFAULT_LONG_CONTEXT_MODEL

    def test_medium_neutral_prompt_falls_back_to_the_default(self):
        prompt = "Tell me something interesting about the history of maritime navigation " * 3
        assert ModelRouter.determine_model(prompt) == DEFAULT_MODEL

    def test_empty_prompt_does_not_crash(self):
        assert ModelRouter.determine_model("") in MODEL_REGISTRY


class TestUserPreference:
    def test_explicit_supported_model_wins(self):
        assert ModelRouter.determine_model("hello", "claude-4.5-haiku") == "claude-4.5-haiku"

    def test_auto_is_not_treated_as_a_model_id(self):
        assert ModelRouter.determine_model("hello", "auto") == DEFAULT_FAST_MODEL

    @pytest.mark.parametrize(
        "model",
        ["gpt-9-ultra", "claude-does-not-exist", "gemini-secret", "../etc/passwd"],
    )
    def test_unknown_models_are_rejected(self, model: str):
        """An unvalidated id used to be forwarded straight to a provider."""
        with pytest.raises(UnknownModelError):
            ModelRouter.determine_model("hello", model)


class TestRegistry:
    def test_every_registered_model_resolves_to_an_adapter(self):
        for model_id in MODEL_REGISTRY:
            assert LLMFactory.get_provider(model_id) is not None

    def test_every_routing_default_is_registered(self):
        defaults = [
            DEFAULT_MODEL,
            DEFAULT_FAST_MODEL,
            DEFAULT_CODING_MODEL,
            DEFAULT_REASONING_MODEL,
            DEFAULT_LONG_CONTEXT_MODEL,
        ]
        for model_id in defaults:
            assert model_id in MODEL_REGISTRY

    def test_api_model_ids_are_present(self):
        for spec in MODEL_REGISTRY.values():
            assert spec.api_model
            assert spec.input_price > 0
            assert spec.output_price > 0

    def test_unknown_model_lookup_raises(self):
        with pytest.raises(UnknownModelError):
            get_spec("not-a-model")
