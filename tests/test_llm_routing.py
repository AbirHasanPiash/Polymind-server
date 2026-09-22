"""Model routing and the registry that backs it."""

from __future__ import annotations

import pytest

from app.services.llm.factory import LLMFactory
from app.services.llm.models import (
    DEFAULT_CODING_MODEL,
    DEFAULT_CREATIVE_MODEL,
    DEFAULT_FAST_MODEL,
    DEFAULT_LONG_CONTEXT_MODEL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_MODEL,
    LEGACY_ALIASES,
    MODEL_REGISTRY,
    UTILITY_MODEL,
    UnknownModelError,
    get_spec,
    list_models,
)
from app.services.llm.router import ModelRouter


class TestAutoRouting:
    def test_coding_prompt_routes_to_the_coding_model(self):
        route = ModelRouter.route("Refactor this Python function to use async/await")
        assert route.model == DEFAULT_CODING_MODEL
        assert route.intent == "coding"

    def test_math_prompt_routes_to_the_reasoning_model(self):
        model = ModelRouter.determine_model(
            "Solve this calculus problem and prove the theorem step by step"
        )
        assert model == DEFAULT_REASONING_MODEL

    def test_short_smalltalk_routes_to_the_fast_model(self):
        assert ModelRouter.determine_model("hi there") == DEFAULT_FAST_MODEL

    def test_very_long_prompt_routes_to_the_long_context_model(self):
        assert ModelRouter.determine_model("lorem ipsum " * 500) == DEFAULT_LONG_CONTEXT_MODEL

    def test_creative_prompt_routes_to_the_creative_model(self):
        assert ModelRouter.determine_model("Write a poem about the sea") == DEFAULT_CREATIVE_MODEL

    def test_medium_neutral_prompt_falls_back_to_the_default(self):
        prompt = "Tell me something interesting about the history of maritime navigation " * 3
        assert ModelRouter.determine_model(prompt) == DEFAULT_MODEL

    def test_empty_prompt_does_not_crash(self):
        assert ModelRouter.determine_model("") in MODEL_REGISTRY

    def test_every_route_has_a_reason(self):
        assert "routed" in ModelRouter.route("hi").reason


class TestUserPreference:
    def test_explicit_supported_model_wins(self):
        route = ModelRouter.route("hello", "claude-haiku-4-5")
        assert route.model == "claude-haiku-4-5"
        assert route.intent == "pinned"

    def test_auto_is_not_treated_as_a_model_id(self):
        assert ModelRouter.determine_model("hello", "auto") == DEFAULT_FAST_MODEL

    def test_legacy_ids_resolve_to_current_models(self):
        for old, new in LEGACY_ALIASES.items():
            assert ModelRouter.determine_model("hello", old) == new

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
            DEFAULT_CREATIVE_MODEL,
            UTILITY_MODEL,
        ]
        for model_id in defaults:
            assert model_id in MODEL_REGISTRY

    def test_specs_are_complete(self):
        for spec in MODEL_REGISTRY.values():
            assert spec.api_model
            assert spec.display_name
            assert spec.input_price > 0
            assert spec.output_price > 0
            assert spec.context_window > 0
            assert spec.tier in {"flagship", "balanced", "fast"}

    def test_public_shape_has_what_the_picker_needs(self):
        public = get_spec("gpt-5.5").to_public()
        assert {
            "id",
            "provider",
            "display_name",
            "tier",
            "pricing",
            "context_window",
            "reasoning",
        } <= set(public)
        assert float(public["pricing"]["output_credits_per_million"]) > float(
            public["pricing"]["input_credits_per_million"]
        )

    def test_hidden_models_are_not_listed(self):
        assert all(not spec.hidden for spec in list_models())

    def test_unknown_model_lookup_raises(self):
        with pytest.raises(UnknownModelError):
            get_spec("not-a-model")

    def test_all_three_providers_are_present(self):
        assert {spec.provider for spec in MODEL_REGISTRY.values()} == {
            "openai",
            "anthropic",
            "google",
        }


class TestRouteReasons:
    def test_pinned_routes_have_their_own_reason(self):
        assert ModelRouter.route("hi", "gpt-5.5").reason == "Model chosen by you"

    def test_auto_routes_explain_the_intent(self):
        assert "coding" in ModelRouter.route("Refactor this Python function").reason.lower()
