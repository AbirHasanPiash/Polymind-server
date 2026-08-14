"""Maps a model id to the adapter that can serve it."""

from __future__ import annotations

from app.services.llm.base import LLMProvider
from app.services.llm.claude_adapter import ClaudeAdapter
from app.services.llm.gemini_adapter import GeminiAdapter
from app.services.llm.models import (
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    ModelSpec,
    get_spec,
    list_models,
)
from app.services.llm.openai_adapter import OpenAIAdapter

# Adapters are stateless in front of a shared client, so one instance each is enough.
_ADAPTERS: dict[str, LLMProvider] = {
    PROVIDER_OPENAI: OpenAIAdapter(),
    PROVIDER_ANTHROPIC: ClaudeAdapter(),
    PROVIDER_GOOGLE: GeminiAdapter(),
}


class LLMFactory:
    """Resolves models to providers, rejecting anything outside the registry."""

    @staticmethod
    def get_provider(model: str) -> LLMProvider:
        """Return the adapter for ``model``.

        Raises ``UnknownModelError`` for ids that are not in the registry, so an
        arbitrary client-supplied string can never reach a provider.
        """
        return _ADAPTERS[get_spec(model).provider]

    @staticmethod
    def get_all_models() -> list[ModelSpec]:
        return list_models()
