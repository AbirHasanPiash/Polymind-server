"""Contract every LLM adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from decimal import Decimal

from app.services.llm.models import ModelSpec, get_spec, price_in_credits
from app.services.llm.schema import ChatMessage
from app.services.llm.usage import Usage

PromptType = str | list[ChatMessage] | list[dict[str, str]]

# Cap on a single completion, guarding against a runaway (and expensive) response.
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class ProviderNotConfiguredError(RuntimeError):
    """Raised when a provider is selected but its API key is not configured."""

    def __init__(self, provider: str, env_var: str) -> None:
        super().__init__(f"{provider} is not configured. Set {env_var} to enable it.")
        self.provider = provider
        self.env_var = env_var


class LLMProvider(ABC):
    """Uniform interface over OpenAI, Anthropic and Gemini."""

    @abstractmethod
    def generate_stream(
        self, prompt: PromptType, model: str, usage: Usage
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks as they arrive, updating ``usage`` in place."""

    @abstractmethod
    async def generate_text(self, prompt: PromptType, model: str, usage: Usage) -> str:
        """Return the whole completion at once, updating ``usage`` in place."""

    @staticmethod
    def spec(model: str) -> ModelSpec:
        return get_spec(model)

    def calculate_cost(self, usage: Usage, model: str) -> Decimal:
        """Price the call in wallet credits.

        Shared by all providers so pricing can never diverge per adapter.
        """
        return price_in_credits(usage, model)
