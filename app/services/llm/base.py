"""Contract every LLM adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from decimal import Decimal

from app.services.llm.models import DEFAULT_EFFORT, Effort, ModelSpec, get_spec, price_in_credits
from app.services.llm.schema import ChatMessage
from app.services.llm.usage import Outcome, Usage

PromptType = str | list[ChatMessage] | list[dict[str, str]]

# Ceiling on a single completion. Thinking tokens count against it on every
# provider, so it is generous enough for a deep answer while still bounding a
# runaway (and expensive) response. The chat endpoint lowers it further when the
# caller's balance cannot cover it.
DEFAULT_MAX_OUTPUT_TOKENS = 8192
MAX_OUTPUT_TOKENS_BY_EFFORT: dict[str, int] = {"low": 4096, "medium": 8192, "high": 16384}


@dataclass(slots=True)
class GenerationOptions:
    """Per-turn settings that every adapter understands."""

    effort: Effort = DEFAULT_EFFORT
    system_prompt: str | None = None
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


class ProviderNotConfiguredError(RuntimeError):
    """Raised when a provider is selected but its API key is not configured."""

    def __init__(self, provider: str, env_var: str) -> None:
        super().__init__(f"{provider} is not configured. Set {env_var} to enable it.")
        self.provider = provider
        self.env_var = env_var


class ModelRefusedError(RuntimeError):
    """The provider's safety layer declined the request outright."""

    def __init__(self, model: str, category: str | None = None) -> None:
        detail = f" ({category})" if category else ""
        super().__init__(f"{model} declined to answer this request{detail}.")
        self.model = model
        self.category = category


class LLMProvider(ABC):
    """Uniform interface over OpenAI, Anthropic and Gemini."""

    @abstractmethod
    def generate_stream(
        self,
        prompt: PromptType,
        model: str,
        usage: Usage,
        options: GenerationOptions | None = None,
        outcome: Outcome | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks as they arrive, updating ``usage`` in place."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: PromptType,
        model: str,
        usage: Usage,
        options: GenerationOptions | None = None,
    ) -> str:
        """Return the whole completion at once, updating ``usage`` in place."""

    @staticmethod
    def spec(model: str) -> ModelSpec:
        return get_spec(model)

    def calculate_cost(self, usage: Usage, model: str) -> Decimal:
        """Price the call in wallet credits.

        Shared by all providers so pricing can never diverge per adapter.
        """
        return price_in_credits(usage, model)


def split_system(
    prompt: PromptType, options: GenerationOptions | None
) -> tuple[str, list[ChatMessage]]:
    """Normalise any accepted prompt shape into (system text, chat messages)."""
    system_chunks: list[str] = []
    if options and options.system_prompt:
        system_chunks.append(options.system_prompt)

    messages: list[ChatMessage] = []
    if isinstance(prompt, str):
        messages.append(ChatMessage.from_text("user", prompt))
    else:
        for item in prompt:
            if isinstance(item, ChatMessage):
                message = item
            elif isinstance(item, dict):
                message = ChatMessage.from_text(item.get("role", "user"), item.get("content", ""))
            else:
                continue
            if message.role == "system":
                if message.text:
                    system_chunks.append(message.text)
                continue
            if not message.text and not message.attachments:
                continue
            messages.append(message)

    return "\n\n".join(system_chunks).strip(), messages
