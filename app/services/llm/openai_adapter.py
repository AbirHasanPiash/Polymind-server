"""OpenAI adapter, built on the Responses API.

The Responses API is the only endpoint every current model supports (the Pro
tier rejects Chat Completions), and it is where reasoning effort, tools and
future features land first.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

import openai

from app.core.config import settings
from app.services.llm.base import (
    GenerationOptions,
    LLMProvider,
    PromptType,
    ProviderNotConfiguredError,
    split_system,
)
from app.services.llm.usage import Outcome, Usage

logger = logging.getLogger(__name__)

# The Pro models only accept medium and above.
_PRO_MIN_EFFORT = {"low": "medium", "medium": "medium", "high": "high"}


class OpenAIAdapter(LLMProvider):
    # One client per process: it owns an HTTP connection pool that should be reused.
    _client: openai.AsyncOpenAI | None = None

    @property
    def client(self) -> openai.AsyncOpenAI:
        if OpenAIAdapter._client is None:
            if not settings.OPENAI_API_KEY:
                raise ProviderNotConfiguredError("OpenAI", "OPENAI_API_KEY")
            OpenAIAdapter._client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                max_retries=2,
                timeout=180.0,
            )
        return OpenAIAdapter._client

    def _request(
        self, prompt: PromptType, model: str, options: GenerationOptions | None
    ) -> dict[str, Any]:
        spec = self.spec(model)  # rejects unknown models before any network call
        options = options or GenerationOptions()
        system_prompt, messages = split_system(prompt, options)

        kwargs: dict[str, Any] = {
            "model": spec.api_model,
            "input": [message.to_responses_input() for message in messages],
            "max_output_tokens": options.max_output_tokens,
            # Conversations are stored in our own database; nothing needs to
            # persist on the provider side.
            "store": False,
        }
        if system_prompt:
            kwargs["instructions"] = system_prompt
        if spec.reasoning:
            effort = options.effort
            if spec.api_model.endswith("-pro"):
                effort = _PRO_MIN_EFFORT[effort]
            kwargs["reasoning"] = {"effort": effort}
        return kwargs

    async def generate_stream(
        self,
        prompt: PromptType,
        model: str,
        usage: Usage,
        options: GenerationOptions | None = None,
        outcome: Outcome | None = None,
    ) -> AsyncGenerator[str, None]:
        stream = await self.client.responses.create(
            **self._request(prompt, model, options), stream=True
        )

        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                yield event.delta
            elif event_type == "response.completed":
                response = event.response
                if response.usage:
                    usage.record(response.usage.input_tokens, response.usage.output_tokens)
                if outcome is not None:
                    outcome.served_model = response.model
                    if getattr(response, "incomplete_details", None):
                        outcome.finish_reason = "length"
            elif event_type == "response.incomplete":
                if outcome is not None:
                    outcome.finish_reason = "length"
                response = event.response
                if response.usage:
                    usage.record(response.usage.input_tokens, response.usage.output_tokens)
            elif event_type in ("response.failed", "error"):
                message = getattr(getattr(event, "response", None), "error", None) or getattr(
                    event, "message", "OpenAI request failed"
                )
                raise RuntimeError(str(message))

    async def generate_text(
        self,
        prompt: PromptType,
        model: str,
        usage: Usage,
        options: GenerationOptions | None = None,
    ) -> str:
        response = await self.client.responses.create(**self._request(prompt, model, options))
        if response.usage:
            usage.record(response.usage.input_tokens, response.usage.output_tokens)
        return response.output_text or ""
