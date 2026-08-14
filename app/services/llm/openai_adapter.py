"""OpenAI chat-completions adapter."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import openai

from app.core.config import settings
from app.services.llm.base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    LLMProvider,
    PromptType,
    ProviderNotConfiguredError,
)
from app.services.llm.schema import ChatMessage
from app.services.llm.usage import Usage


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
                timeout=120.0,
            )
        return OpenAIAdapter._client

    @staticmethod
    def _to_messages(prompt: PromptType) -> list[dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        messages: list[dict] = []
        for item in prompt:
            if isinstance(item, ChatMessage):
                messages.append(item.to_openai_format())
            elif isinstance(item, dict):
                role = "assistant" if item.get("role") == "ai" else item.get("role", "user")
                messages.append({"role": role, "content": item.get("content", "")})
        return messages

    async def generate_stream(
        self, prompt: PromptType, model: str, usage: Usage
    ) -> AsyncGenerator[str, None]:
        spec = self.spec(model)  # rejects unknown models before any network call

        stream = await self.client.chat.completions.create(
            model=spec.api_model,
            messages=self._to_messages(prompt),
            max_completion_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                usage.record(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)

    async def generate_text(self, prompt: PromptType, model: str, usage: Usage) -> str:
        spec = self.spec(model)

        response = await self.client.chat.completions.create(
            model=spec.api_model,
            messages=self._to_messages(prompt),
            max_completion_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )

        if response.usage:
            usage.record(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content or ""
