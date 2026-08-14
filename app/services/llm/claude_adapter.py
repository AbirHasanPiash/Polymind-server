"""Anthropic (Claude) adapter."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import anthropic

from app.core.config import settings
from app.services.llm.base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    LLMProvider,
    PromptType,
    ProviderNotConfiguredError,
)
from app.services.llm.schema import ChatMessage
from app.services.llm.usage import Usage


class ClaudeAdapter(LLMProvider):
    _client: anthropic.AsyncAnthropic | None = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if ClaudeAdapter._client is None:
            if not settings.ANTHROPIC_API_KEY:
                raise ProviderNotConfiguredError("Anthropic", "ANTHROPIC_API_KEY")
            ClaudeAdapter._client = anthropic.AsyncAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                max_retries=2,
                timeout=120.0,
            )
        return ClaudeAdapter._client

    @staticmethod
    def _prepare(prompt: PromptType) -> tuple[str, list[dict[str, Any]]]:
        """Split the conversation into (system prompt, messages).

        Claude takes the system prompt as a top-level parameter rather than as a
        message, and requires alternating user/assistant turns.
        """
        if isinstance(prompt, str):
            return "", [{"role": "user", "content": prompt}]

        system_chunks: list[str] = []
        messages: list[dict[str, Any]] = []

        for item in prompt:
            blocks: list[dict[str, Any]] = []

            if isinstance(item, ChatMessage):
                role = item.role
                for attachment in item.attachments:
                    if attachment.type == "image":
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": attachment.mime_type or "image/jpeg",
                                    "data": attachment.content,
                                },
                            }
                        )
                    else:
                        blocks.append(
                            {
                                "type": "text",
                                "text": f"<file_context>\n{attachment.content}\n</file_context>",
                            }
                        )
                if item.text:
                    blocks.append({"type": "text", "text": item.text})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                if item.get("content"):
                    blocks.append({"type": "text", "text": item["content"]})
            else:
                continue

            if not blocks:
                continue

            if role == "system":
                system_chunks.extend(b["text"] for b in blocks if b["type"] == "text")
            elif role == "user":
                messages.append({"role": "user", "content": blocks})
            elif role in ("ai", "assistant"):
                messages.append({"role": "assistant", "content": blocks})

        return "\n".join(system_chunks).strip(), messages

    def _request_kwargs(self, prompt: PromptType, model: str) -> dict[str, Any]:
        system_prompt, messages = self._prepare(prompt)
        kwargs: dict[str, Any] = {
            "model": self.spec(model).api_model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        }
        # Sending system="" is rejected by the API, so only include it when set.
        if system_prompt:
            kwargs["system"] = system_prompt
        return kwargs

    async def generate_stream(
        self, prompt: PromptType, model: str, usage: Usage
    ) -> AsyncGenerator[str, None]:
        async with self.client.messages.stream(**self._request_kwargs(prompt, model)) as stream:
            async for text in stream.text_stream:
                yield text

            final_message = await stream.get_final_message()
            if final_message.usage:
                usage.record(
                    final_message.usage.input_tokens,
                    final_message.usage.output_tokens,
                )

    async def generate_text(self, prompt: PromptType, model: str, usage: Usage) -> str:
        response = await self.client.messages.create(**self._request_kwargs(prompt, model))

        if response.usage:
            usage.record(response.usage.input_tokens, response.usage.output_tokens)

        return "".join(block.text for block in response.content if block.type == "text")
