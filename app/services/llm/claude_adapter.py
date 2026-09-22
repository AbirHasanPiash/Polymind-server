"""Anthropic (Claude) adapter."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

import anthropic

from app.core.config import settings
from app.services.llm.base import (
    GenerationOptions,
    LLMProvider,
    ModelRefusedError,
    PromptType,
    ProviderNotConfiguredError,
    split_system,
)
from app.services.llm.schema import ChatMessage
from app.services.llm.usage import Outcome, Usage

logger = logging.getLogger(__name__)

# Server-side refusal fallbacks: when the safety layer on a frontier model
# declines a request, the API re-runs it on a fallback model inside the same
# call instead of returning nothing. The served model is reported back and
# billed as such.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5"}


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
                timeout=180.0,
            )
        return ClaudeAdapter._client

    @staticmethod
    def _blocks(message: ChatMessage) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for attachment in message.attachments:
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
        if message.text:
            blocks.append({"type": "text", "text": message.text})
        return blocks

    def _request(
        self, prompt: PromptType, model: str, options: GenerationOptions | None
    ) -> dict[str, Any]:
        spec = self.spec(model)
        options = options or GenerationOptions()
        system_prompt, messages = split_system(prompt, options)

        # Claude requires alternating user/assistant turns; merge neighbours.
        turns: list[dict[str, Any]] = []
        for message in messages:
            role = "assistant" if message.is_assistant else "user"
            blocks = self._blocks(message)
            if not blocks:
                continue
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"].extend(blocks)
            else:
                turns.append({"role": role, "content": blocks})

        kwargs: dict[str, Any] = {
            "model": spec.api_model,
            "messages": turns,
            "max_tokens": options.max_output_tokens,
        }
        # Sending system="" is rejected by the API, so only include it when set.
        if system_prompt:
            kwargs["system"] = system_prompt
        if spec.reasoning:
            # Adaptive thinking: the model decides how much to think, bounded by
            # the effort level. Older models (Haiku 4.5) take neither setting.
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": options.effort}
        if spec.api_model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    @staticmethod
    def _finish(final_message: Any, model: str, usage: Usage, outcome: Outcome | None) -> None:
        if final_message.usage:
            usage.record(final_message.usage.input_tokens, final_message.usage.output_tokens)

        served = getattr(final_message, "model", None)
        if outcome is not None:
            outcome.served_model = served
            for block in final_message.content:
                if getattr(block, "type", "") == "fallback":
                    declined = getattr(getattr(block, "from_", None), "model", "the model")
                    continued = getattr(getattr(block, "to", None), "model", served)
                    outcome.notes.append(
                        f"{declined} declined this request; answered by {continued}."
                    )
            if final_message.stop_reason == "max_tokens":
                outcome.finish_reason = "length"

        if final_message.stop_reason == "refusal":
            details = getattr(final_message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            if outcome is not None:
                outcome.finish_reason = "refusal"
                outcome.refusal_category = category
            raise ModelRefusedError(model, category)

    async def generate_stream(
        self,
        prompt: PromptType,
        model: str,
        usage: Usage,
        options: GenerationOptions | None = None,
        outcome: Outcome | None = None,
    ) -> AsyncGenerator[str, None]:
        async with self.client.beta.messages.stream(
            **self._request(prompt, model, options)
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final_message = await stream.get_final_message()
        self._finish(final_message, model, usage, outcome)

    async def generate_text(
        self,
        prompt: PromptType,
        model: str,
        usage: Usage,
        options: GenerationOptions | None = None,
    ) -> str:
        response = await self.client.beta.messages.create(**self._request(prompt, model, options))
        self._finish(response, model, usage, None)
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
