"""Google Gemini adapter."""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import AsyncGenerator

from google import genai
from google.genai import types

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


class GeminiAdapter(LLMProvider):
    _client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if GeminiAdapter._client is None:
            if not settings.GOOGLE_API_KEY:
                raise ProviderNotConfiguredError("Gemini", "GOOGLE_API_KEY")
            GeminiAdapter._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        return GeminiAdapter._client

    @staticmethod
    def _parts(message: ChatMessage) -> list[types.Part]:
        parts: list[types.Part] = []
        if message.text:
            parts.append(types.Part.from_text(text=message.text))
        for attachment in message.attachments:
            if attachment.type == "text":
                parts.append(types.Part.from_text(text=f"\n[Attachment]\n{attachment.content}"))
            else:
                try:
                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(attachment.content),
                            mime_type=attachment.mime_type or "image/jpeg",
                        )
                    )
                except (binascii.Error, ValueError):
                    # A corrupt attachment must not abort the whole turn.
                    logger.warning("Skipping attachment with invalid base64 payload")
        return parts

    def _prepare(
        self, prompt: PromptType, model: str, options: GenerationOptions | None
    ) -> tuple[str, list[types.Content], types.GenerateContentConfig]:
        spec = self.spec(model)
        options = options or GenerationOptions()
        system_prompt, messages = split_system(prompt, options)

        contents: list[types.Content] = []
        for message in messages:
            parts = self._parts(message)
            if not parts:
                continue
            role = "model" if message.is_assistant else "user"
            contents.append(types.Content(role=role, parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            max_output_tokens=options.max_output_tokens,
        )
        if spec.reasoning:
            # Gemini 3 exposes thinking as a level rather than a token budget.
            config.thinking_config = types.ThinkingConfig(thinking_level=options.effort)
        return spec.api_model, contents, config

    @staticmethod
    def _record(response, usage: Usage, outcome: Outcome | None, model: str) -> None:
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            # Thinking tokens are billed as output; fold them in.
            completion = (response.usage_metadata.candidates_token_count or 0) + (
                getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
            )
            usage.record(prompt_tokens, completion or None)
        if outcome is not None:
            outcome.served_model = getattr(response, "model_version", None) or model
            candidates = response.candidates or []
            if candidates:
                reason = str(getattr(candidates[0], "finish_reason", "") or "")
                if "MAX_TOKENS" in reason:
                    outcome.finish_reason = "length"
                elif "SAFETY" in reason or "PROHIBITED" in reason:
                    outcome.finish_reason = "refusal"

    async def generate_stream(
        self,
        prompt: PromptType,
        model: str,
        usage: Usage,
        options: GenerationOptions | None = None,
        outcome: Outcome | None = None,
    ) -> AsyncGenerator[str, None]:
        api_model, contents, config = self._prepare(prompt, model, options)

        response_stream = await self.client.aio.models.generate_content_stream(
            model=api_model, contents=contents, config=config
        )

        produced = False
        last_chunk = None
        async for chunk in response_stream:
            last_chunk = chunk
            if chunk.text:
                produced = True
                yield chunk.text
        if last_chunk is not None:
            self._record(last_chunk, usage, outcome, model)
            if not produced and outcome is not None and outcome.finish_reason == "refusal":
                raise ModelRefusedError(model, "safety")

    async def generate_text(
        self,
        prompt: PromptType,
        model: str,
        usage: Usage,
        options: GenerationOptions | None = None,
    ) -> str:
        api_model, contents, config = self._prepare(prompt, model, options)
        response = await self.client.aio.models.generate_content(
            model=api_model, contents=contents, config=config
        )
        self._record(response, usage, None, model)
        return response.text or ""
