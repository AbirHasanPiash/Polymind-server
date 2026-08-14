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
    DEFAULT_MAX_OUTPUT_TOKENS,
    LLMProvider,
    PromptType,
    ProviderNotConfiguredError,
)
from app.services.llm.schema import ChatMessage
from app.services.llm.usage import Usage

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
    def _prepare(prompt: PromptType) -> tuple[str | None, list[types.Content]]:
        """Split into (system instruction, contents). Gemini calls the AI role 'model'."""
        if isinstance(prompt, str):
            return None, [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

        system_chunks: list[str] = []
        contents: list[types.Content] = []

        for item in prompt:
            parts: list[types.Part] = []

            if isinstance(item, ChatMessage):
                role = item.role
                if item.text:
                    parts.append(types.Part.from_text(text=item.text))
                for attachment in item.attachments:
                    if attachment.type == "text":
                        parts.append(
                            types.Part.from_text(text=f"\n[Attachment]\n{attachment.content}")
                        )
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
            elif isinstance(item, dict):
                role = item.get("role", "user")
                if item.get("content"):
                    parts.append(types.Part.from_text(text=item["content"]))
            else:
                continue

            if not parts:
                continue

            if role == "system":
                system_chunks.extend(part.text for part in parts if part.text)
            elif role == "user":
                contents.append(types.Content(role="user", parts=parts))
            elif role in ("ai", "assistant", "model"):
                contents.append(types.Content(role="model", parts=parts))

        return ("\n".join(system_chunks) or None), contents

    @staticmethod
    def _config(system_instruction: str | None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )

    async def generate_stream(
        self, prompt: PromptType, model: str, usage: Usage
    ) -> AsyncGenerator[str, None]:
        spec = self.spec(model)
        system_instruction, contents = self._prepare(prompt)

        # Usage arrives with the response chunks, so no separate count_tokens call.
        response_stream = await self.client.aio.models.generate_content_stream(
            model=spec.api_model,
            contents=contents,
            config=self._config(system_instruction),
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text
            if chunk.usage_metadata:
                usage.record(
                    chunk.usage_metadata.prompt_token_count,
                    chunk.usage_metadata.candidates_token_count,
                )

    async def generate_text(self, prompt: PromptType, model: str, usage: Usage) -> str:
        spec = self.spec(model)
        system_instruction, contents = self._prepare(prompt)

        response = await self.client.aio.models.generate_content(
            model=spec.api_model,
            contents=contents,
            config=self._config(system_instruction),
        )

        if response.usage_metadata:
            usage.record(
                response.usage_metadata.prompt_token_count,
                response.usage_metadata.candidates_token_count,
            )
        return response.text or ""
