"""Google Cloud Text-to-Speech."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from decimal import Decimal
from functools import cached_property

from google.cloud import texttospeech
from google.oauth2 import service_account

from app.core.config import settings
from app.services.llm.models import CREDIT_PRECISION, PROFIT_MARGIN, USD_TO_CREDITS_RATE
from app.services.storage import storage

logger = logging.getLogger(__name__)

# Google TTS voice names look like "en-US-Neural2-F"; the first two segments are
# the BCP-47 language code the API expects alongside the voice.
_VOICE_PATTERN = re.compile(r"^[a-z]{2}-[A-Z]{2}-[A-Za-z0-9]+(-[A-Za-z0-9]+)*$")

DEFAULT_VOICE = "en-US-Neural2-F"
MAX_TTS_CHARS = 4096  # Google's per-request limit for a single synthesis


class InvalidVoiceError(ValueError):
    def __init__(self, voice_name: str) -> None:
        super().__init__(f"Invalid voice name: {voice_name!r}")
        self.voice_name = voice_name


class GoogleTTSService:
    PROVIDER_COST_PER_CHAR = Decimal("0.000016")

    @cached_property
    def client(self) -> texttospeech.TextToSpeechClient:
        """Created on first use so a missing key file cannot break startup."""
        path = settings.GOOGLE_APPLICATION_CREDENTIALS
        if path and os.path.exists(path):
            credentials = service_account.Credentials.from_service_account_file(path)
            return texttospeech.TextToSpeechClient(credentials=credentials)

        if path:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS points to a missing file: %s", path)
        # Falls back to Application Default Credentials (workload identity, gcloud…).
        return texttospeech.TextToSpeechClient()

    def calculate_cost(self, text: str) -> Decimal:
        """Price in wallet credits: (chars x provider rate x margin) x credit rate."""
        provider_cost = Decimal(len(text)) * self.PROVIDER_COST_PER_CHAR
        return (provider_cost * PROFIT_MARGIN * USD_TO_CREDITS_RATE).quantize(CREDIT_PRECISION)

    @staticmethod
    def validate_voice(voice_name: str) -> str:
        """Reject malformed voice ids before they reach the API."""
        if not voice_name or not _VOICE_PATTERN.match(voice_name):
            raise InvalidVoiceError(voice_name)
        return voice_name

    @staticmethod
    def _language_code(voice_name: str) -> str:
        return "-".join(voice_name.split("-")[:2])

    def _synthesize(self, text: str, voice_name: str) -> bytes:
        response = self.client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=self._language_code(voice_name),
                name=voice_name,
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
            ),
        )
        return response.audio_content

    async def generate_audio(self, text: str, voice_name: str = DEFAULT_VOICE) -> str:
        """Synthesize MP3 audio, store it, and return its public URL."""
        self.validate_voice(voice_name)
        text = text[:MAX_TTS_CHARS]

        # The Google client is synchronous; keep it off the event loop.
        audio_bytes = await asyncio.to_thread(self._synthesize, text, voice_name)

        return await storage.upload_file_async(
            file_bytes=audio_bytes,
            destination_path=f"tts/{uuid.uuid4()}.mp3",
            content_type="audio/mpeg",
        )


tts_service = GoogleTTSService()
