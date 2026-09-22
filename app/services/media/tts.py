"""Text to speech across providers.

Voices are a catalogue rather than a free-form string: each entry knows its
provider, language and price, so the client can list them and the biller can
price them. Google Cloud voices keep their native ids (``en-US-Neural2-F``) for
backwards compatibility; OpenAI voices are namespaced (``openai/coral``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property

from google.cloud import texttospeech
from google.oauth2 import service_account
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.llm.base import ProviderNotConfiguredError
from app.services.llm.models import PROVIDER_GOOGLE, PROVIDER_OPENAI, usd_to_credits
from app.services.storage import storage

logger = logging.getLogger(__name__)

# Google TTS voice names look like "en-US-Neural2-F"; the first two segments are
# the BCP-47 language code the API expects alongside the voice.
_GOOGLE_VOICE_PATTERN = re.compile(r"^[a-z]{2}-[A-Z]{2}-[A-Za-z0-9]+(-[A-Za-z0-9]+)*$")

DEFAULT_VOICE = "en-US-Neural2-F"
MAX_TTS_CHARS = 4096  # Google's per-request limit for a single synthesis
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"


class InvalidVoiceError(ValueError):
    def __init__(self, voice_name: str) -> None:
        super().__init__(f"Unknown voice: {voice_name!r}")
        self.voice_name = voice_name


@dataclass(frozen=True, slots=True)
class Voice:
    id: str
    provider: str
    name: str
    description: str
    language: str
    gender: str  # female | male | neutral
    cost_per_char_usd: Decimal
    tier: str = "standard"  # standard | premium

    def to_public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "provider": self.provider,
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "gender": self.gender,
            "tier": self.tier,
            "credits_per_1k_chars": str(usd_to_credits(self.cost_per_char_usd * 1000)),
        }


_GOOGLE_NEURAL2 = Decimal("0.000016")
_GOOGLE_CHIRP3 = Decimal("0.000030")
# gpt-4o-mini-tts is billed per token (text in, audio out); at typical speaking
# rates that lands at roughly this much per character.
_OPENAI_MINI_TTS = Decimal("0.000017")


def _google(
    voice_id: str, name: str, description: str, gender: str, tier: str = "standard"
) -> Voice:
    language = "-".join(voice_id.split("-")[:2])
    cost = _GOOGLE_CHIRP3 if "Chirp3" in voice_id else _GOOGLE_NEURAL2
    return Voice(voice_id, PROVIDER_GOOGLE, name, description, language, gender, cost, tier)


def _openai(voice: str, name: str, description: str, gender: str) -> Voice:
    return Voice(
        f"openai/{voice}",
        PROVIDER_OPENAI,
        name,
        description,
        "multilingual",
        gender,
        _OPENAI_MINI_TTS,
        "premium",
    )


VOICES: dict[str, Voice] = {
    voice.id: voice
    for voice in (
        # Google Neural2 (English)
        _google("en-US-Neural2-F", "Ava", "Energetic, upbeat", "female"),
        _google("en-US-Neural2-C", "Claire", "Professional, clear", "female"),
        _google("en-US-Neural2-E", "Ella", "Soft, warm", "female"),
        _google("en-US-Neural2-H", "Harper", "Bright, friendly", "female"),
        _google("en-US-Neural2-A", "Adam", "Calm, measured", "male"),
        _google("en-US-Neural2-D", "Dean", "Deep, authoritative", "male"),
        _google("en-US-Neural2-I", "Ian", "Assertive, crisp", "male"),
        _google("en-US-Neural2-J", "James", "Steady narrator", "male"),
        _google("en-GB-Neural2-A", "Alice", "British, polished", "female"),
        _google("en-GB-Neural2-B", "Ben", "British, relaxed", "male"),
        _google("en-AU-Neural2-A", "Amelia", "Australian, friendly", "female"),
        _google("en-IN-Neural2-A", "Anaya", "Indian English, clear", "female"),
        # Google Chirp 3 HD (studio quality)
        _google("en-US-Chirp3-HD-Aoede", "Aoede", "Studio, expressive", "female", "premium"),
        _google("en-US-Chirp3-HD-Charon", "Charon", "Studio, grounded", "male", "premium"),
        _google("en-US-Chirp3-HD-Kore", "Kore", "Studio, confident", "female", "premium"),
        _google("en-US-Chirp3-HD-Puck", "Puck", "Studio, lively", "male", "premium"),
        # OpenAI gpt-4o-mini-tts (steerable, multilingual)
        _openai("coral", "Coral", "Warm, conversational", "female"),
        _openai("sage", "Sage", "Calm, thoughtful", "female"),
        _openai("shimmer", "Shimmer", "Bright, energetic", "female"),
        _openai("nova", "Nova", "Friendly, natural", "female"),
        _openai("alloy", "Alloy", "Neutral, balanced", "neutral"),
        _openai("ash", "Ash", "Clear, confident", "male"),
        _openai("echo", "Echo", "Smooth, deep", "male"),
        _openai("onyx", "Onyx", "Rich, authoritative", "male"),
        _openai("cedar", "Cedar", "Grounded, warm", "male"),
        _openai("marin", "Marin", "Expressive storyteller", "female"),
    )
}


class TTSService:
    """Synthesises speech with whichever provider the chosen voice belongs to."""

    VOICES = VOICES

    @cached_property
    def google_client(self) -> texttospeech.TextToSpeechClient:
        """Created on first use so a missing key file cannot break startup."""
        path = settings.GOOGLE_APPLICATION_CREDENTIALS
        if path and os.path.exists(path):
            credentials = service_account.Credentials.from_service_account_file(path)
            return texttospeech.TextToSpeechClient(credentials=credentials)

        if path:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS points to a missing file: %s", path)
        # Falls back to Application Default Credentials (workload identity, gcloud…).
        return texttospeech.TextToSpeechClient()

    @staticmethod
    def _openai_client() -> AsyncOpenAI:
        if not settings.OPENAI_API_KEY:
            raise ProviderNotConfiguredError("OpenAI", "OPENAI_API_KEY")
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2, timeout=120.0)

    @classmethod
    def voices(cls) -> list[dict[str, object]]:
        features = settings.public_features()
        result = []
        for voice in cls.VOICES.values():
            payload = voice.to_public()
            enabled = bool(features.get(voice.provider, False))
            if voice.provider == PROVIDER_GOOGLE:
                enabled = bool(settings.GOOGLE_APPLICATION_CREDENTIALS)
            payload["enabled"] = enabled and settings.storage_enabled
            result.append(payload)
        return result

    @classmethod
    def validate_voice(cls, voice_id: str) -> Voice:
        """Resolve a voice id, rejecting anything not in the catalogue."""
        voice = cls.VOICES.get(voice_id)
        if voice is not None:
            return voice
        # Any well-formed Google voice name is accepted, priced at the Neural2
        # rate, so the catalogue does not have to list every locale.
        if voice_id and _GOOGLE_VOICE_PATTERN.match(voice_id):
            return _google(voice_id, voice_id, "Google Cloud voice", "neutral")
        raise InvalidVoiceError(voice_id)

    def calculate_cost(self, text: str, voice_id: str = DEFAULT_VOICE) -> Decimal:
        """Price in wallet credits: (chars x provider rate x margin) x credit rate."""
        voice = self.validate_voice(voice_id)
        return usd_to_credits(Decimal(len(text)) * voice.cost_per_char_usd)

    @staticmethod
    def _language_code(voice_name: str) -> str:
        return "-".join(voice_name.split("-")[:2])

    def _synthesize_google(self, text: str, voice_name: str) -> bytes:
        response = self.google_client.synthesize_speech(
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

    async def _synthesize_openai(self, text: str, voice: str, instructions: str | None) -> bytes:
        async with self._openai_client() as client:
            response = await client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice=voice,
                input=text,
                instructions=instructions or "Speak naturally and clearly.",
                response_format="mp3",
            )
            return response.content

    async def generate_audio(
        self, text: str, voice_name: str = DEFAULT_VOICE, instructions: str | None = None
    ) -> str:
        """Synthesize MP3 audio, store it, and return its public URL."""
        voice = self.validate_voice(voice_name)
        text = text[:MAX_TTS_CHARS]

        if voice.provider == PROVIDER_OPENAI:
            audio_bytes = await self._synthesize_openai(
                text, voice.id.split("/", 1)[1], instructions
            )
        else:
            # The Google client is synchronous; keep it off the event loop.
            audio_bytes = await asyncio.to_thread(self._synthesize_google, text, voice.id)

        return await storage.upload_file_async(
            file_bytes=audio_bytes,
            destination_path=f"tts/{uuid.uuid4()}.mp3",
            content_type="audio/mpeg",
        )


tts_service = TTSService()
