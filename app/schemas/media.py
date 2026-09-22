from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.services.media.tts import DEFAULT_VOICE, MAX_TTS_CHARS

MAX_PROMPT_LENGTH = 4000
MAX_TTS_INSTRUCTIONS = 300


class AudioGenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TTS_CHARS)
    voice_name: str = DEFAULT_VOICE
    # Delivery hints for steerable voices ("cheerful", "slow and calm").
    instructions: str | None = Field(None, max_length=MAX_TTS_INSTRUCTIONS)


class GeneratedAudioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    public_url: str
    text_prompt: str
    voice_name: str | None = None
    provider: str
    source_message_id: UUID | None = None
    cost: float  # display value; the ledger keeps full Decimal precision
    created_at: datetime | None = None


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    model: str = "gpt-image-2"
    size: str = "1024x1024"
    quality: str = "medium"
    reference_image_url: str | None = None


class GeneratedImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    public_url: str
    prompt: str
    reference_image_url: str | None = None
    revised_prompt: str | None = None
    model: str
    size: str
    quality: str
    cost: float  # display value; the ledger keeps full Decimal precision
    created_at: datetime | None = None


class VideoGenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TTS_CHARS)
    voice_name: str = DEFAULT_VOICE
    avatar_url: str


class GeneratedVideoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    public_url: str | None = None
    thumbnail_url: str | None = None
    status: str
    script_text: str
    avatar_image_url: str
    error_message: str | None = None
    cost: float  # display value; the ledger keeps full Decimal precision
    created_at: datetime | None = None
