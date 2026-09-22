"""Speech, image and avatar-video generation.

Generation is asynchronous: the endpoint prices the request, debits the wallet
atomically, queues a Celery task and returns 202. The worker refunds if the job
fails, so credits are never taken for an asset that was never delivered.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.core.ratelimit import limit_by_user
from app.core.security import get_current_user
from app.models.chat import Message
from app.models.media import (
    PENDING_URL,
    STATUS_PROCESSING,
    GeneratedAudio,
    GeneratedImage,
    GeneratedVideo,
)
from app.models.user import User
from app.schemas.media import (
    AudioGenerationRequest,
    GeneratedAudioResponse,
    GeneratedImageResponse,
    GeneratedVideoResponse,
    ImageGenerationRequest,
    VideoGenerationRequest,
)
from app.services import billing
from app.services.media.images import UnsupportedImageOptionError, image_service
from app.services.media.tts import DEFAULT_VOICE, MAX_TTS_CHARS, InvalidVoiceError, tts_service
from app.services.media.video_did import DIDNotConfiguredError, did_service
from app.services.storage import storage
from app.workers.tasks import generate_avatar_task, generate_image_task, generate_tts_task

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_TYPES = {"image/jpeg", "image/png", "image/webp"}
UPLOAD_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}

_media_limit = Depends(limit_by_user("media", settings.RATE_LIMIT_MEDIA_PER_HOUR, 3600))


def _require_storage() -> None:
    if not settings.storage_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Media storage is not configured on this server",
        )


async def _reserve_credits(db: AsyncSession, user: User, cost: Decimal, what: str) -> None:
    """Debit up front so two concurrent requests cannot spend the same balance."""
    try:
        await billing.debit(db, user.id, cost)
    except billing.InsufficientCreditsError as exc:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient credits for {what}. Cost: {exc.required}, balance: {exc.balance}",
        ) from None


# ── catalogues ────────────────────────────────────────────────────────────


@router.get("/voices")
async def list_voices() -> dict:
    """Every voice the speech endpoints accept, with prices in credits."""
    return {"voices": tts_service.voices(), "default": DEFAULT_VOICE, "max_chars": MAX_TTS_CHARS}


@router.get("/images/options")
async def list_image_options() -> dict:
    """Model, quality and size combinations the image endpoint accepts.

    Served so the client can build its picker from the same table that prices
    and validates the request, instead of hard-coding a list that drifts.
    """
    return {"models": image_service.options(), "default": "gpt-image-2"}


# ── audio ─────────────────────────────────────────────────────────────────


@router.get("/list", response_model=list[GeneratedAudioResponse])
async def list_generated_audio(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[GeneratedAudio]:
    result = await db.execute(
        select(GeneratedAudio)
        .where(GeneratedAudio.user_id == current_user.id)
        .order_by(GeneratedAudio.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED, dependencies=[_media_limit])
async def generate_audio_direct(
    request: AudioGenerationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Generate speech from arbitrary text."""
    _require_storage()
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text is required")

    try:
        voice = tts_service.validate_voice(request.voice_name)
    except InvalidVoiceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    cost = tts_service.calculate_cost(text, voice.id)
    await _reserve_credits(db, current_user, cost, "speech generation")
    await db.commit()

    task = generate_tts_task.delay(
        text=text,
        chat_id=None,
        message_id=None,
        user_id=str(current_user.id),
        cost=float(cost),
        voice_name=voice.id,
        instructions=request.instructions,
    )
    return {"task_id": task.id, "status": "processing", "cost_deducted": float(cost)}


@router.post("/tts/{message_id}", status_code=status.HTTP_202_ACCEPTED, dependencies=[_media_limit])
async def trigger_tts_generation(
    message_id: uuid.UUID,
    voice: str = Query(DEFAULT_VOICE, max_length=80),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Read one of the caller's chat messages aloud."""
    _require_storage()
    message = await db.scalar(
        select(Message).options(selectinload(Message.chat)).where(Message.id == message_id)
    )
    if not message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    if message.chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this message",
        )

    existing = await db.scalar(
        select(GeneratedAudio).where(GeneratedAudio.source_message_id == message_id)
    )
    if existing:
        return {
            "status": "exists",
            "message_id": str(message_id),
            "audio_id": str(existing.id),
            "audio_url": existing.public_url,
            "info": "Audio already generated.",
        }

    try:
        voice_spec = tts_service.validate_voice(voice)
    except InvalidVoiceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    text = message.content.strip()[:MAX_TTS_CHARS]
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message has no text")

    cost = tts_service.calculate_cost(text, voice_spec.id)
    await _reserve_credits(db, current_user, cost, "speech generation")
    await db.commit()

    task = generate_tts_task.delay(
        text=text,
        chat_id=str(message.chat_id),
        message_id=str(message.id),
        user_id=str(current_user.id),
        cost=float(cost),
        voice_name=voice_spec.id,
    )
    return {
        "status": "processing",
        "task_id": task.id,
        "message_id": str(message_id),
        "cost_deducted": float(cost),
        "message": "TTS generation started.",
    }


@router.get("/tts/{message_id}", response_model=GeneratedAudioResponse | None)
async def read_message_audio(
    message_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> GeneratedAudio | None:
    """The audio generated for a chat message, once the worker has stored it."""
    return await db.scalar(
        select(GeneratedAudio).where(
            GeneratedAudio.source_message_id == message_id,
            GeneratedAudio.user_id == current_user.id,
        )
    )


@router.delete("/audio/{audio_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_audio(
    audio_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    audio = await db.scalar(
        select(GeneratedAudio).where(
            GeneratedAudio.id == audio_id, GeneratedAudio.user_id == current_user.id
        )
    )
    if not audio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio not found")

    await db.delete(audio)
    await db.commit()
    # Storage is cleaned up after the row is gone: an orphaned object is
    # recoverable, a dangling database row pointing at nothing is not.
    await storage.delete_file_async(audio.public_url)


# ── images ────────────────────────────────────────────────────────────────


@router.get("/images/list", response_model=list[GeneratedImageResponse])
async def list_generated_images(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[GeneratedImage]:
    result = await db.execute(
        select(GeneratedImage)
        .where(GeneratedImage.user_id == current_user.id)
        .order_by(GeneratedImage.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


@router.post("/generate-image", status_code=status.HTTP_202_ACCEPTED, dependencies=[_media_limit])
async def generate_image(
    request: ImageGenerationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Generate an image, optionally editing a reference image."""
    _require_storage()
    try:
        spec = image_service.get_model(request.model)
        cost = image_service.calculate_cost(
            model=request.model, quality=request.quality, size=request.size
        )
    except UnsupportedImageOptionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    if request.reference_image_url and not spec.supports_reference:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{spec.display_name} does not support reference images",
        )

    await _reserve_credits(db, current_user, cost, "image generation")
    await db.commit()

    task = generate_image_task.delay(
        prompt=request.prompt,
        user_id=str(current_user.id),
        model=request.model,
        cost=float(cost),
        size=request.size,
        quality=request.quality,
        reference_image_url=request.reference_image_url,
    )
    return {
        "task_id": task.id,
        "status": "processing",
        "estimated_cost": float(cost),
        "message": "Image generation started.",
    }


@router.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image(
    image_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    image = await db.scalar(
        select(GeneratedImage).where(
            GeneratedImage.id == image_id, GeneratedImage.user_id == current_user.id
        )
    )
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    await db.delete(image)
    await db.commit()
    await storage.delete_file_async(image.public_url)


# ── avatar video ──────────────────────────────────────────────────────────


@router.get("/videos/list", response_model=list[GeneratedVideoResponse])
async def list_generated_videos(
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[GeneratedVideo]:
    result = await db.execute(
        select(GeneratedVideo)
        .where(GeneratedVideo.user_id == current_user.id)
        .order_by(GeneratedVideo.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


@router.post("/generate-avatar", status_code=status.HTTP_202_ACCEPTED, dependencies=[_media_limit])
async def generate_avatar(
    request: VideoGenerationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Render a talking-avatar video from a script and an avatar image."""
    _require_storage()
    try:
        voice = tts_service.validate_voice(request.voice_name)
    except InvalidVoiceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    try:
        # Priced from the script length plus the narration itself.
        cost = did_service.calculate_cost_for_script(request.text) + tts_service.calculate_cost(
            request.text, voice.id
        )
    except DIDNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from None

    await _reserve_credits(db, current_user, cost, "avatar generation")

    video = GeneratedVideo(
        user_id=current_user.id,
        storage_path=PENDING_URL,
        public_url=PENDING_URL,
        script_text=request.text,
        source_audio_url=PENDING_URL,
        avatar_image_url=request.avatar_url,
        provider="d-id",
        model="talks",
        cost=cost,
        status=STATUS_PROCESSING,
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)

    task = generate_avatar_task.delay(
        script_text=request.text,
        voice_name=voice.id,
        avatar_url=request.avatar_url,
        user_id=str(current_user.id),
        video_db_id=str(video.id),
        cost=float(cost),
    )
    return {
        "task_id": task.id,
        "video_id": str(video.id),
        "status": "processing",
        "estimated_cost": float(cost),
        "message": "Avatar generation started.",
    }


@router.delete("/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete a video along with the audio track generated for it."""
    video = await db.scalar(
        select(GeneratedVideo).where(
            GeneratedVideo.id == video_id, GeneratedVideo.user_id == current_user.id
        )
    )
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    public_url, audio_url = video.public_url, video.source_audio_url

    await db.delete(video)
    await db.commit()

    for url in (public_url, audio_url):
        if url and url != PENDING_URL:
            await storage.delete_file_async(url)


# ── uploads ───────────────────────────────────────────────────────────────


@router.post(
    "/upload",
    dependencies=[Depends(limit_by_user("uploads", settings.RATE_LIMIT_UPLOADS_PER_HOUR, 3600))],
)
async def upload_media_asset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Upload an avatar or reference image and return its public URL."""
    _require_storage()
    if file.content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only JPEG, PNG or WebP are allowed.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit",
        )

    # The extension comes from the validated content type, never from the
    # client-supplied filename, which could contain path traversal characters.
    extension = UPLOAD_EXTENSIONS[file.content_type]
    public_url = await storage.upload_file_async(
        file_bytes=file_bytes,
        destination_path=f"uploads/{current_user.id}/{uuid.uuid4()}.{extension}",
        content_type=file.content_type,
    )
    return {"public_url": public_url}
