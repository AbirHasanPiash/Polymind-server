"""Background media generation.

Credits are debited by the API endpoint *before* a task is queued, so every
failure path here refunds them (:func:`_refund`). Tasks are not retried
automatically: a half-finished generation has usually already cost money at the
provider, so the safe response to a failure is to refund the user and let them
decide whether to try again.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal

import httpx

from app.core.config import settings
from app.core.database import session_scope
from app.models.media import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING_EXTERNAL,
    GeneratedAudio,
    GeneratedImage,
    GeneratedVideo,
)
from app.services import billing
from app.services.media.images import image_service
from app.services.media.tts import tts_service
from app.services.media.video_did import DIDError, did_service
from app.services.storage import storage
from app.workers.celery_app import celery_app, run_async

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024


async def _refund(user_id: str, amount: Decimal, reason: str) -> None:
    """Return credits for work that was paid for but never delivered."""
    try:
        async with session_scope() as db:
            await billing.refund(db, uuid.UUID(user_id), amount, reason)
    except Exception:
        # A lost refund needs a human, so make it loud and searchable.
        logger.exception("REFUND FAILED user=%s amount=%s reason=%s", user_id, amount, reason)


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        if len(response.content) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"Downloaded asset exceeds {MAX_DOWNLOAD_BYTES} bytes")
        return response.content


# Text to speech


@celery_app.task(name="generate_tts_task")
def generate_tts_task(
    text: str,
    chat_id: str | None,
    message_id: str | None,
    user_id: str,
    cost: float,
    voice_name: str = "en-US-Neural2-F",
    instructions: str | None = None,
) -> dict:
    """Synthesize speech, store it, and record the result."""

    async def _process() -> dict:
        public_url = await tts_service.generate_audio(
            text, voice_name=voice_name, instructions=instructions
        )
        storage_path = storage.key_from_url(public_url) or public_url

        async with session_scope() as db:
            db.add(
                GeneratedAudio(
                    user_id=uuid.UUID(user_id),
                    storage_path=storage_path,
                    public_url=public_url,
                    text_prompt=text[:500],
                    source_message_id=uuid.UUID(message_id) if message_id else None,
                    provider="openai" if voice_name.startswith("openai/") else "google",
                    voice_name=voice_name,
                    cost=Decimal(str(cost)),
                )
            )
        return {"status": "success", "audio_url": public_url}

    try:
        return run_async(_process())
    except Exception as exc:
        logger.exception("TTS generation failed for user %s", user_id)
        run_async(_refund(user_id, Decimal(str(cost)), "TTS generation failed"))
        return {"status": "failed", "error": str(exc)}


# Image generation


@celery_app.task(name="generate_image_task")
def generate_image_task(
    prompt: str,
    user_id: str,
    model: str,
    cost: float,
    size: str = "1024x1024",
    quality: str = "medium",
    reference_image_url: str | None = None,
) -> dict:
    """Generate an image, store it, and record the result."""

    async def _process() -> dict:
        result = await image_service.generate_and_upload(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            user_id=user_id,
            reference_image_url=reference_image_url,
        )

        async with session_scope() as db:
            db.add(
                GeneratedImage(
                    user_id=uuid.UUID(user_id),
                    storage_path=result["storage_path"],
                    public_url=result["public_url"],
                    prompt=prompt,
                    revised_prompt=result["revised_prompt"],
                    model=model,
                    size=size,
                    quality=quality,
                    cost=Decimal(str(cost)),
                    reference_image_url=reference_image_url,
                )
            )
        return {"status": "success", "image_url": result["public_url"]}

    try:
        return run_async(_process())
    except Exception as exc:
        logger.exception("Image generation failed for user %s", user_id)
        run_async(_refund(user_id, Decimal(str(cost)), "Image generation failed"))
        return {"status": "failed", "error": str(exc)}


# Avatar video
#
# D-ID renders take minutes. Rather than sleeping inside a worker slot for the
# whole render, the job is submitted here and polled by a separate task that
# reschedules itself, so the worker stays free between checks.


@celery_app.task(name="generate_avatar_task")
def generate_avatar_task(
    script_text: str,
    voice_name: str,
    avatar_url: str,
    user_id: str,
    video_db_id: str,
    cost: float,
) -> dict:
    """Narrate the script, submit the D-ID job, and hand off to the poller."""

    async def _process() -> dict:
        audio_url = await tts_service.generate_audio(script_text, voice_name=voice_name)

        # D-ID must fetch the avatar from a stable URL; mirror foreign images.
        avatar_public_url = avatar_url
        if not storage.key_from_url(avatar_url):
            avatar_bytes = await _download(avatar_url)
            avatar_public_url = await storage.upload_file_async(
                file_bytes=avatar_bytes,
                destination_path=f"avatars/{user_id}/{uuid.uuid4()}.png",
                content_type="image/png",
            )

        job_id = await did_service.create_talk(source_url=avatar_public_url, audio_url=audio_url)

        async with session_scope() as db:
            video = await db.get(GeneratedVideo, uuid.UUID(video_db_id))
            if video is None:
                raise RuntimeError(f"Video record {video_db_id} disappeared")
            video.source_audio_url = audio_url
            video.avatar_image_url = avatar_public_url
            video.external_job_id = job_id
            video.status = STATUS_PROCESSING_EXTERNAL

        poll_avatar_task.apply_async(
            args=[video_db_id, job_id, user_id, cost, 1],
            countdown=settings.DID_POLL_INTERVAL_SECONDS,
        )
        return {"status": "processing", "job_id": job_id}

    try:
        return run_async(_process())
    except Exception as exc:
        logger.exception("Avatar submission failed for user %s", user_id)
        run_async(_fail_video(video_db_id, user_id, cost, str(exc)))
        return {"status": "failed", "error": str(exc)}


@celery_app.task(name="poll_avatar_task")
def poll_avatar_task(
    video_db_id: str,
    job_id: str,
    user_id: str,
    cost: float,
    attempt: int,
) -> dict:
    """Check a D-ID job; finish it, reschedule the next check, or give up."""

    def _reschedule() -> dict:
        poll_avatar_task.apply_async(
            args=[video_db_id, job_id, user_id, cost, attempt + 1],
            countdown=settings.DID_POLL_INTERVAL_SECONDS,
        )
        return {"status": "processing", "attempt": attempt}

    async def _process() -> dict:
        result_url = await did_service.check_status(job_id)
        if result_url is None:
            return {"status": "pending"}

        video_bytes = await _download(result_url)
        storage_path = f"generated_videos/{user_id}/{uuid.uuid4()}.mp4"
        public_url = await storage.upload_file_async(video_bytes, storage_path, "video/mp4")

        async with session_scope() as db:
            video = await db.get(GeneratedVideo, uuid.UUID(video_db_id))
            if video is None:
                raise RuntimeError(f"Video record {video_db_id} disappeared")
            video.status = STATUS_COMPLETED
            video.public_url = public_url
            video.storage_path = storage_path

        return {"status": "success", "video_url": public_url}

    exhausted = attempt >= settings.DID_POLL_MAX_ATTEMPTS

    try:
        result = run_async(_process())
    except DIDError as exc:
        # The render itself failed — retrying cannot help.
        logger.error("D-ID job %s failed: %s", job_id, exc)
        run_async(_fail_video(video_db_id, user_id, cost, str(exc)))
        return {"status": "failed", "error": str(exc)}
    except Exception as exc:
        # A network blip while polling must not throw away a running render.
        if not exhausted:
            logger.warning("Poll %s for job %s failed (%s); retrying", attempt, job_id, exc)
            return _reschedule()
        logger.exception("Avatar rendering failed for user %s", user_id)
        run_async(_fail_video(video_db_id, user_id, cost, str(exc)))
        return {"status": "failed", "error": str(exc)}

    if result["status"] != "pending":
        return result

    if exhausted:
        timeout = settings.DID_POLL_MAX_ATTEMPTS * settings.DID_POLL_INTERVAL_SECONDS
        message = f"D-ID job {job_id} did not finish within {timeout}s"
        logger.error(message)
        run_async(_fail_video(video_db_id, user_id, cost, message))
        return {"status": "failed", "error": message}

    return _reschedule()


async def _fail_video(video_db_id: str, user_id: str, cost: float, error: str) -> None:
    """Mark the video failed and refund the user."""
    try:
        async with session_scope() as db:
            video = await db.get(GeneratedVideo, uuid.UUID(video_db_id))
            if video is not None:
                video.status = STATUS_FAILED
                video.error_message = error[:1000]
    except Exception:
        logger.exception("Could not mark video %s as failed", video_db_id)

    await _refund(user_id, Decimal(str(cost)), "Avatar generation failed")
