"""Chat history endpoints and the streaming WebSocket.

The socket is long-lived while database work is short-lived, so each turn opens
its own session instead of holding one open for the whole conversation — an idle
transaction pinned to a socket is what exhausts a connection pool.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import datetime
from decimal import Decimal

import redis.asyncio as redis
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db, session_scope
from app.core.redis import ChatCache, get_redis
from app.core.security import get_current_user, verify_token_socket
from app.models.chat import ROLE_ASSISTANT, ROLE_USER, Chat, Message
from app.models.user import User
from app.services import billing
from app.services.file_processing import process_file
from app.services.llm.factory import LLMFactory
from app.services.llm.models import UnknownModelError
from app.services.llm.router import ModelRouter
from app.services.llm.schema import Attachment, ChatMessage
from app.services.llm.usage import Usage

logger = logging.getLogger(__name__)
router = APIRouter()

# WebSocket close codes
WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011

MAX_MESSAGE_CHARS = 32_000
CHAT_TITLE_CHARS = 40


# Schemas


class AttachmentSchema(BaseModel):
    id: str | None = None
    name: str = ""
    type: str = ""
    size: int = 0
    mime_type: str | None = None


class MessageSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    role: str
    content: str
    model: str | None = None
    attachments: list[AttachmentSchema] = Field(default_factory=list)
    created_at: datetime | None = None


class ChatSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    title: str | None = None
    created_at: datetime | None = None


class UserMessagePayload(BaseModel):
    type: str
    content: str = Field("", max_length=MAX_MESSAGE_CHARS)
    attachments: list[AttachmentSchema] = Field(default_factory=list)
    # Optional per-turn override. Without it a client has to reconnect to switch
    # models, which drops the conversation mid-session.
    model: str | None = None


# HTTP endpoints


@router.get("/list", response_model=list[ChatSchema])
async def get_user_chats(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Chat]:
    """The caller's chats, newest first."""
    result = await db.execute(
        select(Chat)
        .where(Chat.user_id == current_user.id)
        .order_by(desc(Chat.created_at))
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all())


@router.get("/history/{chat_id}", response_model=list[MessageSchema])
async def get_chat_history(
    chat_id: uuid.UUID,
    limit: int = Query(200, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Message]:
    """Full message history for one of the caller's chats."""
    owns_chat = await db.scalar(
        select(Chat.id).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    if not owns_chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    result = await db.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    return list(result.scalars().all())


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    redis_client: redis.Redis = Depends(get_redis),
) -> None:
    """Delete a chat, its messages and its cached history."""
    chat = await db.scalar(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    await db.delete(chat)
    await db.commit()

    try:
        await ChatCache(redis_client).clear_history(str(chat_id))
    except Exception as exc:
        # The durable copy is gone; a stale cache entry only wastes memory
        # until its TTL expires.
        logger.warning("Could not clear cached history for chat %s: %s", chat_id, exc)


@router.post("/upload")
async def upload_files_for_context(
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    redis_client: redis.Redis = Depends(get_redis),
) -> dict:
    """Stage files for the next chat message.

    Extracted content is held in Redis under a short-lived id; only the id and
    metadata go back to the client, so a large PDF never travels through the
    WebSocket.
    """
    if len(files) > settings.MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At most {settings.MAX_UPLOAD_FILES} files can be uploaded at once",
        )

    cache = ChatCache(redis_client)
    processed: list[dict] = []

    for file in files:
        try:
            result = await process_file(file)
            file_id = str(uuid.uuid4())
            await cache.save_temp_file(
                file_id,
                {
                    "type": result["type"],
                    "content": result["content"],
                    "mime_type": result.get("mime_type"),
                },
            )
            processed.append(
                {
                    "id": file_id,
                    "name": result.get("filename", file.filename),
                    "type": result["type"],
                    "size": file.size,
                    "mime_type": file.content_type,
                }
            )
        except HTTPException as exc:
            processed.append({"name": file.filename, "error": exc.detail})
        except Exception:
            logger.exception("Failed to process upload %s", file.filename)
            processed.append({"name": file.filename, "error": "Could not process this file"})

    return {"files": processed}


# WebSocket helpers


async def _send(websocket: WebSocket, data: dict) -> bool:
    """Send JSON, reporting whether the client is still there."""
    if websocket.client_state == WebSocketState.DISCONNECTED:
        return False
    try:
        await websocket.send_json(data)
        return True
    except (WebSocketDisconnect, ConnectionResetError, RuntimeError):
        return False
    except Exception as exc:
        logger.warning("WebSocket send failed: %s", exc)
        return False


async def _close(websocket: WebSocket, code: int = 1000, reason: str = "") -> None:
    try:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=code, reason=reason)
    except Exception:
        pass  # already gone


async def _send_error(websocket: WebSocket, message: str) -> bool:
    return await _send(websocket, {"type": "error", "message": message})


def _chat_title(text: str, attachments: list[AttachmentSchema]) -> str:
    """Derive a title from the first message."""
    text = (text or "").strip()
    if text:
        return f"{text[:CHAT_TITLE_CHARS]}…" if len(text) > CHAT_TITLE_CHARS else text
    if attachments:
        # Attachments are validated models here; the previous dict-style lookup
        # raised AttributeError whenever a file arrived without a name.
        return f"File: {attachments[0].name or 'Attachment'}"
    return "New Chat"


async def _resolve_attachments(
    cache: ChatCache, attachments: list[AttachmentSchema]
) -> list[Attachment]:
    """Swap staged upload ids for the content held in Redis."""
    resolved: list[Attachment] = []
    for item in attachments:
        if not item.id:
            continue
        staged = await cache.get_temp_file(item.id)
        if not staged:
            logger.info("Staged file %s expired before it was sent", item.id)
            continue
        resolved.append(
            Attachment(
                type=staged["type"],
                content=staged["content"],
                mime_type=staged.get("mime_type"),
            )
        )
    return resolved


async def _load_history(
    db: AsyncSession, cache: ChatCache, chat_id: uuid.UUID
) -> list[ChatMessage]:
    """Recent turns, from Redis when warm and from PostgreSQL otherwise."""
    try:
        history = await cache.get_history(str(chat_id))
        if history:
            return history
    except Exception as exc:
        logger.warning("Chat cache unavailable, falling back to the database: %s", exc)

    result = await db.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(desc(Message.created_at))
        .limit(settings.CHAT_HISTORY_LIMIT)
    )
    return [ChatMessage.from_text(m.role, m.content) for m in reversed(result.scalars().all())]


# WebSocket endpoint


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    model: str = "auto",
    chat_id: str | None = None,
    redis_client: redis.Redis = Depends(get_redis),
) -> None:
    """Streaming chat.

    Protocol — client sends ``{"type": "user_message", "content": str,
    "attachments": [...]}``; the server replies with ``system`` events
    (chat_id, route, cost), ``content`` deltas and ``error`` messages.
    """
    await websocket.accept()
    cache = ChatCache(redis_client)

    user_id: uuid.UUID | None = None
    user_email = "unknown"
    current_chat_id: uuid.UUID | None = None

    try:
        # Authentication and session setup
        token = websocket.query_params.get("token")
        if not token:
            await _close(websocket, WS_POLICY_VIOLATION, "Missing token")
            return

        async with session_scope() as db:
            user = await verify_token_socket(token, db)
            if not user:
                await _close(websocket, WS_POLICY_VIOLATION, "Invalid token")
                return
            user_id, user_email = user.id, user.email

            if not await billing.has_credits(db, user_id):
                await _send_error(websocket, "Insufficient credits.")
                await _close(websocket, WS_POLICY_VIOLATION, "Insufficient credits")
                return

            if chat_id:
                try:
                    requested = uuid.UUID(chat_id)
                except ValueError:
                    requested = None
                if requested:
                    current_chat_id = await db.scalar(
                        select(Chat.id).where(Chat.id == requested, Chat.user_id == user_id)
                    )

        logger.info("WebSocket connected for %s", user_email)

        while True:
            raw_data = await websocket.receive_text()

            try:
                payload = UserMessagePayload(**json.loads(raw_data))
            except (json.JSONDecodeError, TypeError, ValidationError) as exc:
                logger.info("Rejected malformed WebSocket payload: %s", exc)
                await _send_error(websocket, "Invalid message format")
                continue

            if payload.type != "user_message":
                continue

            user_text = payload.content

            # Balance is re-checked every turn: the original code only checked at
            # connect time, so one socket could keep generating on an empty wallet.
            async with session_scope() as db:
                if not await billing.has_credits(db, user_id):
                    await _send_error(websocket, "Insufficient credits. Please top up.")
                    await _close(websocket, WS_POLICY_VIOLATION, "Insufficient credits")
                    return

            try:
                selected_model = ModelRouter.determine_model(user_text, payload.model or model)
            except UnknownModelError as exc:
                await _send_error(websocket, str(exc))
                continue

            # Lazily create the chat on the first message.
            if current_chat_id is None:
                async with session_scope() as db:
                    chat = Chat(user_id=user_id, title=_chat_title(user_text, payload.attachments))
                    db.add(chat)
                    await db.flush()
                    current_chat_id = chat.id

                if not await _send(
                    websocket,
                    {"type": "system", "event": "chat_id", "payload": str(current_chat_id)},
                ):
                    return

            # Sending happens outside the transaction: a write held open across
            # a network round-trip to the client is how connections get pinned.
            if not await _send(
                websocket, {"type": "system", "event": "route", "payload": selected_model}
            ):
                return

            llm_attachments = await _resolve_attachments(cache, payload.attachments)

            async with session_scope() as db:
                db.add(
                    Message(
                        chat_id=current_chat_id,
                        role=ROLE_USER,
                        content=user_text,
                        model=selected_model,
                        attachments=[a.model_dump(exclude={"id"}) for a in payload.attachments],
                    )
                )
                # autoflush is off, so this query returns the previous turns only.
                history = await _load_history(db, cache, current_chat_id)

            try:
                await cache.add_message(str(current_chat_id), ROLE_USER, user_text)
            except Exception as exc:
                logger.warning("Could not cache the user message: %s", exc)

            # The cached/stored copy of this turn has no attachment payloads, so
            # replace it with the rich version before sending it to the model.
            latest = ChatMessage.from_text(ROLE_USER, user_text)
            latest.attachments = llm_attachments
            if history and history[-1].role == ROLE_USER and history[-1].text == user_text:
                history[-1] = latest
            else:
                history.append(latest)

            # Generate
            usage = Usage()
            full_response = ""
            client_gone = False
            stream = None

            try:
                provider = LLMFactory.get_provider(selected_model)
                stream = provider.generate_stream(history, selected_model, usage)

                async with asyncio.timeout(settings.CHAT_STREAM_TIMEOUT_SECONDS):
                    async for chunk in stream:
                        full_response += chunk
                        if not await _send(websocket, {"type": "content", "delta": chunk}):
                            client_gone = True
                            break
            except (WebSocketDisconnect, ConnectionResetError):
                client_gone = True
            except TimeoutError:
                await _send_error(websocket, "The model took too long to respond.")
            except Exception as exc:
                logger.exception("Generation failed with %s", selected_model)
                # Provider errors can name keys and endpoints; only echo them
                # back to the client outside production.
                detail = "Generation failed. Please try again."
                if not settings.is_production:
                    detail = f"Generation failed: {exc}"
                await _send_error(websocket, detail)
            finally:
                if stream is not None:
                    # Closing the generator ends the upstream HTTP request rather
                    # than leaving it streaming tokens nobody reads.
                    with contextlib.suppress(Exception):
                        await stream.aclose()

            # Bill for whatever was produced, even if the client vanished mid-stream.
            usage.ensure_validity(prompt_text=user_text, completion_text=full_response)
            if not full_response.strip():
                if client_gone:
                    return
                continue

            cost = await _record_turn(
                user_id=user_id,
                chat_id=current_chat_id,
                model=selected_model,
                response=full_response,
                usage=usage,
            )

            try:
                await cache.add_message(str(current_chat_id), ROLE_ASSISTANT, full_response)
            except Exception as exc:
                logger.warning("Could not cache the assistant message: %s", exc)

            if client_gone:
                return

            await _send(websocket, {"type": "system", "event": "cost", "payload": str(cost)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for %s", user_email)
    except ConnectionResetError:
        logger.info("WebSocket reset by %s", user_email)
    except Exception:
        logger.exception("Unhandled WebSocket error for %s", user_email)
        await _close(websocket, WS_INTERNAL_ERROR)
    finally:
        logger.info("WebSocket closed for %s", user_email)


async def _record_turn(
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
    model: str,
    response: str,
    usage: Usage,
) -> Decimal:
    """Charge for a completed turn and store the assistant message.

    The response has already been delivered, so the debit is allowed to overdraw;
    the next turn's balance check stops the user before they can spend more.
    """
    provider = LLMFactory.get_provider(model)
    cost = provider.calculate_cost(usage, model)

    try:
        async with session_scope() as db:
            await billing.debit(db, user_id, cost, allow_overdraft=True)
            db.add(
                Message(
                    chat_id=chat_id,
                    role=ROLE_ASSISTANT,
                    content=response,
                    model=model,
                    cost=cost,
                    tokens=usage.total_tokens,
                )
            )
    except Exception:
        # Never fail the conversation over bookkeeping — but make it findable.
        logger.exception("Failed to record turn for user %s (cost %s)", user_id, cost)

    return cost
