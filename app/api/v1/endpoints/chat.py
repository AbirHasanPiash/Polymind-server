"""Chat: conversation management over HTTP and streaming generation over a
WebSocket.

The socket is long-lived while database work is short-lived, so each turn opens
its own session instead of holding one open for the whole conversation — an idle
transaction pinned to a socket is what exhausts a connection pool.

WebSocket protocol (v2)
-----------------------
Client → server
  ``{"type": "user_message", "content", "attachments": [{"id"}], "model" | "models",
     "effort", "edit_message_id"}``
  ``{"type": "regenerate", "model" | "models", "effort"}``
  ``{"type": "stop"}`` · ``{"type": "ping"}``

Server → client
  ``{"type": "system", "event": "ready" | "chat_id" | "title" | "stopped" | "truncated"}``
  ``{"type": "turn_start", "user_message_id", "slots": [{"slot", "model", "intent", "reason"}]}``
  ``{"type": "content", "slot", "delta"}``
  ``{"type": "message_done", "slot", "message_id", "model", "served_model", "cost",
     "prompt_tokens", "completion_tokens", "duration_ms", "finish_reason", "notes"}``
  ``{"type": "turn_done", "balance"}``
  ``{"type": "error", "message", "code", "slot"}`` · ``{"type": "pong"}``
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import redis.asyncio as redis
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.websockets import WebSocketState
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db, session_scope
from app.core.ratelimit import limit_by_user
from app.core.redis import ChatCache, get_redis
from app.core.security import get_current_user, verify_token_socket
from app.models.chat import (
    FINISH_ERROR,
    FINISH_INTERRUPTED,
    FINISH_REFUSAL,
    MODE_ARENA,
    MODE_CHAT,
    ROLE_ASSISTANT,
    ROLE_USER,
    Chat,
    Message,
)
from app.models.user import User
from app.schemas.chat import (
    AttachmentSchema,
    ChatSchema,
    ChatSummarySchema,
    ChatUpdateSchema,
    ControlPayload,
    MessageSchema,
    RegeneratePayload,
    SearchHitSchema,
    SharedChatSchema,
    ShareSchema,
    UserMessagePayload,
)
from app.services import billing, chat_service
from app.services.file_processing import process_file
from app.services.llm.base import GenerationOptions, ModelRefusedError, ProviderNotConfiguredError
from app.services.llm.factory import LLMFactory
from app.services.llm.models import UnknownModelError, get_spec
from app.services.llm.router import ModelRouter, Route
from app.services.llm.schema import Attachment, ChatMessage
from app.services.llm.titles import fallback_title, generate_title
from app.services.llm.usage import Outcome, Usage

logger = logging.getLogger(__name__)
router = APIRouter()

# WebSocket close codes
WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011

_INCOMING = TypeAdapter(UserMessagePayload | RegeneratePayload | ControlPayload)


# ── HTTP: conversations ───────────────────────────────────────────────────


async def _owned_chat(db: AsyncSession, chat_id: uuid.UUID, user_id: uuid.UUID) -> Chat:
    chat = await db.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


@router.get("/list", response_model=list[ChatSummarySchema])
async def list_chats(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None, max_length=100),
    pinned: bool | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ChatSummarySchema]:
    """The caller's chats: pinned first, then most recently active."""
    message_count = (
        select(func.count(Message.id)).where(Message.chat_id == Chat.id).scalar_subquery()
    )
    stmt = select(Chat, message_count.label("message_count")).where(Chat.user_id == current_user.id)
    if q:
        stmt = stmt.where(Chat.title.ilike(f"%{q}%"))
    if pinned is not None:
        stmt = stmt.where(Chat.pinned.is_(pinned))
    stmt = (
        stmt.order_by(desc(Chat.pinned), desc(func.coalesce(Chat.updated_at, Chat.created_at)))
        .offset(offset)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [
        ChatSummarySchema(**ChatSchema.model_validate(chat).model_dump(), message_count=count or 0)
        for chat, count in rows
    ]


@router.get("/search", response_model=list[SearchHitSchema])
async def search_messages(
    q: str = Query(..., min_length=2, max_length=100),
    limit: int = Query(30, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[SearchHitSchema]:
    """Full-text-ish search across the caller's messages and titles."""
    pattern = f"%{q}%"
    stmt = (
        select(Message, Chat.title)
        .join(Chat, Chat.id == Message.chat_id)
        .where(Chat.user_id == current_user.id)
        .where(or_(Message.content.ilike(pattern), Chat.title.ilike(pattern)))
        .order_by(desc(Message.created_at))
        .limit(limit)
    )
    hits: list[SearchHitSchema] = []
    for message, title in (await db.execute(stmt)).all():
        content = message.content or ""
        index = content.lower().find(q.lower())
        start = max(0, index - 60) if index >= 0 else 0
        snippet = content[start : start + 180].replace("\n", " ").strip()
        hits.append(
            SearchHitSchema(
                chat_id=message.chat_id,
                chat_title=title,
                message_id=message.id,
                role="user" if message.role == ROLE_USER else "assistant",
                snippet=("…" if start > 0 else "")
                + snippet
                + ("…" if len(content) > start + 180 else ""),
                created_at=message.created_at,
            )
        )
    return hits


@router.get("/history/{chat_id}", response_model=list[MessageSchema])
async def get_chat_history(
    chat_id: uuid.UUID,
    limit: int = Query(300, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Message]:
    """Full message history for one of the caller's chats."""
    await _owned_chat(db, chat_id, current_user.id)
    result = await db.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    return list(result.scalars().all())


@router.get("/{chat_id}", response_model=ChatSchema)
async def get_chat(
    chat_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Chat:
    return await _owned_chat(db, chat_id, current_user.id)


@router.patch("/{chat_id}", response_model=ChatSchema)
async def update_chat(
    chat_id: uuid.UUID,
    payload: ChatUpdateSchema,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Chat:
    """Rename, pin or set per-conversation instructions."""
    chat = await _owned_chat(db, chat_id, current_user.id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(chat, field, value)
    await db.commit()
    await db.refresh(chat)
    return chat


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    redis_client: redis.Redis = Depends(get_redis),
) -> None:
    """Delete a chat, its messages and its cached history."""
    chat = await _owned_chat(db, chat_id, current_user.id)
    await db.delete(chat)
    await db.commit()

    try:
        await ChatCache(redis_client).clear_history(str(chat_id))
    except Exception as exc:
        # The durable copy is gone; a stale cache entry only wastes memory
        # until its TTL expires.
        logger.warning("Could not clear cached history for chat %s: %s", chat_id, exc)


@router.get("/{chat_id}/export")
async def export_chat(
    chat_id: uuid.UUID,
    format: str = Query("markdown", pattern="^(markdown|json)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    """Download the conversation as Markdown or JSON."""
    chat = await _owned_chat(db, chat_id, current_user.id)
    messages = list(
        (
            await db.execute(
                select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    stem = "".join(c if c.isalnum() or c in " -_" else "" for c in (chat.title or "conversation"))
    stem = (stem.strip().replace(" ", "-") or "conversation")[:60]
    if format == "json":
        body, media_type, ext = chat_service.export_json(chat, messages), "application/json", "json"
    else:
        body, media_type, ext = chat_service.export_markdown(chat, messages), "text/markdown", "md"
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{stem}.{ext}"'},
    )


@router.post("/{chat_id}/share", response_model=ShareSchema)
async def share_chat(
    chat_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ShareSchema:
    """Create (or return) a public read-only link for the conversation."""
    chat = await _owned_chat(db, chat_id, current_user.id)
    if not chat.share_token:
        chat.share_token = chat_service.new_share_token()
        chat.shared_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(chat)
    return ShareSchema(
        share_token=chat.share_token,
        url=chat_service.share_url(chat.share_token),
        shared_at=chat.shared_at,
    )


@router.delete("/{chat_id}/share", status_code=status.HTTP_204_NO_CONTENT)
async def unshare_chat(
    chat_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    chat = await _owned_chat(db, chat_id, current_user.id)
    chat.share_token = None
    chat.shared_at = None
    await db.commit()


@router.get("/shared/{token}", response_model=SharedChatSchema)
async def read_shared_chat(token: str, db: AsyncSession = Depends(get_db)) -> SharedChatSchema:
    """Public view of a shared conversation. No authentication."""
    chat = await db.scalar(select(Chat).where(Chat.share_token == token))
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="This link is no longer active"
        )
    messages = (
        (
            await db.execute(
                select(Message).where(Message.chat_id == chat.id).order_by(Message.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    return SharedChatSchema(
        title=chat.title,
        mode=chat.mode,
        created_at=chat.created_at,
        shared_at=chat.shared_at,
        messages=[
            {
                "id": m.id,
                "role": "user" if m.role == ROLE_USER else "assistant",
                "content": m.content,
                "model": m.model,
                "parent_id": m.parent_id,
                "created_at": m.created_at,
            }
            for m in messages
        ],
    )


@router.post(
    "/upload",
    dependencies=[Depends(limit_by_user("uploads", settings.RATE_LIMIT_UPLOADS_PER_HOUR, 3600))],
)
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


# ── WebSocket helpers ─────────────────────────────────────────────────────


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


async def _send_error(
    websocket: WebSocket, message: str, code: str = "error", slot: int | None = None
) -> bool:
    return await _send(websocket, {"type": "error", "message": message, "code": code, "slot": slot})


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
                type=staged["type"], content=staged["content"], mime_type=staged.get("mime_type")
            )
        )
    return resolved


async def _stream_until(stream, stop: asyncio.Event):
    """Iterate a provider stream, ending early when ``stop`` is set.

    Racing each chunk against the stop event means "Stop" takes effect at once
    rather than after the provider's next chunk, and closing the generator ends
    the upstream request instead of leaving it streaming tokens nobody reads.
    """
    stop_task = asyncio.ensure_future(stop.wait())
    try:
        while True:
            next_task = asyncio.ensure_future(stream.__anext__())
            done, _ = await asyncio.wait(
                {next_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if stop_task in done and next_task not in done:
                next_task.cancel()
                with contextlib.suppress(BaseException):
                    await next_task
                return
            try:
                chunk = next_task.result()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        stop_task.cancel()
        with contextlib.suppress(BaseException):
            await stop_task
        with contextlib.suppress(Exception):
            await stream.aclose()


class ChatSession:
    """State for one open socket: who is talking, which chat, and the turn in flight."""

    def __init__(self, websocket: WebSocket, cache: ChatCache, user: User) -> None:
        self.websocket = websocket
        self.cache = cache
        self.user_id: uuid.UUID = user.id
        self.user_email = user.email
        prefs = user.preferences if isinstance(user.preferences, dict) else {}
        self.custom_instructions: str | None = prefs.get("custom_instructions")
        self.chat_id: uuid.UUID | None = None
        self.chat_mode: str = MODE_CHAT
        self.chat_prompt: str | None = None
        self.chat_title: str | None = None
        self.turn_task: asyncio.Task | None = None
        self.stop_event = asyncio.Event()
        self.background: set[asyncio.Task] = set()

    @property
    def busy(self) -> bool:
        return self.turn_task is not None and not self.turn_task.done()

    async def adopt_chat(self, requested: str | None) -> None:
        if not requested:
            return
        try:
            chat_id = uuid.UUID(requested)
        except ValueError:
            return
        async with session_scope() as db:
            chat = await db.scalar(
                select(Chat).where(Chat.id == chat_id, Chat.user_id == self.user_id)
            )
            if chat:
                self.chat_id = chat.id
                self.chat_mode = chat.mode or MODE_CHAT
                self.chat_prompt = chat.system_prompt
                self.chat_title = chat.title

    # ── turn lifecycle ────────────────────────────────────────────────

    def start(self, coro) -> None:
        self.stop_event = asyncio.Event()
        self.turn_task = asyncio.create_task(coro)

    async def stop(self) -> None:
        if self.busy:
            self.stop_event.set()

    async def wait_for_turn(self, grace_seconds: float = 30.0) -> None:
        if self.turn_task is None:
            return
        with contextlib.suppress(TimeoutError, asyncio.CancelledError, Exception):
            async with asyncio.timeout(grace_seconds):
                await asyncio.shield(self.turn_task)

    def spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    async def run_turn(self, payload: UserMessagePayload | RegeneratePayload) -> None:
        """One user turn: persist the prompt, stream one reply per model, bill."""
        ws = self.websocket
        regenerate = isinstance(payload, RegeneratePayload)
        user_text = "" if regenerate else payload.content
        attachments_meta = [] if regenerate else payload.attachments

        if regenerate:
            if self.chat_id is None:
                await _send_error(
                    ws, "There is nothing to regenerate yet.", "nothing_to_regenerate"
                )
                return
            async with session_scope() as db:
                previous = await chat_service.last_user_message(db, self.chat_id)
            if previous is None:
                await _send_error(
                    ws, "There is nothing to regenerate yet.", "nothing_to_regenerate"
                )
                return
            # Route on the prompt being answered again, not on an empty string.
            user_text = previous.content

        # 1. Which models, and can the wallet afford them?
        try:
            requested = payload.models or None
            model_ids = chat_service.resolve_models(requested, payload.model, user_text)
        except UnknownModelError as exc:
            await _send_error(ws, str(exc), "unknown_model")
            return

        arena = len(model_ids) > 1
        if self.chat_id is not None and arena != (self.chat_mode == MODE_ARENA) and not regenerate:
            # A conversation keeps its mode; the client starts a new chat to switch.
            if self.chat_mode == MODE_ARENA:
                await _send_error(
                    ws, "This is an arena conversation: pick two or more models.", "mode"
                )
            else:
                await _send_error(ws, "Start a new chat to compare models side by side.", "mode")
            return

        async with session_scope() as db:
            balance = await billing.get_balance(db, self.user_id)
        if balance <= 0:
            await _send_error(
                ws, "You are out of credits. Top up to keep chatting.", "insufficient_credits"
            )
            return

        # 2. Persist the user message (or rewind for edit / regenerate).
        llm_attachments = await _resolve_attachments(self.cache, attachments_meta)
        try:
            async with session_scope() as db:
                if self.chat_id is None:
                    chat = Chat(
                        user_id=self.user_id,
                        title=fallback_title(
                            user_text, attachments_meta[0].name if attachments_meta else None
                        ),
                        mode=MODE_ARENA if arena else MODE_CHAT,
                    )
                    db.add(chat)
                    await db.flush()
                    self.chat_id = chat.id
                    self.chat_mode = chat.mode
                    self.chat_title = chat.title
                    created = True
                else:
                    created = False

                truncated_from: uuid.UUID | None = None
                if regenerate:
                    user_message = await chat_service.last_user_message(db, self.chat_id)
                    if user_message is None:
                        await _send_error(
                            ws, "There is nothing to regenerate yet.", "nothing_to_regenerate"
                        )
                        return
                    await chat_service.delete_replies_after(db, self.chat_id, user_message)
                    user_text = user_message.content
                    truncated_from = user_message.id
                else:
                    if payload.edit_message_id is not None:
                        target = await db.get(Message, payload.edit_message_id)
                        if (
                            target is None
                            or target.chat_id != self.chat_id
                            or target.role != ROLE_USER
                        ):
                            await _send_error(ws, "That message cannot be edited.", "bad_edit")
                            return
                        await chat_service.truncate_from(db, self.chat_id, target)
                        truncated_from = target.id
                    user_message = Message(
                        chat_id=self.chat_id,
                        role=ROLE_USER,
                        content=user_text,
                        model=None,
                        attachments=[a.model_dump(exclude={"id"}) for a in attachments_meta],
                    )
                    db.add(user_message)
                    await db.flush()
                user_message_id = user_message.id
                # autoflush is off, so this query returns the previous turns only.
                histories = {
                    model: await chat_service.load_history(
                        db, self.chat_id, model_filter=model if arena else None
                    )
                    for model in model_ids
                }
        except Exception:
            logger.exception("Failed to record the user turn for %s", self.user_email)
            await _send_error(ws, "Could not save your message. Please try again.", "storage")
            return

        if created and not await _send(
            ws, {"type": "system", "event": "chat_id", "payload": str(self.chat_id)}
        ):
            return
        if truncated_from is not None:
            await self.cache_clear()
            await _send(
                ws, {"type": "system", "event": "truncated", "payload": str(truncated_from)}
            )
        else:
            await self.cache_add(ROLE_USER, user_text)

        # The cached/stored copy of this turn has no attachment payloads, so
        # replace it with the rich version before sending it to the model.
        latest = ChatMessage.from_text(ROLE_USER, user_text)
        latest.attachments = llm_attachments
        for history in histories.values():
            if history and history[-1].role == ROLE_USER and history[-1].text == user_text:
                history[-1] = latest
            else:
                history.append(latest)

        # 3. Budget each slot against the balance.
        slots = []
        for index, model in enumerate(model_ids):
            spec = get_spec(model)
            # Arena slots and explicit picks are "pinned"; only an "auto" turn is
            # scored, and its intent travels to the client as the reason shown.
            route = (
                Route(spec.id, "pinned")
                if requested
                else ModelRouter.route(user_text, payload.model)
            )
            try:
                budget = chat_service.plan_budget(
                    spec,
                    chat_service.history_chars(histories[model]),
                    balance,
                    payload.effort,
                    share=len(model_ids),
                )
            except chat_service.CannotAffordError as exc:
                await _send_error(ws, str(exc), "insufficient_credits", slot=index)
                return
            slots.append(
                {
                    "slot": index,
                    "model": spec.id,
                    "intent": route.intent,
                    "reason": route.reason,
                    "budget": budget,
                }
            )

        if not await _send(
            ws,
            {
                "type": "turn_start",
                "user_message_id": str(user_message_id),
                "slots": [{k: v for k, v in s.items() if k != "budget"} for s in slots],
            },
        ):
            return

        # 4. Generate every slot concurrently.
        system_prompt = chat_service.build_system_prompt(self.custom_instructions, self.chat_prompt)
        results = await asyncio.gather(
            *(
                self._generate_slot(
                    slot=s["slot"],
                    model=s["model"],
                    history=histories[s["model"]],
                    options=GenerationOptions(
                        effort=payload.effort,
                        system_prompt=system_prompt,
                        max_output_tokens=s["budget"].max_output_tokens,
                    ),
                    user_text=user_text,
                    parent_id=user_message_id,
                )
                for s in slots
            ),
            return_exceptions=True,
        )

        # 5. Close the turn.
        async with session_scope() as db:
            await chat_service.touch_chat(db, self.chat_id)
            new_balance = await billing.get_balance(db, self.user_id)
        await _send(ws, {"type": "turn_done", "balance": str(new_balance)})

        replies = [r for r in results if isinstance(r, str) and r.strip()]
        if created and replies and settings.CHAT_TITLE_GENERATION:
            self.spawn(self._title_chat(user_text, replies[0]))

    async def _generate_slot(
        self,
        *,
        slot: int,
        model: str,
        history: list[ChatMessage],
        options: GenerationOptions,
        user_text: str,
        parent_id: uuid.UUID,
    ) -> str:
        """Stream one model's reply, then bill and store it. Returns the text."""
        ws = self.websocket
        usage, outcome = Usage(), Outcome()
        full_response = ""
        started = time.perf_counter()
        error_message: str | None = None
        error_code = "generation_failed"

        try:
            provider = LLMFactory.get_provider(model)
            stream = provider.generate_stream(history, model, usage, options, outcome)
            async with asyncio.timeout(settings.CHAT_STREAM_TIMEOUT_SECONDS):
                async for chunk in _stream_until(stream, self.stop_event):
                    full_response += chunk
                    if not await _send(ws, {"type": "content", "slot": slot, "delta": chunk}):
                        self.stop_event.set()
                        break
            if self.stop_event.is_set():
                outcome.finish_reason = FINISH_INTERRUPTED
        except ModelRefusedError as exc:
            outcome.finish_reason = FINISH_REFUSAL
            error_message, error_code = str(exc), "refused"
        except ProviderNotConfiguredError as exc:
            error_message, error_code = (
                f"{exc.provider} is not available right now.",
                "provider_unavailable",
            )
        except TimeoutError:
            outcome.finish_reason = FINISH_ERROR
            error_message, error_code = "The model took too long to respond.", "timeout"
        except Exception as exc:
            logger.exception("Generation failed with %s", model)
            outcome.finish_reason = FINISH_ERROR
            # Provider errors can name keys and endpoints; only echo them back
            # outside production.
            error_message = "Generation failed. Please try again."
            if not settings.is_production:
                error_message = f"Generation failed: {exc}"

        duration_ms = int((time.perf_counter() - started) * 1000)

        # Bill for whatever was produced, even if the client vanished mid-stream.
        usage.ensure_validity(prompt_text=user_text, completion_text=full_response)
        if not full_response.strip():
            await _send_error(
                ws, error_message or "The model returned an empty reply.", error_code, slot=slot
            )
            return ""

        billed_model = model
        if outcome.served_model:
            with contextlib.suppress(UnknownModelError):
                billed_model = get_spec(outcome.served_model).id
        cost, message_id = await self._record_reply(
            model=billed_model,
            response=full_response,
            usage=usage,
            outcome=outcome,
            parent_id=parent_id,
            duration_ms=duration_ms,
        )
        await self.cache_add(ROLE_ASSISTANT, full_response)

        await _send(
            ws,
            {
                "type": "message_done",
                "slot": slot,
                "message_id": str(message_id) if message_id else None,
                "model": billed_model,
                "served_model": outcome.served_model,
                "cost": str(cost),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "duration_ms": duration_ms,
                "finish_reason": outcome.finish_reason,
                "notes": outcome.notes + ([error_message] if error_message else []),
            },
        )
        return full_response

    async def _record_reply(
        self,
        *,
        model: str,
        response: str,
        usage: Usage,
        outcome: Outcome,
        parent_id: uuid.UUID,
        duration_ms: int,
    ) -> tuple[Decimal, uuid.UUID | None]:
        """Charge for a completed reply and store it.

        The response has already been delivered, so the debit is allowed to
        overdraw; the next turn's balance check stops the user before they can
        spend more.
        """
        provider = LLMFactory.get_provider(model)
        cost = provider.calculate_cost(usage, model)
        message_id: uuid.UUID | None = None
        try:
            async with session_scope() as db:
                await billing.debit(db, self.user_id, cost, allow_overdraft=True)
                message = Message(
                    chat_id=self.chat_id,
                    role=ROLE_ASSISTANT,
                    content=response,
                    model=model,
                    parent_id=parent_id,
                    cost=cost,
                    tokens=usage.total_tokens,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    duration_ms=duration_ms,
                    finish_reason=outcome.finish_reason,
                )
                db.add(message)
                await db.flush()
                message_id = message.id
        except Exception:
            # Never fail the conversation over bookkeeping — but make it findable.
            logger.exception("Failed to record reply for user %s (cost %s)", self.user_id, cost)
        return cost, message_id

    async def _title_chat(self, user_text: str, reply: str) -> None:
        title = await generate_title(user_text, reply)
        if not title or self.chat_id is None:
            return
        async with session_scope() as db:
            chat = await db.get(Chat, self.chat_id)
            if chat is None or chat.title != self.chat_title:
                return  # the user renamed it meanwhile
            chat.title = title
        self.chat_title = title
        await _send(self.websocket, {"type": "system", "event": "title", "payload": title})

    # ── cache ─────────────────────────────────────────────────────────

    async def cache_add(self, role: str, content: str) -> None:
        if self.chat_id is None or self.chat_mode == MODE_ARENA:
            return
        try:
            await self.cache.add_message(str(self.chat_id), role, content)
        except Exception as exc:
            logger.warning("Could not cache a chat message: %s", exc)

    async def cache_clear(self) -> None:
        if self.chat_id is None:
            return
        try:
            await self.cache.clear_history(str(self.chat_id))
        except Exception as exc:
            logger.warning("Could not clear the chat cache: %s", exc)


# ── WebSocket endpoint ────────────────────────────────────────────────────


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    chat_id: str | None = None,
    redis_client: redis.Redis = Depends(get_redis),
) -> None:
    """Streaming chat. See the module docstring for the protocol."""
    await websocket.accept()

    # Preferred: the first frame carries the token, so it never appears in a
    # request log or browser history. The query parameter is kept for older
    # clients.
    token = websocket.query_params.get("token")
    if not token:
        try:
            async with asyncio.timeout(10):
                first = json.loads(await websocket.receive_text())
        except (TimeoutError, json.JSONDecodeError, WebSocketDisconnect, RuntimeError):
            await _close(websocket, WS_POLICY_VIOLATION, "Missing token")
            return
        token = (
            first.get("token") if isinstance(first, dict) and first.get("type") == "auth" else None
        )
    if not token:
        await _close(websocket, WS_POLICY_VIOLATION, "Missing token")
        return

    async with session_scope() as db:
        user = await verify_token_socket(token, db)
        if not user:
            await _close(websocket, WS_POLICY_VIOLATION, "Invalid token")
            return
        session = ChatSession(websocket, ChatCache(redis_client), user)

    await session.adopt_chat(chat_id)
    logger.info("WebSocket connected for %s", session.user_email)
    await _send(
        websocket,
        {
            "type": "system",
            "event": "ready",
            "payload": {
                "chat_id": str(session.chat_id) if session.chat_id else None,
                "mode": session.chat_mode,
            },
        },
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = _INCOMING.validate_python(json.loads(raw))
            except (json.JSONDecodeError, TypeError, ValidationError) as exc:
                logger.info("Rejected malformed WebSocket payload: %s", exc)
                await _send_error(websocket, "Invalid message format", "invalid_payload")
                continue

            if isinstance(payload, ControlPayload):
                if payload.type == "ping":
                    await _send(websocket, {"type": "pong"})
                elif payload.type == "stop":
                    await session.stop()
                    await session.wait_for_turn()
                    await _send(websocket, {"type": "system", "event": "stopped", "payload": None})
                continue

            if session.busy:
                await _send_error(
                    websocket, "Wait for the current reply to finish (or stop it).", "busy"
                )
                continue

            session.start(session.run_turn(payload))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for %s", session.user_email)
    except ConnectionResetError:
        logger.info("WebSocket reset by %s", session.user_email)
    except Exception:
        logger.exception("Unhandled WebSocket error for %s", session.user_email)
        await _close(websocket, WS_INTERNAL_ERROR)
    finally:
        # Let a reply in flight finish its bookkeeping before the socket goes away.
        await session.stop()
        await session.wait_for_turn()
        for task in list(session.background):
            task.cancel()
        logger.info("WebSocket closed for %s", session.user_email)
