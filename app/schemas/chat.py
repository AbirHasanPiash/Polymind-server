"""Chat request/response shapes for the HTTP and WebSocket APIs."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.llm.models import DEFAULT_EFFORT, Effort

MAX_MESSAGE_CHARS = 32_000
MAX_SYSTEM_PROMPT_CHARS = 4_000
MAX_TITLE_CHARS = 120
ARENA_MAX_MODELS = 3


class AttachmentSchema(BaseModel):
    id: str | None = None
    name: str = ""
    type: str = ""
    size: int = 0
    mime_type: str | None = None


class MessageSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    model: str | None = None
    parent_id: uuid.UUID | None = None
    attachments: list[AttachmentSchema] = Field(default_factory=list)
    tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: Decimal | None = None
    duration_ms: int | None = None
    finish_reason: str | None = None
    created_at: datetime | None = None


class ChatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None = None
    mode: str = "chat"
    pinned: bool = False
    system_prompt: str | None = None
    share_token: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatSummarySchema(ChatSchema):
    message_count: int = 0


class ChatUpdateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(None, min_length=1, max_length=MAX_TITLE_CHARS)
    pinned: bool | None = None
    system_prompt: str | None = Field(None, max_length=MAX_SYSTEM_PROMPT_CHARS)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str | None) -> str | None:
        return value.strip() if value else value


class SearchHitSchema(BaseModel):
    chat_id: uuid.UUID
    chat_title: str | None
    message_id: uuid.UUID
    role: str
    snippet: str
    created_at: datetime | None = None


class ShareSchema(BaseModel):
    share_token: str
    url: str
    shared_at: datetime | None = None


class SharedMessageSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    model: str | None = None
    parent_id: uuid.UUID | None = None
    created_at: datetime | None = None


class SharedChatSchema(BaseModel):
    title: str | None
    mode: str
    created_at: datetime | None
    shared_at: datetime | None
    messages: list[SharedMessageSchema]


# ── WebSocket payloads ────────────────────────────────────────────────────


class UserMessagePayload(BaseModel):
    type: Literal["user_message"]
    content: str = Field("", max_length=MAX_MESSAGE_CHARS)
    attachments: list[AttachmentSchema] = Field(default_factory=list, max_length=10)
    # Optional per-turn override. Without it a client has to reconnect to switch
    # models, which drops the conversation mid-session.
    model: str | None = None
    # Two or three models answering side by side (arena mode).
    models: list[str] | None = Field(None, min_length=2, max_length=ARENA_MAX_MODELS)
    effort: Effort = DEFAULT_EFFORT
    # Replace an earlier user message (and everything after it) with this one.
    edit_message_id: uuid.UUID | None = None

    @field_validator("models")
    @classmethod
    def _distinct(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("Arena models must be distinct")
        return value


class RegeneratePayload(BaseModel):
    type: Literal["regenerate"]
    model: str | None = None
    models: list[str] | None = Field(None, min_length=2, max_length=ARENA_MAX_MODELS)
    effort: Effort = DEFAULT_EFFORT


class ControlPayload(BaseModel):
    type: Literal["stop", "ping"]
