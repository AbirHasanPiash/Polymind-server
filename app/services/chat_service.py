"""Conversation helpers shared by the chat endpoints: history, truncation,
affordability checks, export formatting and the platform system prompt."""

from __future__ import annotations

import json
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chat import ROLE_ASSISTANT, ROLE_USER, Chat, Message
from app.services.llm.base import MAX_OUTPUT_TOKENS_BY_EFFORT
from app.services.llm.models import (
    PROFIT_MARGIN,
    USD_TO_CREDITS_RATE,
    ModelSpec,
    get_spec,
)
from app.services.llm.schema import ChatMessage
from app.services.llm.usage import estimate_tokens

MILLION = Decimal("1000000")
# A reply shorter than this is not worth starting; the turn is refused instead
# of being cut off after a sentence.
MIN_OUTPUT_TOKENS = 512

PLATFORM_PROMPT = (
    "You are Polymind, a helpful, precise assistant with access to several frontier "
    "models. Answer directly and accurately; say when you are unsure. Format replies "
    "in Markdown and put code in fenced blocks tagged with the language. "
    "Today's date is {today}."
)


def build_system_prompt(custom_instructions: str | None, chat_prompt: str | None) -> str:
    """Compose the platform prompt with the user's and the conversation's instructions."""
    parts = [PLATFORM_PROMPT.format(today=datetime.now(UTC).strftime("%Y-%m-%d"))]
    if custom_instructions and custom_instructions.strip():
        parts.append(
            "The user has set these standing instructions; follow them unless a message "
            f"asks otherwise:\n{custom_instructions.strip()}"
        )
    if chat_prompt and chat_prompt.strip():
        parts.append(f"Instructions for this conversation:\n{chat_prompt.strip()}")
    return "\n\n".join(parts)


def new_share_token() -> str:
    return secrets.token_urlsafe(24)


def share_url(token: str) -> str:
    return f"{settings.FRONTEND_URL.rstrip('/')}/share/{token}"


# ── history ───────────────────────────────────────────────────────────────


async def load_history(
    db: AsyncSession,
    chat_id: uuid.UUID,
    *,
    limit: int | None = None,
    model_filter: str | None = None,
) -> list[ChatMessage]:
    """Recent turns from PostgreSQL, oldest first.

    ``model_filter`` keeps only the assistant replies produced by that model, so
    in an arena each model continues its own thread of the conversation.
    """
    stmt = select(Message).where(Message.chat_id == chat_id)
    if model_filter:
        stmt = stmt.where((Message.role == ROLE_USER) | (Message.model == model_filter))
    stmt = stmt.order_by(desc(Message.created_at)).limit(limit or settings.CHAT_HISTORY_LIMIT)
    result = await db.execute(stmt)
    rows = list(reversed(result.scalars().all()))
    return [ChatMessage.from_text(row.role, row.content) for row in rows if row.content]


async def truncate_from(db: AsyncSession, chat_id: uuid.UUID, message: Message) -> int:
    """Delete ``message`` and everything after it. Returns the number removed."""
    victims = await db.execute(
        select(Message.id).where(
            Message.chat_id == chat_id, Message.created_at >= message.created_at
        )
    )
    ids = [row[0] for row in victims.all()]
    if not ids:
        return 0
    await db.execute(delete(Message).where(Message.id.in_(ids)))
    return len(ids)


async def last_user_message(db: AsyncSession, chat_id: uuid.UUID) -> Message | None:
    return await db.scalar(
        select(Message)
        .where(Message.chat_id == chat_id, Message.role == ROLE_USER)
        .order_by(desc(Message.created_at))
        .limit(1)
    )


async def delete_replies_after(db: AsyncSession, chat_id: uuid.UUID, user_message: Message) -> int:
    """Remove the assistant replies that followed ``user_message`` (for regenerate)."""
    victims = await db.execute(
        select(Message.id).where(
            Message.chat_id == chat_id,
            Message.role == ROLE_ASSISTANT,
            Message.created_at >= user_message.created_at,
        )
    )
    ids = [row[0] for row in victims.all()]
    if ids:
        await db.execute(delete(Message).where(Message.id.in_(ids)))
    return len(ids)


async def touch_chat(db: AsyncSession, chat_id: uuid.UUID) -> None:
    chat = await db.get(Chat, chat_id)
    if chat is not None:
        chat.updated_at = datetime.now(UTC)


# ── affordability ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class Budget:
    """What a turn on one model may spend, given the caller's balance."""

    model: str
    max_output_tokens: int
    estimated_prompt_tokens: int
    minimum_credits: Decimal


class CannotAffordError(Exception):
    def __init__(self, model: str, required: Decimal, balance: Decimal) -> None:
        super().__init__(
            f"Not enough credits for {model}: a reply needs about {required:.4f} "
            f"credits and the balance is {balance:.4f}."
        )
        self.model = model
        self.required = required
        self.balance = balance


def _credits_per_token(price_per_million: Decimal) -> Decimal:
    return price_per_million * PROFIT_MARGIN * USD_TO_CREDITS_RATE / MILLION


def plan_budget(
    spec: ModelSpec,
    prompt_chars: int,
    balance: Decimal,
    effort: str,
    share: int = 1,
) -> Budget:
    """Cap the reply length by what the wallet can pay for.

    The balance check used to be "greater than zero", which let a single turn
    on an expensive model overdraw the wallet by a large amount. Now a turn on
    a model the balance cannot cover is refused up front, and an affordable one
    has its output ceiling lowered to fit.
    """
    prompt_tokens = estimate_tokens("x" * prompt_chars) + 64  # + system prompt slack
    in_rate = _credits_per_token(spec.input_price)
    out_rate = _credits_per_token(spec.output_price)

    available = balance / share
    prompt_cost = prompt_tokens * in_rate
    minimum = prompt_cost + MIN_OUTPUT_TOKENS * out_rate
    if available < minimum:
        raise CannotAffordError(spec.id, minimum, balance)

    ceiling = MAX_OUTPUT_TOKENS_BY_EFFORT.get(effort, MAX_OUTPUT_TOKENS_BY_EFFORT["medium"])
    affordable = int((available - prompt_cost) / out_rate) if out_rate > 0 else ceiling
    return Budget(
        model=spec.id,
        max_output_tokens=max(MIN_OUTPUT_TOKENS, min(ceiling, affordable, spec.max_output_tokens)),
        estimated_prompt_tokens=prompt_tokens,
        minimum_credits=minimum.quantize(Decimal("0.000001")),
    )


def history_chars(history: list[ChatMessage]) -> int:
    total = 0
    for message in history:
        total += len(message.text)
        for attachment in message.attachments:
            # Images are billed by the provider in tokens per tile; a fixed
            # allowance keeps the estimate in the right range.
            total += len(attachment.content) if attachment.type == "text" else 4000
    return total


def resolve_models(requested: list[str] | None, single: str | None, prompt: str) -> list[str]:
    """Validate the requested model(s) against the registry."""
    from app.services.llm.router import ModelRouter  # local import: avoids a cycle at import time

    if requested:
        return [get_spec(model).id for model in requested]
    return [ModelRouter.determine_model(prompt, single)]


# ── export ────────────────────────────────────────────────────────────────


def export_markdown(chat: Chat, messages: list[Message]) -> str:
    lines = [f"# {chat.title or 'Conversation'}", ""]
    if chat.created_at:
        lines.append(
            f"_Started {chat.created_at.strftime('%Y-%m-%d %H:%M UTC')} · exported from Polymind_"
        )
        lines.append("")
    for message in messages:
        if message.role == ROLE_USER:
            lines.append("## You")
        else:
            lines.append(f"## {message.model or 'Assistant'}")
        lines.append("")
        lines.append(message.content.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_json(chat: Chat, messages: list[Message]) -> str:
    payload = {
        "id": str(chat.id),
        "title": chat.title,
        "mode": chat.mode,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
        "messages": [
            {
                "id": str(message.id),
                "role": "user" if message.role == ROLE_USER else "assistant",
                "model": message.model,
                "content": message.content,
                "parent_id": str(message.parent_id) if message.parent_id else None,
                "tokens": message.tokens,
                "cost": str(message.cost) if message.cost is not None else None,
                "created_at": message.created_at.isoformat() if message.created_at else None,
            }
            for message in messages
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
