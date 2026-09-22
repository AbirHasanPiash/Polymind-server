"""Per-user spend analytics, computed from the ledgers each feature writes."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Date, cast, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ROLE_ASSISTANT, Chat, Message
from app.models.media import GeneratedAudio, GeneratedImage, GeneratedVideo
from app.models.transaction import STATUS_COMPLETED, Transaction
from app.schemas.user import ActivityItem, ModelUsage, UsagePoint, UsageSummary
from app.services import billing

ZERO = Decimal("0")


def _dec(value) -> Decimal:
    return Decimal(str(value)) if value is not None else ZERO


async def _daily(db: AsyncSession, model, cost_column, user_filter, since) -> dict[str, Decimal]:
    rows = await db.execute(
        select(cast(model.created_at, Date).label("day"), func.coalesce(func.sum(cost_column), 0))
        .where(user_filter, model.created_at >= since)
        .group_by("day")
    )
    return {str(day): _dec(total) for day, total in rows.all()}


async def summarize(db: AsyncSession, user_id: uuid.UUID, days: int) -> UsageSummary:
    since = datetime.now(UTC) - timedelta(days=days)
    chat_filter = Chat.user_id == user_id

    # Chat spend lives on assistant messages, joined through the chat's owner.
    chat_rows = await db.execute(
        select(
            cast(Message.created_at, Date).label("day"),
            func.coalesce(func.sum(Message.cost), 0),
        )
        .join(Chat, Chat.id == Message.chat_id)
        .where(chat_filter, Message.role == ROLE_ASSISTANT, Message.created_at >= since)
        .group_by("day")
    )
    chat_daily = {str(day): _dec(total) for day, total in chat_rows.all()}
    image_daily = await _daily(
        db, GeneratedImage, GeneratedImage.cost, GeneratedImage.user_id == user_id, since
    )
    audio_daily = await _daily(
        db, GeneratedAudio, GeneratedAudio.cost, GeneratedAudio.user_id == user_id, since
    )
    video_daily = await _daily(
        db, GeneratedVideo, GeneratedVideo.cost, GeneratedVideo.user_id == user_id, since
    )

    by_day: list[UsagePoint] = []
    for offset in range(days - 1, -1, -1):
        day = (datetime.now(UTC) - timedelta(days=offset)).date().isoformat()
        by_day.append(
            UsagePoint(
                date=day,
                chat=chat_daily.get(day, ZERO),
                image=image_daily.get(day, ZERO),
                audio=audio_daily.get(day, ZERO),
                video=video_daily.get(day, ZERO),
            )
        )

    spent_by_category = {
        "chat": sum(chat_daily.values(), ZERO),
        "image": sum(image_daily.values(), ZERO),
        "audio": sum(audio_daily.values(), ZERO),
        "video": sum(video_daily.values(), ZERO),
    }

    counts_row = (
        await db.execute(
            select(
                select(func.count(Message.id))
                .join(Chat, Chat.id == Message.chat_id)
                .where(chat_filter, Message.role == ROLE_ASSISTANT, Message.created_at >= since)
                .scalar_subquery()
                .label("messages"),
                select(func.count(Chat.id))
                .where(chat_filter, Chat.created_at >= since)
                .scalar_subquery()
                .label("chats"),
                select(func.count(GeneratedImage.id))
                .where(GeneratedImage.user_id == user_id, GeneratedImage.created_at >= since)
                .scalar_subquery()
                .label("images"),
                select(func.count(GeneratedAudio.id))
                .where(GeneratedAudio.user_id == user_id, GeneratedAudio.created_at >= since)
                .scalar_subquery()
                .label("audio"),
                select(func.count(GeneratedVideo.id))
                .where(GeneratedVideo.user_id == user_id, GeneratedVideo.created_at >= since)
                .scalar_subquery()
                .label("videos"),
            )
        )
    ).one()

    model_rows = await db.execute(
        select(
            Message.model,
            func.count(Message.id),
            func.coalesce(func.sum(Message.tokens), 0),
            func.coalesce(func.sum(Message.cost), 0),
        )
        .join(Chat, Chat.id == Message.chat_id)
        .where(chat_filter, Message.role == ROLE_ASSISTANT, Message.created_at >= since)
        .group_by(Message.model)
        .order_by(desc(func.sum(Message.cost)))
    )
    by_model = [
        ModelUsage(
            model=model or "unknown", messages=int(count), tokens=int(tokens), credits=_dec(credits)
        )
        for model, count, tokens, credits in model_rows.all()
    ]

    recent: list[ActivityItem] = []
    for message, title in (
        await db.execute(
            select(Message, Chat.title)
            .join(Chat, Chat.id == Message.chat_id)
            .where(chat_filter, Message.role == ROLE_ASSISTANT)
            .order_by(desc(Message.created_at))
            .limit(10)
        )
    ).all():
        recent.append(
            ActivityItem(
                kind="chat",
                title=title or "Conversation",
                credits=_dec(message.cost),
                reference_id=message.chat_id,
                created_at=message.created_at,
            )
        )
    for image in (
        await db.execute(
            select(GeneratedImage)
            .where(GeneratedImage.user_id == user_id)
            .order_by(desc(GeneratedImage.created_at))
            .limit(5)
        )
    ).scalars():
        recent.append(
            ActivityItem(
                kind="image",
                title=image.prompt[:80],
                credits=_dec(image.cost),
                reference_id=image.id,
                created_at=image.created_at,
            )
        )
    for audio in (
        await db.execute(
            select(GeneratedAudio)
            .where(GeneratedAudio.user_id == user_id)
            .order_by(desc(GeneratedAudio.created_at))
            .limit(5)
        )
    ).scalars():
        recent.append(
            ActivityItem(
                kind="audio",
                title=audio.text_prompt[:80],
                credits=_dec(audio.cost),
                reference_id=audio.id,
                created_at=audio.created_at,
            )
        )
    for video in (
        await db.execute(
            select(GeneratedVideo)
            .where(GeneratedVideo.user_id == user_id)
            .order_by(desc(GeneratedVideo.created_at))
            .limit(5)
        )
    ).scalars():
        recent.append(
            ActivityItem(
                kind="video",
                title=video.script_text[:80],
                credits=_dec(video.cost),
                reference_id=video.id,
                created_at=video.created_at,
            )
        )
    for purchase in (
        await db.execute(
            select(Transaction)
            .where(Transaction.user_id == user_id, Transaction.status == STATUS_COMPLETED)
            .order_by(desc(Transaction.created_at))
            .limit(5)
        )
    ).scalars():
        recent.append(
            ActivityItem(
                kind="purchase",
                title=f"Purchased {purchase.credits_added} credits",
                credits=-_dec(purchase.credits_added),
                reference_id=purchase.id,
                created_at=purchase.created_at,
            )
        )
    recent.sort(key=lambda item: item.created_at or datetime.min.replace(tzinfo=UTC), reverse=True)

    return UsageSummary(
        days=days,
        balance=await billing.get_balance(db, user_id),
        spent_credits=sum(spent_by_category.values(), ZERO),
        spent_by_category=spent_by_category,
        counts={
            "messages": int(counts_row.messages or 0),
            "chats": int(counts_row.chats or 0),
            "images": int(counts_row.images or 0),
            "audio": int(counts_row.audio or 0),
            "videos": int(counts_row.videos or 0),
        },
        by_day=by_day,
        by_model=by_model,
        recent=recent[:20],
    )
