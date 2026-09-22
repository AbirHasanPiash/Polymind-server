"""Admin analytics. Every route here requires a superuser."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Date, cast, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_admin
from app.models import (
    Chat,
    GeneratedAudio,
    GeneratedImage,
    GeneratedVideo,
    Message,
    Package,
    Transaction,
    User,
)
from app.models.base import utc_now
from app.models.chat import ROLE_ASSISTANT
from app.models.transaction import STATUS_COMPLETED
from app.schemas.admin_stats import AdminOverviewStats, AdminTransaction

router = APIRouter(dependencies=[Depends(get_current_admin)])

DEFAULT_TREND_DAYS = 30
MAX_TREND_DAYS = 365


def _total(column) -> object:
    return select(func.coalesce(func.sum(column), 0)).scalar_subquery()


def _count(model) -> object:
    return select(func.count(model.id)).scalar_subquery()


def _dec(value) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal("0")


@router.get("/overview", response_model=AdminOverviewStats)
async def get_admin_overview(
    days: int = Query(DEFAULT_TREND_DAYS, ge=1, le=MAX_TREND_DAYS),
    db: AsyncSession = Depends(get_db),
) -> AdminOverviewStats:
    """Platform-wide totals plus daily revenue, signup and usage trends."""
    since = utc_now() - timedelta(days=days)

    # All the independent aggregates travel in one statement: many sequential
    # round-trips to the database was the dominant cost of this endpoint.
    totals = (
        await db.execute(
            select(
                select(func.coalesce(func.sum(Transaction.amount), 0))
                .where(Transaction.status == STATUS_COMPLETED)
                .scalar_subquery()
                .label("revenue"),
                _count(User).label("users"),
                select(func.count(User.id))
                .where(User.last_login_at >= since)
                .scalar_subquery()
                .label("active_users"),
                select(func.count(User.id))
                .where(User.created_at >= since)
                .scalar_subquery()
                .label("new_users"),
                _count(Chat).label("chats"),
                select(func.count(Message.id))
                .where(Message.role == ROLE_ASSISTANT)
                .scalar_subquery()
                .label("messages"),
                _count(GeneratedImage).label("images"),
                _count(GeneratedAudio).label("audio"),
                _count(GeneratedVideo).label("videos"),
                _total(Message.tokens).label("tokens"),
                _total(Message.cost).label("message_cost"),
                _total(GeneratedImage.cost).label("image_cost"),
                _total(GeneratedAudio.cost).label("audio_cost"),
                _total(GeneratedVideo.cost).label("video_cost"),
            )
        )
    ).one()

    revenue_trend = (
        await db.execute(
            select(cast(Transaction.created_at, Date).label("day"), func.sum(Transaction.amount))
            .where(Transaction.status == STATUS_COMPLETED, Transaction.created_at >= since)
            .group_by("day")
            .order_by("day")
        )
    ).all()

    user_trend = (
        await db.execute(
            select(cast(User.created_at, Date).label("day"), func.count(User.id))
            .where(User.created_at >= since)
            .group_by("day")
            .order_by("day")
        )
    ).all()

    usage_trend = (
        await db.execute(
            select(
                cast(Message.created_at, Date).label("day"),
                func.coalesce(func.sum(Message.cost), 0),
            )
            .where(Message.role == ROLE_ASSISTANT, Message.created_at >= since)
            .group_by("day")
            .order_by("day")
        )
    ).all()

    usage_by_model = (
        await db.execute(
            select(
                Message.model,
                func.count(Message.id),
                func.coalesce(func.sum(Message.tokens), 0),
                func.coalesce(func.sum(Message.cost), 0),
            )
            .where(Message.role == ROLE_ASSISTANT, Message.created_at >= since)
            .group_by(Message.model)
            .order_by(desc(func.sum(Message.cost)))
            .limit(20)
        )
    ).all()

    top_users = (
        await db.execute(
            select(
                User.id,
                User.email,
                func.coalesce(func.sum(Message.cost), 0),
                func.count(Message.id),
            )
            .join(Chat, Chat.user_id == User.id)
            .join(Message, Message.chat_id == Chat.id)
            .where(Message.role == ROLE_ASSISTANT, Message.created_at >= since)
            .group_by(User.id, User.email)
            .order_by(desc(func.sum(Message.cost)))
            .limit(10)
        )
    ).all()

    return AdminOverviewStats(
        total_revenue=_dec(totals.revenue),
        total_users=totals.users,
        total_chats=totals.chats,
        active_users=totals.active_users or 0,
        new_users=totals.new_users or 0,
        total_images_generated=totals.images,
        total_audio_generated=totals.audio,
        total_videos_generated=totals.videos,
        total_messages=totals.messages or 0,
        total_tokens_consumed=totals.tokens,
        total_ai_cost=_dec(totals.message_cost)
        + _dec(totals.image_cost)
        + _dec(totals.audio_cost)
        + _dec(totals.video_cost),
        spend_by_category={
            "chat": _dec(totals.message_cost),
            "image": _dec(totals.image_cost),
            "audio": _dec(totals.audio_cost),
            "video": _dec(totals.video_cost),
        },
        revenue_trend=[{"date": str(day), "value": value} for day, value in revenue_trend],
        user_growth_trend=[{"date": str(day), "value": value} for day, value in user_trend],
        usage_trend=[{"date": str(day), "value": value} for day, value in usage_trend],
        usage_by_model=[
            {
                "model": model or "unknown",
                "messages": int(count),
                "tokens": int(tokens),
                "credits": _dec(credits),
            }
            for model, count, tokens, credits in usage_by_model
        ],
        top_users=[
            {"user_id": user_id, "email": email, "credits": _dec(credits), "messages": int(count)}
            for user_id, email, credits, count in top_users
        ],
    )


@router.get("/transactions", response_model=list[AdminTransaction])
async def list_transactions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status_filter: str | None = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
) -> list[AdminTransaction]:
    """Recent purchases across all users, newest first."""
    stmt = (
        select(Transaction, User.email, Package.name)
        .join(User, User.id == Transaction.user_id)
        .outerjoin(Package, Package.id == Transaction.package_id)
        .order_by(desc(Transaction.created_at))
        .offset(offset)
        .limit(limit)
    )
    if status_filter:
        stmt = stmt.where(Transaction.status == status_filter)
    rows = (await db.execute(stmt)).all()
    return [
        AdminTransaction(
            id=tx.id,
            user_id=tx.user_id,
            email=email,
            package_name=package_name,
            payment_gateway=tx.payment_gateway,
            amount=_dec(tx.amount),
            currency=tx.currency,
            credits_added=_dec(tx.credits_added),
            status=tx.status,
            created_at=tx.created_at,
            completed_at=tx.completed_at,
        )
        for tx, email, package_name in rows
    ]
