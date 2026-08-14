"""Admin analytics. Every route here requires a superuser."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_admin
from app.models import (
    Chat,
    GeneratedAudio,
    GeneratedImage,
    GeneratedVideo,
    Message,
    Transaction,
    User,
)
from app.models.base import utc_now
from app.models.transaction import STATUS_COMPLETED
from app.schemas.admin_stats import AdminOverviewStats

router = APIRouter(dependencies=[Depends(get_current_admin)])

DEFAULT_TREND_DAYS = 30
MAX_TREND_DAYS = 365


def _total(column) -> object:
    return select(func.coalesce(func.sum(column), 0)).scalar_subquery()


def _count(model) -> object:
    return select(func.count(model.id)).scalar_subquery()


@router.get("/overview", response_model=AdminOverviewStats)
async def get_admin_overview(
    days: int = Query(DEFAULT_TREND_DAYS, ge=1, le=MAX_TREND_DAYS),
    db: AsyncSession = Depends(get_db),
) -> AdminOverviewStats:
    """Platform-wide totals plus daily revenue and signup trends."""
    since = utc_now() - timedelta(days=days)

    # All the independent aggregates travel in one statement: eight sequential
    # round-trips to the database was the dominant cost of this endpoint.
    totals = (
        await db.execute(
            select(
                select(func.coalesce(func.sum(Transaction.amount), 0))
                .where(Transaction.status == STATUS_COMPLETED)
                .scalar_subquery()
                .label("revenue"),
                _count(User).label("users"),
                _count(Chat).label("chats"),
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

    return AdminOverviewStats(
        total_revenue=totals.revenue,
        total_users=totals.users,
        total_chats=totals.chats,
        total_images_generated=totals.images,
        total_audio_generated=totals.audio,
        total_videos_generated=totals.videos,
        total_tokens_consumed=totals.tokens,
        total_ai_cost=totals.message_cost
        + totals.image_cost
        + totals.audio_cost
        + totals.video_cost,
        revenue_trend=[{"date": str(day), "value": value} for day, value in revenue_trend],
        user_growth_trend=[{"date": str(day), "value": value} for day, value in user_trend],
    )
