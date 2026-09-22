from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class StatTrend(BaseModel):
    date: str
    value: Decimal


class ModelUsageStat(BaseModel):
    model: str
    messages: int
    tokens: int
    credits: Decimal


class TopUserStat(BaseModel):
    user_id: UUID
    email: str
    credits: Decimal
    messages: int


class AdminOverviewStats(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    # Totals
    total_revenue: Decimal
    total_users: int
    total_chats: int
    active_users: int  # signed in within the trend window
    new_users: int  # created within the trend window

    # Multimodal breakdown
    total_images_generated: int
    total_audio_generated: int
    total_videos_generated: int
    total_messages: int

    # Economics
    total_tokens_consumed: int
    total_ai_cost: Decimal
    spend_by_category: dict[str, Decimal]

    # Time series
    revenue_trend: list[StatTrend]
    user_growth_trend: list[StatTrend]
    usage_trend: list[StatTrend]

    # Breakdown
    usage_by_model: list[ModelUsageStat]
    top_users: list[TopUserStat]


class AdminTransaction(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    email: str | None = None
    package_name: str | None = None
    payment_gateway: str | None = None
    amount: Decimal
    currency: str | None = None
    credits_added: Decimal
    status: str
    created_at: datetime | None = None
    completed_at: datetime | None = None
