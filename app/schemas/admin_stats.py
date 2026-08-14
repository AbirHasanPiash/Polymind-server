from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class StatTrend(BaseModel):
    date: str
    value: Decimal


class AdminOverviewStats(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    # Totals
    total_revenue: Decimal
    total_users: int
    total_chats: int

    # Multimodal breakdown
    total_images_generated: int
    total_audio_generated: int
    total_videos_generated: int

    # Economics
    total_tokens_consumed: int
    total_ai_cost: Decimal

    # Time series
    revenue_trend: list[StatTrend]
    user_growth_trend: list[StatTrend]
