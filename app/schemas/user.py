from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.services.llm.models import Effort, is_supported

# Long enough to resist guessing, short enough that bcrypt-style backends do not
# silently truncate.
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
MAX_CUSTOM_INSTRUCTIONS = 4000
MAX_SAVED_PROMPTS = 50


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    full_name: str | None = Field(None, max_length=120)


class UserLogin(UserBase):
    password: str = Field(..., max_length=MAX_PASSWORD_LENGTH)


class GoogleLogin(BaseModel):
    token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=16, max_length=256)
    new_password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)


class ChangePasswordRequest(BaseModel):
    current_password: str | None = Field(None, max_length=MAX_PASSWORD_LENGTH)
    new_password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)


class DeleteAccountRequest(BaseModel):
    password: str | None = Field(None, max_length=MAX_PASSWORD_LENGTH)
    confirmation: Literal["DELETE"]


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token lifetime in seconds")


class SavedPrompt(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=1, max_length=80)
    content: str = Field(..., min_length=1, max_length=4000)


class Preferences(BaseModel):
    """Everything a user can tune about how the assistant behaves for them."""

    model_config = ConfigDict(extra="forbid")

    custom_instructions: str | None = Field(None, max_length=MAX_CUSTOM_INSTRUCTIONS)
    default_model: str = "auto"
    default_effort: Effort = "medium"
    saved_prompts: list[SavedPrompt] = Field(default_factory=list, max_length=MAX_SAVED_PROMPTS)
    # Small UI conveniences the client persists across devices.
    send_on_enter: bool = True
    show_costs: bool = True

    @field_validator("default_model")
    @classmethod
    def _model_exists(cls, value: str) -> str:
        if value != "auto" and not is_supported(value):
            raise ValueError(f"Unknown model: {value}")
        return value


class WalletResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    credits: Decimal = Field(..., description="Current balance")


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    full_name: str | None = None
    is_active: bool
    is_superuser: bool
    has_password: bool = False
    preferences: dict = Field(default_factory=dict)
    created_at: datetime | None = None
    wallet: WalletResponse | None = None


class UserUpdateProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(None, max_length=120)
    preferences: Preferences | None = None


# ── usage analytics ───────────────────────────────────────────────────────


class UsagePoint(BaseModel):
    date: str
    chat: Decimal = Decimal("0")
    image: Decimal = Decimal("0")
    audio: Decimal = Decimal("0")
    video: Decimal = Decimal("0")


class ModelUsage(BaseModel):
    model: str
    messages: int
    tokens: int
    credits: Decimal


class ActivityItem(BaseModel):
    kind: Literal["chat", "image", "audio", "video", "purchase"]
    title: str
    credits: Decimal
    reference_id: UUID | None = None
    created_at: datetime | None = None


class UsageSummary(BaseModel):
    days: int
    balance: Decimal
    spent_credits: Decimal
    spent_by_category: dict[str, Decimal]
    counts: dict[str, int]
    by_day: list[UsagePoint]
    by_model: list[ModelUsage]
    recent: list[ActivityItem]
