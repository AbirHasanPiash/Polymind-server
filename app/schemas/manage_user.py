from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserWalletSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    credits: Decimal
    updated_at: datetime | None = None


class UserAdminResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    full_name: str | None = None
    is_active: bool
    is_superuser: bool
    created_at: datetime | None = None
    wallet: UserWalletSchema | None = None


class UserUpdateAdmin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(None, max_length=120)
    is_active: bool | None = None
    is_superuser: bool | None = None
    credits: Decimal | None = Field(None, ge=0)


class UserListResponse(BaseModel):
    users: list[UserAdminResponse]
    total_count: int
    page: int
    size: int
