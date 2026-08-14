from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PackageBase(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    description: str | None = None
    price: Decimal = Field(..., gt=0, description="Price in the package currency")
    credits: Decimal = Field(..., gt=0, description="Credits the buyer receives")
    is_active: bool = True
    is_featured: bool = False


class PackageCreate(PackageBase):
    pass


class PackageUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, min_length=3, max_length=50)
    description: str | None = None
    price: Decimal | None = Field(None, gt=0)
    credits: Decimal | None = Field(None, gt=0)
    is_active: bool | None = None
    is_featured: bool | None = None


class PackageResponse(PackageBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    currency: str | None = None
    # Nullable in the database, so legacy rows must not fail validation.
    created_at: datetime | None = None
    updated_at: datetime | None = None
