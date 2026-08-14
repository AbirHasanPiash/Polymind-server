from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# Long enough to resist guessing, short enough that bcrypt-style backends do not
# silently truncate.
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    full_name: str | None = Field(None, max_length=120)


class UserLogin(UserBase):
    password: str = Field(..., max_length=MAX_PASSWORD_LENGTH)


class GoogleLogin(BaseModel):
    token: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token lifetime in seconds")


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
    wallet: WalletResponse | None = None


class UserUpdateProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(None, max_length=120)
