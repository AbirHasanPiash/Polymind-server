"""Current-user profile, preferences, security and usage."""

from __future__ import annotations

import logging
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_user, get_password_hash, verify_password
from app.models.user import User
from app.schemas.user import (
    ChangePasswordRequest,
    DeleteAccountRequest,
    Preferences,
    UsageSummary,
    UserResponse,
    UserUpdateProfile,
)
from app.services import billing, usage

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_DEV_TOPUP = Decimal("10000")


def _response(user: User) -> UserResponse:
    payload = UserResponse.model_validate(user)
    payload.has_password = bool(user.hashed_password)
    payload.preferences = Preferences.model_validate(user.preferences or {}).model_dump()
    return payload


@router.get("/me", response_model=UserResponse)
async def read_user_me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Profile plus the current credit balance."""
    await db.refresh(current_user, attribute_names=["wallet"])
    return _response(current_user)


@router.patch("/me", response_model=UserResponse)
async def update_user_me(
    payload: UserUpdateProfile,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Update the caller's profile and preferences."""
    update_data = payload.model_dump(exclude_unset=True)
    if "preferences" in update_data and payload.preferences is not None:
        current_user.preferences = payload.preferences.model_dump()
        update_data.pop("preferences")
    for field, value in update_data.items():
        setattr(current_user, field, value)

    await db.commit()
    await db.refresh(current_user, attribute_names=["wallet"])
    return _response(current_user)


@router.post("/me/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Change the password. The current one is required whenever one is set.

    Accounts that only ever signed in with Google can set a first password
    through the email reset flow instead.
    """
    if current_user.hashed_password and (
        not payload.current_password
        or not verify_password(payload.current_password, current_user.hashed_password)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect"
        )
    current_user.hashed_password = get_password_hash(payload.new_password)
    await db.commit()
    logger.info("Password changed for %s", current_user.email)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    payload: DeleteAccountRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Permanently delete the account and everything it owns."""
    if current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin accounts cannot delete themselves; ask another admin.",
        )
    if current_user.hashed_password and not (
        payload.password and verify_password(payload.password, current_user.hashed_password)
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password is incorrect")

    email = current_user.email
    await db.delete(current_user)
    await db.commit()
    logger.warning("Account deleted by its owner: %s", email)


@router.get("/me/usage", response_model=UsageSummary)
async def read_usage(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UsageSummary:
    """Spend over time, by model and by category, plus recent activity."""
    return await usage.summarize(db, current_user.id, days)


@router.post("/topup")
async def dev_top_up_credits(
    amount: Decimal = Query(..., gt=0, le=MAX_DEV_TOPUP),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Grant free credits to the caller — local development only.

    This endpoint mints credits out of nothing, so it is disabled unless
    ``ALLOW_DEV_TOPUP`` is set, and the settings validator refuses to let that
    flag be enabled in production.
    """
    if not settings.ALLOW_DEV_TOPUP:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    new_balance = await billing.credit(db, current_user.id, amount)
    await db.commit()

    logger.warning("DEV top-up: %s credits granted to %s", amount, current_user.email)
    return {"message": "Credits updated", "new_credits": new_balance}
