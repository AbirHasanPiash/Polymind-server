"""Current-user profile and wallet."""

from __future__ import annotations

import logging
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.user import UserResponse, UserUpdateProfile
from app.services import billing

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_DEV_TOPUP = Decimal("10000")


@router.get("/me", response_model=UserResponse)
async def read_user_me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Profile plus the current credit balance."""
    await db.refresh(current_user, attribute_names=["wallet"])
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_user_me(
    payload: UserUpdateProfile,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Update the caller's own profile."""
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(current_user, field, value)

    await db.commit()
    await db.refresh(current_user, attribute_names=["wallet"])
    return current_user


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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    new_balance = await billing.credit(db, current_user.id, amount)
    await db.commit()

    logger.warning("DEV top-up: %s credits granted to %s", amount, current_user.email)
    return {"message": "Credits updated", "new_credits": new_balance}
