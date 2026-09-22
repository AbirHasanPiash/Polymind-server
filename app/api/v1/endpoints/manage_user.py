"""Admin user management. Every route here requires a superuser."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.database import get_db
from app.core.security import get_current_admin
from app.models import User, Wallet
from app.schemas.manage_user import UserAdminResponse, UserListResponse, UserUpdateAdmin

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_admin)])


@router.get("", response_model=UserListResponse)
async def list_users_admin(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None, max_length=100),
    is_active: bool | None = None,
    is_superuser: bool | None = None,
    db: AsyncSession = Depends(get_db),
) -> UserListResponse:
    """Paginated user list with optional search and status filters."""
    filters = []
    if search:
        pattern = f"%{search}%"
        filters.append(or_(User.email.ilike(pattern), User.full_name.ilike(pattern)))
    if is_active is not None:
        filters.append(User.is_active == is_active)
    if is_superuser is not None:
        filters.append(User.is_superuser == is_superuser)

    total_count = await db.scalar(select(func.count(User.id)).where(*filters))

    result = await db.execute(
        select(User)
        .options(joinedload(User.wallet))
        .where(*filters)
        .order_by(User.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
    )

    return UserListResponse(
        users=result.unique().scalars().all(),
        total_count=total_count or 0,
        page=page,
        size=size,
    )


@router.patch("/{user_id}", response_model=UserAdminResponse)
async def update_user_admin(
    user_id: UUID,
    obj_in: UserUpdateAdmin,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> User:
    """Update a user's profile, status, role or credit balance."""
    user = await db.get(User, user_id, options=[joinedload(User.wallet)])
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    update_data = obj_in.model_dump(exclude_unset=True)

    # Guard against an admin locking themselves out of the admin surface.
    if user.id == admin.id:
        if update_data.get("is_superuser") is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot remove your own admin privileges",
            )
        if update_data.get("is_active") is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot deactivate your own account",
            )

    if "credits" in update_data:
        new_credits = update_data.pop("credits")
        if user.wallet:
            user.wallet.credits = new_credits
        else:
            db.add(Wallet(user_id=user.id, credits=new_credits))

    for field, value in update_data.items():
        setattr(user, field, value)

    await db.commit()
    await db.refresh(user, attribute_names=["wallet"])

    logger.info(
        "Admin %s updated user %s (%s)",
        admin.email,
        user.email,
        ", ".join(update_data) or "credits",
    )
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_admin(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> None:
    """Delete a user and everything they own (chats, media, transactions)."""
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account",
        )

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await db.delete(user)
    await db.commit()
    logger.warning("Admin %s deleted user %s", admin.email, user.email)
