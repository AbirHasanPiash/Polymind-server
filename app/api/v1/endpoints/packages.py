"""Credit packages. Reading is open to any user; writing is admin-only."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_admin, get_current_user
from app.models.package import Package
from app.models.user import User
from app.schemas.package import PackageCreate, PackageResponse, PackageUpdate

router = APIRouter()


@router.get("/", response_model=list[PackageResponse])
async def read_packages(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    include_inactive: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Package]:
    """List packages available for purchase.

    Inactive packages are hidden by default — they used to be returned here even
    though checkout rejects them — and only an admin may ask to see them.
    """
    stmt = select(Package).order_by(Package.is_active.desc(), Package.price.asc())
    if not (include_inactive and current_user.is_superuser):
        stmt = stmt.where(Package.is_active.is_(True))

    result = await db.execute(stmt.offset(skip).limit(limit))
    return list(result.scalars().all())


@router.post(
    "/",
    response_model=PackageResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(get_current_admin)],
)
async def create_package(
    package_in: PackageCreate,
    db: AsyncSession = Depends(get_db),
) -> Package:
    """Create a package (admin only)."""
    duplicate = await db.scalar(select(Package.id).where(Package.name == package_in.name))
    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A package with this name already exists",
        )

    package = Package(**package_in.model_dump())
    db.add(package)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A package with this name already exists",
        ) from None
    await db.refresh(package)
    return package


@router.put(
    "/{package_id}",
    response_model=PackageResponse,
    dependencies=[Depends(get_current_admin)],
)
async def update_package(
    package_id: UUID,
    package_in: PackageUpdate,
    db: AsyncSession = Depends(get_db),
) -> Package:
    """Update a package (admin only)."""
    package = await db.get(Package, package_id)
    if not package:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Package not found")

    for field, value in package_in.model_dump(exclude_unset=True).items():
        setattr(package, field, value)

    await db.commit()
    await db.refresh(package)
    return package


@router.delete(
    "/{package_id}",
    response_model=PackageResponse,
    dependencies=[Depends(get_current_admin)],
)
async def delete_package(
    package_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Package:
    """Deactivate a package (admin only).

    Soft delete: transactions reference packages, so rows are never removed.
    """
    package = await db.get(Package, package_id)
    if not package:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Package not found")

    package.is_active = False
    await db.commit()
    await db.refresh(package)
    return package
