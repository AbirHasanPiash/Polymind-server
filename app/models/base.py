"""Shared building blocks for ORM models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base

__all__ = ["Base", "utc_now", "uuid_pk", "created_at_column", "updated_at_column"]


def utc_now() -> datetime:
    """Timezone-aware "now". Naive datetimes are a bug source, so never use utcnow()."""
    return datetime.now(UTC)


def uuid_pk() -> Column:
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)


def created_at_column() -> Column:
    return Column(DateTime(timezone=True), default=utc_now, nullable=True)


def updated_at_column() -> Column:
    return Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=True)
