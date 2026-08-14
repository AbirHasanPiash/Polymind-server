"""Async SQLAlchemy engine, session factory and FastAPI dependency."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Server-side statement caching and prepared statements must be disabled when
# talking to a transaction-mode pooler (PgBouncer, Supabase pooler); NullPool is
# the matching pool strategy because the pooler already owns the connections.
_connect_args: dict[str, Any] = {
    "timeout": settings.DB_CONNECT_TIMEOUT,
    "command_timeout": settings.DB_COMMAND_TIMEOUT,
    "statement_cache_size": 0,
    "server_settings": {"application_name": settings.PROJECT_NAME},
}

_pool_args: dict[str, Any] = (
    {"poolclass": NullPool}
    if settings.DB_USE_NULL_POOL
    else {
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_timeout": settings.DB_POOL_TIMEOUT,
        "pool_recycle": settings.DB_POOL_RECYCLE,
        "pool_pre_ping": True,
    }
)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    connect_args=_connect_args,
    **_pool_args,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a session that is rolled back on failure."""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await _safe_rollback(session)
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Session for code outside the request cycle (WebSockets, Celery tasks).

    Commits on success, rolls back on failure, always closes.
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await _safe_rollback(session)
            raise


async def _safe_rollback(session: AsyncSession) -> None:
    """Roll back without masking the original error if the link is already gone."""
    try:
        await session.rollback()
    except Exception as exc:  # pragma: no cover - only on a dead connection
        logger.debug("Rollback failed (connection likely closed): %s", exc)


async def check_db_connection() -> bool:
    """Return True when the database answers a trivial query."""
    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return False


async def warmup_db_connection(attempts: int = 3, delay: float = 2.0) -> bool:
    """Open a connection at startup so the first request isn't the slow one."""
    for attempt in range(1, attempts + 1):
        if await check_db_connection():
            logger.info("Database connection established")
            return True
        if attempt < attempts:
            logger.warning("Database warmup attempt %s/%s failed, retrying…", attempt, attempts)
            await asyncio.sleep(delay)
    logger.error("Could not establish a database connection during startup")
    return False
