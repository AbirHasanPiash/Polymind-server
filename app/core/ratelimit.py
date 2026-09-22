"""Redis-backed rate limiting.

A fixed window counter per key: cheap, one round-trip, good enough to stop
credential stuffing on the auth endpoints and runaway loops on the expensive
media endpoints. Redis being unavailable never blocks a request — the limiter
fails open and logs, because a cache outage must not become an outage.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, status

from app.core.redis import get_redis_client
from app.core.security import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)


async def hit(key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
    """Count a hit against ``key``. Returns (allowed, seconds until reset)."""
    redis = get_redis_client()
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.ttl(key)
            count, ttl = await pipe.execute()
        if ttl is None or ttl < 0:
            await redis.expire(key, window_seconds)
            ttl = window_seconds
        return count <= limit, int(ttl)
    except Exception as exc:  # fail open
        logger.warning("Rate limiter unavailable (%s); allowing request", exc)
        return True, 0


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _too_many(retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too many requests. Please slow down and try again shortly.",
        headers={"Retry-After": str(max(retry_after, 1))},
    )


def limit_by_ip(scope: str, limit: int, window_seconds: int) -> Callable:
    """Dependency: at most ``limit`` requests per client IP per window."""

    async def dependency(request: Request) -> None:
        allowed, retry_after = await hit(
            f"ratelimit:{scope}:ip:{_client_ip(request)}", limit, window_seconds
        )
        if not allowed:
            raise _too_many(retry_after)

    return dependency


def limit_by_user(scope: str, limit: int, window_seconds: int) -> Callable:
    """Dependency: at most ``limit`` requests per signed-in user per window."""

    async def dependency(current_user: User = Depends(get_current_user)) -> None:
        allowed, retry_after = await hit(
            f"ratelimit:{scope}:user:{current_user.id}", limit, window_seconds
        )
        if not allowed:
            raise _too_many(retry_after)

    return dependency
