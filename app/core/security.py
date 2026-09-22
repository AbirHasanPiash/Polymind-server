"""Password hashing, JWT issuing/verification and the auth dependencies."""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.redis import get_redis_client
from app.models.user import User

logger = logging.getLogger(__name__)

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

TOKEN_TYPE_ACCESS = "access"

CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def verify_password(plain_password: str, hashed_password: str | None) -> bool:
    """Constant-ish time password check.

    Accounts created through Google have no password hash; those must never be
    able to log in with an empty password, hence the explicit guard.
    """
    if not hashed_password:
        return False
    try:
        return password_hash.verify(plain_password, hashed_password)
    except Exception:  # malformed/legacy hash
        return False


def get_password_hash(password: str) -> str:
    return password_hash.hash(password)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """Issue a signed access token for ``subject`` (the user's email)."""
    now = datetime.now(UTC)
    expire = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    claims = {
        "sub": subject,
        "type": TOKEN_TYPE_ACCESS,
        "iat": now,
        "exp": expire,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict[str, Any] | None:
    """Return the token payload, or None when it is invalid/expired."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"require_exp": True, "require_sub": True},
        )
    except JWTError:
        return None

    # ``type`` is absent from tokens issued before it was introduced; treat those
    # as access tokens so existing sessions survive the upgrade.
    if payload.get("type", TOKEN_TYPE_ACCESS) != TOKEN_TYPE_ACCESS:
        return None
    return payload


async def _load_active_user(email: str, db: AsyncSession) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        return None
    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    payload = decode_token(token)
    if payload is None:
        raise CREDENTIALS_EXCEPTION

    user = await _load_active_user(payload["sub"], db)
    if user is None:
        raise CREDENTIALS_EXCEPTION
    return user


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require a superuser. Every /admin route must depend on this."""
    if not current_user.is_superuser:
        logger.warning("Forbidden admin access attempt by %s", current_user.email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user does not have enough privileges",
        )
    return current_user


async def verify_token_socket(token: str, db: AsyncSession) -> User | None:
    """Authenticate a WebSocket handshake. Returns None instead of raising."""
    payload = decode_token(token)
    if payload is None:
        return None
    return await _load_active_user(payload["sub"], db)


# ── password reset tokens ─────────────────────────────────────────────────
#
# Single-use, short-lived, stored hashed in Redis: a leaked database dump or a
# log line cannot be replayed into a reset.


def _reset_key(token: str) -> str:
    digest = hashlib.sha256(token.encode()).hexdigest()
    return f"password-reset:{digest}"


async def issue_password_reset_token(user_id: uuid.UUID) -> str:
    token = secrets.token_urlsafe(32)
    await get_redis_client().setex(
        _reset_key(token), settings.PASSWORD_RESET_TOKEN_MINUTES * 60, str(user_id)
    )
    return token


async def consume_password_reset_token(token: str) -> uuid.UUID | None:
    """Return the user id for a valid token and invalidate it, or None."""
    redis = get_redis_client()
    key = _reset_key(token)
    async with redis.pipeline(transaction=True) as pipe:
        pipe.get(key)
        pipe.delete(key)
        value, _ = await pipe.execute()
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None
