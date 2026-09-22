"""Signup, login, Google OAuth, session refresh and password reset."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TypeVar

from fastapi import APIRouter, Depends, HTTPException, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError

from app.core.config import settings
from app.core.database import async_session_maker
from app.core.ratelimit import limit_by_ip
from app.core.security import (
    consume_password_reset_token,
    create_access_token,
    get_current_user,
    get_password_hash,
    issue_password_reset_token,
    verify_password,
)
from app.models.user import SIGNUP_BONUS_CREDITS, User, Wallet
from app.schemas.user import (
    ForgotPasswordRequest,
    GoogleLogin,
    ResetPasswordRequest,
    Token,
    UserCreate,
    UserLogin,
)
from app.services.email import password_reset_email, send_email

logger = logging.getLogger(__name__)
router = APIRouter()

T = TypeVar("T")

# Managed Postgres services drop idle connections; the auth path is the one place
# where a transient reconnect should not surface as a failed login.
RETRYABLE_ERRORS = (OperationalError, InterfaceError, ConnectionRefusedError, TimeoutError)

INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail="Incorrect email or password",
)


def _token_response(subject: str) -> Token:
    return Token(
        access_token=create_access_token(subject),
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


async def execute_with_retry(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> T:
    """Run a database operation, retrying transient connection failures.

    Only connection-level errors are retried — a failed operation with a real
    error (integrity, programming) is raised immediately. ``CancelledError`` is
    never caught: swallowing it would keep work running after a client hangs up.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await operation()
        except RETRYABLE_ERRORS as exc:
            if attempt == max_retries:
                logger.error("Database unavailable after %s attempts: %s", max_retries, exc)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Database temporarily unavailable. Please try again shortly.",
                ) from exc
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                "Database attempt %s/%s failed (%s), retrying in %ss",
                attempt,
                max_retries,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


async def get_user_by_email(email: str) -> User | None:
    async def operation() -> User | None:
        async with async_session_maker() as db:
            result = await db.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()

    return await execute_with_retry(operation)


async def create_user_with_wallet(
    email: str,
    hashed_password: str | None,
    full_name: str,
) -> User:
    """Create the user and their wallet in a single transaction.

    Both rows commit together, so a user can never exist without a wallet. A
    concurrent signup with the same email loses the unique-index race and is
    reported as a duplicate rather than a 500.
    """

    async def operation() -> User:
        async with async_session_maker() as db:
            user = User(
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                is_active=True,
                last_login_at=datetime.now(UTC),
            )
            user.wallet = Wallet(credits=SIGNUP_BONUS_CREDITS)
            db.add(user)
            try:
                await db.commit()
            except IntegrityError:
                await db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered",
                ) from None
            await db.refresh(user)
            return user

    return await execute_with_retry(operation)


async def _mark_login(user_id) -> None:
    """Best-effort bookkeeping; a failure here must never fail the login."""
    try:
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            if user is not None:
                user.last_login_at = datetime.now(UTC)
                await db.commit()
    except Exception as exc:  # pragma: no cover - only on a flaky connection
        logger.debug("Could not record last login: %s", exc)


# Note: 201 would be the more correct status for a create, but existing clients
# check for 200, so the status is kept as-is.
@router.post(
    "/signup",
    response_model=Token,
    dependencies=[Depends(limit_by_ip("signup", settings.RATE_LIMIT_SIGNUP_PER_HOUR, 3600))],
)
async def signup(user_in: UserCreate) -> Token:
    """Register with email and password."""
    if await get_user_by_email(user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    user = await create_user_with_wallet(
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name or "New User",
    )
    logger.info("New user registered: %s", user.email)
    return _token_response(user.email)


@router.post(
    "/login",
    response_model=Token,
    dependencies=[Depends(limit_by_ip("login", settings.RATE_LIMIT_LOGIN_PER_MINUTE, 60))],
)
async def login(user_in: UserLogin) -> Token:
    """Exchange email and password for an access token."""
    user = await get_user_by_email(user_in.email)

    # Same error for unknown email and wrong password: revealing which one is
    # wrong turns the endpoint into an account-enumeration oracle.
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise INVALID_CREDENTIALS

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated",
        )

    await _mark_login(user.id)
    return _token_response(user.email)


@router.post(
    "/google",
    response_model=Token,
    dependencies=[Depends(limit_by_ip("google", settings.RATE_LIMIT_LOGIN_PER_MINUTE * 2, 60))],
)
async def google_login(login_data: GoogleLogin) -> Token:
    """Log in (or register) with a Google ID token."""
    if not settings.google_login_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google login is not configured",
        )

    try:
        id_info = await asyncio.to_thread(
            id_token.verify_oauth2_token,
            login_data.token,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except ValueError as exc:
        logger.warning("Rejected Google token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Google token",
        ) from None

    email = id_info.get("email")
    if not email or not id_info.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account has no verified email address",
        )

    user = await get_user_by_email(email)

    if user is None:
        logger.info("Registering new user from Google login: %s", email)
        user = await create_user_with_wallet(
            email=email,
            # No password: the account signs in with Google until the user sets
            # one through the reset flow. A null hash can never verify.
            hashed_password=None,
            full_name=id_info.get("name") or "Google User",
        )
    elif not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated",
        )
    else:
        await _mark_login(user.id)

    return _token_response(user.email)


@router.post("/refresh", response_model=Token)
async def refresh(current_user: User = Depends(get_current_user)) -> Token:
    """Issue a fresh token for a still-valid session.

    The client calls this while the app is in use, so an active user is never
    logged out mid-conversation by the hard expiry.
    """
    return _token_response(current_user.email)


@router.post(
    "/forgot-password",
    dependencies=[Depends(limit_by_ip("forgot-password", 5, 900))],
)
async def forgot_password(payload: ForgotPasswordRequest) -> dict:
    """Email a single-use reset link. Always answers the same way, so the
    endpoint cannot be used to check whether an address is registered."""
    user = await get_user_by_email(payload.email)
    if user is not None and user.is_active:
        token = await issue_password_reset_token(user.id)
        reset_url = f"{settings.FRONTEND_URL.rstrip('/')}/reset-password?token={token}"
        subject, text, html = password_reset_email(reset_url)
        delivered = await send_email(user.email, subject, text, html)
        if not delivered and not settings.is_production:
            logger.warning("DEV password reset link for %s: %s", user.email, reset_url)
    return {"message": "If that email is registered, a reset link is on its way."}


@router.post("/reset-password", response_model=Token)
async def reset_password(payload: ResetPasswordRequest) -> Token:
    """Set a new password from a reset token and sign the user in."""
    user_id = await consume_password_reset_token(payload.token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This reset link is invalid or has expired",
        )

    async def operation() -> User | None:
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            if user is None or not user.is_active:
                return None
            user.hashed_password = get_password_hash(payload.new_password)
            user.last_login_at = datetime.now(UTC)
            await db.commit()
            await db.refresh(user)
            return user

    user = await execute_with_retry(operation)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="This account is not available"
        )

    logger.info("Password reset completed for %s", user.email)
    return _token_response(user.email)
