"""Credit purchases through Stripe (hosted checkout) and Razorpay (client-side).

Both flows converge on the same rule: credits are only ever granted while the
matching transaction row is locked ``FOR UPDATE`` and still ``pending``. That
makes a replayed webhook, a double-clicked verify call and two concurrent
deliveries of the same event all no-ops instead of duplicate credit.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from functools import cache
from uuid import UUID

import razorpay
import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.base import utc_now
from app.models.package import Package
from app.models.transaction import (
    GATEWAY_RAZORPAY,
    GATEWAY_STRIPE,
    STATUS_COMPLETED,
    STATUS_PENDING,
    Transaction,
)
from app.models.user import User
from app.schemas.transaction import RazorpayVerification, TransactionResponse
from app.services import billing

logger = logging.getLogger(__name__)
router = APIRouter()

# Gateways expect the smallest currency unit (cents, paise).
MINOR_UNITS = 100


def _gateway_unavailable(name: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"{name} payments are not configured",
    )


@cache
def get_razorpay_client() -> razorpay.Client:
    """Built on first use so a deployment without Razorpay still starts."""
    if not settings.razorpay_enabled:
        raise _gateway_unavailable("Razorpay")
    return razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))


def _require_stripe() -> None:
    if not settings.stripe_enabled:
        raise _gateway_unavailable("Stripe")
    stripe.api_key = settings.STRIPE_SECRET_KEY


def _to_minor_units(amount: Decimal) -> int:
    """Convert a decimal price to integer minor units without truncating."""
    return int((amount * MINOR_UNITS).to_integral_value(rounding="ROUND_HALF_UP"))


async def _get_active_package(db: AsyncSession, package_id: UUID) -> Package:
    package = await db.get(Package, package_id)
    if not package or not package.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Package not found or inactive",
        )
    return package


# Stripe


@router.post("/create-checkout-session/{package_id}")
async def create_checkout_session(
    package_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Start a Stripe Checkout session for a credit package."""
    _require_stripe()
    package = await _get_active_package(db, package_id)

    try:
        checkout_session = await asyncio.to_thread(
            stripe.checkout.Session.create,
            payment_method_types=["card"],
            billing_address_collection="required",
            customer_email=current_user.email,
            line_items=[
                {
                    "price_data": {
                        "currency": (package.currency or "usd").lower(),
                        "product_data": {
                            "name": package.name,
                            "description": package.description or package.name,
                        },
                        "unit_amount": _to_minor_units(package.price),
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url=(
                f"{settings.FRONTEND_URL}/dashboard/payment/success"
                "?session_id={CHECKOUT_SESSION_ID}"
            ),
            cancel_url=f"{settings.FRONTEND_URL}/dashboard/payment/cancel",
            metadata={
                "user_id": str(current_user.id),
                "package_id": str(package.id),
                "credits": str(package.credits),
            },
            client_reference_id=str(current_user.id),
        )
    except stripe.StripeError as exc:
        logger.error("Stripe session creation failed for %s: %s", current_user.email, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not start the checkout session",
        ) from exc

    db.add(
        Transaction(
            user_id=current_user.id,
            package_id=package.id,
            payment_gateway=GATEWAY_STRIPE,
            stripe_session_id=checkout_session.id,
            amount=package.price,
            currency=(package.currency or "usd").lower(),
            credits_added=package.credits,
            status=STATUS_PENDING,
        )
    )
    await db.commit()

    return {"checkout_url": checkout_session.url, "session_id": checkout_session.id}


@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Receive Stripe events. Only signed events are trusted."""
    _require_stripe()
    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid payload") from exc
    except stripe.SignatureVerificationError as exc:
        logger.warning("Rejected Stripe webhook with an invalid signature")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature") from exc

    if event["type"] == "checkout.session.completed":
        await _handle_checkout_completed(event["data"]["object"], db)
    else:
        logger.debug("Ignoring Stripe event %s", event["type"])

    return {"status": "success"}


async def _handle_checkout_completed(session: dict, db: AsyncSession) -> None:
    """Grant credits for a paid checkout session, exactly once."""
    session_id = session.get("id")

    # "completed" also fires for asynchronous methods that have not settled yet.
    if session.get("payment_status") != "paid":
        logger.info("Checkout %s completed but not paid (%s)", session_id, session.get("payment_status"))
        return

    transaction = await _lock_or_create_stripe_transaction(session, db)
    if transaction is None or transaction.status == STATUS_COMPLETED:
        return  # unknown session, or already credited by an earlier delivery

    await billing.credit(db, transaction.user_id, transaction.credits_added)
    transaction.status = STATUS_COMPLETED
    transaction.completed_at = utc_now()
    await db.commit()

    logger.info(
        "Credited %s credits to user %s for Stripe session %s",
        transaction.credits_added, transaction.user_id, session_id,
    )


async def _lock_or_create_stripe_transaction(
    session: dict, db: AsyncSession
) -> Transaction | None:
    """Return the locked transaction for a session, creating it if it is missing.

    The row is normally written when checkout starts. It can be absent if that
    write failed after Stripe accepted the payment, so the webhook rebuilds it
    from the (signature-verified) metadata rather than dropping the purchase.
    """
    session_id = session["id"]
    locked = (
        select(Transaction).where(Transaction.stripe_session_id == session_id).with_for_update()
    )

    transaction = await db.scalar(locked)
    if transaction is not None:
        return transaction

    metadata = session.get("metadata") or {}
    user_id, credits = metadata.get("user_id"), metadata.get("credits")
    if not user_id or not credits:
        logger.error("Stripe session %s has no usable metadata; skipping", session_id)
        return None

    package_id = metadata.get("package_id")
    transaction = Transaction(
        user_id=UUID(user_id),
        package_id=UUID(package_id) if package_id else None,
        payment_gateway=GATEWAY_STRIPE,
        stripe_session_id=session_id,
        amount=Decimal(session.get("amount_total", 0)) / MINOR_UNITS,
        currency=(session.get("currency") or "usd").lower(),
        credits_added=Decimal(credits),
        status=STATUS_PENDING,
    )
    db.add(transaction)
    try:
        # The INSERT itself holds the row, so no extra lock is needed here.
        await db.flush()
    except IntegrityError:
        # A concurrent delivery inserted it first; take the lock on that row.
        await db.rollback()
        return await db.scalar(locked)

    logger.warning("Rebuilt missing transaction record for Stripe session %s", session_id)
    return transaction


# Razorpay


@router.post("/create-razorpay-order/{package_id}")
async def create_razorpay_order(
    package_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Create a Razorpay order for a credit package."""
    client = get_razorpay_client()
    package = await _get_active_package(db, package_id)

    currency = (package.currency or "USD").upper()
    amount_minor = _to_minor_units(package.price)

    try:
        order = await asyncio.to_thread(
            client.order.create,
            data={
                "amount": amount_minor,
                "currency": currency,
                "receipt": f"rcpt_{str(current_user.id)[:8]}_{int(utc_now().timestamp())}",
                "notes": {
                    "user_id": str(current_user.id),
                    "package_id": str(package.id),
                    "credits": str(package.credits),
                },
            },
        )
    except Exception as exc:
        logger.error("Razorpay order creation failed for %s: %s", current_user.email, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not create the payment order",
        ) from exc

    db.add(
        Transaction(
            user_id=current_user.id,
            package_id=package.id,
            payment_gateway=GATEWAY_RAZORPAY,
            razorpay_order_id=order["id"],
            amount=package.price,
            currency=currency.lower(),
            credits_added=package.credits,
            status=STATUS_PENDING,
        )
    )
    await db.commit()

    return {
        "order_id": order["id"],
        "amount": amount_minor,
        "currency": currency,
        "key_id": settings.RAZORPAY_KEY_ID,
    }


@router.post("/verify-razorpay-payment")
async def verify_razorpay_payment(
    verification: RazorpayVerification,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Verify a client-side Razorpay payment and credit the wallet."""
    client = get_razorpay_client()

    try:
        client.utility.verify_payment_signature(
            {
                "razorpay_order_id": verification.razorpay_order_id,
                "razorpay_payment_id": verification.razorpay_payment_id,
                "razorpay_signature": verification.razorpay_signature,
            }
        )
    except razorpay.errors.SignatureVerificationError:
        logger.warning("Razorpay signature verification failed for %s", current_user.email)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment signature verification failed",
        ) from None

    # FOR UPDATE serialises concurrent verifications of the same order.
    transaction = await db.scalar(
        select(Transaction)
        .where(Transaction.razorpay_order_id == verification.razorpay_order_id)
        .with_for_update()
    )

    if transaction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")

    # A valid signature proves the payment; it does not prove who is calling.
    if transaction.user_id != current_user.id:
        logger.warning(
            "User %s tried to verify a payment belonging to %s",
            current_user.email, transaction.user_id,
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Order does not belong to you")

    if transaction.status == STATUS_COMPLETED:
        return {"status": "success", "message": "Payment already processed"}

    await billing.credit(db, transaction.user_id, transaction.credits_added)
    transaction.status = STATUS_COMPLETED
    transaction.razorpay_payment_id = verification.razorpay_payment_id
    transaction.completed_at = utc_now()

    try:
        await db.commit()
    except IntegrityError:
        # razorpay_payment_id is unique: the same payment cannot settle twice.
        await db.rollback()
        logger.warning("Duplicate Razorpay payment id %s", verification.razorpay_payment_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This payment has already been processed",
        ) from None

    logger.info(
        "Credited %s credits to user %s for Razorpay order %s",
        transaction.credits_added, transaction.user_id, transaction.razorpay_order_id,
    )
    return {"status": "success", "message": "Payment verified and credits added"}


# History


@router.get("/history", response_model=list[TransactionResponse])
async def read_payment_history(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Transaction]:
    """The caller's own purchase history, newest first."""
    result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .order_by(Transaction.created_at.desc())
        .limit(min(limit, 200))
    )
    return list(result.scalars().all())
