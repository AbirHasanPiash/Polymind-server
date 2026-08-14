"""Wallet accounting.

Every credit movement goes through this module. The rules it enforces:

* **Atomic** — balances change with a single conditional SQL statement, never
  read-modify-write in Python. Two concurrent requests can therefore not both
  see the same balance and both spend it.
* **Exact** — amounts are ``Decimal`` end to end; floats are never allowed to
  touch money.
* **Guarded** — a debit that would overdraw fails instead of going negative,
  unless the caller explicitly opts into an overdraft (used for usage that has
  already been delivered, such as a completed LLM response).
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import utc_now
from app.models.user import Wallet

logger = logging.getLogger(__name__)

ZERO = Decimal("0")


class InsufficientCreditsError(Exception):
    """Raised when a wallet cannot cover a debit."""

    def __init__(self, required: Decimal, balance: Decimal) -> None:
        super().__init__(f"Insufficient credits: need {required}, balance {balance}")
        self.required = required
        self.balance = balance


async def get_balance(db: AsyncSession, user_id: uuid.UUID) -> Decimal:
    """Current balance, or 0 when the user has no wallet yet."""
    balance = await db.scalar(select(Wallet.credits).where(Wallet.user_id == user_id))
    return balance if balance is not None else ZERO


async def has_credits(db: AsyncSession, user_id: uuid.UUID, amount: Decimal = ZERO) -> bool:
    """Cheap pre-flight check. Not a reservation — always debit before delivering."""
    return await get_balance(db, user_id) > amount


async def debit(
    db: AsyncSession,
    user_id: uuid.UUID,
    amount: Decimal,
    *,
    allow_overdraft: bool = False,
) -> Decimal:
    """Deduct ``amount`` atomically and return the new balance.

    ``allow_overdraft`` is for work already performed (the response was streamed,
    the audio was generated): refusing the charge there would give it away free.
    The balance can then dip slightly negative, bounded by one unit of work, and
    the next request is rejected up front.

    The caller owns the transaction — commit after this returns.
    """
    if amount <= ZERO:
        return await get_balance(db, user_id)

    stmt = (
        update(Wallet)
        .where(Wallet.user_id == user_id)
        .values(credits=Wallet.credits - amount, updated_at=utc_now())
        .returning(Wallet.credits)
    )
    if not allow_overdraft:
        # The guard lives in the WHERE clause, so the check and the write are one
        # statement and cannot interleave with a competing debit.
        stmt = stmt.where(Wallet.credits >= amount)

    new_balance = await db.scalar(stmt)

    if new_balance is None:
        raise InsufficientCreditsError(required=amount, balance=await get_balance(db, user_id))

    return new_balance


async def credit(db: AsyncSession, user_id: uuid.UUID, amount: Decimal) -> Decimal:
    """Add ``amount`` to the wallet, creating it if the user has none.

    Used for purchases, refunds and admin adjustments. The upsert makes a
    concurrent double-credit impossible to lose and a missing wallet harmless.
    """
    if amount <= ZERO:
        return await get_balance(db, user_id)

    stmt = (
        insert(Wallet)
        .values(id=uuid.uuid4(), user_id=user_id, credits=amount, updated_at=utc_now())
        .on_conflict_do_update(
            index_elements=[Wallet.user_id],
            set_={"credits": Wallet.credits + amount, "updated_at": utc_now()},
        )
        .returning(Wallet.credits)
    )
    return await db.scalar(stmt)


async def refund(db: AsyncSession, user_id: uuid.UUID, amount: Decimal, reason: str) -> Decimal:
    """Give credits back when paid-for work failed. Always logged."""
    new_balance = await credit(db, user_id, amount)
    logger.info("Refunded %s credits to user %s (%s)", amount, user_id, reason)
    return new_balance
