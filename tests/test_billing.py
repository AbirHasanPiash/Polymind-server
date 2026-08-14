"""Wallet accounting rules.

The database enforces atomicity; these tests pin the contract around it — that a
debit which cannot be covered raises instead of silently going negative, and that
an overdraft only happens where it is explicitly allowed.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from app.services import billing

USER_ID = uuid.uuid4()


class FakeSession:
    """Stands in for AsyncSession, recording the statements it is given.

    ``scalar`` returns the queued results in order, mirroring how the real
    session answers the conditional UPDATE (None when no row matched).
    """

    def __init__(self, *results):
        self.results = list(results)
        self.statements = []

    async def scalar(self, statement):
        self.statements.append(statement)
        return self.results.pop(0) if self.results else None


class TestDebit:
    async def test_returns_the_new_balance(self):
        db = FakeSession(Decimal("7.5"))
        assert await billing.debit(db, USER_ID, Decimal("2.5")) == Decimal("7.5")

    async def test_guards_the_balance_in_the_where_clause(self):
        """The check and the write must be one statement, not read-then-write."""
        db = FakeSession(Decimal("1"))
        await billing.debit(db, USER_ID, Decimal("5"))
        assert "credits >=" in str(db.statements[0]).replace("wallets.", "")

    async def test_overdraft_drops_the_guard(self):
        db = FakeSession(Decimal("-0.2"))
        await billing.debit(db, USER_ID, Decimal("5"), allow_overdraft=True)
        assert "credits >=" not in str(db.statements[0]).replace("wallets.", "")

    async def test_insufficient_balance_raises(self):
        # No row matched the conditional update, then the balance lookup.
        db = FakeSession(None, Decimal("1.0"))
        with pytest.raises(billing.InsufficientCreditsError) as exc:
            await billing.debit(db, USER_ID, Decimal("5"))
        assert exc.value.required == Decimal("5")
        assert exc.value.balance == Decimal("1.0")

    @pytest.mark.parametrize("amount", [Decimal("0"), Decimal("-1")])
    async def test_non_positive_amounts_do_not_write(self, amount):
        db = FakeSession(Decimal("3"))
        await billing.debit(db, USER_ID, amount)
        assert "UPDATE" not in str(db.statements[0]).upper()


class TestCredit:
    async def test_returns_the_new_balance(self):
        db = FakeSession(Decimal("110"))
        assert await billing.credit(db, USER_ID, Decimal("10")) == Decimal("110")

    async def test_creates_the_wallet_when_missing(self):
        """An upsert means a purchase never fails because a wallet row is absent."""
        db = FakeSession(Decimal("10"))
        await billing.credit(db, USER_ID, Decimal("10"))
        statement = str(db.statements[0]).upper()
        assert "INSERT INTO WALLETS" in statement
        assert "ON CONFLICT" in statement

    async def test_refund_is_a_credit(self):
        db = FakeSession(Decimal("42"))
        assert await billing.refund(db, USER_ID, Decimal("2"), "task failed") == Decimal("42")


class TestBalanceChecks:
    async def test_missing_wallet_reads_as_zero(self):
        assert await billing.get_balance(FakeSession(None), USER_ID) == Decimal("0")

    async def test_has_credits_is_strictly_positive(self):
        assert await billing.has_credits(FakeSession(Decimal("0.000001")), USER_ID)
        assert not await billing.has_credits(FakeSession(Decimal("0")), USER_ID)
        assert not await billing.has_credits(FakeSession(Decimal("-1")), USER_ID)
