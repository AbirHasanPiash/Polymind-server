"""Payment settlement rules.

The expensive failure mode here is granting credits twice for one payment, so
these tests drive the Stripe handler with a fake session and assert that only a
paid, still-pending transaction is ever credited.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from app.api.v1.endpoints import payments
from app.models.transaction import STATUS_COMPLETED, STATUS_PENDING


class FakeTransaction:
    def __init__(self, status=STATUS_PENDING, credits=Decimal("100")):
        self.id = uuid4()
        self.user_id = uuid4()
        self.status = status
        self.credits_added = credits
        self.completed_at = None


class FakeSession:
    """Minimal AsyncSession stand-in: returns a fixed row and records commits."""

    def __init__(self, transaction=None):
        self.transaction = transaction
        self.commits = 0
        self.added = []

    async def scalar(self, _statement):
        return self.transaction

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass


@pytest.fixture
def credits_granted(monkeypatch):
    """Capture every call to billing.credit made through the payments module."""
    granted = []

    async def fake_credit(_db, user_id, amount):
        granted.append((user_id, amount))
        return amount

    monkeypatch.setattr(payments.billing, "credit", fake_credit)
    return granted


def _session(payment_status="paid", session_id="cs_test_123", metadata=None):
    return {
        "id": session_id,
        "payment_status": payment_status,
        "amount_total": 1000,
        "currency": "usd",
        "metadata": metadata if metadata is not None else {},
    }


class TestStripeSettlement:
    async def test_paid_pending_transaction_is_credited_once(self, credits_granted):
        transaction = FakeTransaction()
        db = FakeSession(transaction)

        await payments._handle_checkout_completed(_session(), db)

        assert credits_granted == [(transaction.user_id, Decimal("100"))]
        assert transaction.status == STATUS_COMPLETED
        assert transaction.completed_at is not None
        assert db.commits == 1

    async def test_replayed_webhook_does_not_credit_again(self, credits_granted):
        """Stripe retries deliveries; the second one must be a no-op."""
        transaction = FakeTransaction(status=STATUS_COMPLETED)
        db = FakeSession(transaction)

        await payments._handle_checkout_completed(_session(), db)

        assert credits_granted == []
        assert db.commits == 0

    @pytest.mark.parametrize("payment_status", ["unpaid", "no_payment_required", None])
    async def test_unpaid_sessions_are_ignored(self, credits_granted, payment_status):
        """checkout.session.completed also fires before async methods settle."""
        transaction = FakeTransaction()
        db = FakeSession(transaction)

        await payments._handle_checkout_completed(_session(payment_status=payment_status), db)

        assert credits_granted == []
        assert transaction.status == STATUS_PENDING

    async def test_unknown_session_without_metadata_is_ignored(self, credits_granted):
        db = FakeSession(transaction=None)

        await payments._handle_checkout_completed(_session(), db)

        assert credits_granted == []
        assert db.added == []

    async def test_missing_record_is_rebuilt_from_signed_metadata(self, credits_granted):
        """A lost checkout row must not lose the customer's purchase."""
        db = FakeSession(transaction=None)
        metadata = {"user_id": str(uuid4()), "credits": "250", "package_id": str(uuid4())}

        await payments._handle_checkout_completed(_session(metadata=metadata), db)

        assert len(db.added) == 1
        rebuilt = db.added[0]
        assert rebuilt.credits_added == Decimal("250")
        assert rebuilt.status == STATUS_COMPLETED
        assert credits_granted == [(rebuilt.user_id, Decimal("250"))]


class TestAmountConversion:
    @pytest.mark.parametrize(
        ("price", "expected"),
        [
            (Decimal("9.99"), 999),
            (Decimal("10.00"), 1000),
            (Decimal("0.01"), 1),
            (Decimal("19.995"), 2000),  # rounds instead of truncating to 1999
            (Decimal("100"), 10000),
        ],
    )
    def test_prices_convert_to_minor_units(self, price, expected):
        assert payments._to_minor_units(price) == expected
