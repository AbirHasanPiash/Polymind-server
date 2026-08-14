from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TransactionBase(BaseModel):
    amount: Decimal
    currency: str
    credits_added: Decimal
    status: str


class TransactionResponse(TransactionBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    package_id: UUID | None = None
    payment_gateway: str | None = None
    # Exactly one gateway reference is set per transaction, so both are optional.
    stripe_session_id: str | None = None
    razorpay_order_id: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class RazorpayVerification(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
