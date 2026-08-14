from sqlalchemy import Column, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base, created_at_column, uuid_pk

GATEWAY_STRIPE = "stripe"
GATEWAY_RAZORPAY = "razorpay"

STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class Transaction(Base):
    """A credit purchase. One row per checkout attempt, per gateway."""

    __tablename__ = "transactions"

    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    package_id = Column(UUID(as_uuid=True), ForeignKey("packages.id"), nullable=True)

    payment_gateway = Column(String, default=GATEWAY_STRIPE, index=True)

    # Gateway references are unique so a replayed webhook cannot create a duplicate.
    stripe_session_id = Column(String, unique=True, index=True, nullable=True)
    razorpay_order_id = Column(String, unique=True, index=True, nullable=True)
    razorpay_payment_id = Column(String, unique=True, index=True, nullable=True)

    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String, default="usd")
    credits_added = Column(Numeric(18, 6), nullable=False)

    status = Column(String, default=STATUS_PENDING, index=True)

    created_at = created_at_column()
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="transactions")
    package = relationship("Package", back_populates="transactions")

    __table_args__ = (Index("ix_transactions_user_id_created_at", "user_id", "created_at"),)
