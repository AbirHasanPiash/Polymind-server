from sqlalchemy import Boolean, Column, Numeric, String, Text
from sqlalchemy.orm import relationship

from app.models.base import Base, created_at_column, updated_at_column, uuid_pk


class Package(Base):
    """A purchasable bundle of credits."""

    __tablename__ = "packages"

    id = uuid_pk()

    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)

    price = Column(Numeric(10, 2), nullable=False)
    currency = Column(String, default="USD")
    credits = Column(Numeric(18, 6), nullable=False)

    is_active = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)

    created_at = created_at_column()
    updated_at = updated_at_column()

    transactions = relationship("Transaction", back_populates="package")
