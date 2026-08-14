from sqlalchemy import Boolean, Column, ForeignKey, Numeric, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base, created_at_column, updated_at_column, uuid_pk

# Credits granted to a brand new account.
SIGNUP_BONUS_CREDITS = 10


class User(Base):
    __tablename__ = "users"

    id = uuid_pk()
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)  # null for OAuth-only accounts
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True, server_default=text("true"), nullable=False)
    is_superuser = Column(Boolean, default=False, server_default=text("false"), nullable=False)
    created_at = created_at_column()
    updated_at = updated_at_column()

    wallet = relationship("Wallet", back_populates="user", uselist=False, cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    audios = relationship("GeneratedAudio", back_populates="user", cascade="all, delete-orphan")
    images = relationship("GeneratedImage", back_populates="user", cascade="all, delete-orphan")
    videos = relationship("GeneratedVideo", back_populates="user", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.email}>"


class Wallet(Base):
    __tablename__ = "wallets"

    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    # Numeric, never float: credits are money and must not drift.
    credits = Column(
        Numeric(18, 6),
        default=SIGNUP_BONUS_CREDITS,
        server_default=text("10.0"),
        nullable=False,
    )
    updated_at = updated_at_column()

    user = relationship("User", back_populates="wallet")
