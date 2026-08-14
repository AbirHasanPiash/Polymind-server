from sqlalchemy import JSON, Column, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base, created_at_column, uuid_pk

ROLE_USER = "user"
ROLE_ASSISTANT = "ai"


class Chat(Base):
    __tablename__ = "chats"

    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    created_at = created_at_column()

    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    user = relationship("User", back_populates="chats")

    __table_args__ = (Index("ix_chats_user_id_created_at", "user_id", "created_at"),)


class Message(Base):
    __tablename__ = "messages"

    id = uuid_pk()
    chat_id = Column(UUID(as_uuid=True), ForeignKey("chats.id"), nullable=False)

    role = Column(String, nullable=False)  # ROLE_USER | ROLE_ASSISTANT
    content = Column(Text, nullable=False)
    attachments = Column(JSON, default=list)
    model = Column(String, nullable=True)

    tokens = Column(Integer, default=0)
    cost = Column(Numeric(18, 6), default=0)
    created_at = created_at_column()

    chat = relationship("Chat", back_populates="messages")

    __table_args__ = (Index("ix_messages_chat_id_created_at", "chat_id", "created_at"),)
