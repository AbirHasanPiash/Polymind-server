from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base, created_at_column, updated_at_column, uuid_pk

ROLE_USER = "user"
ROLE_ASSISTANT = "ai"

MODE_CHAT = "chat"
MODE_ARENA = "arena"

FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_REFUSAL = "refusal"
FINISH_INTERRUPTED = "interrupted"
FINISH_ERROR = "error"


class Chat(Base):
    __tablename__ = "chats"

    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    # "chat" for a normal conversation, "arena" when several models answer side by side.
    mode = Column(String, nullable=False, default=MODE_CHAT, server_default=text("'chat'"))
    pinned = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    # Optional per-conversation instructions (a persona, a tone, a project brief).
    system_prompt = Column(Text, nullable=True)
    # Public read-only share link; null until the owner shares the conversation.
    share_token = Column(String(64), unique=True, index=True, nullable=True)
    shared_at = Column(DateTime(timezone=True), nullable=True)
    created_at = created_at_column()
    # Bumped on every turn so the sidebar can order by recent activity.
    updated_at = updated_at_column()

    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    user = relationship("User", back_populates="chats")

    __table_args__ = (
        Index("ix_chats_user_id_created_at", "user_id", "created_at"),
        Index("ix_chats_user_id_updated_at", "user_id", "updated_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = uuid_pk()
    chat_id = Column(UUID(as_uuid=True), ForeignKey("chats.id"), nullable=False)

    role = Column(String, nullable=False)  # ROLE_USER | ROLE_ASSISTANT
    content = Column(Text, nullable=False)
    attachments = Column(JSON, default=list)
    model = Column(String, nullable=True)
    # For assistant messages: the user message they answer. Lets several arena
    # replies hang off one prompt.
    parent_id = Column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )

    tokens = Column(Integer, default=0)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    cost = Column(Numeric(18, 6), default=0)
    duration_ms = Column(Integer, nullable=True)
    finish_reason = Column(String(32), nullable=True)
    created_at = created_at_column()

    chat = relationship("Chat", back_populates="messages")

    __table_args__ = (Index("ix_messages_chat_id_created_at", "chat_id", "created_at"),)
