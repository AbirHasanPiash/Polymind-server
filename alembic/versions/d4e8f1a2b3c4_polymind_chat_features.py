"""Polymind: chat modes, sharing, message metadata and user preferences

Adds what the redesigned chat needs — pinned conversations, per-chat system
prompts, an arena mode, public share links, per-message token/cost/timing
metadata with a parent link for side-by-side replies — plus a preferences
document on users.

Revision ID: d4e8f1a2b3c4
Revises: c3a1d7e94b21
Create Date: 2026-09-21

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d4e8f1a2b3c4"
down_revision: str | Sequence[str] | None = "c3a1d7e94b21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chats",
        sa.Column("mode", sa.String(), nullable=False, server_default=sa.text("'chat'")),
    )
    op.add_column(
        "chats",
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column("chats", sa.Column("system_prompt", sa.Text(), nullable=True))
    op.add_column("chats", sa.Column("share_token", sa.String(length=64), nullable=True))
    op.add_column("chats", sa.Column("shared_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("chats", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE chats SET updated_at = created_at WHERE updated_at IS NULL")
    op.create_index("ix_chats_share_token", "chats", ["share_token"], unique=True)
    op.create_index("ix_chats_user_id_updated_at", "chats", ["user_id", "updated_at"], unique=False)

    op.add_column("messages", sa.Column("parent_id", sa.UUID(), nullable=True))
    op.add_column("messages", sa.Column("prompt_tokens", sa.Integer(), nullable=True))
    op.add_column("messages", sa.Column("completion_tokens", sa.Integer(), nullable=True))
    op.add_column("messages", sa.Column("duration_ms", sa.Integer(), nullable=True))
    op.add_column("messages", sa.Column("finish_reason", sa.String(length=32), nullable=True))
    op.create_foreign_key(
        "fk_messages_parent_id_messages",
        "messages",
        "messages",
        ["parent_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.add_column(
        "users",
        sa.Column("preferences", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
    )
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "last_login_at")
    op.drop_column("users", "preferences")

    op.drop_constraint("fk_messages_parent_id_messages", "messages", type_="foreignkey")
    op.drop_column("messages", "finish_reason")
    op.drop_column("messages", "duration_ms")
    op.drop_column("messages", "completion_tokens")
    op.drop_column("messages", "prompt_tokens")
    op.drop_column("messages", "parent_id")

    op.drop_index("ix_chats_user_id_updated_at", table_name="chats")
    op.drop_index("ix_chats_share_token", table_name="chats")
    op.drop_column("chats", "updated_at")
    op.drop_column("chats", "shared_at")
    op.drop_column("chats", "share_token")
    op.drop_column("chats", "system_prompt")
    op.drop_column("chats", "pinned")
    op.drop_column("chats", "mode")
