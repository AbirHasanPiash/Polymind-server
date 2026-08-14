"""Add (user_id, created_at) indexes for the per-user timeline queries

Every list endpoint filters by user_id and orders by created_at desc. Without a
composite index those queries scan the whole table and sort the result; these
indexes turn each one into an index range scan.

Revision ID: c3a1d7e94b21
Revises: bd87a116edce
Create Date: 2026-08-14

"""

from collections.abc import Sequence

from alembic import op

revision: str = "c3a1d7e94b21"
down_revision: str | Sequence[str] | None = "bd87a116edce"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEXES: list[tuple[str, str]] = [
    ("ix_chats_user_id_created_at", "chats"),
    ("ix_generated_audio_user_id_created_at", "generated_audio"),
    ("ix_generated_images_user_id_created_at", "generated_images"),
    ("ix_generated_videos_user_id_created_at", "generated_videos"),
    ("ix_transactions_user_id_created_at", "transactions"),
]


def upgrade() -> None:
    for index_name, table_name in _INDEXES:
        op.create_index(index_name, table_name, ["user_id", "created_at"], unique=False)


def downgrade() -> None:
    for index_name, table_name in reversed(_INDEXES):
        op.drop_index(index_name, table_name=table_name)
