from sqlalchemy import Column, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base, created_at_column, updated_at_column, uuid_pk

# Lifecycle of an asynchronously generated asset.
STATUS_PROCESSING = "processing"
STATUS_PROCESSING_EXTERNAL = "processing_external"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Placeholder stored before the real asset exists.
PENDING_URL = "pending"


class GeneratedAudio(Base):
    __tablename__ = "generated_audio"

    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    storage_path = Column(String, nullable=False)
    public_url = Column(String, nullable=False)
    cost = Column(Numeric(18, 6), nullable=False, default=0)
    text_prompt = Column(Text, nullable=False)
    voice_name = Column(String, nullable=True)
    provider = Column(String, default="google")
    source_message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=True)
    created_at = created_at_column()

    user = relationship("User", back_populates="audios")

    __table_args__ = (Index("ix_generated_audio_user_id_created_at", "user_id", "created_at"),)


class GeneratedImage(Base):
    __tablename__ = "generated_images"

    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    storage_path = Column(String, nullable=False)
    public_url = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    reference_image_url = Column(String, nullable=True)
    revised_prompt = Column(Text, nullable=True)

    model = Column(String, default="gpt-image-1.5")
    size = Column(String, default="1024x1024")
    quality = Column(String, default="standard")
    cost = Column(Numeric(18, 6), nullable=False, default=0)

    created_at = created_at_column()

    user = relationship("User", back_populates="images")

    __table_args__ = (Index("ix_generated_images_user_id_created_at", "user_id", "created_at"),)


class GeneratedVideo(Base):
    __tablename__ = "generated_videos"

    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    storage_path = Column(String, nullable=False)
    public_url = Column(String, nullable=False)
    thumbnail_url = Column(String, nullable=True)

    script_text = Column(Text, nullable=False)
    source_audio_url = Column(String, nullable=False)
    avatar_image_url = Column(String, nullable=False)

    provider = Column(String, default="d-id")
    model = Column(String, default="talks")
    external_job_id = Column(String, nullable=True)

    cost = Column(Numeric(18, 6), nullable=False, default=0)
    status = Column(String, default=STATUS_PROCESSING)
    error_message = Column(Text, nullable=True)

    created_at = created_at_column()
    updated_at = updated_at_column()

    user = relationship("User", back_populates="videos")

    __table_args__ = (Index("ix_generated_videos_user_id_created_at", "user_id", "created_at"),)
