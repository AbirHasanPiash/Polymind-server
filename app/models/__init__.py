"""ORM models.

Importing this package registers every model on ``Base.metadata``, which is what
Alembic autogenerate and the relationship resolver both rely on.
"""

from app.models.base import Base, utc_now
from app.models.chat import Chat, Message
from app.models.media import GeneratedAudio, GeneratedImage, GeneratedVideo
from app.models.package import Package
from app.models.transaction import Transaction
from app.models.user import User, Wallet

__all__ = [
    "Base",
    "Chat",
    "GeneratedAudio",
    "GeneratedImage",
    "GeneratedVideo",
    "Message",
    "Package",
    "Transaction",
    "User",
    "Wallet",
    "utc_now",
]
