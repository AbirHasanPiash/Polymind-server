"""Shared Redis connection pool and the chat cache built on top of it."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

import redis.asyncio as redis

from app.core.config import settings
from app.services.llm.schema import ChatMessage

logger = logging.getLogger(__name__)

# One pool per process: creating a client per request opens (and TLS-handshakes)
# a fresh socket every time, which dominates latency on small requests.
_pool = redis.ConnectionPool.from_url(
    settings.REDIS_URL,
    encoding="utf-8",
    decode_responses=True,
    max_connections=settings.REDIS_MAX_CONNECTIONS,
    health_check_interval=30,
)

_client = redis.Redis(connection_pool=_pool)


def get_redis_client() -> redis.Redis:
    """Return the process-wide Redis client."""
    return _client


async def get_redis() -> AsyncGenerator[redis.Redis, None]:
    """FastAPI dependency. Connections return to the pool, they are not closed."""
    yield _client


async def check_redis_connection() -> bool:
    try:
        await _client.ping()
        return True
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return False


async def close_redis() -> None:
    """Release pooled connections on shutdown."""
    await _client.aclose()
    await _pool.aclose()


class ChatCache:
    """Short-lived chat history and upload staging area.

    Every key carries a TTL, so an abandoned conversation cannot pin memory in
    Redis forever. PostgreSQL remains the source of truth for history.
    """

    def __init__(self, redis_client: redis.Redis, ttl: int | None = None) -> None:
        self.redis = redis_client
        self.ttl = ttl or settings.CACHE_TTL_SECONDS

    @staticmethod
    def _history_key(chat_id: str) -> str:
        return f"chat:{chat_id}:history"

    @staticmethod
    def _file_key(file_id: str) -> str:
        return f"temp_file:{file_id}"

    async def add_message(self, chat_id: str, role: str, content: str) -> None:
        key = self._history_key(chat_id)
        payload = json.dumps({"role": role, "content": content})
        # Pipeline: one round-trip instead of two.
        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.rpush(key, payload)
            pipe.ltrim(key, -settings.CHAT_HISTORY_LIMIT * 2, -1)
            pipe.expire(key, self.ttl)
            await pipe.execute()

    async def get_history(self, chat_id: str, limit: int | None = None) -> list[ChatMessage]:
        limit = limit or settings.CHAT_HISTORY_LIMIT
        raw_messages = await self.redis.lrange(self._history_key(chat_id), -limit, -1)

        history: list[ChatMessage] = []
        for raw in raw_messages:
            try:
                data = json.loads(raw)
                history.append(ChatMessage.from_text(role=data["role"], text=data["content"]))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Discarding malformed cached message for chat %s: %s", chat_id, exc)
        return history

    async def clear_history(self, chat_id: str) -> None:
        await self.redis.delete(self._history_key(chat_id))

    async def save_temp_file(self, file_id: str, file_data: dict) -> None:
        """Stage extracted file content (text or base64) until it is sent in a message."""
        await self.redis.setex(self._file_key(file_id), self.ttl, json.dumps(file_data))

    async def get_temp_file(self, file_id: str) -> dict | None:
        data = await self.redis.get(self._file_key(file_id))
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Discarding malformed staged file %s", file_id)
            return None
