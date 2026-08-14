"""Celery application and shared task plumbing.

Tasks are synchronous functions that drive async service code, so each one runs
its own event loop and disposes of the database engine afterwards — a pooled
connection must never outlive the loop that created it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

from celery import Celery

from app.core.config import settings
from app.core.database import engine
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

configure_logging()

celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    # Listed explicitly so `-A app.workers.celery_app` registers the tasks too.
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Redeliver a task if the worker dies mid-flight instead of losing it.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # One task at a time per process: these tasks are long and IO-heavy.
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    # Ceilings so a stuck provider call cannot hold a worker slot forever.
    task_soft_time_limit=600,
    task_time_limit=660,
    result_expires=86400,
    broker_connection_retry_on_startup=True,
)

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from a synchronous Celery task."""

    async def _runner() -> T:
        try:
            return await coro
        finally:
            # Connections are bound to this loop; drop them before it closes.
            await engine.dispose()

    return asyncio.run(_runner())
