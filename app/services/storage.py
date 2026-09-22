"""Object storage (Cloudflare R2 / any S3-compatible endpoint).

boto3 is synchronous, so every public method has an ``async`` variant that runs
the blocking call in a worker thread. Calling the sync methods directly from a
coroutine would stall the event loop for the entire round-trip.
"""

from __future__ import annotations

import asyncio
import logging
from functools import cached_property

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageNotConfiguredError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "Object storage is not configured. Set the STORAGE_* environment variables."
        )


class StorageError(RuntimeError):
    """Upload or delete failed after retries."""


class StorageService:
    def __init__(self) -> None:
        self.bucket = settings.STORAGE_BUCKET_NAME
        self.public_base_url = (settings.STORAGE_PUBLIC_URL or "").rstrip("/")

    @cached_property
    def client(self):
        """Built on first use so a deployment without storage can still boot."""
        if not settings.storage_enabled:
            raise StorageNotConfiguredError
        return boto3.client(
            "s3",
            endpoint_url=settings.STORAGE_ENDPOINT,
            aws_access_key_id=settings.STORAGE_ACCESS_KEY,
            aws_secret_access_key=settings.STORAGE_SECRET_KEY,
            region_name=settings.STORAGE_REGION,
            config=Config(
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
            ),
        )

    def public_url(self, key: str) -> str:
        return f"{self.public_base_url}/{key.lstrip('/')}"

    def key_from_url(self, url: str) -> str | None:
        """Recover the object key from a public URL, or None if it is foreign.

        Returning None keeps a stray URL from turning into a delete against an
        unrelated key.
        """
        if not url or not self.public_base_url:
            return None
        prefix = f"{self.public_base_url}/"
        if not url.startswith(prefix):
            return None
        key = url[len(prefix) :]
        return key or None

    def upload_file(self, file_bytes: bytes, destination_path: str, content_type: str) -> str:
        """Upload bytes and return the public URL. Blocking."""
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=destination_path,
                Body=file_bytes,
                ContentType=content_type,
            )
        except (ClientError, BotoCoreError) as exc:
            logger.error("Storage upload failed for %s: %s", destination_path, exc)
            raise StorageError(f"Failed to upload {destination_path}") from exc
        return self.public_url(destination_path)

    def delete_file(self, public_url: str) -> bool:
        """Delete the object behind a public URL. Blocking. Never raises."""
        key = self.key_from_url(public_url)
        if key is None:
            logger.warning("Skipping delete for URL outside the configured bucket: %s", public_url)
            return False
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except (ClientError, BotoCoreError) as exc:
            # Orphaned objects are cheaper than a failed user-facing delete.
            logger.error("Storage delete failed for %s: %s", key, exc)
            return False
        logger.info("Deleted object %s", key)
        return True

    async def upload_file_async(
        self, file_bytes: bytes, destination_path: str, content_type: str
    ) -> str:
        return await asyncio.to_thread(self.upload_file, file_bytes, destination_path, content_type)

    async def delete_file_async(self, public_url: str) -> bool:
        return await asyncio.to_thread(self.delete_file, public_url)


storage = StorageService()
