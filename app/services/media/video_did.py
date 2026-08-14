"""D-ID talking-avatar video generation."""

from __future__ import annotations

import logging
from decimal import Decimal
from functools import cached_property

import httpx

from app.core.config import settings
from app.services.llm.models import CREDIT_PRECISION, PROFIT_MARGIN, USD_TO_CREDITS_RATE

logger = logging.getLogger(__name__)

BASE_URL = "https://api.d-id.com"
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# D-ID bills in 15-second blocks.
SECONDS_PER_BLOCK = 15
# Roughly the speaking rate of the default TTS voices, used to estimate duration.
CHARS_PER_SECOND = 15


class DIDNotConfiguredError(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)


class DIDError(RuntimeError):
    """The D-ID API rejected a request or failed a job."""


class DIDService:
    PRICE_PER_BLOCK = Decimal("0.10")

    @cached_property
    def _auth(self) -> httpx.BasicAuth:
        """Parsed on first use; a missing key must not prevent the app from booting."""
        raw_key = settings.DID_API_KEY
        if not raw_key:
            raise DIDNotConfiguredError("Avatar video is disabled. Set DID_API_KEY to enable it.")
        if ":" not in raw_key:
            raise DIDNotConfiguredError("DID_API_KEY must be in 'username:password' form.")
        username, password = raw_key.split(":", 1)
        return httpx.BasicAuth(username=username, password=password)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            auth=self._auth,
            base_url=BASE_URL,
            timeout=REQUEST_TIMEOUT,
            headers={"accept": "application/json", "content-type": "application/json"},
        )

    @staticmethod
    def estimate_duration_seconds(script_text: str) -> int:
        """Approximate spoken length so long scripts are not charged as short ones."""
        return max(SECONDS_PER_BLOCK, -(-len(script_text) // CHARS_PER_SECOND))

    def calculate_cost(self, duration_seconds: int = SECONDS_PER_BLOCK) -> Decimal:
        blocks = max(1, -(-duration_seconds // SECONDS_PER_BLOCK))  # ceiling division
        provider_cost = Decimal(blocks) * self.PRICE_PER_BLOCK
        return (provider_cost * PROFIT_MARGIN * USD_TO_CREDITS_RATE).quantize(CREDIT_PRECISION)

    def calculate_cost_for_script(self, script_text: str) -> Decimal:
        return self.calculate_cost(self.estimate_duration_seconds(script_text))

    async def create_talk(self, source_url: str, audio_url: str) -> str:
        """Submit a job and return its id."""
        payload = {
            "source_url": source_url,
            "script": {"type": "audio", "audio_url": audio_url},
            "config": {"fluent": True, "pad_audio": "0.0", "stitch": True},
        }

        async with self._client() as client:
            response = await client.post("/talks", json=payload)

        if response.status_code not in (200, 201):
            logger.error("D-ID create failed (%s): %s", response.status_code, response.text)
            raise DIDError(f"D-ID rejected the request ({response.status_code})")

        return response.json()["id"]

    async def check_status(self, talk_id: str) -> str | None:
        """Return the result URL when finished, None while still running."""
        async with self._client() as client:
            response = await client.get(f"/talks/{talk_id}")

        if response.status_code != 200:
            logger.error("D-ID status failed (%s): %s", response.status_code, response.text)
            raise DIDError(f"D-ID status check failed ({response.status_code})")

        data = response.json()
        status = data.get("status")

        if status == "done":
            result_url = data.get("result_url")
            if not result_url:
                raise DIDError("D-ID reported 'done' without a result_url")
            return result_url

        if status == "error":
            raise DIDError(f"D-ID job failed: {data.get('error', {})}")

        return None


did_service = DIDService()
