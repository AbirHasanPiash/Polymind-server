"""OpenAI image generation and editing."""

from __future__ import annotations

import base64
import logging
import uuid
from decimal import Decimal

import httpx
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.llm.base import ProviderNotConfiguredError
from app.services.llm.models import CREDIT_PRECISION, PROFIT_MARGIN, USD_TO_CREDITS_RATE
from app.services.storage import storage

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
MAX_REFERENCE_IMAGE_BYTES = 20 * 1024 * 1024


class ImageGenerationError(RuntimeError):
    """Generation failed, or the provider returned an unusable response."""


class UnsupportedImageOptionError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)


class OpenAIImageService:
    """Prices, generates and stores images.

    Provider prices are USD per image; the same margin and credit rate as the
    text models are applied so one credit means the same thing everywhere.
    """

    PRICING_TABLE: dict[str, dict[str, dict[str, Decimal]]] = {
        "gpt-image-1.5": {
            "low": {
                "1024x1024": Decimal("0.009"),
                "1024x1536": Decimal("0.013"),
                "1536x1024": Decimal("0.013"),
            },
            "medium": {
                "1024x1024": Decimal("0.034"),
                "1024x1536": Decimal("0.050"),
                "1536x1024": Decimal("0.050"),
            },
            "high": {
                "1024x1024": Decimal("0.133"),
                "1024x1536": Decimal("0.200"),
                "1536x1024": Decimal("0.200"),
            },
        },
        # DALL·E 3 supports a different set of aspect ratios from gpt-image-*;
        # listing the gpt sizes here priced shapes the model cannot produce.
        "dall-e-3": {
            "standard": {
                "1024x1024": Decimal("0.040"),
                "1024x1792": Decimal("0.080"),
                "1792x1024": Decimal("0.080"),
            },
            "hd": {
                "1024x1024": Decimal("0.080"),
                "1024x1792": Decimal("0.120"),
                "1792x1024": Decimal("0.120"),
            },
        },
    }

    # Only this model supports editing from a reference image.
    REFERENCE_IMAGE_MODEL = "gpt-image-1.5"

    @staticmethod
    def _new_client() -> AsyncOpenAI:
        """A client per call, deliberately not cached.

        This service runs inside Celery tasks, and each task drives its own
        event loop; a cached async client would keep sockets bound to a loop
        that has already been closed. One generation is a single slow request,
        so there is nothing to gain from pooling across tasks.
        """
        if not settings.OPENAI_API_KEY:
            raise ProviderNotConfiguredError("OpenAI", "OPENAI_API_KEY")
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2, timeout=180.0)

    @classmethod
    def supported_models(cls) -> list[str]:
        return list(cls.PRICING_TABLE)

    @classmethod
    def options(cls) -> dict[str, dict[str, list[str]]]:
        """Valid quality/size combinations per model, for clients to render.

        Exposing this keeps the picker in the UI from offering a combination the
        API would reject.
        """
        return {
            model: {
                "qualities": list(qualities),
                "sizes": sorted({size for sizes in qualities.values() for size in sizes}),
            }
            for model, qualities in cls.PRICING_TABLE.items()
        }

    @classmethod
    def validate_options(cls, model: str, quality: str, size: str) -> None:
        """Reject combinations that have no price.

        An unpriced request used to fall back to an arbitrary rate, which meant
        billing one thing and generating another.
        """
        qualities = cls.PRICING_TABLE.get(model)
        if qualities is None:
            raise UnsupportedImageOptionError(
                f"Unsupported image model {model!r}. Supported: {', '.join(cls.supported_models())}"
            )
        sizes = qualities.get(quality)
        if sizes is None:
            raise UnsupportedImageOptionError(
                f"Unsupported quality {quality!r} for {model}. Supported: {', '.join(qualities)}"
            )
        if size not in sizes:
            raise UnsupportedImageOptionError(
                f"Unsupported size {size!r} for {model}/{quality}. Supported: {', '.join(sizes)}"
            )

    def calculate_cost(self, model: str, quality: str, size: str) -> Decimal:
        self.validate_options(model, quality, size)
        provider_cost = self.PRICING_TABLE[model][quality][size]
        return (provider_cost * PROFIT_MARGIN * USD_TO_CREDITS_RATE).quantize(CREDIT_PRECISION)

    @staticmethod
    async def _download(url: str, max_bytes: int) -> bytes:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            if len(response.content) > max_bytes:
                raise ImageGenerationError("Reference image is too large")
            return response.content

    @staticmethod
    def _extract_bytes(image_data) -> bytes:
        """Return raw image bytes from either response shape.

        gpt-image-* returns base64 inline; dall-e-3 returns a short-lived URL.
        """
        b64 = getattr(image_data, "b64_json", None)
        if b64:
            return base64.b64decode(b64)
        return b""

    async def generate_and_upload(
        self,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        user_id: str,
        reference_image_url: str | None = None,
    ) -> dict[str, str]:
        """Generate (or edit) an image, store it, and return its URL and prompt."""
        self.validate_options(model, quality, size)

        if reference_image_url and model != self.REFERENCE_IMAGE_MODEL:
            raise UnsupportedImageOptionError(
                f"Reference images are only supported by {self.REFERENCE_IMAGE_MODEL}"
            )

        try:
            async with self._new_client() as client:
                if reference_image_url:
                    reference_bytes = await self._download(
                        reference_image_url, MAX_REFERENCE_IMAGE_BYTES
                    )
                    # Editing is a different endpoint from generation; the
                    # reference image is uploaded as a file part, and passing it
                    # to generate() raised TypeError before this was corrected.
                    response = await client.images.edit(
                        model=model,
                        image=("reference.png", reference_bytes, "image/png"),
                        prompt=prompt,
                        size=size,
                        quality=quality,
                        n=1,
                    )
                else:
                    response = await client.images.generate(
                        model=model,
                        prompt=prompt,
                        size=size,
                        quality=quality,
                        n=1,
                    )
        except (UnsupportedImageOptionError, ImageGenerationError):
            raise
        except Exception as exc:
            logger.error("Image generation failed: %s", exc)
            raise ImageGenerationError(str(exc)) from exc

        if not response.data:
            raise ImageGenerationError("Provider returned no image")

        image_data = response.data[0]
        file_bytes = self._extract_bytes(image_data)
        if not file_bytes:
            temp_url = getattr(image_data, "url", None)
            if not temp_url:
                raise ImageGenerationError("Provider returned neither image data nor a URL")
            file_bytes = await self._download(temp_url, MAX_REFERENCE_IMAGE_BYTES)

        storage_path = f"generated_images/{user_id}/{uuid.uuid4()}.png"
        public_url = await storage.upload_file_async(
            file_bytes=file_bytes,
            destination_path=storage_path,
            content_type="image/png",
        )

        return {
            "public_url": public_url,
            "storage_path": storage_path,
            "revised_prompt": getattr(image_data, "revised_prompt", None) or prompt,
        }


image_service = OpenAIImageService()
