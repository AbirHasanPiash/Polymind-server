"""Image generation across providers.

One service prices, validates, generates and stores images for every image
model the platform offers. The pricing table is the catalogue: what it lists is
what the API accepts and what the client's picker shows.

* OpenAI (``gpt-image-2``, ``gpt-image-1.5``): priced per image by quality and
  pixel size, generated with the Images API; editing from a reference image
  goes through the edits endpoint.
* Google (``gemini-3.1-flash-image`` "Nano Banana 2", ``gemini-3-pro-image``
  "Nano Banana Pro"): priced per image by output resolution; the shape is an
  aspect ratio rather than a pixel size.
"""

from __future__ import annotations

import base64
import binascii
import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

import httpx
from google import genai
from google.genai import types
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.llm.base import ProviderNotConfiguredError
from app.services.llm.models import PROVIDER_GOOGLE, PROVIDER_OPENAI, usd_to_credits
from app.services.storage import storage

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
MAX_REFERENCE_IMAGE_BYTES = 20 * 1024 * 1024


class ImageGenerationError(RuntimeError):
    """Generation failed, or the provider returned an unusable response."""


class UnsupportedImageOptionError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class ImageModel:
    id: str
    provider: str
    display_name: str
    description: str
    # USD per image: quality -> size -> price
    prices: dict[str, dict[str, Decimal]]
    quality_labels: dict[str, str]
    size_labels: dict[str, str]
    default_quality: str
    default_size: str
    supports_reference: bool = True
    badge: str | None = None
    strengths: tuple[str, ...] = field(default=())

    def qualities(self) -> list[str]:
        return list(self.prices)

    def sizes(self) -> list[str]:
        seen: dict[str, None] = {}
        for sizes in self.prices.values():
            for size in sizes:
                seen.setdefault(size, None)
        return list(seen)


_OPENAI_SIZES = {
    "1024x1024": "Square · 1024×1024",
    "1024x1536": "Portrait · 1024×1536",
    "1536x1024": "Landscape · 1536×1024",
}
_OPENAI_QUALITIES = {"low": "Low", "medium": "Medium", "high": "High"}

# Google shapes: aspect ratios rendered at the chosen resolution.
_GOOGLE_SIZES = {
    "1:1": "Square · 1:1",
    "4:3": "Landscape · 4:3",
    "3:4": "Portrait · 3:4",
    "16:9": "Wide · 16:9",
    "9:16": "Tall · 9:16",
    "3:2": "Photo · 3:2",
    "2:3": "Photo · 2:3",
}


def _flat(price: str, sizes: dict[str, str]) -> dict[str, Decimal]:
    return dict.fromkeys(sizes, Decimal(price))


IMAGE_MODELS: dict[str, ImageModel] = {
    "gpt-image-2": ImageModel(
        id="gpt-image-2",
        provider=PROVIDER_OPENAI,
        display_name="GPT Image 2",
        description="OpenAI's newest image model: reasoning-guided composition and crisp text.",
        # Per-image cost derived from OpenAI's token pricing; non-square sizes
        # use more output tokens, priced at the same ratio as gpt-image-1.5.
        prices={
            "low": {
                "1024x1024": Decimal("0.006"),
                "1024x1536": Decimal("0.009"),
                "1536x1024": Decimal("0.009"),
            },
            "medium": {
                "1024x1024": Decimal("0.053"),
                "1024x1536": Decimal("0.079"),
                "1536x1024": Decimal("0.079"),
            },
            "high": {
                "1024x1024": Decimal("0.211"),
                "1024x1536": Decimal("0.316"),
                "1536x1024": Decimal("0.316"),
            },
        },
        quality_labels=_OPENAI_QUALITIES,
        size_labels=_OPENAI_SIZES,
        default_quality="medium",
        default_size="1024x1024",
        badge="new",
        strengths=("text rendering", "editing", "realism"),
    ),
    "gpt-image-1.5": ImageModel(
        id="gpt-image-1.5",
        provider=PROVIDER_OPENAI,
        display_name="GPT Image 1.5",
        description="Fast, dependable generation and reference-image editing.",
        prices={
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
        quality_labels=_OPENAI_QUALITIES,
        size_labels=_OPENAI_SIZES,
        default_quality="medium",
        default_size="1024x1024",
        strengths=("speed", "editing"),
    ),
    "gemini-3.1-flash-image": ImageModel(
        id="gemini-3.1-flash-image",
        provider=PROVIDER_GOOGLE,
        display_name="Nano Banana 2",
        description="Google's production image model: quick, consistent, great at edits.",
        prices={
            "1K": _flat("0.067", _GOOGLE_SIZES),
            "2K": _flat("0.101", _GOOGLE_SIZES),
            "4K": _flat("0.151", _GOOGLE_SIZES),
        },
        quality_labels={"1K": "1K", "2K": "2K", "4K": "4K"},
        size_labels=_GOOGLE_SIZES,
        default_quality="1K",
        default_size="1:1",
        badge="new",
        strengths=("consistency", "editing", "speed"),
    ),
    "gemini-3-pro-image": ImageModel(
        id="gemini-3-pro-image",
        provider=PROVIDER_GOOGLE,
        display_name="Nano Banana Pro",
        description="Google's flagship: 4K output, complex layouts, precise text.",
        prices={
            "1K": _flat("0.134", _GOOGLE_SIZES),
            "2K": _flat("0.134", _GOOGLE_SIZES),
            "4K": _flat("0.240", _GOOGLE_SIZES),
        },
        quality_labels={"1K": "1K", "2K": "2K", "4K": "4K"},
        size_labels=_GOOGLE_SIZES,
        default_quality="1K",
        default_size="1:1",
        strengths=("4K", "layouts", "text rendering"),
    ),
}

DEFAULT_IMAGE_MODEL = "gpt-image-2"


class ImageService:
    """Prices, generates and stores images for every supported model."""

    MODELS = IMAGE_MODELS
    # Kept for callers that only know the old name.
    REFERENCE_IMAGE_MODEL = "gpt-image-1.5"

    @classmethod
    def supported_models(cls) -> list[str]:
        return list(cls.MODELS)

    @classmethod
    def get_model(cls, model: str) -> ImageModel:
        try:
            return cls.MODELS[model]
        except KeyError:
            raise UnsupportedImageOptionError(
                f"Unsupported image model {model!r}. Supported: {', '.join(cls.MODELS)}"
            ) from None

    @classmethod
    def validate_options(cls, model: str, quality: str, size: str) -> None:
        """Reject combinations that have no price.

        An unpriced request used to fall back to an arbitrary rate, which meant
        billing one thing and generating another.
        """
        spec = cls.get_model(model)
        sizes = spec.prices.get(quality)
        if sizes is None:
            raise UnsupportedImageOptionError(
                f"Unsupported quality {quality!r} for {model}. Supported: {', '.join(spec.prices)}"
            )
        if size not in sizes:
            raise UnsupportedImageOptionError(
                f"Unsupported size {size!r} for {model}/{quality}. Supported: {', '.join(sizes)}"
            )

    def calculate_cost(self, model: str, quality: str, size: str) -> Decimal:
        self.validate_options(model, quality, size)
        return usd_to_credits(self.MODELS[model].prices[quality][size])

    @classmethod
    def options(cls) -> list[dict[str, object]]:
        """The catalogue for the client, with every price in credits.

        Exposing this keeps the picker in the UI from offering a combination the
        API would reject, and lets it quote the exact charge before generating.
        """
        features = settings.public_features()
        catalogue = []
        for spec in cls.MODELS.values():
            catalogue.append(
                {
                    "id": spec.id,
                    "provider": spec.provider,
                    "display_name": spec.display_name,
                    "description": spec.description,
                    "badge": spec.badge,
                    "strengths": list(spec.strengths),
                    "enabled": bool(features.get(spec.provider, False))
                    and settings.storage_enabled,
                    "supports_reference": spec.supports_reference,
                    "default_quality": spec.default_quality,
                    "default_size": spec.default_size,
                    "qualities": [
                        {"value": q, "label": spec.quality_labels.get(q, q)}
                        for q in spec.qualities()
                    ],
                    "sizes": [
                        {"value": s, "label": spec.size_labels.get(s, s)} for s in spec.sizes()
                    ],
                    "prices": {
                        quality: {size: str(usd_to_credits(price)) for size, price in sizes.items()}
                        for quality, sizes in spec.prices.items()
                    },
                }
            )
        return catalogue

    # ── generation ────────────────────────────────────────────────────

    @staticmethod
    def _openai_client() -> AsyncOpenAI:
        """A client per call, deliberately not cached.

        This service runs inside Celery tasks, and each task drives its own
        event loop; a cached async client would keep sockets bound to a loop
        that has already been closed. One generation is a single slow request,
        so there is nothing to gain from pooling across tasks.
        """
        if not settings.OPENAI_API_KEY:
            raise ProviderNotConfiguredError("OpenAI", "OPENAI_API_KEY")
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2, timeout=240.0)

    @staticmethod
    def _google_client() -> genai.Client:
        if not settings.GOOGLE_API_KEY:
            raise ProviderNotConfiguredError("Gemini", "GOOGLE_API_KEY")
        return genai.Client(api_key=settings.GOOGLE_API_KEY)

    @staticmethod
    async def _download(url: str, max_bytes: int) -> bytes:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            if len(response.content) > max_bytes:
                raise ImageGenerationError("Reference image is too large")
            return response.content

    async def _generate_openai(
        self, spec: ImageModel, prompt: str, size: str, quality: str, reference: bytes | None
    ) -> tuple[bytes, str | None]:
        async with self._openai_client() as client:
            if reference is not None:
                # Editing is a different endpoint from generation; the
                # reference image is uploaded as a file part.
                response = await client.images.edit(
                    model=spec.id,
                    image=("reference.png", reference, "image/png"),
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                )
            else:
                response = await client.images.generate(
                    model=spec.id, prompt=prompt, size=size, quality=quality, n=1
                )

        if not response.data:
            raise ImageGenerationError("Provider returned no image")
        image_data = response.data[0]
        b64 = getattr(image_data, "b64_json", None)
        if b64:
            try:
                file_bytes = base64.b64decode(b64)
            except binascii.Error as exc:
                raise ImageGenerationError("Provider returned malformed image data") from exc
        else:
            temp_url = getattr(image_data, "url", None)
            if not temp_url:
                raise ImageGenerationError("Provider returned neither image data nor a URL")
            file_bytes = await self._download(temp_url, MAX_REFERENCE_IMAGE_BYTES)
        return file_bytes, getattr(image_data, "revised_prompt", None)

    async def _generate_google(
        self, spec: ImageModel, prompt: str, size: str, quality: str, reference: bytes | None
    ) -> tuple[bytes, str | None]:
        client = self._google_client()
        parts: list[types.Part] = []
        if reference is not None:
            parts.append(types.Part.from_bytes(data=reference, mime_type="image/png"))
        parts.append(types.Part.from_text(text=prompt))

        response = await client.aio.models.generate_content(
            model=spec.id,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=size, image_size=quality),
            ),
        )
        for candidate in response.candidates or []:
            for part in (candidate.content.parts if candidate.content else []) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and inline.data:
                    data = inline.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    return bytes(data), None
        raise ImageGenerationError("Provider returned no image")

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
        spec = self.MODELS[model]

        if reference_image_url and not spec.supports_reference:
            raise UnsupportedImageOptionError(
                f"{spec.display_name} does not support reference images"
            )

        reference = (
            await self._download(reference_image_url, MAX_REFERENCE_IMAGE_BYTES)
            if reference_image_url
            else None
        )

        try:
            if spec.provider == PROVIDER_OPENAI:
                file_bytes, revised = await self._generate_openai(
                    spec, prompt, size, quality, reference
                )
            else:
                file_bytes, revised = await self._generate_google(
                    spec, prompt, size, quality, reference
                )
        except (UnsupportedImageOptionError, ImageGenerationError, ProviderNotConfiguredError):
            raise
        except Exception as exc:
            logger.error("Image generation failed with %s: %s", model, exc)
            raise ImageGenerationError(str(exc)) from exc

        storage_path = f"generated_images/{user_id}/{uuid.uuid4()}.png"
        public_url = await storage.upload_file_async(
            file_bytes=file_bytes, destination_path=storage_path, content_type="image/png"
        )
        return {
            "public_url": public_url,
            "storage_path": storage_path,
            "revised_prompt": revised or prompt,
        }


image_service = ImageService()
