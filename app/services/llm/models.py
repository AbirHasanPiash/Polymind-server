"""Registry of every model the platform is allowed to call.

This is the single source of truth for model ids, provider routing and pricing.
Keeping them together prevents the class of bug where a request is served by one
model but billed at another model's rate, and it lets the API reject unknown
model ids up front instead of forwarding them to a provider.

Prices are USD per 1M tokens, as published by each provider. Update them here
and every adapter, quote and invoice follows.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.services.llm.usage import Usage

# Markup applied to the provider's cost, then converted into wallet credits.
PROFIT_MARGIN = Decimal("4.0")
USD_TO_CREDITS_RATE = Decimal("10.0")

CREDIT_PRECISION = Decimal("0.000001")  # matches Numeric(18, 6) in the database

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GOOGLE = "google"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str  # public id used by clients
    provider: str
    api_model: str  # id understood by the provider's API
    input_price: Decimal  # USD / 1M input tokens
    output_price: Decimal  # USD / 1M output tokens
    description: str = ""


_SPECS: tuple[ModelSpec, ...] = (
    # OpenAI
    ModelSpec(
        id="gpt-5.2-pro",
        provider=PROVIDER_OPENAI,
        api_model="gpt-5.2-pro",
        input_price=Decimal("21.00"),
        output_price=Decimal("168.00"),
        description="Deep reasoning and hard analysis",
    ),
    ModelSpec(
        id="gpt-5.2",
        provider=PROVIDER_OPENAI,
        api_model="gpt-5.2",
        input_price=Decimal("1.75"),
        output_price=Decimal("14.00"),
        description="Balanced general-purpose default",
    ),
    ModelSpec(
        id="gpt-5-mini",
        provider=PROVIDER_OPENAI,
        api_model="gpt-5-mini",
        input_price=Decimal("0.25"),
        output_price=Decimal("2.00"),
        description="Cheapest OpenAI tier",
    ),
    # Google
    ModelSpec(
        id="gemini-3-pro-preview",
        provider=PROVIDER_GOOGLE,
        api_model="gemini-3-pro-preview",
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        description="Long context, multimodal",
    ),
    ModelSpec(
        id="gemini-2.5-pro",
        provider=PROVIDER_GOOGLE,
        api_model="gemini-2.5-pro",
        input_price=Decimal("1.88"),
        output_price=Decimal("12.50"),
        description="Large context window",
    ),
    ModelSpec(
        id="gemini-3-flash-preview",
        provider=PROVIDER_GOOGLE,
        api_model="gemini-3-flash-preview",
        input_price=Decimal("0.50"),
        output_price=Decimal("3.00"),
        description="Fastest responses",
    ),
    ModelSpec(
        id="gemini-2.5-flash",
        provider=PROVIDER_GOOGLE,
        api_model="gemini-2.5-flash",
        input_price=Decimal("0.30"),
        output_price=Decimal("2.50"),
        description="Fast and inexpensive",
    ),
    # Anthropic
    ModelSpec(
        id="claude-4.5-opus",
        provider=PROVIDER_ANTHROPIC,
        api_model="claude-opus-4-5-20251101",
        input_price=Decimal("5.00"),
        output_price=Decimal("25.00"),
        description="Strongest coding and agentic model",
    ),
    ModelSpec(
        id="claude-4.5-sonnet",
        provider=PROVIDER_ANTHROPIC,
        api_model="claude-sonnet-4-5-20250929",
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        description="Balanced coding model",
    ),
    ModelSpec(
        id="claude-4.5-haiku",
        provider=PROVIDER_ANTHROPIC,
        api_model="claude-haiku-4-5-20251001",
        input_price=Decimal("1.00"),
        output_price=Decimal("5.00"),
        description="Fast, low-cost coding model",
    ),
)

MODEL_REGISTRY: dict[str, ModelSpec] = {spec.id: spec for spec in _SPECS}

# Defaults used when the router has no strong signal.
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_FAST_MODEL = "gemini-3-flash-preview"
DEFAULT_CODING_MODEL = "claude-4.5-opus"
DEFAULT_REASONING_MODEL = "gpt-5.2-pro"
DEFAULT_LONG_CONTEXT_MODEL = "gemini-2.5-pro"


class UnknownModelError(ValueError):
    """Raised when a client asks for a model that is not in the registry."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Unsupported model: {model!r}")
        self.model = model


def get_spec(model: str) -> ModelSpec:
    """Look up a model, rejecting anything not explicitly allowed."""
    try:
        return MODEL_REGISTRY[model]
    except KeyError:
        raise UnknownModelError(model) from None


def is_supported(model: str) -> bool:
    return model in MODEL_REGISTRY


def list_models() -> list[ModelSpec]:
    return list(_SPECS)


def price_in_credits(usage: Usage, model: str) -> Decimal:
    """Convert token usage into wallet credits: (cost x margin) x credit rate."""
    spec = get_spec(model)
    million = Decimal("1000000")

    provider_cost = (Decimal(usage.prompt_tokens) / million) * spec.input_price + (
        Decimal(usage.completion_tokens) / million
    ) * spec.output_price

    return (provider_cost * PROFIT_MARGIN * USD_TO_CREDITS_RATE).quantize(CREDIT_PRECISION)
