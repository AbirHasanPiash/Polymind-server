"""Registry of every model the platform is allowed to call.

This is the single source of truth for model ids, provider routing, pricing and
the metadata the client renders (display name, tier, context window, badges).
Keeping them together prevents the class of bug where a request is served by one
model but billed at another model's rate, lets the API reject unknown ids before
they reach a provider, and means adding a model is one entry here — the API,
the router, the biller and the model picker all follow.

Prices are USD per 1M tokens as published by each provider (verified 2026-09).
Update them here and every quote and invoice follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from app.services.llm.usage import Usage

# Markup applied to the provider's cost, then converted into wallet credits.
PROFIT_MARGIN = Decimal("4.0")
USD_TO_CREDITS_RATE = Decimal("10.0")

CREDIT_PRECISION = Decimal("0.000001")  # matches Numeric(18, 6) in the database

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GOOGLE = "google"

PROVIDER_LABELS: dict[str, str] = {
    PROVIDER_OPENAI: "OpenAI",
    PROVIDER_ANTHROPIC: "Anthropic",
    PROVIDER_GOOGLE: "Google",
}

Tier = Literal["flagship", "balanced", "fast"]
Status = Literal["stable", "preview"]
Effort = Literal["low", "medium", "high"]

EFFORT_LEVELS: tuple[Effort, ...] = ("low", "medium", "high")
DEFAULT_EFFORT: Effort = "medium"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str  # public id used by clients
    provider: str
    api_model: str  # id understood by the provider's API
    display_name: str
    input_price: Decimal  # USD / 1M input tokens
    output_price: Decimal  # USD / 1M output tokens
    context_window: int
    max_output_tokens: int
    tier: Tier
    description: str
    strengths: tuple[str, ...] = ()
    # Whether the model can take a thinking / reasoning-effort setting. Models
    # without it ignore the requested effort level.
    reasoning: bool = True
    supports_vision: bool = True
    status: Status = "stable"
    released: str = ""  # YYYY-MM, for the "new" badge and sorting
    badge: str | None = None
    # Hidden models stay callable (old conversations, admin tests) but are not
    # offered in the picker.
    hidden: bool = False
    aliases: tuple[str, ...] = field(default=())

    @property
    def input_credits_per_million(self) -> Decimal:
        return (self.input_price * PROFIT_MARGIN * USD_TO_CREDITS_RATE).quantize(Decimal("0.01"))

    @property
    def output_credits_per_million(self) -> Decimal:
        return (self.output_price * PROFIT_MARGIN * USD_TO_CREDITS_RATE).quantize(Decimal("0.01"))

    def to_public(self) -> dict[str, object]:
        """Shape served by ``GET /models`` — everything the picker needs."""
        return {
            "id": self.id,
            "provider": self.provider,
            "provider_label": PROVIDER_LABELS.get(self.provider, self.provider),
            "display_name": self.display_name,
            "description": self.description,
            "tier": self.tier,
            "strengths": list(self.strengths),
            "reasoning": self.reasoning,
            "supports_vision": self.supports_vision,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "status": self.status,
            "released": self.released,
            "badge": self.badge,
            "pricing": {
                # Credits per 1M tokens, i.e. what the user actually pays.
                "input_credits_per_million": str(self.input_credits_per_million),
                "output_credits_per_million": str(self.output_credits_per_million),
            },
        }


_SPECS: tuple[ModelSpec, ...] = (
    # ── OpenAI ────────────────────────────────────────────────────────────
    ModelSpec(
        id="gpt-5.5-pro",
        provider=PROVIDER_OPENAI,
        api_model="gpt-5.5-pro",
        display_name="GPT-5.5 Pro",
        input_price=Decimal("30.00"),
        output_price=Decimal("180.00"),
        context_window=1_050_000,
        max_output_tokens=128_000,
        tier="flagship",
        description="Maximum reasoning depth for the hardest analysis and research.",
        strengths=("reasoning", "research", "math"),
        released="2026-04",
    ),
    ModelSpec(
        id="gpt-5.5",
        provider=PROVIDER_OPENAI,
        api_model="gpt-5.5",
        display_name="GPT-5.5",
        input_price=Decimal("5.00"),
        output_price=Decimal("30.00"),
        context_window=1_050_000,
        max_output_tokens=128_000,
        tier="flagship",
        description="OpenAI's flagship: strong reasoning, coding and agentic work.",
        strengths=("reasoning", "coding", "writing"),
        released="2026-04",
        badge="new",
    ),
    ModelSpec(
        id="gpt-5.4",
        provider=PROVIDER_OPENAI,
        api_model="gpt-5.4",
        display_name="GPT-5.4",
        input_price=Decimal("2.50"),
        output_price=Decimal("15.00"),
        context_window=1_050_000,
        max_output_tokens=128_000,
        tier="balanced",
        description="Balanced everyday model with a very large context window.",
        strengths=("general", "coding", "long-context"),
        released="2026-03",
    ),
    ModelSpec(
        id="gpt-5.4-mini",
        provider=PROVIDER_OPENAI,
        api_model="gpt-5.4-mini",
        display_name="GPT-5.4 mini",
        input_price=Decimal("0.75"),
        output_price=Decimal("4.50"),
        context_window=400_000,
        max_output_tokens=128_000,
        tier="fast",
        description="Quick and inexpensive for everyday questions and drafts.",
        strengths=("speed", "general"),
        released="2026-03",
    ),
    ModelSpec(
        id="gpt-5.4-nano",
        provider=PROVIDER_OPENAI,
        api_model="gpt-5.4-nano",
        display_name="GPT-5.4 nano",
        input_price=Decimal("0.20"),
        output_price=Decimal("1.25"),
        context_window=400_000,
        max_output_tokens=128_000,
        tier="fast",
        description="The cheapest OpenAI tier: classification, extraction, quick replies.",
        strengths=("speed", "cost"),
        released="2026-03",
    ),
    # ── Anthropic ─────────────────────────────────────────────────────────
    ModelSpec(
        id="claude-fable-5-1",
        provider=PROVIDER_ANTHROPIC,
        api_model="claude-fable-5-1",
        display_name="Claude Fable 5.1",
        input_price=Decimal("10.00"),
        output_price=Decimal("50.00"),
        context_window=1_000_000,
        max_output_tokens=128_000,
        tier="flagship",
        description="Anthropic's most capable model for demanding reasoning and long tasks.",
        strengths=("reasoning", "coding", "research", "long-context"),
        released="2026-08",
        badge="new",
    ),
    ModelSpec(
        id="claude-opus-5",
        provider=PROVIDER_ANTHROPIC,
        api_model="claude-opus-5",
        display_name="Claude Opus 5",
        input_price=Decimal("5.00"),
        output_price=Decimal("25.00"),
        context_window=1_000_000,
        max_output_tokens=128_000,
        tier="flagship",
        description="Deep, careful work: complex coding, analysis and writing.",
        strengths=("coding", "reasoning", "writing"),
        released="2026-07",
    ),
    ModelSpec(
        id="claude-sonnet-5",
        provider=PROVIDER_ANTHROPIC,
        api_model="claude-sonnet-5",
        display_name="Claude Sonnet 5",
        input_price=Decimal("2.00"),
        output_price=Decimal("10.00"),
        context_window=1_000_000,
        max_output_tokens=128_000,
        tier="balanced",
        description="The best mix of speed and intelligence — a great default.",
        strengths=("general", "coding", "writing"),
        released="2026-06",
    ),
    ModelSpec(
        id="claude-haiku-4-5",
        provider=PROVIDER_ANTHROPIC,
        api_model="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        input_price=Decimal("1.00"),
        output_price=Decimal("5.00"),
        context_window=200_000,
        max_output_tokens=64_000,
        tier="fast",
        description="Fast and affordable for simple tasks and quick replies.",
        strengths=("speed", "cost"),
        reasoning=False,
        released="2025-10",
    ),
    # ── Google ────────────────────────────────────────────────────────────
    ModelSpec(
        id="gemini-3.1-pro-preview",
        provider=PROVIDER_GOOGLE,
        api_model="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro",
        input_price=Decimal("2.00"),
        output_price=Decimal("12.00"),
        context_window=1_048_576,
        max_output_tokens=65_536,
        tier="flagship",
        description="Google's strongest reasoning model with a huge multimodal context.",
        strengths=("reasoning", "long-context", "multimodal"),
        status="preview",
        released="2026-02",
        badge="preview",
    ),
    ModelSpec(
        id="gemini-3.8-flash",
        provider=PROVIDER_GOOGLE,
        api_model="gemini-3.8-flash",
        display_name="Gemini 3.8 Flash",
        input_price=Decimal("0.75"),  # promotional rate through 2026-12-31
        output_price=Decimal("3.75"),
        context_window=1_048_576,
        max_output_tokens=65_536,
        tier="balanced",
        description="Google's newest Flash: frontier quality at high speed.",
        strengths=("speed", "general", "multimodal"),
        released="2026-08",
        badge="new",
    ),
    ModelSpec(
        id="gemini-3.5-flash-lite",
        provider=PROVIDER_GOOGLE,
        api_model="gemini-3.5-flash-lite",
        display_name="Gemini 3.5 Flash-Lite",
        input_price=Decimal("0.30"),
        output_price=Decimal("2.50"),
        context_window=1_048_576,
        max_output_tokens=65_536,
        tier="fast",
        description="The fastest, most economical Gemini for high-volume work.",
        strengths=("speed", "cost"),
        released="2026-06",
    ),
)

MODEL_REGISTRY: dict[str, ModelSpec] = {spec.id: spec for spec in _SPECS}

# Public ids from earlier releases, so a conversation started on a retired model
# keeps working and a stale client preference does not turn into an error.
LEGACY_ALIASES: dict[str, str] = {
    "gpt-5.2-pro": "gpt-5.5-pro",
    "gpt-5.2": "gpt-5.5",
    "gpt-5-mini": "gpt-5.4-mini",
    "claude-4.5-opus": "claude-opus-5",
    "claude-4.5-sonnet": "claude-sonnet-5",
    "claude-4.5-haiku": "claude-haiku-4-5",
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3-flash-preview": "gemini-3.8-flash",
    "gemini-2.5-pro": "gemini-3.1-pro-preview",
    "gemini-2.5-flash": "gemini-3.8-flash",
}

# Defaults used by the router when it has no strong signal.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_FAST_MODEL = "gemini-3.8-flash"
DEFAULT_CODING_MODEL = "claude-opus-5"
DEFAULT_REASONING_MODEL = "gpt-5.5"
DEFAULT_LONG_CONTEXT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_CREATIVE_MODEL = "claude-sonnet-5"
# Cheap model used for background work the user is not charged for (chat titles).
UTILITY_MODEL = "gemini-3.5-flash-lite"

ROUTING_DEFAULTS: dict[str, str] = {
    "coding": DEFAULT_CODING_MODEL,
    "reasoning": DEFAULT_REASONING_MODEL,
    "long_context": DEFAULT_LONG_CONTEXT_MODEL,
    "creative": DEFAULT_CREATIVE_MODEL,
    "fast": DEFAULT_FAST_MODEL,
    "default": DEFAULT_MODEL,
}


class UnknownModelError(ValueError):
    """Raised when a client asks for a model that is not in the registry."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Unsupported model: {model!r}")
        self.model = model


def resolve_model_id(model: str) -> str:
    """Map legacy public ids onto their current replacement."""
    return LEGACY_ALIASES.get(model, model)


def get_spec(model: str) -> ModelSpec:
    """Look up a model, rejecting anything not explicitly allowed."""
    try:
        return MODEL_REGISTRY[resolve_model_id(model)]
    except KeyError:
        raise UnknownModelError(model) from None


def is_supported(model: str) -> bool:
    return resolve_model_id(model) in MODEL_REGISTRY


def list_models(include_hidden: bool = False) -> list[ModelSpec]:
    return [spec for spec in _SPECS if include_hidden or not spec.hidden]


def price_in_credits(usage: Usage, model: str) -> Decimal:
    """Convert token usage into wallet credits: (cost x margin) x credit rate."""
    spec = get_spec(model)
    million = Decimal("1000000")

    provider_cost = (Decimal(usage.prompt_tokens) / million) * spec.input_price + (
        Decimal(usage.completion_tokens) / million
    ) * spec.output_price

    return (provider_cost * PROFIT_MARGIN * USD_TO_CREDITS_RATE).quantize(CREDIT_PRECISION)


def usd_to_credits(usd: Decimal) -> Decimal:
    """Convert a provider's USD cost into wallet credits with the standard markup."""
    return (usd * PROFIT_MARGIN * USD_TO_CREDITS_RATE).quantize(CREDIT_PRECISION)
