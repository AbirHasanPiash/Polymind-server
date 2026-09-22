"""Intent-based model selection for "auto" mode.

The prompt is scored against keyword families; the highest-scoring family picks
the model best suited to it. An explicit user preference always wins, but it is
validated against the model registry first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.llm.models import (
    DEFAULT_CODING_MODEL,
    DEFAULT_CREATIVE_MODEL,
    DEFAULT_FAST_MODEL,
    DEFAULT_LONG_CONTEXT_MODEL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_MODEL,
    get_spec,
)

AUTO = "auto"

# Prompts longer than this are routed to the large-context model regardless of topic.
LONG_PROMPT_CHARS = 4000
# Prompts shorter than this go to the fastest model when no intent is detected.
SHORT_PROMPT_CHARS = 150

INTENT_LABELS: dict[str, str] = {
    "coding": "Code detected — routed to the strongest coding model",
    "long_context": "Large input — routed to the long-context model",
    "data": "Data analysis — routed to the long-context model",
    "reasoning": "Analytical question — routed to the reasoning model",
    "creative": "Writing task — routed to the creative model",
    "fast": "Quick question — routed to the fastest model",
    "default": "General request — routed to the balanced default",
    "pinned": "Model chosen by you",
}


@dataclass(frozen=True, slots=True)
class Route:
    model: str
    intent: str

    @property
    def reason(self) -> str:
        return INTENT_LABELS.get(self.intent, INTENT_LABELS["default"])


def _compile(patterns: list[str]) -> re.Pattern[str]:
    """Merge keyword groups into one case-insensitive pattern.

    One pass over the text per family instead of one pass per keyword group —
    this runs on every chat message, so the difference is worth the merge.
    """
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


class ModelRouter:
    CODING_PATTERNS = [
        # Core syntax & languages
        r"\b(code|codes|def|class|import|function|const|let|var|return|await|async)\b",
        r"\b(struct|impl|interface|package|namespace|void|public|private)\b",
        r"\b(python|javascript|typescript|golang|rust|java|c\+\+|swift|kotlin)\b",
        r"\b(bash|shell|powershell|zsh|chmod|sudo|grep|sed|awk)\b",
        # Web frameworks & libraries
        r"\b(react|vue|angular|svelte|next\.?js|nuxt|node\.?js|express)\b",
        r"\b(fastapi|django|flask|spring boot|laravel|rails|asp\.net)\b",
        r"\b(tailwind|bootstrap|css|sass|html|jsx|tsx|component)\b",
        r"\b(redux|zustand|context api|hooks|middleware|auth)\b",
        # Infrastructure & tooling
        r"\b(docker|kubernetes|k8s|aws|azure|gcp|terraform|ansible)\b",
        r"\b(git|github|gitlab|ci/cd|pipeline|jenkins|actions)\b",
        r"\b(npm|pip|yarn|cargo|maven|gradle|composer)\b",
        r"\b(linux|ubuntu|centos|debian|alpine|ssh|nginx|apache)\b",
        # Debugging & concepts
        r"\b(bug|error|exception|stacktrace|traceback|undefined|null|segfault)\b",
        r"\b(refactor|optimize|complexity|big o|algorithm|structure)\b",
        r"\b(api|rest|graphql|grpc|websocket|endpoint|json|xml|yaml)\b",
        r"\b(db|database|sql|postgres|mysql|mongodb|redis|orm|sqlalchemy)\b",
    ]

    REASONING_PATTERNS = [
        # Math & logic
        r"\b(solve|calculate|compute|prove|derive|evaluate)\b",
        r"\b(math|algebra|calculus|geometry|trigonometry|statistics|probability)\b",
        r"\b(logic|theorem|axiom|lemma|proof|contradiction|fallacy)\b",
        # Analysis & strategy
        r"\b(analyze|analyse|critique|compare|contrast|pros and cons|trade-off)\b",
        r"\b(strategy|plan|roadmap|methodology|framework|approach)\b",
        r"\b(why|how does|explain|implication|consequence|causality)\b",
        r"\b(troubleshoot|diagnose|root cause|investigate)\b",
        # Science & academia
        r"\b(physics|chemistry|biology|quantum|relativity|thermodynamics)\b",
        r"\b(research|hypothesis|experiment|study|citation|reference)\b",
        r"\b(economic|market|financial|investment|crypto|blockchain)\b",
    ]

    CREATIVE_PATTERNS = [
        # Writing formats
        r"\b(write|compose|draft|create|generate|brainstorm)\b",
        r"\b(story|poem|essay|blog|article|email|letter|speech)\b",
        r"\b(script|screenplay|dialogue|lyrics|song|haiku|sonnet)\b",
        r"\b(tweet|post|caption|headline|tagline|slogan|copy)\b",
        # Narrative elements
        r"\b(imagine|scenario|fiction|fantasy|sci-fi|plot|twist)\b",
        r"\b(character|protagonist|antagonist|setting|world-building)\b",
        r"\b(tone|style|voice|mood|atmosphere|metaphor|simile)\b",
        # Professional / marketing
        r"\b(marketing|proposal|pitch|presentation|resume|cover letter)\b",
        r"\b(branding|identity|mission|vision|value proposition)\b",
    ]

    DATA_PATTERNS = [
        # Data actions
        r"\b(summarize|summarise|summary|extract|key points|tl;dr|abstract)\b",
        r"\b(visualize|plot|chart|graph|dashboard|heatmap)\b",
        r"\b(clean|transform|process|parse|scrape|crawl)\b",
        # Formats & tools
        r"\b(dataset|csv|excel|spreadsheet|dataframe|jsonl|parquet)\b",
        r"\b(pandas|numpy|matplotlib|seaborn|scikit|pytorch|tensorflow)\b",
        # Analysis terms
        r"\b(pattern|trend|insight|correlation|outlier|anomaly)\b",
        r"\b(report|audit|review|assessment|log analysis)\b",
    ]

    _CODING = _compile(CODING_PATTERNS)
    _REASONING = _compile(REASONING_PATTERNS)
    _CREATIVE = _compile(CREATIVE_PATTERNS)
    _DATA = _compile(DATA_PATTERNS)

    @staticmethod
    def _score(text: str, pattern: re.Pattern[str]) -> int:
        return sum(1 for _ in pattern.finditer(text))

    @classmethod
    def route(cls, prompt: str, user_preference: str | None = None) -> Route:
        """Pick a model and say why.

        An explicit ``user_preference`` other than "auto" is validated against the
        registry and returned; unknown ids raise ``UnknownModelError`` rather than
        being forwarded to a provider.
        """
        if user_preference and user_preference.lower() != AUTO:
            return Route(model=get_spec(user_preference).id, intent="pinned")

        prompt = prompt or ""
        coding = cls._score(prompt, cls._CODING)
        reasoning = cls._score(prompt, cls._REASONING)
        creative = cls._score(prompt, cls._CREATIVE)
        data = cls._score(prompt, cls._DATA)

        # Coding intent wins ties: a mislabelled coding prompt is the costliest miss.
        if coding > 0 and coding >= max(reasoning, creative, data):
            return Route(DEFAULT_CODING_MODEL, "coding")

        if len(prompt) > LONG_PROMPT_CHARS:
            return Route(DEFAULT_LONG_CONTEXT_MODEL, "long_context")

        if data > 0 and data >= max(reasoning, creative):
            return Route(DEFAULT_LONG_CONTEXT_MODEL, "data")

        if reasoning > 0 and reasoning >= max(creative, data):
            return Route(DEFAULT_REASONING_MODEL, "reasoning")

        if creative > 0:
            return Route(DEFAULT_CREATIVE_MODEL, "creative")

        if len(prompt) < SHORT_PROMPT_CHARS:
            return Route(DEFAULT_FAST_MODEL, "fast")

        return Route(DEFAULT_MODEL, "default")

    @classmethod
    def determine_model(cls, prompt: str, user_preference: str | None = None) -> str:
        """Return the model id to use (see :meth:`route`)."""
        return cls.route(prompt, user_preference).model
