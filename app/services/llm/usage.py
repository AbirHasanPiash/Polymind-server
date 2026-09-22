"""Token accounting shared by every LLM adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Rough industry average for English text; only used when a provider omits usage.
CHARS_PER_TOKEN = 4


@dataclass(slots=True)
class Usage:
    """Mutable token counter passed by reference into the adapters."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(
        self, prompt_tokens: int | None = None, completion_tokens: int | None = None
    ) -> None:
        """Store counts reported by a provider, ignoring the Nones some SDKs send.

        Streaming APIs report usage in a late chunk and may send ``None`` in the
        earlier ones; assigning those directly would break the arithmetic that
        billing depends on.
        """
        if prompt_tokens is not None:
            self.prompt_tokens = int(prompt_tokens)
        if completion_tokens is not None:
            self.completion_tokens = int(completion_tokens)

    def ensure_validity(self, prompt_text: str, completion_text: str) -> None:
        """Fall back to a character-based estimate when the provider reported none.

        Without this a failed usage report would bill the user zero, so the
        estimate is deliberately charged rather than skipped.
        """
        if self.prompt_tokens <= 0 and prompt_text:
            self.prompt_tokens = math.ceil(len(prompt_text) / CHARS_PER_TOKEN)
        if self.completion_tokens <= 0 and completion_text:
            self.completion_tokens = math.ceil(len(completion_text) / CHARS_PER_TOKEN)


def estimate_tokens(text: str) -> int:
    """Cheap upper-ish estimate used for pre-flight affordability checks."""
    return math.ceil(len(text or "") / CHARS_PER_TOKEN)


@dataclass(slots=True)
class Outcome:
    """What happened to a generation, beyond the text it produced.

    Filled in by the adapter as the stream completes: which model actually
    served the request (a provider-side fallback may differ from the one asked
    for), why it stopped, and any note worth surfacing to the user.
    """

    served_model: str | None = None
    finish_reason: str = "stop"  # stop | length | refusal | interrupted | error
    refusal_category: str | None = None
    notes: list[str] = field(default_factory=list)
