"""Conversation titles.

The first message is a poor title ("hey can you help me with something"), so
after the first reply a cheap utility model is asked for a short one. It runs in
the background, is never billed to the user, and falls back to a trimmed copy of
the message when no provider is configured or the call fails.
"""

from __future__ import annotations

import logging
import re

from app.core.config import settings
from app.services.llm.base import GenerationOptions
from app.services.llm.factory import LLMFactory
from app.services.llm.models import UTILITY_MODEL
from app.services.llm.schema import ChatMessage
from app.services.llm.usage import Usage

logger = logging.getLogger(__name__)

TITLE_CHARS = 48
_PROMPT = (
    "Write a title of at most six words for the conversation below. Reply with the "
    "title only: no quotes, no trailing punctuation, in the language of the user.\n\n"
    "User: {user}\n\nAssistant: {assistant}"
)


def fallback_title(text: str, attachment_name: str | None = None) -> str:
    """Derive a title from the first message without calling a model."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if text:
        return f"{text[:TITLE_CHARS].rstrip()}…" if len(text) > TITLE_CHARS else text
    if attachment_name:
        return f"File: {attachment_name}"
    return "New chat"


def clean_title(raw: str) -> str:
    title = raw.strip().splitlines()[0] if raw.strip() else ""
    title = title.strip(" \"'“”‘’.#*-")
    return title[:TITLE_CHARS].rstrip()


async def generate_title(user_text: str, assistant_text: str) -> str | None:
    """Ask the utility model for a title; None when it is unavailable."""
    if not settings.GOOGLE_API_KEY and UTILITY_MODEL.startswith("gemini"):
        return None
    try:
        provider = LLMFactory.get_provider(UTILITY_MODEL)
        prompt = _PROMPT.format(user=user_text[:2000], assistant=assistant_text[:2000])
        raw = await provider.generate_text(
            [ChatMessage.from_text("user", prompt)],
            UTILITY_MODEL,
            Usage(),
            GenerationOptions(effort="low", max_output_tokens=64),
        )
    except Exception as exc:  # best effort by design
        logger.info("Title generation skipped: %s", exc)
        return None
    title = clean_title(raw)
    return title or None
