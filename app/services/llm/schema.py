"""Provider-neutral conversation types.

Adapters translate these into each vendor's wire format, so the rest of the
application never depends on a specific SDK's message shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "ai"]


class ContentBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Attachment(BaseModel):
    """A processed file ready for LLM consumption.

    - ``text``: extracted text (PDF, DOCX, source code…)
    - ``image``: base64 payload; ``mime_type`` is then required by every provider
    """

    type: Literal["text", "image"]
    content: str
    mime_type: str | None = None


class ChatMessage(BaseModel):
    role: Role
    content: list[ContentBlock]
    attachments: list[Attachment] = Field(default_factory=list)

    @classmethod
    def from_text(cls, role: str, text: str) -> ChatMessage:
        """Build a message from a plain string."""
        return cls(role=role, content=[ContentBlock(text=text)], attachments=[])

    @property
    def text(self) -> str:
        """Concatenated text of every content block (empty string if none)."""
        return "".join(block.text for block in self.content)

    @property
    def is_assistant(self) -> bool:
        return self.role in ("ai", "assistant")

    def _text_with_attachments(self) -> str:
        """Prompt text with any text attachments folded in after it."""
        text_attachments = [a.content for a in self.attachments if a.type == "text"]
        return "\n\n".join(filter(None, [self.text, *text_attachments]))

    def to_openai_format(self) -> dict[str, object]:
        """Chat Completions shape, kept for tests and any compatible provider."""
        role = "assistant" if self.role == "ai" else self.role

        if not self.attachments:
            return {"role": role, "content": self.text}

        parts: list[dict[str, object]] = []
        combined = self._text_with_attachments()
        if combined:
            parts.append({"type": "text", "text": combined})

        for attachment in self.attachments:
            if attachment.type == "image":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{attachment.mime_type or 'image/jpeg'};base64,{attachment.content}",
                            "detail": "auto",
                        },
                    }
                )

        return {"role": role, "content": parts}

    def to_responses_input(self) -> dict[str, object]:
        """OpenAI Responses API input item.

        Assistant turns are plain strings; user turns become typed parts so
        images travel as ``input_image`` data URLs.
        """
        if self.is_assistant:
            return {"role": "assistant", "content": self._text_with_attachments()}

        role = "user" if self.role != "system" else "system"
        if not self.attachments:
            return {"role": role, "content": self.text}

        parts: list[dict[str, object]] = []
        combined = self._text_with_attachments()
        if combined:
            parts.append({"type": "input_text", "text": combined})
        for attachment in self.attachments:
            if attachment.type == "image":
                parts.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{attachment.mime_type or 'image/jpeg'};base64,{attachment.content}",
                        "detail": "auto",
                    }
                )
        return {"role": role, "content": parts}
