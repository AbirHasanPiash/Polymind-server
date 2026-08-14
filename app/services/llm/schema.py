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

    def to_openai_format(self) -> dict[str, object]:
        """Convert to the OpenAI chat format, mapping 'ai' to 'assistant'."""
        role = "assistant" if self.role == "ai" else self.role

        if not self.attachments:
            return {"role": role, "content": self.text}

        # With attachments the API requires the multipart "content parts" form.
        parts: list[dict[str, object]] = []
        text_content = self.text

        text_attachments = [a.content for a in self.attachments if a.type == "text"]
        if text_content or text_attachments:
            parts.append({"type": "text", "text": "\n\n".join(filter(None, [text_content, *text_attachments]))})

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
