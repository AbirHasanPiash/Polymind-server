"""Chat WebSocket behaviour that does not need a database.

Also covers the two crash paths found in the audit: a title derived from an
attachment whose name is empty, and a history entry whose content is empty.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.v1.endpoints.chat import AttachmentSchema, _chat_title
from app.services.llm.schema import ChatMessage

WS_POLICY_VIOLATION = 1008


class TestHandshake:
    def test_connection_without_a_token_is_closed(self, client: TestClient):
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect("/api/v1/chat/ws") as ws,
        ):
            ws.receive_text()
        assert exc.value.code == WS_POLICY_VIOLATION

    def test_connection_with_a_forged_token_is_closed(self, client: TestClient):
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect("/api/v1/chat/ws?token=forged.token.here") as ws,
        ):
            ws.receive_text()
        assert exc.value.code == WS_POLICY_VIOLATION


class TestChatTitle:
    def test_short_message_becomes_the_title(self):
        assert _chat_title("Hello there", []) == "Hello there"

    def test_long_message_is_truncated(self):
        title = _chat_title("x" * 200, [])
        assert len(title) <= 41
        assert title.endswith("…")

    def test_attachment_name_is_used_when_there_is_no_text(self):
        attachment = AttachmentSchema(id="1", name="report.pdf", type="text", size=10)
        assert _chat_title("   ", [attachment]) == "File: report.pdf"

    def test_attachment_without_a_name_does_not_crash(self):
        """The previous dict-style lookup raised AttributeError here."""
        attachment = AttachmentSchema(id="1", name="", type="text", size=10)
        assert _chat_title("", [attachment]) == "File: Attachment"

    def test_empty_message_with_no_attachments(self):
        assert _chat_title("", []) == "New Chat"


class TestHistoryComparison:
    def test_text_of_an_empty_message_is_safe(self):
        """Indexing content[0] used to raise IndexError on an empty message."""
        assert ChatMessage(role="user", content=[]).text == ""

    def test_text_matches_the_original_string(self):
        assert ChatMessage.from_text("user", "hello").text == "hello"
