"""Chat WebSocket behaviour that does not need a database."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.schemas.chat import RegeneratePayload, UserMessagePayload
from app.services.llm.schema import ChatMessage
from app.services.llm.titles import clean_title, fallback_title

WS_POLICY_VIOLATION = 1008


class TestHandshake:
    def test_connection_without_a_token_is_closed(self, client: TestClient):
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect("/api/v1/chat/ws") as ws,
        ):
            # A first frame that is not an auth frame ends the handshake at once.
            ws.send_json({"type": "user_message", "content": "hi"})
            ws.receive_text()
        assert exc.value.code == WS_POLICY_VIOLATION

    def test_auth_frame_with_a_forged_token_is_closed(self, client: TestClient):
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect("/api/v1/chat/ws") as ws,
        ):
            ws.send_json({"type": "auth", "token": "forged.token.here"})
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
        assert fallback_title("Hello there") == "Hello there"

    def test_long_message_is_truncated(self):
        title = fallback_title("x" * 200)
        assert len(title) <= 49
        assert title.endswith("…")

    def test_whitespace_is_collapsed(self):
        assert fallback_title("  what\n\nis   this ") == "what is this"

    def test_attachment_name_is_used_when_there_is_no_text(self):
        assert fallback_title("   ", "report.pdf") == "File: report.pdf"

    def test_empty_message_with_no_attachments(self):
        assert fallback_title("", None) == "New chat"

    def test_model_title_is_cleaned(self):
        assert clean_title('"Async Refactor Plan."\nignored') == "Async Refactor Plan"


class TestPayloads:
    def test_arena_models_must_be_distinct(self):
        with pytest.raises(ValueError):
            UserMessagePayload(type="user_message", content="hi", models=["gpt-5.5", "gpt-5.5"])

    def test_arena_needs_at_least_two_models(self):
        with pytest.raises(ValueError):
            UserMessagePayload(type="user_message", content="hi", models=["gpt-5.5"])

    def test_effort_defaults_to_medium(self):
        assert RegeneratePayload(type="regenerate").effort == "medium"

    def test_unknown_effort_is_rejected(self):
        with pytest.raises(ValueError):
            UserMessagePayload(type="user_message", content="hi", effort="ultra")


class TestHistoryComparison:
    def test_text_of_an_empty_message_is_safe(self):
        assert ChatMessage(role="user", content=[]).text == ""

    def test_text_matches_the_original_string(self):
        assert ChatMessage.from_text("user", "hello").text == "hello"
