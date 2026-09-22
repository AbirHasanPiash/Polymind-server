"""Budgeting, prompt assembly and export formatting for the chat service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.services import chat_service
from app.services.llm.base import MAX_OUTPUT_TOKENS_BY_EFFORT
from app.services.llm.models import get_spec
from app.services.llm.schema import Attachment, ChatMessage


class TestBudget:
    def test_rich_balance_gets_the_full_effort_ceiling(self):
        budget = chat_service.plan_budget(get_spec("gpt-5.4-nano"), 400, Decimal("100"), "high")
        assert budget.max_output_tokens == MAX_OUTPUT_TOKENS_BY_EFFORT["high"]

    def test_small_balance_lowers_the_output_ceiling(self):
        spec = get_spec("gpt-5.5-pro")  # 180 USD / 1M output tokens
        budget = chat_service.plan_budget(spec, 400, Decimal("10"), "high")
        assert (
            chat_service.MIN_OUTPUT_TOKENS
            <= budget.max_output_tokens
            < MAX_OUTPUT_TOKENS_BY_EFFORT["high"]
        )

    def test_unaffordable_turn_is_refused_up_front(self):
        with pytest.raises(chat_service.CannotAffordError) as exc:
            chat_service.plan_budget(get_spec("gpt-5.5-pro"), 400, Decimal("0.01"), "medium")
        assert exc.value.required > exc.value.balance

    def test_arena_splits_the_balance_between_slots(self):
        spec = get_spec("gpt-5.5")
        alone = chat_service.plan_budget(spec, 400, Decimal("2"), "high")
        shared = chat_service.plan_budget(spec, 400, Decimal("2"), "high", share=2)
        assert shared.max_output_tokens <= alone.max_output_tokens

    def test_history_chars_counts_text_and_images(self):
        message = ChatMessage.from_text("user", "hello")
        message.attachments = [
            Attachment(type="text", content="x" * 100),
            Attachment(type="image", content="abc", mime_type="image/png"),
        ]
        assert chat_service.history_chars([message]) == 5 + 100 + 4000


class TestModels:
    def test_arena_models_are_validated(self):
        assert chat_service.resolve_models(["gpt-5.5", "claude-4.5-sonnet"], None, "") == [
            "gpt-5.5",
            "claude-sonnet-5",
        ]

    def test_single_model_falls_back_to_routing(self):
        assert chat_service.resolve_models(None, "auto", "hi") == ["gemini-3.8-flash"]


class TestSystemPrompt:
    def test_platform_prompt_carries_the_date(self):
        prompt = chat_service.build_system_prompt(None, None)
        assert "Polymind" in prompt
        assert datetime.now(UTC).strftime("%Y-%m-%d") in prompt

    def test_user_and_chat_instructions_are_appended(self):
        prompt = chat_service.build_system_prompt("Be brief.", "Speak like a pirate.")
        assert "Be brief." in prompt
        assert "Speak like a pirate." in prompt
        assert prompt.index("Be brief.") < prompt.index("pirate")

    def test_blank_instructions_are_ignored(self):
        assert chat_service.build_system_prompt("   ", "") == chat_service.build_system_prompt(
            None, None
        )


def _chat(title="Plan"):
    return SimpleNamespace(
        id=uuid.uuid4(), title=title, mode="chat", created_at=datetime(2026, 9, 21, tzinfo=UTC)
    )


def _message(role, content, model=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        role=role,
        content=content,
        model=model,
        parent_id=None,
        tokens=12,
        cost=Decimal("0.5"),
        created_at=datetime(2026, 9, 21, 10, 0, tzinfo=UTC),
    )


class TestExport:
    def test_markdown_has_headings_per_turn(self):
        text = chat_service.export_markdown(
            _chat(), [_message("user", "Hello"), _message("ai", "Hi!", "gpt-5.5")]
        )
        assert text.startswith("# Plan")
        assert "## You\n\nHello" in text
        assert "## gpt-5.5\n\nHi!" in text

    def test_json_uses_public_roles(self):
        import json

        payload = json.loads(
            chat_service.export_json(
                _chat(), [_message("user", "Hello"), _message("ai", "Hi!", "gpt-5.5")]
            )
        )
        assert [m["role"] for m in payload["messages"]] == ["user", "assistant"]
        assert payload["messages"][1]["cost"] == "0.5"

    def test_share_url_points_at_the_frontend(self):
        assert chat_service.share_url("abc").endswith("/share/abc")
