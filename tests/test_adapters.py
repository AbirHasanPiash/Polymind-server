"""Request shapes each adapter sends, checked without any network access."""

from __future__ import annotations

from app.services.llm.base import GenerationOptions, split_system
from app.services.llm.claude_adapter import ClaudeAdapter
from app.services.llm.gemini_adapter import GeminiAdapter
from app.services.llm.openai_adapter import OpenAIAdapter
from app.services.llm.schema import Attachment, ChatMessage


def _conversation() -> list[ChatMessage]:
    user = ChatMessage.from_text("user", "what is this?")
    user.attachments = [Attachment(type="image", content="QUJD", mime_type="image/png")]
    return [
        ChatMessage.from_text("system", "You are terse."),
        user,
        ChatMessage.from_text("ai", "A test image."),
        ChatMessage.from_text("user", "thanks"),
    ]


class TestSplitSystem:
    def test_system_messages_and_options_are_merged(self):
        system, messages = split_system(
            _conversation(), GenerationOptions(system_prompt="Platform rules.")
        )
        assert system.startswith("Platform rules.")
        assert "You are terse." in system
        assert [m.role for m in messages] == ["user", "ai", "user"]

    def test_plain_string_prompt(self):
        system, messages = split_system("hello", None)
        assert system == ""
        assert messages[0].text == "hello"


class TestOpenAI:
    def test_uses_the_responses_api_shape(self):
        request = OpenAIAdapter()._request(
            _conversation(), "gpt-5.5", GenerationOptions(effort="low")
        )
        assert request["model"] == "gpt-5.5"
        assert request["store"] is False
        assert request["instructions"] == "You are terse."
        assert request["reasoning"] == {"effort": "low"}
        first = request["input"][0]
        assert first["role"] == "user"
        assert first["content"][0]["type"] == "input_text"
        assert first["content"][1]["type"] == "input_image"
        assert request["input"][1] == {"role": "assistant", "content": "A test image."}

    def test_pro_models_never_get_low_effort(self):
        request = OpenAIAdapter()._request("hi", "gpt-5.5-pro", GenerationOptions(effort="low"))
        assert request["reasoning"] == {"effort": "medium"}

    def test_legacy_ids_map_to_current_api_models(self):
        assert OpenAIAdapter()._request("hi", "gpt-5.2", None)["model"] == "gpt-5.5"


class TestClaude:
    def test_alternating_turns_with_system_and_thinking(self):
        request = ClaudeAdapter()._request(
            _conversation(), "claude-opus-5", GenerationOptions(effort="high")
        )
        assert request["system"] == "You are terse."
        assert request["thinking"] == {"type": "adaptive"}
        assert request["output_config"] == {"effort": "high"}
        assert [turn["role"] for turn in request["messages"]] == ["user", "assistant", "user"]
        image_block = request["messages"][0]["content"][0]
        assert image_block["type"] == "image"
        assert image_block["source"]["media_type"] == "image/png"

    def test_frontier_models_opt_into_server_side_fallbacks(self):
        request = ClaudeAdapter()._request("hi", "claude-fable-5-1", None)
        assert request["fallbacks"] == "default"
        assert request["betas"] == ["server-side-fallback-2026-07-01"]

    def test_haiku_gets_no_thinking_or_effort(self):
        request = ClaudeAdapter()._request("hi", "claude-haiku-4-5", None)
        assert "thinking" not in request
        assert "output_config" not in request
        assert "fallbacks" not in request

    def test_consecutive_user_turns_are_merged(self):
        prompt = [ChatMessage.from_text("user", "a"), ChatMessage.from_text("user", "b")]
        request = ClaudeAdapter()._request(prompt, "claude-sonnet-5", None)
        assert len(request["messages"]) == 1
        assert [b["text"] for b in request["messages"][0]["content"]] == ["a", "b"]


class TestGemini:
    def test_thinking_level_follows_effort(self):
        api_model, contents, config = GeminiAdapter()._prepare(
            _conversation(), "gemini-3.8-flash", GenerationOptions(effort="low")
        )
        assert api_model == "gemini-3.8-flash"
        assert config.system_instruction == "You are terse."
        assert str(config.thinking_config.thinking_level.value).lower() == "low"
        assert [c.role for c in contents] == ["user", "model", "user"]

    def test_image_attachments_become_inline_parts(self):
        _, contents, _ = GeminiAdapter()._prepare(_conversation(), "gemini-3.8-flash", None)
        parts = contents[0].parts
        assert parts[0].text == "what is this?"
        assert parts[1].inline_data.mime_type == "image/png"
