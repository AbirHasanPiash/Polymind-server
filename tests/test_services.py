"""Service-layer units: message formatting, storage keys and file processing."""

from __future__ import annotations

import io

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.services.file_processing import process_file
from app.services.llm.schema import Attachment, ChatMessage
from app.services.storage import storage


def _upload(name: str, data: bytes, content_type: str) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(data), headers={"content-type": content_type})


class TestChatMessage:
    def test_plain_message_uses_the_simple_format(self):
        message = ChatMessage.from_text("user", "hello")
        assert message.to_openai_format() == {"role": "user", "content": "hello"}

    def test_ai_role_maps_to_assistant(self):
        assert ChatMessage.from_text("ai", "hi").to_openai_format()["role"] == "assistant"

    def test_image_attachment_becomes_a_data_url_part(self):
        message = ChatMessage.from_text("user", "what is this?")
        message.attachments = [Attachment(type="image", content="QUJD", mime_type="image/png")]

        parts = message.to_openai_format()["content"]
        assert parts[0]["type"] == "text"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_text_attachment_is_folded_into_the_prompt(self):
        message = ChatMessage.from_text("user", "summarise")
        message.attachments = [Attachment(type="text", content="file body")]

        parts = message.to_openai_format()["content"]
        assert len(parts) == 1
        assert "summarise" in parts[0]["text"]
        assert "file body" in parts[0]["text"]

    def test_text_property_concatenates_blocks(self):
        assert ChatMessage.from_text("user", "abc").text == "abc"

    def test_responses_input_uses_typed_parts(self):
        message = ChatMessage.from_text("user", "look")
        message.attachments = [Attachment(type="image", content="QUJD", mime_type="image/png")]
        item = message.to_responses_input()
        assert item["role"] == "user"
        assert item["content"][0] == {"type": "input_text", "text": "look"}
        assert item["content"][1]["type"] == "input_image"

    def test_responses_input_for_assistant_is_a_string(self):
        assert ChatMessage.from_text("ai", "hi").to_responses_input() == {
            "role": "assistant",
            "content": "hi",
        }


class TestStorageKeys:
    """Pinned to a known base URL so the result does not depend on local config."""

    @pytest.fixture(autouse=True)
    def _bucket(self, monkeypatch):
        monkeypatch.setattr(storage, "public_base_url", "https://cdn.example.com")

    def test_key_is_recovered_from_a_public_url(self):
        url = storage.public_url("tts/abc.mp3")
        assert url == "https://cdn.example.com/tts/abc.mp3"
        assert storage.key_from_url(url) == "tts/abc.mp3"

    def test_nested_keys_survive_the_round_trip(self):
        key = "generated_images/user-1/abc-def.png"
        assert storage.key_from_url(storage.public_url(key)) == key

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.example.com/secret.mp3",
            "https://cdn.example.com.evil.com/x.mp3",
            "https://cdn.example.com/",
            "",
            "not a url",
        ],
    )
    def test_foreign_urls_yield_no_key(self, url):
        """A delete must never be derived from a URL outside the bucket."""
        assert storage.key_from_url(url) is None


class TestFileProcessing:
    async def test_text_file_is_wrapped_with_delimiters(self):
        result = await process_file(_upload("notes.txt", b"hello world", "text/plain"))
        assert result["type"] == "text"
        assert "hello world" in result["content"]
        assert "START FILE: notes.txt" in result["content"]

    async def test_source_file_is_detected_by_extension(self):
        result = await process_file(_upload("main.py", b"print(1)", "application/octet-stream"))
        assert result["type"] == "text"

    async def test_image_is_re_encoded_as_base64(self):
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10), "red").save(buffer, format="PNG")
        result = await process_file(_upload("pic.png", buffer.getvalue(), "image/png"))
        assert result["type"] == "image"
        assert result["mime_type"] == "image/png"
        assert result["content"]

    async def test_corrupt_image_is_a_client_error(self):
        with pytest.raises(HTTPException) as exc:
            await process_file(_upload("pic.png", b"definitely not an image", "image/png"))
        assert exc.value.status_code == 400

    async def test_empty_file_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            await process_file(_upload("empty.txt", b"", "text/plain"))
        assert exc.value.status_code == 400

    async def test_oversized_file_is_rejected(self):
        from app.core.config import settings

        payload = b"x" * (settings.max_upload_size_bytes + 1)
        with pytest.raises(HTTPException) as exc:
            await process_file(_upload("big.txt", payload, "text/plain"))
        assert exc.value.status_code == 413

    async def test_unsupported_type_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            await process_file(_upload("app.bin", b"\x00\x01", "application/x-binary"))
        assert exc.value.status_code == 400
