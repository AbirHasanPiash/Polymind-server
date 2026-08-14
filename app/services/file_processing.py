"""Turn an uploaded file into something an LLM can consume.

Images become resized base64 payloads; documents and source files become text.
Every path is bounded — size, pixel count, page count and character count — so a
single upload cannot exhaust memory or blow up a prompt.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import PyPDF2
from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import docx
except ImportError:  # pragma: no cover - optional dependency
    docx = None

MAX_TEXT_LENGTH = 100_000  # characters kept from any single document
MAX_PDF_PAGES = 20
MAX_IMAGE_DIMENSION = 2048  # longest edge after downscaling
# Refuse absurd pixel counts before decoding: a small file can expand to GBs.
Image.MAX_IMAGE_PIXELS = 50_000_000

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".c", ".cpp", ".h", ".hpp",
    ".java", ".rs", ".go", ".rb", ".php", ".sh", ".bat", ".ps1",
    ".html", ".css", ".scss", ".sql", ".json", ".yaml", ".yml",
    ".xml", ".md", ".txt", ".env", ".gitignore", ".dockerfile", ".conf", ".ini",
}

IMAGE_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


async def process_file(file: UploadFile) -> dict:
    """Extract content from ``file``.

    Returns ``{"type": "text"|"image", "content": ..., ...}``.
    Raises ``HTTPException`` for anything the caller should see as a 400/413.
    """
    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"
    extension = Path(filename).suffix.lower()

    file_bytes = await file.read()
    if not file_bytes:
        raise _bad_request(f"{filename} is empty")
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{filename} exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit",
        )

    if content_type.startswith("image/"):
        return _process_image(file_bytes, content_type)

    if content_type == "application/pdf" or extension == ".pdf":
        return _wrap_text(filename, _extract_pdf_text(file_bytes))

    if extension == ".docx":
        return _wrap_text(filename, _extract_docx_text(file_bytes))

    is_text_mime = content_type.startswith("text/") or any(
        marker in content_type for marker in ("json", "javascript", "xml", "csv")
    )
    if is_text_mime or extension in CODE_EXTENSIONS or extension == ".csv":
        return _wrap_text(filename, _decode_text(file_bytes))

    raise _bad_request(f"Unsupported file type: {content_type} ({extension or 'no extension'})")


def _wrap_text(filename: str, text: str) -> dict:
    """Delimit file content so the model can tell it apart from the user's message."""
    return {
        "type": "text",
        "content": f"--- START FILE: {filename} ---\n{text[:MAX_TEXT_LENGTH]}\n--- END FILE ---",
        "filename": filename,
    }


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _process_image(file_bytes: bytes, mime_type: str) -> dict:
    """Downscale and re-encode an image, returning it as base64."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except Image.DecompressionBombError:
        raise _bad_request("Image resolution is too large to process") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise _bad_request("Invalid or corrupted image file") from None

    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

    save_format = IMAGE_FORMATS.get(mime_type, "PNG")
    if save_format == "JPEG" and image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format=save_format, optimize=True)

    return {
        "type": "image",
        "mime_type": mime_type,
        "content": base64.b64encode(buffer.getvalue()).decode("utf-8"),
    }


def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = (page.extract_text() for page in reader.pages[:MAX_PDF_PAGES])
        text = "\n".join(page for page in pages if page)
    except Exception as exc:
        logger.warning("PDF extraction failed: %s", exc)
        return "[Error extracting PDF text]"
    return text or "[PDF contained no readable text]"


def _extract_docx_text(file_bytes: bytes) -> str:
    if docx is None:  # pragma: no cover - optional dependency
        return "[Error: python-docx is not installed on the server]"
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as exc:
        logger.warning("DOCX extraction failed: %s", exc)
        return "[Error extracting Word document text]"
