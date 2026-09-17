"""Extract raw text from uploaded PDF or TXT files."""

from __future__ import annotations

import io

from pypdf import PdfReader


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file is neither a PDF nor a plain-text file."""


def extract_text(file_bytes: bytes, content_type: str, filename: str) -> str:
    """Extract raw text from file bytes based on content type / extension.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        content_type: The MIME type reported by the upload (may be unreliable).
        filename: Original filename, used as a fallback signal for file type.

    Returns:
        The extracted plain text.

    Raises:
        UnsupportedFileTypeError: If the file is not a recognized PDF or TXT file.
    """
    lower_name = filename.lower()

    if content_type == "application/pdf" or lower_name.endswith(".pdf"):
        return _extract_pdf_text(file_bytes)

    if content_type in ("text/plain", "") or lower_name.endswith(".txt"):
        return _extract_txt_text(file_bytes)

    raise UnsupportedFileTypeError(f"Unsupported file type for '{filename}' ({content_type})")


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages_text).strip()


def _extract_txt_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace").strip()
