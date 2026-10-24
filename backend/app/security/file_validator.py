"""Security utilities for file validation and sanitization."""

import re
import os
import hashlib
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.exceptions import PDFValidationError, SecurityError
from app.core.logging_config import get_logger

logger = get_logger("security")

ALLOWED_MIME_TYPES = {
    "application/pdf",
}

ALLOWED_EXTENSIONS = {
    ".pdf",
}

# Characters allowed in sanitized filenames
SAFE_FILENAME_PATTERN = re.compile(r'[^a-zA-Z0-9_\-.]')


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and injection attacks."""
    # Remove path separators
    filename = os.path.basename(filename)
    # Remove null bytes
    filename = filename.replace('\x00', '')
    # Replace unsafe characters
    name, ext = os.path.splitext(filename)
    name = SAFE_FILENAME_PATTERN.sub('_', name)
    # Limit length
    name = name[:200]
    ext = ext[:10].lower()
    if not name:
        name = "document"
    return f"{name}{ext}"


def generate_safe_filename(original_filename: str) -> str:
    """Generate a unique safe filename."""
    import uuid
    safe_name = sanitize_filename(original_filename)
    name, ext = os.path.splitext(safe_name)
    unique_id = uuid.uuid4().hex[:12]
    return f"{name}_{unique_id}{ext}"


def validate_file_extension(filename: str) -> None:
    """Validate file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise PDFValidationError(
            f"File extension '{ext}' is not allowed. Only PDF files are accepted.",
            details={"extension": ext, "allowed": list(ALLOWED_EXTENSIONS)}
        )


def validate_mime_type(file_content: bytes) -> str:
    """Validate file MIME type from content bytes."""
    # Check magic bytes for PDF
    if file_content[:5] != b'%PDF-':
        raise PDFValidationError(
            "File does not appear to be a valid PDF (invalid magic bytes).",
            details={"magic_bytes": file_content[:10].hex()}
        )
    return "application/pdf"


def validate_file_size(file_size: int) -> None:
    """Validate file size against configured maximum."""
    settings = get_settings()
    max_size = settings.max_file_size_bytes
    if file_size > max_size:
        raise PDFValidationError(
            f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed ({settings.max_file_size_mb} MB).",
            details={"file_size": file_size, "max_size": max_size}
        )


def validate_page_count(page_count: int) -> None:
    """Validate page count against configured maximum."""
    settings = get_settings()
    if page_count > settings.max_pages:
        raise PDFValidationError(
            f"Document has {page_count} pages, exceeding the maximum of {settings.max_pages}.",
            details={"page_count": page_count, "max_pages": settings.max_pages}
        )
    if page_count == 0:
        raise PDFValidationError(
            "Document appears to have no pages.",
            details={"page_count": 0}
        )


def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()
