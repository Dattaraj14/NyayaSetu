"""Custom exception classes for NyayaSetu."""

from typing import Any, Optional


class NyayaSetuError(Exception):
    """Base exception for NyayaSetu."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DocumentNotFoundError(NyayaSetuError):
    """Document not found."""
    pass


class DocumentProcessingError(NyayaSetuError):
    """Error during document processing."""
    pass


class PDFValidationError(NyayaSetuError):
    """Invalid PDF file."""
    pass


class OCRError(NyayaSetuError):
    """OCR processing error."""
    pass


class LLMError(NyayaSetuError):
    """LLM inference error."""
    pass


class EmbeddingError(NyayaSetuError):
    """Embedding generation error."""
    pass


class RetrievalError(NyayaSetuError):
    """Vector retrieval error."""
    pass


class TranslationError(NyayaSetuError):
    """Translation error."""
    pass


class SecurityError(NyayaSetuError):
    """Security validation error."""
    pass


class RateLimitError(NyayaSetuError):
    """Rate limit exceeded."""
    pass
