"""Document-related Pydantic schemas."""
from typing import Optional, Any, Literal
from datetime import datetime
from typing import Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field


# ─── Document Schemas ────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    id: UUID
    filename: str
    status: str
    message: str


class DocumentPageInfo(BaseModel):
    page_number: int
    extraction_method: Optional[str] = None
    ocr_confidence: Optional[float] = None
    is_low_confidence: bool = False
    has_text: bool = True


class DocumentSectionInfo(BaseModel):
    section_type: str
    section_title: Optional[str] = None
    content: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None


class LegalEntityInfo(BaseModel):
    entity_type: str
    entity_value: str
    normalized_value: Optional[str] = None
    confidence: float = 1.0


class DocumentDetail(BaseModel):
    id: UUID
    filename: str
    original_filename: str
    file_size: int
    page_count: int
    pdf_type: Optional[str] = None
    extraction_method: Optional[str] = None
    status: str
    status_message: Optional[str] = None
    processing_progress: float = 0.0

    # Metadata
    court: Optional[str] = None
    case_name: Optional[str] = None
    case_number: Optional[str] = None
    document_type: str = "judgment"
    language: str = "en"
    document_date: Optional[datetime] = None
    judges: Optional[list[str]] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None

    # Pages and sections
    pages: list[DocumentPageInfo] = []
    sections: list[DocumentSectionInfo] = []
    entities: list[LegalEntityInfo] = []

    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListItem(BaseModel):
    id: UUID
    filename: str
    original_filename: str
    status: str
    page_count: int
    pdf_type: Optional[str] = None
    court: Optional[str] = None
    case_name: Optional[str] = None
    language: str = "en"
    processing_progress: float = 0.0
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentStatusResponse(BaseModel):
    id: UUID
    status: str
    status_message: Optional[str] = None
    processing_progress: float = 0.0


# ─── Summary Schemas ─────────────────────────────────────────

class SummarizeRequest(BaseModel):
    summary_type: str = Field(..., pattern="^(short|medium|detailed)$")
    language: str = Field(default="en", pattern="^(en|hi|mr)$")


class SummaryResponse(BaseModel):
    id: UUID
    document_id: UUID
    summary_type: str
    language: str
    content: str
    simplified_content: Optional[str] = None
    model_used: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Simplification Schemas ──────────────────────────────────

class SimplifyRequest(BaseModel):
    language: str = Field(default="en", pattern="^(en|hi|mr)$")


class SimplificationResponse(BaseModel):
    document_id: UUID
    language: str
    original_summary: str
    simplified_text: str
    terminology: list[dict[str, str]] = []
    meaning_preservation_score: Optional[float] = None


# ─── Translation Schemas ─────────────────────────────────────

class TranslateRequest(BaseModel):
    target_language: str = Field(..., pattern="^(hi|mr)$")
    content_type: str = Field(default="summary_medium", pattern="^(summary_short|summary_medium|summary_detailed|simplification)$")


class TranslationResponse(BaseModel):
    id: UUID
    document_id: UUID
    source_language: str
    target_language: str
    content_type: str
    translated_text: str
    terminology_preserved: Optional[list[dict[str, str]]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Chat Schemas ────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    session_id: Optional[UUID] = None

    language: str = Field(
        default="en",
        pattern="^(en|hi|mr)$",
    )

    mode: Literal[
        "document_qa",
        "general_legal",
    ] = "document_qa"


class SourceReference(BaseModel):
    document_name: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    paragraph: Optional[str] = None
    content: Optional[str] = None
    knowledge_source: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: UUID
    message: str
    sources: list[SourceReference] = []
    confidence: Optional[float] = None
    was_refused: bool = False
    refusal_reason: Optional[str] = None
    mode: str  # document_qa or general_legal
    disclaimer: str = "This is informational assistance and not a substitute for professional legal advice."


# ─── Legal Search Schemas ────────────────────────────────────

class LegalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    language: str = Field(default="en", pattern="^(en|hi|mr)$")


class LegalSearchResult(BaseModel):
    title: str
    content: str
    source: Optional[str] = None
    authority: Optional[str] = None
    relevance_score: float = 0.0


class LegalSearchResponse(BaseModel):
    query: str
    results: list[LegalSearchResult] = []
    total: int = 0


# ─── Health Schema ───────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    database: str = "connected"
    llm: str = "available"
    embeddings: str = "available"
    ocr: str = "available"
