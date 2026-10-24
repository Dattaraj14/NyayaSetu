"""Document-related database models."""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, Boolean,
    ForeignKey, Enum, JSON, Index, Uuid as UUID
)
from sqlalchemy.orm import relationship
from app.models.database import Base


class DocumentStatus(str, PyEnum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    OCR_PROCESSING = "ocr_processing"
    PREPROCESSING = "preprocessing"
    ANALYZING = "analyzing"
    EMBEDDING = "embedding"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractionMethod(str, PyEnum):
    TEXT = "text"
    OCR = "ocr"
    HYBRID = "hybrid"


class PDFType(str, PyEnum):
    TEXT = "text"
    SCANNED = "scanned"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    page_count = Column(Integer, default=0)
    pdf_type = Column(Enum(PDFType), default=PDFType.UNKNOWN)
    extraction_method = Column(Enum(ExtractionMethod), nullable=True)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.UPLOADED, index=True)
    status_message = Column(Text, nullable=True)
    processing_progress = Column(Float, default=0.0)

    # Metadata
    source = Column(String(500), nullable=True)
    court = Column(String(500), nullable=True)
    case_name = Column(String(1000), nullable=True)
    case_number = Column(String(200), nullable=True)
    document_type = Column(String(100), default="judgment")
    language = Column(String(50), default="en")
    document_date = Column(DateTime, nullable=True)
    judges = Column(JSON, nullable=True)
    petitioner = Column(String(1000), nullable=True)
    respondent = Column(String(1000), nullable=True)

    # Provenance
    collection_date = Column(DateTime, default=datetime.utcnow)
    source_url = Column(String(2000), nullable=True)

    # Versioning
    document_version = Column(Integer, default=1)
    model_version = Column(String(100), nullable=True)
    prompt_version = Column(String(100), nullable=True)
    embedding_version = Column(String(100), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    pages = relationship("DocumentPage", back_populates="document", cascade="all, delete-orphan")
    sections = relationship("DocumentSection", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("LegalEntity", back_populates="document", cascade="all, delete-orphan")
    summaries = relationship("Summary", back_populates="document", cascade="all, delete-orphan")
    translations = relationship("Translation", back_populates="document", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="document", cascade="all, delete-orphan")
    processing_logs = relationship("ProcessingLog", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_status_created", "status", "created_at"),
    )


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False)
    raw_text = Column(Text, nullable=True)
    processed_text = Column(Text, nullable=True)
    extraction_method = Column(Enum(ExtractionMethod), nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    has_images = Column(Boolean, default=False)
    is_low_confidence = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="pages")

    __table_args__ = (
        Index("ix_doc_pages_doc_page", "document_id", "page_number", unique=True),
    )


class DocumentSection(Base):
    __tablename__ = "document_sections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    section_type = Column(String(100), nullable=False)  # background, facts, issues, arguments, reasoning, decision, order
    section_title = Column(String(500), nullable=True)
    content = Column(Text, nullable=True)
    start_page = Column(Integer, nullable=True)
    end_page = Column(Integer, nullable=True)
    order_index = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="sections")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    page_number = Column(Integer, nullable=True)
    section_type = Column(String(100), nullable=True)
    language = Column(String(50), default="en")
    metadata_json = Column(JSON, nullable=True)

    # Vector reference
    vector_id = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_doc_chunks_doc_idx", "document_id", "chunk_index"),
    )


class LegalEntity(Base):
    __tablename__ = "legal_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False)  # person, court, judge, act, section, article, date, location, case_number
    entity_value = Column(String(2000), nullable=False)
    normalized_value = Column(String(2000), nullable=True)
    page_number = Column(Integer, nullable=True)
    start_pos = Column(Integer, nullable=True)
    end_pos = Column(Integer, nullable=True)
    confidence = Column(Float, default=1.0)

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="entities")

    __table_args__ = (
        Index("ix_legal_entities_type", "entity_type"),
    )


class Summary(Base):
    __tablename__ = "summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    summary_type = Column(String(50), nullable=False)  # short, medium, detailed
    language = Column(String(50), default="en")
    content = Column(Text, nullable=False)
    simplified_content = Column(Text, nullable=True)
    model_used = Column(String(200), nullable=True)
    prompt_version = Column(String(50), nullable=True)
    metadata_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="summaries")

    __table_args__ = (
        Index("ix_summaries_doc_type_lang", "document_id", "summary_type", "language"),
    )


class Translation(Base):
    __tablename__ = "translations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    source_language = Column(String(50), nullable=False)
    target_language = Column(String(50), nullable=False)
    content_type = Column(String(50), nullable=False)  # summary_short, summary_medium, summary_detailed, simplification
    source_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    model_used = Column(String(200), nullable=True)
    terminology_preserved = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="translations")

    __table_args__ = (
        Index("ix_translations_doc_lang", "document_id", "target_language", "content_type"),
    )
