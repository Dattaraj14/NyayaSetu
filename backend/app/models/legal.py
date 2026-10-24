"""Legal knowledge and chat database models."""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, Boolean,
    ForeignKey, Enum, JSON, Index, Uuid as UUID
)
from sqlalchemy.orm import relationship
from app.models.database import Base


class LegalTermCategory(str, PyEnum):
    PROCEDURE = "procedure"
    CONSTITUTIONAL = "constitutional"
    CRIMINAL = "criminal"
    CIVIL = "civil"
    ADMINISTRATIVE = "administrative"
    GENERAL = "general"


class LegalTerm(Base):
    """Legal terminology registry with multilingual support."""
    __tablename__ = "legal_terms"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    term_en = Column(String(500), nullable=False, unique=True, index=True)
    term_hi = Column(String(500), nullable=True)
    term_mr = Column(String(500), nullable=True)
    explanation_en = Column(Text, nullable=True)
    explanation_hi = Column(Text, nullable=True)
    explanation_mr = Column(Text, nullable=True)
    category = Column(Enum(LegalTermCategory), default=LegalTermCategory.GENERAL)
    preserve_in_translation = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeDocumentStatus(str, PyEnum):
    CURRENT = "current"
    AMENDED = "amended"
    REPEALED = "repealed"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class KnowledgeDocument(Base):
    """Authoritative legal knowledge documents."""
    __tablename__ = "knowledge_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(1000), nullable=False)
    authority = Column(String(500), nullable=True)
    source = Column(String(500), nullable=True)
    source_url = Column(String(2000), nullable=True)
    document_type = Column(String(100), nullable=False)  # constitution, act, rule, regulation, notification
    language = Column(String(50), default="en")
    publication_date = Column(DateTime, nullable=True)
    effective_date = Column(DateTime, nullable=True)
    amendment_date = Column(DateTime, nullable=True)
    status = Column(Enum(KnowledgeDocumentStatus), default=KnowledgeDocumentStatus.CURRENT)
    version = Column(Integer, default=1)
    retrieval_date = Column(DateTime, default=datetime.utcnow)
    content = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("KnowledgeChunk", back_populates="knowledge_document", cascade="all, delete-orphan")


class KnowledgeChunk(Base):
    """Chunked legal knowledge for vector retrieval."""
    __tablename__ = "knowledge_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    knowledge_document_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    section = Column(String(500), nullable=True)
    language = Column(String(50), default="en")
    metadata_json = Column(JSON, nullable=True)
    vector_id = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    knowledge_document = relationship("KnowledgeDocument", back_populates="chunks")


class ChatMode(str, PyEnum):
    DOCUMENT_QA = "document_qa"
    GENERAL_LEGAL = "general_legal"


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True)
    mode = Column(Enum(ChatMode), nullable=False)
    title = Column(String(500), nullable=True)
    language = Column(String(50), default="en")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    language = Column(String(50), default="en")

    # Evidence/Sources
    sources = Column(JSON, nullable=True)  # List of source references
    confidence = Column(Float, nullable=True)
    was_refused = Column(Boolean, default=False)
    refusal_reason = Column(String(500), nullable=True)

    # Metadata
    model_used = Column(String(200), nullable=True)
    retrieval_count = Column(Integer, default=0)
    latency_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class ProcessingLog(Base):
    """Processing logs for audit and debugging."""
    __tablename__ = "processing_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True)
    stage = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)  # started, completed, failed
    duration_ms = Column(Integer, nullable=True)
    model = Column(String(200), nullable=True)
    error_message = Column(Text, nullable=True)
    details = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="processing_logs")


class EvaluationRecord(Base):
    """Evaluation results storage."""
    __tablename__ = "evaluation_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    evaluation_type = Column(String(100), nullable=False)  # summarization, simplification, translation, retrieval, chatbot
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    details = Column(JSON, nullable=True)
    model_version = Column(String(100), nullable=True)
    prompt_version = Column(String(50), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
