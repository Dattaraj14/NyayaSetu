"""Structured document summarization routes powered by LangGraph."""

from __future__ import annotations

import json
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.summary_graph import run_summary_graph
from app.core.logging_config import get_logger
from app.models.database import get_db
from app.models.document import (
    Document,
    DocumentChunk,
    DocumentStatus,
    LegalEntity,
    Summary,
)
from app.pipelines.rag.llm_service import GeminiServiceUnavailableError


logger = get_logger("summary_routes")
router = APIRouter()


class StructuredSummarizeRequest(BaseModel):
    summary_type: Literal["short", "medium", "detailed"] = "medium"
    language: Literal["en", "hi", "mr"] = "en"
    force_refresh: bool = Field(
        default=False,
        description="Regenerate even if a matching summary already exists.",
    )


def _serialize_summary(summary: Summary) -> dict[str, Any]:
    meta = summary.metadata_json if isinstance(summary.metadata_json, dict) else {}
    return {
        "id": str(summary.id),
        "document_id": str(summary.document_id),
        "summary_type": summary.summary_type,
        "language": summary.language,
        "document_type": meta.get("document_type", "other"),
        "content": summary.content,
        "structured_summary": meta.get("structured_summary", {}),
        "claims": meta.get("claims", []),
        "verification_status": meta.get("verification_status", "verified"),
        "trace": meta.get("trace", []),
        "model_used": summary.model_used,
        "created_at": summary.created_at,
    }


@router.post("/{document_id}/summarize")
async def summarize_document(
    document_id: UUID,
    request: StructuredSummarizeRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Generate and persist a verified structured legal summary using LangGraph."""
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = doc_result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    if document.status != DocumentStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=(
                "Document processing is not complete. "
                f"Current status: {document.status.value}."
            ),
        )

    # Return cached summary unless force_refresh is requested.
    if not request.force_refresh:
        existing_res = await db.execute(
            select(Summary)
            .where(
                Summary.document_id == document_id,
                Summary.summary_type == request.summary_type,
                Summary.language == request.language,
            )
            .order_by(Summary.created_at.desc())
            .limit(1)
        )
        existing = existing_res.scalar_one_or_none()
        if existing is not None:
            return _serialize_summary(existing)

    # Load document chunks in order.
    chunks_res = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    db_chunks = chunks_res.scalars().all()
    if not db_chunks:
        raise HTTPException(
            status_code=409,
            detail="No processed chunks found for this document.",
        )

    chunks_payload: list[dict[str, Any]] = []
    for chunk in db_chunks:
        meta = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
        chunks_payload.append(
            {
                "chunk_id": str(chunk.id),
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "page_number": chunk.page_number,
                "section_type": chunk.section_type,
                "section_title": meta.get("section_title", ""),
            }
        )

    try:
        result = await run_summary_graph(
            document_id=str(document_id),
            chunks=chunks_payload,
            summary_type=request.summary_type,
            language=request.language,
        )
    except GeminiServiceUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Gemini is temporarily unavailable. Please retry in a moment.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "document_summarization_failed",
            document_id=str(document_id),
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate document summary.",
        ) from exc

    # Persist summary into existing summaries table.
    summary_record = Summary(
        document_id=document_id,
        summary_type=request.summary_type,
        language=request.language,
        content=result["markdown_summary"],
        model_used=result.get("model_used"),
        prompt_version="langgraph-summary-v1",
        metadata_json={
            "document_type": result["document_type"],
            "structured_summary": result["structured_summary"],
            "claims": result["claims"],
            "verification_status": result["verification_status"],
            "trace": result["trace"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
    )
    db.add(summary_record)

    # Update document metadata and extracted legal entities.
    doc_type = result["document_type"]
    structured = result["structured_summary"]
    document.document_type = doc_type
    if doc_type == "judgment":
        court = str(structured.get("court") or "").strip()
        case_number = str(structured.get("case_number") or "").strip()
        if court and court.lower() != "not specified":
            document.court = court[:500]
        if case_number and case_number.lower() != "not specified":
            document.case_number = case_number[:200]

    extracted_entities = result.get("extracted_entities") or []
    if extracted_entities:
        await db.execute(
            delete(LegalEntity).where(LegalEntity.document_id == document_id)
        )
        for ent in extracted_entities:
            db.add(
                LegalEntity(
                    document_id=document_id,
                    entity_type=ent["entity_type"],
                    entity_value=ent["entity_value"],
                    confidence=1.0,
                )
            )

    await db.commit()
    await db.refresh(summary_record)

    return _serialize_summary(summary_record)


@router.get("/{document_id}/summaries")
async def list_document_summaries(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Fetch all saved summaries for a document."""
    result = await db.execute(
        select(Summary)
        .where(Summary.document_id == document_id)
        .order_by(Summary.created_at.desc())
    )
    summaries = result.scalars().all()
    return [_serialize_summary(s) for s in summaries]


@router.get("/{document_id}/summaries/export")
async def export_document_summary(
    document_id: UUID,
    summary_type: Literal["short", "medium", "detailed"] = Query(default="medium"),
    language: Literal["en", "hi", "mr"] = Query(default="en"),
    export_format: Literal["md", "json"] = Query(default="md"),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download a saved document summary as Markdown or JSON."""
    result = await db.execute(
        select(Summary)
        .where(
            Summary.document_id == document_id,
            Summary.summary_type == summary_type,
            Summary.language == language,
        )
        .order_by(Summary.created_at.desc())
        .limit(1)
    )
    summary = result.scalar_one_or_none()
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="Summary not found. Generate it first via POST /{document_id}/summarize.",
        )

    serialized = _serialize_summary(summary)
    filename_base = f"nyayasetu-summary-{document_id}-{summary_type}-{language}"

    if export_format == "json":
        body = json.dumps(serialized, ensure_ascii=False, indent=2, default=str)
        return Response(
            content=body,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename_base}.json"'
            },
        )

    md_content = (
        f"# NyayaSetu Legal Summary ({summary_type.title()} | {language.upper()})\n\n"
        f"**Document Type:** {serialized['document_type']}  \n"
        f"**Verification Status:** {serialized['verification_status']}\n\n"
        f"{serialized['content']}\n\n"
        "---\n"
        "*Disclaimer: Generated for informational purposes only and is not legal advice.*\n"
    )
    return Response(
        content=md_content,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename_base}.md"'
        },
    )