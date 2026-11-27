"""Chat routes (standard + SSE streaming) powered by LangGraph."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.models.database import get_db
from app.models.document import Document, DocumentStatus
from app.pipelines.rag.chat_service import get_chat_service
from app.pipelines.rag.llm_service import GeminiServiceUnavailableError
from app.schemas.document import (
    ChatRequest,
    ChatResponse,
    SourceReference,
    VerifiedClaim,
)


logger = get_logger("chat_routes")
router = APIRouter()


def _format_sources(
    document: Document,
    raw_sources: list[dict[str, Any]],
) -> list[SourceReference]:
    source_references: list[SourceReference] = []
    for source in raw_sources:
        paragraph_value = source.get("paragraph_numbers")
        if isinstance(paragraph_value, list):
            paragraph_text = ", ".join(str(value) for value in paragraph_value)
        elif paragraph_value is None:
            paragraph_text = None
        else:
            paragraph_text = str(paragraph_value)

        source_references.append(
            SourceReference(
                document_name=document.original_filename or document.filename,
                page=source.get("page_number"),
                section=source.get("section_title") or source.get("section_type"),
                paragraph=paragraph_text,
                content=source.get("content"),
                knowledge_source="Uploaded legal document",
            )
        )
    return source_references


async def _validate_document_for_chat(
    document_id: UUID,
    request: ChatRequest,
    db: AsyncSession,
) -> Document:
    if request.mode == "general_legal":
        raise HTTPException(
            status_code=501,
            detail=(
                "General legal information mode is not implemented yet. "
                "Please use document_qa."
            ),
        )

    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

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

    return document


@router.post(
    "/{document_id}",
    response_model=ChatResponse,
)
async def document_chat_interaction(
    document_id: UUID,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Run stateful LangGraph QA over an uploaded legal document."""
    document = await _validate_document_for_chat(document_id, request, db)
    session_uuid = request.session_id or uuid4()
    service = get_chat_service()

    try:
        result_data = await service.answer(
            document_id=str(document_id),
            question=request.message,
            language=request.language,
            top_k=5,
            session_id=str(session_uuid),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Vector index for this document was not found. Please index it first.",
        ) from exc
    except GeminiServiceUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Gemini is temporarily unavailable. Please retry in a moment.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "chat_endpoint_failed",
            document_id=str(document_id),
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the legal-document response.",
        ) from exc

    sources = _format_sources(document, result_data.get("sources") or [])
    claims = [VerifiedClaim(**c) for c in (result_data.get("claims") or [])]

    return ChatResponse(
        session_id=session_uuid,
        message=result_data.get("answer", ""),
        sources=sources,
        confidence=None,
        was_refused=bool(result_data.get("was_refused", False)),
        refusal_reason=result_data.get("refusal_reason"),
        mode="document_qa",
        intent=result_data.get("intent"),
        verification_status=result_data.get("verification_status"),
        claims=claims,
        trace=result_data.get("trace") or [],
    )


@router.post("/{document_id}/stream")
async def document_chat_stream(
    document_id: UUID,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint streaming LangGraph node progress
    (`routing`, `retrieving`, `grading`, `generating`, `verifying`)
    followed by answer chunks and the final structured response.
    """
    document = await _validate_document_for_chat(document_id, request, db)
    session_uuid = request.session_id or uuid4()
    service = get_chat_service()

    async def event_generator():
        try:
            async for event in service.answer_stream(
                document_id=str(document_id),
                question=request.message,
                language=request.language,
                top_k=5,
                session_id=str(session_uuid),
            ):
                if event.get("type") == "final":
                    sources = _format_sources(document, event.get("sources") or [])
                    payload = {
                        "type": "final",
                        "session_id": str(session_uuid),
                        "message": event.get("answer", ""),
                        "sources": [s.model_dump() for s in sources],
                        "was_refused": bool(event.get("was_refused", False)),
                        "refusal_reason": event.get("refusal_reason"),
                        "mode": "document_qa",
                        "intent": event.get("intent"),
                        "verification_status": event.get("verification_status"),
                        "claims": event.get("claims") or [],
                        "trace": event.get("trace") or [],
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except FileNotFoundError:
            err = {
                "type": "error",
                "detail": "Vector index for this document was not found.",
            }
            yield f"data: {json.dumps(err)}\n\n"
        except GeminiServiceUnavailableError:
            err = {
                "type": "error",
                "detail": "Gemini is temporarily unavailable. Please retry.",
            }
            yield f"data: {json.dumps(err)}\n\n"
        except Exception as exc:
            logger.exception(
                "chat_stream_failed",
                document_id=str(document_id),
                error=str(exc),
            )
            err = {
                "type": "error",
                "detail": "An unexpected error occurred during streaming.",
            }
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )