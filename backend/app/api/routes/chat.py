"""Chat routes for NyayaSetu."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.document import Document, DocumentStatus
from app.pipelines.rag.chat_service import get_chat_service
from app.pipelines.rag.llm_service import (
    GeminiServiceUnavailableError,
)
from app.schemas.document import (
    ChatRequest,
    ChatResponse,
    SourceReference,
)


router = APIRouter()


@router.post(
    "/{document_id}",
    response_model=ChatResponse,
)
async def document_chat_interaction(
    document_id: UUID,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:

    # ---------------------------------------------------------
    # General legal mode is intentionally blocked here.
    #
    # It will later use a separate authoritative legal-knowledge
    # pipeline instead of the uploaded-document RAG pipeline.
    # ---------------------------------------------------------
    if request.mode == "general_legal":

        raise HTTPException(
            status_code=501,
            detail=(
                "General legal information mode is not "
                "implemented yet. Please use document_qa."
            ),
        )

    # ---------------------------------------------------------
    # DOCUMENT QA MODE
    # ---------------------------------------------------------

    result = await db.execute(
        select(Document).where(
            Document.id == document_id
        )
    )

    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if document.status != DocumentStatus.COMPLETED:

        raise HTTPException(
            status_code=409,
            detail=(
                "Document processing is not complete. "
                f"Current status: "
                f"{document.status.value}."
            ),
        )

    service = get_chat_service()

    try:

        result_data = await service.answer(
            document_id=str(document_id),
            question=request.message,
            language=request.language,
            top_k=5,
        )

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail=(
                "Vector index for this document "
                "was not found."
            ),
        )

    except GeminiServiceUnavailableError:

        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is temporarily unavailable. "
                "Please retry in a moment."
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while "
                "generating the legal-document response."
            ),
        ) from exc

    # ---------------------------------------------------------
    # Convert retrieved sources to API response format.
    # ---------------------------------------------------------

    source_references: list[SourceReference] = []

    for source in result_data["sources"]:

        paragraph_value = source.get(
            "paragraph_numbers"
        )

        if isinstance(
            paragraph_value,
            list,
        ):

            paragraph_text = ", ".join(
                str(value)
                for value in paragraph_value
            )

        elif paragraph_value is None:

            paragraph_text = None

        else:

            paragraph_text = str(
                paragraph_value
            )

        source_references.append(
            SourceReference(
                document_name=document.filename,
                page=source.get(
                    "page_number"
                ),
                section=(
                    source.get("section_title")
                    or source.get("section_type")
                ),
                paragraph=paragraph_text,
                content=source.get(
                    "content"
                ),
                knowledge_source=(
                    "Uploaded legal document"
                ),
            )
        )

    return ChatResponse(
        session_id=(
            request.session_id
            or uuid4()
        ),
        message=result_data["answer"],
        sources=source_references,
        confidence=None,
        was_refused=result_data[
            "was_refused"
        ],
        refusal_reason=result_data[
            "refusal_reason"
        ],
        mode="document_qa",
    )