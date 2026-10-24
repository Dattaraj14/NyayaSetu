"""Document vector-indexing route."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.document import (
    Document,
    DocumentStatus,
)
from app.pipelines.rag.indexer import index_document


router = APIRouter()


@router.post("/{document_id}/index")
async def build_document_index(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Build the BGE-M3 + FAISS index for a processed document."""

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
                f"Current status: {document.status.value}."
            ),
        )

    document.status = DocumentStatus.EMBEDDING
    document.status_message = (
        "Building semantic search index..."
    )
    document.processing_progress = 95.0

    await db.commit()

    try:
        chunks_indexed = await index_document(
            document_id
        )

    except Exception as exc:

        document.status = DocumentStatus.FAILED
        document.status_message = (
            f"Vector indexing failed: {exc}"
        )
        await db.commit()

        raise HTTPException(
            status_code=500,
            detail="Failed to build document search index.",
        ) from exc

    document.status = DocumentStatus.COMPLETED
    document.status_message = (
        f"Document ready. {chunks_indexed} chunks indexed."
    )
    document.processing_progress = 100.0

    await db.commit()

    return {
        "document_id": str(document_id),
        "chunks_indexed": chunks_indexed,
        "status": "completed",
        "message": (
            "Document semantic search index built successfully."
        ),
    }