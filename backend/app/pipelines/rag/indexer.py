"""Build FAISS indexes from stored legal document chunks."""

from __future__ import annotations

import ast
import asyncio
import json
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.core.logging_config import get_logger
from app.models.database import async_session_factory
from app.models.document import DocumentChunk
from app.pipelines.rag.embeddings import LegalEmbeddingModel
from app.pipelines.rag.vector_store import LegalVectorStore


logger = get_logger("rag_indexer")


def parse_chunk_metadata(value: Any) -> dict[str, Any]:
    """
    Safely normalize chunk metadata into a dictionary.

    Supports:
    - dictionaries
    - JSON strings
    - Python-literal dictionary strings
    - empty/invalid values
    """

    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return {}

        # Try valid JSON first.
        try:
            parsed = json.loads(value)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # Some older records may contain Python-style dict strings.
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, dict):
                return parsed

        except (ValueError, SyntaxError):
            pass

    return {}


async def index_document(document_id: UUID) -> int:
    """Generate embeddings and build a FAISS index for one document."""

    document_id_str = str(document_id)

    logger.info(
        "document_indexing_started",
        document_id=document_id_str,
    )

    # ------------------------------------------------------------------
    # Load chunks from SQLite
    # ------------------------------------------------------------------

    async with async_session_factory() as db:
        result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )

        chunks = result.scalars().all()

    if not chunks:
        logger.warning(
            "no_chunks_found_for_indexing",
            document_id=document_id_str,
        )
        return 0

    logger.info(
        "document_chunks_loaded",
        document_id=document_id_str,
        chunks=len(chunks),
    )

    # ------------------------------------------------------------------
    # Generate embeddings
    # ------------------------------------------------------------------

    embedding_model = LegalEmbeddingModel()

    texts = [chunk.content for chunk in chunks]

    embeddings = await asyncio.to_thread(
        embedding_model.encode,
        texts,
    )

    logger.info(
        "document_embeddings_generated",
        document_id=document_id_str,
        chunks=len(chunks),
        dimension=embeddings.shape[1],
    )

    # ------------------------------------------------------------------
    # Build metadata for FAISS
    # ------------------------------------------------------------------

    metadata: list[dict[str, Any]] = []

    for chunk in chunks:

        chunk_metadata: dict[str, Any] = {
            "chunk_id": str(chunk.id),
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "page_number": chunk.page_number,
            "section_type": chunk.section_type,
            "language": chunk.language,
        }

        stored_metadata = parse_chunk_metadata(
            chunk.metadata_json
        )

        chunk_metadata.update(stored_metadata)

        metadata.append(chunk_metadata)

    # ------------------------------------------------------------------
    # Create document-specific FAISS index
    # ------------------------------------------------------------------

    vector_store = LegalVectorStore(
        document_id=document_id_str,
        dimension=embedding_model.dimension,
    )

    vector_store.add(
        embeddings,
        metadata,
    )

    await asyncio.to_thread(
        vector_store.save
    )

    # ------------------------------------------------------------------
    # Save FAISS vector IDs back to SQLite
    # ------------------------------------------------------------------

    async with async_session_factory() as db:

        result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )

        db_chunks = result.scalars().all()

        for vector_id, chunk in enumerate(db_chunks):
            chunk.vector_id = vector_id

        await db.commit()

    logger.info(
        "document_indexing_complete",
        document_id=document_id_str,
        chunks=len(chunks),
        vectors=vector_store.index.ntotal,
    )

    return len(chunks)


async def main() -> None:
    """Index the newest document that contains legal chunks."""

    async with async_session_factory() as db:

        result = await db.execute(
            select(DocumentChunk.document_id)
            .order_by(DocumentChunk.created_at.desc())
            .limit(1)
        )

        document_id = result.scalar_one_or_none()

    if document_id is None:
        print("No document chunks found.")
        return

    print(f"Document: {document_id}")
    print("Starting real document indexing...")
    print()

    count = await index_document(
        document_id
    )

    print()
    print(f"Chunks indexed: {count}")
    print("REAL DOCUMENT INDEXING COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())