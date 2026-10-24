"""
Indexer for the NyayaSetu general legal knowledge base.

Reads KnowledgeDocument + KnowledgeChunk records from SQLite,
generates BGE-M3 embeddings, and stores them in the shared
legal knowledge FAISS index.
"""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import select

from app.models.database import async_session_factory
from app.models.legal import (
    KnowledgeChunk,
    KnowledgeDocument,
)
from app.pipelines.rag.embeddings import LegalEmbeddingModel
from app.pipelines.rag.knowledge_vector_store import (
    KnowledgeVectorStore,
)
from app.core.logging_config import get_logger


logger = get_logger("knowledge_indexer")


async def load_knowledge_chunks() -> list[dict[str, Any]]:
    """
    Load all legal knowledge chunks together with their
    parent document metadata.
    """

    async with async_session_factory() as db:

        result = await db.execute(
            select(
                KnowledgeChunk,
                KnowledgeDocument,
            )
            .join(
                KnowledgeDocument,
                KnowledgeChunk.knowledge_document_id
                == KnowledgeDocument.id,
            )
            .order_by(
                KnowledgeDocument.title,
                KnowledgeChunk.chunk_index,
            )
        )

        rows = result.all()

    records: list[dict[str, Any]] = []

    for chunk, document in rows:

        records.append(
            {
                "chunk": chunk,
                "document": document,
            }
        )

    return records


def build_metadata(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build FAISS metadata for every KnowledgeChunk.
    """

    metadata: list[dict[str, Any]] = []

    for record in records:

        chunk: KnowledgeChunk = record["chunk"]
        document: KnowledgeDocument = record["document"]

        metadata.append(
            {
                "knowledge_chunk_id": str(
                    chunk.id
                ),
                "knowledge_document_id": str(
                    document.id
                ),
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "section": chunk.section,
                "language": chunk.language,

                # Provenance
                "title": document.title,
                "authority": document.authority,
                "source": document.source,
                "source_url": document.source_url,
                "document_type": document.document_type,
                "publication_date": (
                    document.publication_date.isoformat()
                    if document.publication_date
                    else None
                ),
                "effective_date": (
                    document.effective_date.isoformat()
                    if document.effective_date
                    else None
                ),
                "amendment_date": (
                    document.amendment_date.isoformat()
                    if document.amendment_date
                    else None
                ),
                "status": (
                    document.status.value
                    if document.status
                    else None
                ),
                "version": document.version,
                "retrieval_date": (
                    document.retrieval_date.isoformat()
                    if document.retrieval_date
                    else None
                ),
                "document_metadata": (
                    document.metadata_json or {}
                ),
            }
        )

    return metadata


async def save_vector_ids(
    records: list[dict[str, Any]],
) -> None:
    """
    Save the FAISS vector position back to each KnowledgeChunk.
    """

    async with async_session_factory() as db:

        for vector_id, record in enumerate(records):

            chunk: KnowledgeChunk = record["chunk"]

            result = await db.execute(
                select(KnowledgeChunk).where(
                    KnowledgeChunk.id == chunk.id
                )
            )

            db_chunk = result.scalar_one_or_none()

            if db_chunk is not None:
                db_chunk.vector_id = vector_id

        await db.commit()


async def build_knowledge_index() -> int:
    """
    Rebuild the complete shared legal knowledge index.

    Returns:
        Number of indexed knowledge chunks.
    """

    logger.info(
        "knowledge_indexing_started"
    )

    records = await load_knowledge_chunks()

    if not records:
        raise RuntimeError(
            "No legal knowledge chunks found in the database."
        )

    logger.info(
        "knowledge_chunks_loaded",
        chunks=len(records),
    )

    texts = [
        record["chunk"].content
        for record in records
    ]

    embedding_model = LegalEmbeddingModel()

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
    )

    logger.info(
        "knowledge_embeddings_generated",
        chunks=len(texts),
        dimension=embeddings.shape[1],
    )

    # Rebuild from scratch so the FAISS index and metadata
    # always correspond exactly to the database records.
    vector_store = KnowledgeVectorStore(
        dimension=embedding_model.dimension
    )

    metadata = build_metadata(records)

    vector_store.add(
        embeddings,
        metadata,
    )

    vector_store.save()

    await save_vector_ids(records)

    logger.info(
        "knowledge_indexing_complete",
        chunks=len(records),
        dimension=embedding_model.dimension,
    )

    return len(records)


async def main() -> None:
    count = await build_knowledge_index()

    print(
        f"Knowledge index built successfully: "
        f"{count} chunks"
    )


if __name__ == "__main__":
    asyncio.run(main())