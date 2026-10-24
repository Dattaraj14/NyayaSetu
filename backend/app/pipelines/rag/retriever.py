"""Semantic retrieval over document-specific FAISS indexes."""

from __future__ import annotations

from typing import Any

from app.core.logging_config import get_logger
from app.pipelines.rag.embeddings import LegalEmbeddingModel
from app.pipelines.rag.vector_store import LegalVectorStore

logger = get_logger("retriever")


class LegalRetriever:
    """Retrieve relevant legal chunks for a user query."""

    def __init__(self) -> None:
        self.embedding_model = LegalEmbeddingModel()

    def retrieve(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.
        """

        if not query.strip():
            return []

        logger.info(
            "retrieval_started",
            document_id=document_id,
            query=query,
            top_k=top_k,
        )

        # Load the already-built document-specific FAISS index.
        vector_store = LegalVectorStore.load(
            document_id
        )

        # Convert user query into a normalized BGE-M3 embedding.
        query_embedding = self.embedding_model.encode_single(
            query
        )

        # Search FAISS.
        results = vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        logger.info(
            "retrieval_complete",
            document_id=document_id,
            results=len(results),
        )

        return results