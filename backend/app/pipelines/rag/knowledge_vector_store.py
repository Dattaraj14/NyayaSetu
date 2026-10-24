"""
FAISS vector store for the NyayaSetu legal knowledge base.

Unlike document-grounded RAG, which creates one index per uploaded
document, the general legal knowledge base uses one shared FAISS index.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.logging_config import get_logger


logger = get_logger("knowledge_vector_store")
settings = get_settings()


class KnowledgeVectorStore:
    """
    Shared FAISS vector store for authoritative legal knowledge.

    The index stores KnowledgeChunk embeddings along with metadata
    identifying the source legal document, authority, version,
    effective date, and other provenance information.
    """

    INDEX_DIRECTORY = "legal_knowledge"

    def __init__(
        self,
        dimension: int,
    ) -> None:

        self.dimension = dimension

        self.base_path = (
            Path(settings.vector_db_path)
            / self.INDEX_DIRECTORY
        )

        self.base_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = (
            self.base_path / "index.faiss"
        )

        self.metadata_path = (
            self.base_path / "metadata.json"
        )

        # Normalized embeddings + inner product =
        # cosine similarity.
        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.metadata: list[dict] = []

    # ==================================================================
    # ADD
    # ==================================================================

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """Add knowledge embeddings and their metadata."""

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Embeddings and metadata must contain "
                "the same number of items."
            )

        if embeddings.size == 0:
            return

        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.dimension}, "
                f"received {vectors.shape[1]}."
            )

        self.index.add(vectors)

        self.metadata.extend(metadata)

        logger.info(
            "knowledge_vectors_added",
            vectors=len(vectors),
            total_vectors=self.index.ntotal,
        )

    # ==================================================================
    # SAVE
    # ==================================================================

    def save(self) -> None:
        """Persist the FAISS index and metadata."""

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as metadata_file:

            json.dump(
                self.metadata,
                metadata_file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        logger.info(
            "knowledge_vector_store_saved",
            vectors=self.index.ntotal,
            path=str(self.base_path),
        )

    # ==================================================================
    # SEARCH
    # ==================================================================

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search the shared legal knowledge index.
        """

        if self.index.ntotal == 0:
            return []

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: "
                f"expected {self.dimension}, "
                f"received {query.shape[1]}."
            )

        top_k = min(
            max(top_k, 1),
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query,
            top_k,
        )

        results: list[dict] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            item = dict(
                self.metadata[index]
            )

            item["score"] = float(score)

            results.append(item)

        return results

    # ==================================================================
    # LOAD
    # ==================================================================

    @classmethod
    def load(cls) -> "KnowledgeVectorStore":
        """Load the persisted legal knowledge index."""

        index_path = (
            Path(settings.vector_db_path)
            / cls.INDEX_DIRECTORY
            / "index.faiss"
        )

        metadata_path = (
            Path(settings.vector_db_path)
            / cls.INDEX_DIRECTORY
            / "metadata.json"
        )

        if not index_path.exists():
            raise FileNotFoundError(
                "Legal knowledge FAISS index was not found."
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                "Legal knowledge vector metadata was not found."
            )

        index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as metadata_file:

            metadata = json.load(
                metadata_file
            )

        store = cls(
            dimension=index.d,
        )

        store.index = index
        store.metadata = metadata

        logger.info(
            "knowledge_vector_store_loaded",
            vectors=index.ntotal,
        )

        return store

    # ==================================================================
    # DELETE
    # ==================================================================

    def delete(self) -> None:
        """Delete the complete legal knowledge vector store."""

        if not self.base_path.exists():
            return

        for path in self.base_path.iterdir():

            if path.is_file():
                path.unlink()

        self.base_path.rmdir()

        logger.info(
            "knowledge_vector_store_deleted"
        )