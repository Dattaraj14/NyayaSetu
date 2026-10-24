"""FAISS vector store for legal document retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.logging_config import get_logger


logger = get_logger("vector_store")
settings = get_settings()


class LegalVectorStore:
    """
    FAISS-based vector store.

    Each document receives its own FAISS index so that chunks
    from different uploaded documents cannot accidentally mix
    during document-grounded retrieval.
    """

    def __init__(
        self,
        document_id: str,
        dimension: int,
    ) -> None:

        self.document_id = str(
            document_id
        )

        self.dimension = dimension

        self.base_path = Path(
            settings.vector_db_path
        )

        self.document_path = (
            self.base_path
            / self.document_id
        )

        self.document_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = (
            self.document_path
            / "index.faiss"
        )

        self.metadata_path = (
            self.document_path
            / "metadata.json"
        )

        # --------------------------------------------------------------
        # Inner-product similarity.
        #
        # Embeddings are normalized, so this corresponds to
        # cosine similarity.
        # --------------------------------------------------------------

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
        """Add embeddings and corresponding metadata."""

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

        self.index.add(
            vectors
        )

        self.metadata.extend(
            metadata
        )

        logger.info(
            "vectors_added",
            document_id=self.document_id,
            vectors=len(vectors),
            total_vectors=self.index.ntotal,
        )

    # ==================================================================
    # SAVE
    # ==================================================================

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""

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
            )

        logger.info(
            "vector_store_saved",
            document_id=self.document_id,
            vectors=self.index.ntotal,
            path=str(self.document_path),
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
        Search the FAISS index.

        Returns metadata together with similarity scores.
        """

        if self.index.ntotal == 0:
            return []

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(
                1,
                -1,
            )

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: "
                f"expected {self.dimension}, "
                f"received {query.shape[1]}."
            )

        top_k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            item = dict(
                self.metadata[index]
            )

            item["score"] = float(
                score
            )

            results.append(
                item
            )

        return results

    # ==================================================================
    # LOAD
    # ==================================================================

    @classmethod
    def load(
        cls,
        document_id: str,
    ) -> "LegalVectorStore":
        """Load an existing document vector store."""

        document_id = str(
            document_id
        )

        base_path = Path(
            settings.vector_db_path
        )

        document_path = (
            base_path
            / document_id
        )

        index_path = (
            document_path
            / "index.faiss"
        )

        metadata_path = (
            document_path
            / "metadata.json"
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found for "
                f"document {document_id}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector metadata not found for "
                f"document {document_id}"
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
            document_id=document_id,
            dimension=index.d,
        )

        store.index = index
        store.metadata = metadata

        logger.info(
            "vector_store_loaded",
            document_id=document_id,
            vectors=index.ntotal,
        )

        return store

    # ==================================================================
    # DELETE
    # ==================================================================

    def delete(self) -> None:
        """Delete the persisted vector store."""

        if self.document_path.exists():

            for path in self.document_path.iterdir():

                if path.is_file():
                    path.unlink()

            self.document_path.rmdir()

        logger.info(
            "vector_store_deleted",
            document_id=self.document_id,
        )