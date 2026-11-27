"""FAISS vector store for legal document retrieval with in-memory LRU caching."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import threading
import time

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.logging_config import get_logger


logger = get_logger("vector_store")
settings = get_settings()


class LegalVectorStore:
    """
    FAISS-based vector store.

    Each document receives its own FAISS index so chunks from different
    documents never mix. Loaded indexes are kept in a bounded LRU cache
    and invalidated automatically on re-index or delete.
    """

    _cache: "OrderedDict[str, LegalVectorStore]" = OrderedDict()
    _cache_lock = threading.Lock()

    def __init__(
        self,
        document_id: str,
        dimension: int,
    ) -> None:
        self.document_id = str(document_id)
        self.dimension = dimension

        self.base_path = Path(settings.vector_db_path)
        self.document_path = self.base_path / self.document_id
        self.document_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.document_path / "index.faiss"
        self.metadata_path = self.document_path / "metadata.json"

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: list[dict] = []

    @classmethod
    def invalidate_cache(cls, document_id: str) -> None:
        """Remove a document's vector store from the in-memory cache."""
        with cls._cache_lock:
            cls._cache.pop(str(document_id), None)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached vector stores."""
        with cls._cache_lock:
            cls._cache.clear()

    @classmethod
    def _cache_put(cls, document_id: str, store: "LegalVectorStore") -> None:
        max_size = max(1, settings.vector_store_cache_size)
        key = str(document_id)
        with cls._cache_lock:
            if key in cls._cache:
                cls._cache.move_to_end(key)
            cls._cache[key] = store
            while len(cls._cache) > max_size:
                cls._cache.popitem(last=False)

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """Add embeddings and corresponding metadata."""
        if len(embeddings) != len(metadata):
            raise ValueError(
                "Embeddings and metadata must contain the same number of items."
            )
        if embeddings.size == 0:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"received {vectors.shape[1]}."
            )

        self.index.add(vectors)
        self.metadata.extend(metadata)

        logger.info(
            "vectors_added",
            document_id=self.document_id,
            vectors=len(vectors),
            total_vectors=self.index.ntotal,
        )

    def save(self) -> None:
        """Persist FAISS index and metadata to disk and refresh cache."""
        save_start = time.perf_counter()

        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(self.metadata, metadata_file, ensure_ascii=False, indent=2)

        self.invalidate_cache(self.document_id)
        self._cache_put(self.document_id, self)

        logger.info(
            "vector_store_saved",
            document_id=self.document_id,
            vectors=self.index.ntotal,
            path=str(self.document_path),
            elapsed_seconds=round(time.perf_counter() - save_start, 4),
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """Search the FAISS index and return metadata with similarity scores."""
        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"received {query.shape[1]}."
            )

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            item = dict(self.metadata[index])
            item["score"] = float(score)
            results.append(item)

        return results

    @classmethod
    def load(
        cls,
        document_id: str,
        *,
        use_cache: bool = True,
    ) -> "LegalVectorStore":
        """Load an existing document vector store (from LRU cache if available)."""
        document_id = str(document_id)

        if use_cache:
            with cls._cache_lock:
                cached = cls._cache.get(document_id)
                if cached is not None:
                    cls._cache.move_to_end(document_id)
                    logger.info(
                        "vector_store_cache_hit",
                        document_id=document_id,
                        vectors=cached.index.ntotal,
                    )
                    return cached

        base_path = Path(settings.vector_db_path)
        document_path = base_path / document_id
        index_path = document_path / "index.faiss"
        metadata_path = document_path / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found for document {document_id}"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector metadata not found for document {document_id}"
            )

        load_start = time.perf_counter()

        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        store = cls(document_id=document_id, dimension=index.d)
        store.index = index
        store.metadata = metadata

        if use_cache:
            cls._cache_put(document_id, store)

        logger.info(
            "vector_store_loaded",
            document_id=document_id,
            vectors=index.ntotal,
            elapsed_seconds=round(time.perf_counter() - load_start, 4),
        )
        return store

    def delete(self) -> None:
        """Delete the persisted vector store and evict from cache."""
        self.invalidate_cache(self.document_id)

        if self.document_path.exists():
            for path in self.document_path.iterdir():
                if path.is_file():
                    path.unlink()
            self.document_path.rmdir()

        logger.info(
            "vector_store_deleted",
            document_id=self.document_id,
        )