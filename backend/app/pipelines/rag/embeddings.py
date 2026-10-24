"""
Shared BGE-M3 embedding service.

The embedding model is loaded only once per backend process and
reused for document indexing and chat retrieval.
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging_config import get_logger


logger = get_logger(__name__)
settings = get_settings()


class LegalEmbeddingModel:
    """
    Singleton wrapper around BAAI/bge-m3.

    Any part of the backend can safely do:

        LegalEmbeddingModel()

    and receive the same loaded model instance.
    """

    _instance: "LegalEmbeddingModel | None" = None
    _instance_lock = threading.Lock()
    _encode_lock = threading.Lock()

    def __new__(
        cls,
    ) -> "LegalEmbeddingModel":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True

        self.settings = settings

        requested_device = (
            self.settings.embedding_device
        )

        device = requested_device

        if requested_device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"

                    logger.warning(
                        "cuda_unavailable_using_cpu"
                    )

            except Exception:
                device = "cpu"

                logger.warning(
                    "cuda_detection_failed_using_cpu"
                )

        self.device = device

        logger.info(
            "loading_embedding_model",
            model=self.settings.embedding_model,
            device=self.device,
        )

        self.model = SentenceTransformer(
            self.settings.embedding_model,
            device=self.device,
        )

        self._dimension = (
            self.model.get_embedding_dimension()
        )

        logger.info(
            "embedding_model_loaded",
            model=self.settings.embedding_model,
            device=self.device,
            dimension=self._dimension,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Encode multiple texts.

        A small lock prevents concurrent calls from trying to use
        the same SentenceTransformer instance simultaneously.
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        batch = (
            batch_size
            or self.settings.embedding_batch_size
        )

        logger.info(
            "embedding_generation_started",
            texts=len(texts),
            batch_size=batch,
        )

        with self._encode_lock:
            embeddings = self.model.encode(
                list(texts),
                batch_size=batch,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        logger.info(
            "embedding_generation_complete",
            texts=len(texts),
            dimension=embeddings.shape[1]
            if embeddings.ndim == 2
            else self.dimension,
        )

        return embeddings

    def encode_single(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Encode a single query/document string.
        """

        embeddings = self.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
        )

        return embeddings[0]

    def search(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        In-memory cosine similarity search.

        Embeddings are expected to be normalized.
        """

        if len(candidate_embeddings) == 0:
            return []

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        candidates = np.asarray(
            candidate_embeddings,
            dtype=np.float32,
        )

        scores = candidates @ query

        top_k = min(
            max(top_k, 1),
            len(scores),
        )

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            (
                int(index),
                float(scores[index]),
            )
            for index in indices
        ]