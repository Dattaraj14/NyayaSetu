"""
Shared embedding service.

The embedding model is loaded once per backend process and reused for
document indexing and chat retrieval.
"""

from __future__ import annotations

import threading
import time
from typing import Literal, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging_config import get_logger


logger = get_logger(__name__)


class LegalEmbeddingModel:
    """
    Thread-safe singleton wrapper around the configured SentenceTransformer model.

    Any part of the backend can safely call `LegalEmbeddingModel()` and receive
    the same loaded model instance.
    """

    _instance: "LegalEmbeddingModel | None" = None
    _instance_lock = threading.Lock()
    _encode_lock = threading.Lock()

    def __new__(cls) -> "LegalEmbeddingModel":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Resolve 'auto', 'cuda', or 'cpu' to a valid PyTorch device string."""
        normalized = (requested_device or "auto").strip().lower()

        if normalized == "cpu":
            return "cpu"

        if normalized in {"auto", "cuda"}:
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                if normalized == "cuda":
                    logger.warning("cuda_unavailable_using_cpu")
                return "cpu"
            except Exception:
                logger.warning("cuda_detection_failed_using_cpu")
                return "cpu"

        logger.warning(
            "unknown_embedding_device_using_cpu",
            requested_device=requested_device,
        )
        return "cpu"

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        with self._instance_lock:
            if getattr(self, "_initialized", False):
                return

            self.settings = get_settings()
            self.device = self._resolve_device(self.settings.embedding_device)

            logger.info(
                "loading_embedding_model",
                model=self.settings.embedding_model,
                device=self.device,
            )

            load_start = time.perf_counter()

            model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.device,
            )

            if self.settings.embedding_max_seq_length:
                model.max_seq_length = int(self.settings.embedding_max_seq_length)

            if self.device == "cuda":
                try:
                    model = model.half()
                except Exception as exc:
                    logger.warning("embedding_fp16_failed", error=str(exc))

            self.model = model
            self._dimension = int(self.model.get_embedding_dimension())
            self._initialized = True

            logger.info(
                "embedding_model_loaded",
                model=self.settings.embedding_model,
                device=self.device,
                dimension=self._dimension,
                elapsed_seconds=round(time.perf_counter() - load_start, 4),
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def _apply_prefix(
        self,
        texts: Sequence[str],
        kind: Literal["query", "passage"],
    ) -> list[str]:
        prefix = (
            self.settings.embedding_query_prefix
            if kind == "query"
            else self.settings.embedding_passage_prefix
        )
        if not prefix:
            return list(texts)
        return [f"{prefix}{text}" for text in texts]

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: Literal["query", "passage"] = "passage",
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode multiple texts with optional query/passage prefixes."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        batch = batch_size or self.settings.embedding_batch_size
        prepared_texts = self._apply_prefix(texts, kind)
        encode_start = time.perf_counter()

        logger.info(
            "embedding_generation_started",
            texts=len(prepared_texts),
            batch_size=batch,
            kind=kind,
            device=self.device,
        )

        with self._encode_lock:
            embeddings = self.model.encode(
                prepared_texts,
                batch_size=batch,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        logger.info(
            "embedding_generation_complete",
            texts=len(prepared_texts),
            kind=kind,
            device=self.device,
            dimension=embeddings.shape[1]
            if embeddings.ndim == 2
            else self.dimension,
            elapsed_seconds=round(time.perf_counter() - encode_start, 4),
        )

        return embeddings

    def encode_single(
        self,
        text: str,
        *,
        kind: Literal["query", "passage"] = "query",
    ) -> np.ndarray:
        """Encode a single query or passage string."""
        embeddings = self.encode(
            [text],
            kind=kind,
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
        """In-memory cosine similarity search over normalized embeddings."""
        if len(candidate_embeddings) == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        candidates = np.asarray(candidate_embeddings, dtype=np.float32)

        scores = candidates @ query
        top_k = min(max(top_k, 1), len(scores))
        indices = np.argsort(scores)[::-1][:top_k]

        return [(int(index), float(scores[index])) for index in indices]