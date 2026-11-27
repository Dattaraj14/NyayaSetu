"""Hybrid legal retrieval: Dense FAISS + BM25 lexical search with Reciprocal Rank Fusion."""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from app.core.logging_config import get_logger
from app.pipelines.rag.embeddings import LegalEmbeddingModel
from app.pipelines.rag.vector_store import LegalVectorStore

logger = get_logger("retriever")

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _tokenize_multilingual(text: str) -> list[str]:
    """Tokenize English, Hindi, Marathi, and legal section numbers cleanly."""
    return _TOKEN_RE.findall(text.lower())


class LegalRetriever:
    """Retrieve relevant legal chunks using dense + BM25 hybrid RRF search."""

    _bm25_cache: dict[tuple[str, int], tuple[Optional[BM25Okapi], list[dict[str, Any]]]] = {}
    _bm25_lock = threading.Lock()

    def __init__(self) -> None:
        self.embedding_model = LegalEmbeddingModel()

    @classmethod
    def _get_bm25_index(
        cls,
        document_id: str,
        vector_store: LegalVectorStore,
    ) -> tuple[Optional[BM25Okapi], list[dict[str, Any]]]:
        """
        Build or return a cached BM25 index tied to the current LegalVectorStore instance.
        If the document is re-indexed and reloaded, `id(vector_store)` changes and
        BM25 rebuilds automatically.
        """
        cache_key = (str(document_id), id(vector_store))
        with cls._bm25_lock:
            cached = cls._bm25_cache.get(cache_key)
            if cached is not None:
                return cached

            metadata = list(vector_store.metadata)
            tokenized_corpus = [
                _tokenize_multilingual(str(item.get("content", "")))
                for item in metadata
            ]

            bm25 = (
                BM25Okapi(tokenized_corpus)
                if tokenized_corpus and any(tokenized_corpus)
                else None
            )

            # Keep cache bounded so old document store IDs don't accumulate.
            if len(cls._bm25_cache) >= 32:
                cls._bm25_cache.clear()
            cls._bm25_cache[cache_key] = (bm25, metadata)
            return bm25, metadata

    def retrieve_hybrid(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
        section_refs: Optional[list[str]] = None,
        page_refs: Optional[list[int]] = None,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Retrieve chunks by fusing FAISS dense rankings and BM25 lexical rankings
        via Reciprocal Rank Fusion, with optional legal section/page boosts.
        """
        if not query.strip():
            return []

        logger.info(
            "retrieval_started",
            document_id=document_id,
            top_k=top_k,
            mode="hybrid_rrf",
        )

        vector_store = LegalVectorStore.load(document_id)
        total_chunks = vector_store.index.ntotal
        if total_chunks == 0:
            return []

        pool_k = min(max(top_k * 3, 15), total_chunks)

        # 1. Dense retrieval (FAISS)
        query_embedding = self.embedding_model.encode_single(query, kind="query")
        dense_results = vector_store.search(query_embedding, top_k=pool_k)

        # 2. Lexical retrieval (BM25)
        bm25, all_metadata = self._get_bm25_index(document_id, vector_store)
        bm25_ranked: list[tuple[int, float]] = []
        if bm25 is not None:
            query_tokens = _tokenize_multilingual(query)
            if query_tokens:
                bm25_scores = bm25.get_scores(query_tokens)
                ranked_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda idx: float(bm25_scores[idx]),
                    reverse=True,
                )[:pool_k]
                bm25_ranked = [
                    (idx, float(bm25_scores[idx]))
                    for idx in ranked_indices
                    if float(bm25_scores[idx]) > 0.0
                ]

        # 3. Reciprocal Rank Fusion
        fused: dict[str, dict[str, Any]] = {}

        def _chunk_key(item: dict[str, Any]) -> str:
            return str(item.get("chunk_id") or item.get("chunk_index"))

        for rank, item in enumerate(dense_results, start=1):
            key = _chunk_key(item)
            entry = dict(item)
            entry["dense_score"] = float(item.get("score", 0.0))
            entry["bm25_score"] = 0.0
            entry["rrf_score"] = 1.0 / (rrf_k + rank)
            fused[key] = entry

        for rank, (meta_idx, bm25_score) in enumerate(bm25_ranked, start=1):
            meta_item = all_metadata[meta_idx]
            key = _chunk_key(meta_item)
            rrf_add = 1.0 / (rrf_k + rank)

            if key in fused:
                fused[key]["bm25_score"] = bm25_score
                fused[key]["rrf_score"] += rrf_add
            else:
                entry = dict(meta_item)
                entry["dense_score"] = 0.0
                entry["bm25_score"] = bm25_score
                entry["rrf_score"] = rrf_add
                entry["score"] = rrf_add
                fused[key] = entry

        # 4. Metadata boost for explicit section / page references
        normalized_sections = [s.lower() for s in (section_refs or []) if s]
        page_set = set(page_refs or [])

        if normalized_sections or page_set:
            for entry in fused.values():
                content_lower = str(entry.get("content", "")).lower()
                if any(ref in content_lower for ref in normalized_sections):
                    entry["rrf_score"] += 0.015
                start_page = entry.get("start_page", entry.get("page_number"))
                end_page = entry.get("end_page", start_page)
                if page_set and start_page is not None:
                    try:
                        if any(int(start_page) <= p <= int(end_page) for p in page_set):
                            entry["rrf_score"] += 0.02
                    except (TypeError, ValueError):
                        pass

        ranked_results = sorted(
            fused.values(),
            key=lambda item: float(item.get("rrf_score", 0.0)),
            reverse=True,
        )[:top_k]

        for item in ranked_results:
            item["score"] = round(float(item.get("rrf_score", 0.0)), 6)

        logger.info(
            "retrieval_complete",
            document_id=document_id,
            results=len(ranked_results),
            dense_candidates=len(dense_results),
            bm25_candidates=len(bm25_ranked),
        )

        return ranked_results

    def retrieve(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Default retrieval entry point (hybrid dense + BM25 + RRF)."""
        return self.retrieve_hybrid(
            document_id=document_id,
            query=query,
            top_k=top_k,
        )