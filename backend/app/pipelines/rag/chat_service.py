"""Document-grounded legal chat service."""

from __future__ import annotations

import asyncio
from typing import Any

from app.pipelines.rag.context_builder import RAGContextBuilder
from app.pipelines.rag.llm_service import GeminiLegalLLM
from app.pipelines.rag.retriever import LegalRetriever


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}


class DocumentChatService:
    """Orchestrates retrieval, context building and LLM generation."""

    def __init__(self) -> None:
        self.context_builder = RAGContextBuilder()
        self._retriever: LegalRetriever | None = None
        self._llm: GeminiLegalLLM | None = None

    def _get_retriever(self) -> LegalRetriever:
        """Create the embedding/retrieval service lazily."""

        if self._retriever is None:
            self._retriever = LegalRetriever()

        return self._retriever

    def _get_llm(self) -> GeminiLegalLLM:
        """Create the Gemini service lazily."""

        if self._llm is None:
            self._llm = GeminiLegalLLM()

        return self._llm

    async def answer(
        self,
        document_id: str,
        question: str,
        language: str = "en",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Retrieve document evidence and generate a grounded answer.
        """

        retriever = self._get_retriever()

        # Retrieval uses CPU-heavy embedding generation, so don't
        # block FastAPI's async event loop.
        results = await asyncio.to_thread(
            retriever.retrieve,
            document_id,
            question,
            top_k,
        )

        if not results:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the uploaded document to answer "
                    "this question."
                ),
                "sources": [],
                "was_refused": True,
                "refusal_reason": (
                    "No relevant document context was retrieved."
                ),
            }

        rag = self.context_builder.build(
            question=question,
            results=results,
            max_chunks=top_k,
        )

        language_name = LANGUAGE_NAMES.get(
            language,
            "English",
        )

        language_instruction = (
            f"\n\nIMPORTANT RESPONSE LANGUAGE:\n"
            f"Answer the user in {language_name}."
        )

        prompt = (
            rag["prompt"]
            + language_instruction
        )

        llm = self._get_llm()

        answer = await llm.generate(
            prompt
        )

        return {
            "answer": answer,
            "sources": results,
            "was_refused": False,
            "refusal_reason": None,
        }


_chat_service = DocumentChatService()


def get_chat_service() -> DocumentChatService:
    """Return the shared chat service."""

    return _chat_service