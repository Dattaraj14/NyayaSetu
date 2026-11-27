"""Document-grounded legal chat service powered by LangGraph."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from app.agents.qa_graph import run_qa_graph, stream_qa_graph
from app.pipelines.rag.llm_service import GeminiLegalLLM
from app.pipelines.rag.retriever import LegalRetriever


class DocumentChatService:
    """Orchestrates stateful LangGraph QA and streaming over legal documents."""

    def __init__(self) -> None:
        self._retriever: LegalRetriever | None = None
        self._llm: GeminiLegalLLM | None = None

    def _get_retriever(self) -> LegalRetriever:
        if self._retriever is None:
            self._retriever = LegalRetriever()
        return self._retriever

    def _get_llm(self) -> GeminiLegalLLM:
        if self._llm is None:
            self._llm = GeminiLegalLLM()
        return self._llm

    async def answer(
        self,
        document_id: str,
        question: str,
        language: str = "en",
        top_k: int = 5,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the stateful LangGraph QA pipeline for a question."""
        resolved_session_id = session_id or str(uuid4())
        return await run_qa_graph(
            document_id=document_id,
            question=question,
            session_id=resolved_session_id,
            language=language,
            top_k=top_k,
            retriever=self._get_retriever(),
            llm=self._get_llm(),
        )

    async def answer_stream(
        self,
        document_id: str,
        question: str,
        language: str = "en",
        top_k: int = 5,
        session_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream LangGraph node progress events, tokens, and final payload."""
        resolved_session_id = session_id or str(uuid4())
        async for event in stream_qa_graph(
            document_id=document_id,
            question=question,
            session_id=resolved_session_id,
            language=language,
            top_k=top_k,
            retriever=self._get_retriever(),
            llm=self._get_llm(),
        ):
            yield event


_chat_service = DocumentChatService()


def get_chat_service() -> DocumentChatService:
    """Return the shared chat service."""
    return _chat_service