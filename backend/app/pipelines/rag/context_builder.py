"""Build grounded RAG context from retrieved legal chunks."""

from __future__ import annotations

from typing import Any


class RAGContextBuilder:
    """Convert retrieval results into structured LLM-ready context."""

    def build_context(
        self,
        results: list[dict[str, Any]],
        max_chunks: int = 5,
    ) -> str:
        """
        Build a structured context string from retrieved chunks.

        Each source contains:
        - relevance score
        - chunk index
        - page
        - section
        - legal text
        """

        if not results:
            return "No relevant document context was retrieved."

        selected_results = results[:max_chunks]

        sections: list[str] = []

        for source_number, result in enumerate(
            selected_results,
            start=1,
        ):
            score = float(result.get("score", 0.0))
            chunk_index = result.get("chunk_index", "unknown")
            page_number = result.get("page_number", "unknown")
            section_type = result.get(
                "section_type",
                "unknown",
            )
            section_title = result.get(
                "section_title",
                "",
            )
            content = str(
                result.get("content", "")
            ).strip()

            section_label = (
                section_title
                if section_title
                else section_type
            )

            source_block = (
                f"[SOURCE {source_number}]\n"
                f"Relevance score: {score:.4f}\n"
                f"Chunk: {chunk_index}\n"
                f"Page: {page_number}\n"
                f"Section: {section_label}\n"
                f"Content:\n{content}"
            )

            sections.append(source_block)

        return "\n\n".join(sections)

    def build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Build a grounded legal-answer prompt.

        The model is explicitly instructed to answer only
        from the retrieved document context.
        """

        return f"""You are NyayaSetu, an Indian legal document
understanding assistant.

Your task is to answer the user's question using ONLY the
provided document context.

IMPORTANT RULES:
1. Do not invent facts, arguments, dates, names, sections,
   or conclusions that are not supported by the context.
2. If the answer cannot be determined from the supplied
   context, clearly say that the information is not available
   in the retrieved portion of the document.
3. Prefer precise and simple language.
4. Preserve important legal terminology.
5. When possible, mention the page number of the supporting
   source.
6. Distinguish what the document states from your own explanation.
7. Do not present the response as legal advice.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

ANSWER:
"""

    def build(
        self,
        question: str,
        results: list[dict[str, Any]],
        max_chunks: int = 5,
    ) -> dict[str, Any]:
        """Build both structured context and the final prompt."""

        context = self.build_context(
            results=results,
            max_chunks=max_chunks,
        )

        prompt = self.build_prompt(
            question=question,
            context=context,
        )

        return {
            "question": question,
            "context": context,
            "prompt": prompt,
            "sources": results[:max_chunks],
        }