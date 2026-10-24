from app.pipelines.rag.context_builder import RAGContextBuilder
from app.pipelines.rag.llm_service import GeminiLegalLLM
from app.pipelines.rag.retriever import LegalRetriever


DOCUMENT_ID = "026f44fe-430d-4522-985f-f12cff9de1b2"


async def main():
    query = "What arguments were presented before the court?"

    print("Loading retriever...")
    retriever = LegalRetriever()

    print("Retrieving relevant legal context...")
    results = retriever.retrieve(
        document_id=DOCUMENT_ID,
        query=query,
        top_k=5,
    )

    if not results:
        print("No relevant legal context was found.")
        return

    print(f"Retrieved chunks: {len(results)}")

    builder = RAGContextBuilder()

    rag = builder.build(
        question=query,
        results=results,
        max_chunks=5,
    )

    print()
    print("Loading Gemini...")
    llm = GeminiLegalLLM()

    print("Generating grounded answer...")
    answer = await llm.generate(
        rag["prompt"]
    )

    print()
    print("=" * 80)
    print("NYAYASETU ANSWER")
    print("=" * 80)
    print()
    print(answer)

    print()
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for index, source in enumerate(
        rag["sources"],
        start=1,
    ):
        print(
            f"{index}. "
            f"Page {source.get('page_number')} | "
            f"Section {source.get('section_type')} | "
            f"Score {source.get('score', 0):.4f}"
        )

    print()
    print("LLM RAG TEST PASSED")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())