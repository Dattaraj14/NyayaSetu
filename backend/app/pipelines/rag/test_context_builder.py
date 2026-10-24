from app.pipelines.rag.context_builder import RAGContextBuilder
from app.pipelines.rag.retriever import LegalRetriever


DOCUMENT_ID = "026f44fe-430d-4522-985f-f12cff9de1b2"


def main():
    query = "What arguments were presented before the court?"

    print("Loading retriever...")
    retriever = LegalRetriever()

    print()
    print("Retrieving legal chunks...")
    results = retriever.retrieve(
        document_id=DOCUMENT_ID,
        query=query,
        top_k=5,
    )

    print()
    print("Building RAG context...")

    builder = RAGContextBuilder()

    rag = builder.build(
        question=query,
        results=results,
        max_chunks=5,
    )

    print()
    print("=" * 80)
    print("RAG CONTEXT")
    print("=" * 80)
    print()
    print(rag["context"])

    print()
    print("=" * 80)
    print("LLM PROMPT")
    print("=" * 80)
    print()
    print(rag["prompt"])

    print()
    print("CONTEXT BUILDER TEST COMPLETE")


if __name__ == "__main__":
    main()