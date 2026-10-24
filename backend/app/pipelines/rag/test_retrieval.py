from app.pipelines.rag.retriever import LegalRetriever


DOCUMENT_ID = "026f44fe-430d-4522-985f-f12cff9de1b2"


def main():
    print("Loading legal retriever...")
    retriever = LegalRetriever()

    print()
    print("Testing semantic retrieval...")
    print()

    query = "What arguments were presented before the court?"

    print(f"Query: {query}")
    print()

    results = retriever.retrieve(
        document_id=DOCUMENT_ID,
        query=query,
        top_k=5,
    )

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Page: {result['page_number']}")
        print(f"Section: {result['section_type']}")
        print(f"Content: {result['content']}")
        print()

    print("RETRIEVAL TEST COMPLETE")


if __name__ == "__main__":
    main()