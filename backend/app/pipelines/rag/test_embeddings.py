from app.pipelines.rag.embeddings import LegalEmbeddingModel


def main():
    print("Loading embedding model...")

    model = LegalEmbeddingModel()

    print()
    print("Embedding model loaded successfully!")
    print(f"Model: {model.model_name}")
    print(f"Device: {model.device}")
    print(f"Dimension: {model.dimension}")

    texts = [
        "The court dismissed the appeal.",
        "The appellant challenged the decision of the tribunal.",
        "The weather is pleasant today.",
    ]

    print()
    print("Generating embeddings...")

    embeddings = model.encode(texts)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")

    print()
    print("Testing semantic similarity...")

    results = model.search(
        "The appellant challenged the tribunal decision.",
        embeddings,
        top_k=3,
    )

    for index, score in results:
        print(
            f"Text {index}: "
            f"score={score:.4f} | "
            f"{texts[index]}"
        )

    print()
    print("EMBEDDING TEST PASSED")


if __name__ == "__main__":
    main()