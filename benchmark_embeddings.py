import asyncio
import sys
import time
from pathlib import Path

# Add backend to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_PATH = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_PATH))

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.models.database import async_session_factory
from app.models.document import Document, DocumentChunk


QUESTIONS = [
    "What were the main arguments of the appellant?",
    "What was the final decision of the Court?",
    "What did the Court say about AI-generated fake judgments?",
    "What happened before the NCLT and NCLAT?",
    "Why did the Court set aside the earlier judgments?",
]


async def load_latest_chunks():
    async with async_session_factory() as db:
        result = await db.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(1)
        )
        document = result.scalar_one_or_none()

        if not document:
            raise RuntimeError("No document found in the database.")

        result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
        chunks = result.scalars().all()

        if not chunks:
            raise RuntimeError("No chunks found for the latest document.")

        return document, chunks


def benchmark_model(model_name, chunks, questions, is_e5=False):
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    start = time.perf_counter()

    model = SentenceTransformer(model_name)

    load_time = time.perf_counter() - start

    document_texts = [chunk.content for chunk in chunks]

    if is_e5:
        document_texts = [
            f"passage: {text}" for text in document_texts
        ]

    start = time.perf_counter()

    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embedding_time = time.perf_counter() - start

    print(f"Model load time:       {load_time:.2f} sec")
    print(f"Document embedding:    {embedding_time:.2f} sec")
    print(f"Embedding dimension:   {document_embeddings.shape[1]}")
    print(f"Number of chunks:      {len(chunks)}")

    for question in questions:
        query_text = question

        if is_e5:
            query_text = f"query: {question}"

        query_embedding = model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        scores = document_embeddings @ query_embedding

        ranked = scores.argsort()[::-1][:3]

        print("\n" + "-" * 80)
        print(f"QUESTION: {question}")
        print("-" * 80)

        for rank, idx in enumerate(ranked, start=1):
            chunk = chunks[idx]

            preview = chunk.content.replace("\n", " ")[:250]

            print(
                f"\n#{rank}  Score: {scores[idx]:.4f}"
            )
            print(
                f"Chunk: {chunk.chunk_index}"
                f" | Pages: {chunk.page_number}"
            )
            print(f"Text: {preview}...")


async def main():
    document, chunks = await load_latest_chunks()

    print("=" * 80)
    print("NYAYASETU EMBEDDING BENCHMARK")
    print("=" * 80)
    print(f"Document ID: {document.id}")
    print(f"Filename:    {document.original_filename}")
    print(f"Chunks:      {len(chunks)}")

    benchmark_model(
        "BAAI/bge-m3",
        chunks,
        QUESTIONS,
        is_e5=False,
    )

    benchmark_model(
        "intfloat/multilingual-e5-small",
        chunks,
        QUESTIONS,
        is_e5=True,
    )


if __name__ == "__main__":
    asyncio.run(main())