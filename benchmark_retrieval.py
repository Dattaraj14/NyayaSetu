import asyncio
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------
# Make the backend package and backend-relative database
# available even when this script is run from project root.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_PATH = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_PATH))
os.chdir(BACKEND_PATH)

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.models.database import async_session_factory
from app.models.document import Document, DocumentChunk


# ---------------------------------------------------------
# Questions + expected relevant chunk
#
# Ground truth is based on the actual 7 chunks from the
# benchmarked judgment.
# ---------------------------------------------------------
QUESTIONS = [
    # Chunk 0 — AI-generated fake judgments
    {
        "question": "What did the Court say about AI-generated fake judgments?",
        "expected": 0,
    },
    {
        "question": "What kind of material did the Tribunal rely on?",
        "expected": 0,
    },
    {
        "question": "What did the Court call the AI-generated material relied upon by the Tribunal?",
        "expected": 0,
    },

    # Chunk 1 — delegation of thinking
    {
        "question": "What concern did the Court express about delegating thinking?",
        "expected": 1,
    },
    {
        "question": "What capacity did the Court say lies at the core of human existence?",
        "expected": 1,
    },
    {
        "question": "What serious consequence did the Court associate with making delegation of thinking a habit?",
        "expected": 1,
    },

    # Chunk 2 — Bar Council of India
    {
        "question": "What did the Court direct the Bar Council of India to do?",
        "expected": 2,
    },
    {
        "question": "Which body was directed to constitute a committee?",
        "expected": 2,
    },
    {
        "question": "What issue was the committee asked to deliberate on?",
        "expected": 2,
    },

    # Chunk 3 — NCLT / insolvency proceedings
    {
        "question": "What application did Respondent No. 1 file?",
        "expected": 3,
    },
    {
        "question": "Before which tribunal was the Section 7 application filed?",
        "expected": 3,
    },
    {
        "question": "What process was Respondent No. 1 seeking to initiate?",
        "expected": 3,
    },

    # Chunk 4 — examination of fake judgments
    {
        "question": "What was not in dispute regarding the judgments relied upon by the NCLT?",
        "expected": 4,
    },
    {
        "question": "What did the independent examination reveal about the judgments?",
        "expected": 4,
    },
    {
        "question": "What were some AI-generated paragraphs wrongly attributed to?",
        "expected": 4,
    },

    # Chunk 5 — affidavit / alleged precedents
    {
        "question": "What did Respondent No. 1's affidavit indicate about the alleged judgments?",
        "expected": 5,
    },
    {
        "question": "Who said that the alleged judgments relied on by the NCLT were not cited by counsel at the bar?",
        "expected": 5,
    },
    {
        "question": "What did the affidavit say about the so-called precedents relied upon by the adjudicating authority?",
        "expected": 5,
    },

    # Chunk 6 — final decision
    {
        "question": "What was the Court's final decision regarding the NCLT and NCLAT orders?",
        "expected": 6,
    },
    {
        "question": "What happened to the Section 7 application after the judgments were set aside?",
        "expected": 6,
    },
]


def load_model(model_name: str):
    start = time.perf_counter()
    model = SentenceTransformer(model_name)
    elapsed = time.perf_counter() - start

    return model, elapsed


def prepare_documents(model_name: str, chunks):
    texts = [chunk.content for chunk in chunks]

    if model_name == "intfloat/multilingual-e5-small":
        texts = [f"passage: {text}" for text in texts]

    return texts


def prepare_query(model_name: str, question: str):
    if model_name == "intfloat/multilingual-e5-small":
        return f"query: {question}"

    return question


def benchmark_model(model_name: str, chunks):
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    model, load_time = load_model(model_name)

    document_texts = prepare_documents(model_name, chunks)

    start = time.perf_counter()

    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    document_embedding_time = time.perf_counter() - start

    print(f"Model load time:       {load_time:.2f} sec")
    print(f"Document embedding:    {document_embedding_time:.2f} sec")
    print(f"Embedding dimension:   {document_embeddings.shape[1]}")
    print(f"Number of chunks:      {len(chunks)}")

    recall_at_1 = 0
    recall_at_3 = 0
    total_query_time = 0

    results = []

    for item in QUESTIONS:
        question = item["question"]
        expected_chunk = item["expected"]

        query_text = prepare_query(model_name, question)

        start = time.perf_counter()

        query_embedding = model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_time = time.perf_counter() - start
        total_query_time += query_time

        scores = document_embeddings @ query_embedding
        ranked_indices = scores.argsort()[::-1]

        top1 = int(ranked_indices[0])
        top3 = [int(x) for x in ranked_indices[:3]]

        hit1 = top1 == expected_chunk
        hit3 = expected_chunk in top3

        if hit1:
            recall_at_1 += 1

        if hit3:
            recall_at_3 += 1

        results.append(
            {
                "question": question,
                "expected": expected_chunk,
                "top1": top1,
                "top3": top3,
                "hit1": hit1,
                "hit3": hit3,
                "score": float(scores[top1]),
            }
        )

    total_questions = len(QUESTIONS)

    recall1_percent = (recall_at_1 / total_questions) * 100
    recall3_percent = (recall_at_3 / total_questions) * 100
    avg_query_time = total_query_time / total_questions

    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)

    print(f"Recall@1:             {recall_at_1}/{total_questions} "
          f"({recall1_percent:.1f}%)")

    print(f"Recall@3:             {recall_at_3}/{total_questions} "
          f"({recall3_percent:.1f}%)")

    print(f"Average query time:    {avg_query_time:.4f} sec")

    print("\nQUESTION RESULTS")
    print("-" * 80)

    for i, result in enumerate(results, start=1):
        status1 = "PASS" if result["hit1"] else "MISS"
        status3 = "PASS" if result["hit3"] else "MISS"

        print(
            f"{i:02d}. "
            f"Expected={result['expected']} | "
            f"Top1={result['top1']} [{status1}] | "
            f"Top3={result['top3']} [{status3}]"
        )

        print(f"    {result['question']}")

    return {
        "model": model_name,
        "load_time": load_time,
        "document_embedding_time": document_embedding_time,
        "dimension": document_embeddings.shape[1],
        "recall1": recall1_percent,
        "recall3": recall3_percent,
        "avg_query_time": avg_query_time,
    }


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


async def main():
    document, chunks = await load_latest_chunks()

    print("=" * 80)
    print("NYAYASETU RETRIEVAL BENCHMARK")
    print("=" * 80)
    print(f"Document ID: {document.id}")
    print(f"Filename:    {document.original_filename}")
    print(f"Chunks:      {len(chunks)}")
    print(f"Questions:   {len(QUESTIONS)}")

    bge_result = benchmark_model(
        "BAAI/bge-m3",
        chunks,
    )

    e5_result = benchmark_model(
        "intfloat/multilingual-e5-small",
        chunks,
    )

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<28}"
        f"{'BGE-M3':>18}"
        f"{'E5-small':>18}"
    )

    print("-" * 64)

    print(
        f"{'Document embedding time':<28}"
        f"{bge_result['document_embedding_time']:>15.2f}s"
        f"{e5_result['document_embedding_time']:>18.2f}s"
    )

    print(
        f"{'Embedding dimension':<28}"
        f"{bge_result['dimension']:>18}"
        f"{e5_result['dimension']:>18}"
    )

    print(
        f"{'Recall@1':<28}"
        f"{bge_result['recall1']:>17.1f}%"
        f"{e5_result['recall1']:>17.1f}%"
    )

    print(
        f"{'Recall@3':<28}"
        f"{bge_result['recall3']:>17.1f}%"
        f"{e5_result['recall3']:>17.1f}%"
    )

    print(
        f"{'Average query time':<28}"
        f"{bge_result['avg_query_time']:>15.4f}s"
        f"{e5_result['avg_query_time']:>18.4f}s"
    )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    asyncio.run(main())