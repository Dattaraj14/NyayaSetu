"""Offline unit and LangGraph workflow tests for NyayaSetu backend upgrades."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from app.agents.qa_graph import (
    ClaimVerdict,
    DraftAnswerOutput,
    DraftClaim,
    EvidenceGradeOutput,
    QueryRewriteOutput,
    VerificationOutput,
    build_qa_graph,
)
from app.agents.summary_graph import (
    DocTypeClassification,
    JudgmentSummaryOutput,
    SummaryClaim,
    SummaryClaimVerdict,
    SummaryVerificationOutput,
    build_summary_graph,
)
from app.pipelines.ocr.ocr_engine import reconstruct_text_from_tsv_data
from app.pipelines.rag import vector_store as vector_store_module
from app.pipelines.rag.vector_store import LegalVectorStore


# ============================================================================
# MOCKS
# ============================================================================


SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk-1",
        "chunk_index": 0,
        "page_number": 6,
        "section_type": "facts",
        "section_title": "Facts",
        "paragraph_numbers": [9, 10],
        "content": (
            "The Bar Council of India is directed to constitute a committee "
            "to frame guidelines regarding AI-generated fake citations."
        ),
        "score": 0.91,
        "rrf_score": 0.032,
    },
    {
        "chunk_id": "chunk-2",
        "chunk_index": 1,
        "page_number": 11,
        "section_type": "decision",
        "section_title": "Decision",
        "paragraph_numbers": [18],
        "content": (
            "The orders of the NCLT and NCLAT are set aside and the Section 7 "
            "application is restored for fresh decision within two weeks."
        ),
        "score": 0.88,
        "rrf_score": 0.030,
    },
]


class FakeRetriever:
    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
        self.chunks = chunks if chunks is not None else list(SAMPLE_CHUNKS)
        self.queries: list[str] = []

    def retrieve_hybrid(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
        section_refs: list[str] | None = None,
        page_refs: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        return self.chunks[:top_k]


class ScriptedLLM:
    """Returns pre-scripted Pydantic responses keyed by schema class."""

    def __init__(self, responses: dict[type, list[Any]]) -> None:
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[str] = []

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type,
        *,
        temperature: float = 0.1,
    ) -> Any:
        self.calls.append(response_schema.__name__)
        queue = self.responses.get(response_schema)
        if not queue:
            raise AssertionError(
                f"No scripted response remaining for {response_schema.__name__}"
            )
        return queue.pop(0)


# ============================================================================
# 1. OCR TEXT RECONSTRUCTION
# ============================================================================


def test_reconstruct_text_from_tsv_data_preserves_lines_and_paragraphs() -> None:
    tsv_data = {
        "text": ["IN", "THE", "COURT", "", "Paragraph", "one.", "Paragraph", "two."],
        "block_num": [1, 1, 1, 1, 2, 2, 2, 2],
        "par_num": [1, 1, 1, 1, 1, 1, 2, 2],
        "line_num": [1, 1, 2, 2, 1, 1, 1, 1],
    }

    reconstructed = reconstruct_text_from_tsv_data(tsv_data)
    assert reconstructed == "IN THE\nCOURT\n\nParagraph one.\n\nParagraph two."


# ============================================================================
# 2. VECTOR STORE LRU CACHE & DELETE
# ============================================================================


def test_vector_store_cache_and_delete(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(vector_store_module.settings, "vector_db_path", str(tmp_path))
    LegalVectorStore.clear_cache()

    store = LegalVectorStore(document_id="doc-test-1", dimension=4)
    vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    store.add(vectors, [{"chunk_id": "c1", "content": "Section 7 IBC"}])
    store.save()

    loaded_1 = LegalVectorStore.load("doc-test-1")
    loaded_2 = LegalVectorStore.load("doc-test-1")
    assert loaded_1 is loaded_2
    assert loaded_1.index.ntotal == 1

    store.delete()
    assert not (tmp_path / "doc-test-1").exists()


# ============================================================================
# 3. LANGGRAPH QA WORKFLOW TESTS
# ============================================================================


def _base_qa_input(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "document_id": "doc-1",
        "session_id": "sess-1",
        "language": "en",
        "top_k": 5,
        "history": [],
        "retry_count": 0,
        "regen_count": 0,
        "trace": [],
    }


def test_qa_graph_out_of_scope_refusal() -> None:
    retriever = FakeRetriever()
    llm = ScriptedLLM({})
    graph = build_qa_graph(retriever=retriever, llm=llm).compile()

    result = asyncio.run(graph.ainvoke(_base_qa_input("Write a poem about rain")))

    assert result["intent"] == "out_of_scope"
    assert result["was_refused"] is True
    assert result["trace"] == ["router", "refuse"]
    assert llm.calls == []


def test_qa_graph_verified_happy_path() -> None:
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        {
            EvidenceGradeOutput: [
                EvidenceGradeOutput(
                    sufficient=True, reason="Order is present.", missing_info=""
                )
            ],
            DraftAnswerOutput: [
                DraftAnswerOutput(
                    cannot_answer=False,
                    refusal_reason="",
                    answer="The Supreme Court set aside the NCLT and NCLAT orders (Page 11).",
                    claims=[
                        DraftClaim(
                            text="NCLT and NCLAT orders were set aside.",
                            source_ids=["S1"],
                        )
                    ],
                )
            ],
            VerificationOutput: [
                VerificationOutput(
                    all_supported=True,
                    verdicts=[
                        ClaimVerdict(
                            claim_index=0, supported=True, reason="Directly stated in S1."
                        )
                    ],
                )
            ],
        }
    )

    graph = build_qa_graph(retriever=retriever, llm=llm).compile()
    result = asyncio.run(
        graph.ainvoke(_base_qa_input("What was the final order under Section 7?"))
    )

    assert result["intent"] == "section_lookup"
    assert result["was_refused"] is False
    assert result["verification_status"] == "verified"
    assert result["trace"] == [
        "router",
        "analyze_query",
        "retrieve",
        "rerank",
        "grade_evidence",
        "generate",
        "verify",
        "finalize",
    ]
    # Reranker should promote chunk-2 (which mentions Section 7) to rank 1 (S1).
    assert result["reranked"][0]["chunk_id"] == "chunk-2"
    assert result["claims"][0]["chunk_ids"] == ["chunk-2"]
    assert result["claims"][0]["pages"] == [11]


def test_qa_graph_retry_then_success() -> None:
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        {
            EvidenceGradeOutput: [
                EvidenceGradeOutput(
                    sufficient=False,
                    reason="Need exact direction.",
                    missing_info="Bar Council direction",
                ),
                EvidenceGradeOutput(
                    sufficient=True, reason="Found direction.", missing_info=""
                ),
            ],
            QueryRewriteOutput: [
                QueryRewriteOutput(
                    rewritten_query="Bar Council of India committee guidelines"
                )
            ],
            DraftAnswerOutput: [
                DraftAnswerOutput(
                    cannot_answer=False,
                    refusal_reason="",
                    answer="The Bar Council of India must form a committee (Page 6).",
                    claims=[
                        DraftClaim(
                            text="Bar Council of India must form a committee.",
                            source_ids=["S1"],
                        )
                    ],
                )
            ],
            VerificationOutput: [
                VerificationOutput(
                    all_supported=True,
                    verdicts=[
                        ClaimVerdict(claim_index=0, supported=True, reason="Supported.")
                    ],
                )
            ],
        }
    )

    graph = build_qa_graph(retriever=retriever, llm=llm).compile()
    result = asyncio.run(graph.ainvoke(_base_qa_input("What about the bar body?")))

    assert "rewrite_and_retry" in result["trace"]
    assert result["retry_count"] == 1
    assert result["was_refused"] is False
    assert len(retriever.queries) == 2


def test_qa_graph_retry_exhausted_refuses() -> None:
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        {
            EvidenceGradeOutput: [
                EvidenceGradeOutput(sufficient=False, reason="Missing", missing_info="X"),
                EvidenceGradeOutput(sufficient=False, reason="Missing", missing_info="X"),
                EvidenceGradeOutput(sufficient=False, reason="Not in doc", missing_info="X"),
            ],
            QueryRewriteOutput: [
                QueryRewriteOutput(rewritten_query="retry 1"),
                QueryRewriteOutput(rewritten_query="retry 2"),
            ],
        }
    )

    graph = build_qa_graph(retriever=retriever, llm=llm).compile()
    result = asyncio.run(
        graph.ainvoke(_base_qa_input("What was the stamp duty paid in 2010?"))
    )

    assert result["was_refused"] is True
    assert result["retry_count"] == 2
    assert result["trace"][-1] == "refuse"


def test_qa_graph_verification_failure_triggers_regeneration() -> None:
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        {
            EvidenceGradeOutput: [
                EvidenceGradeOutput(sufficient=True, reason="Ok", missing_info="")
            ],
            DraftAnswerOutput: [
                DraftAnswerOutput(
                    cannot_answer=False,
                    refusal_reason="",
                    answer="Draft with unsupported claim.",
                    claims=[DraftClaim(text="Unsupported claim.", source_ids=["S1"])],
                ),
                DraftAnswerOutput(
                    cannot_answer=False,
                    refusal_reason="",
                    answer="Corrected grounded answer (Page 6).",
                    claims=[DraftClaim(text="Grounded claim.", source_ids=["S1"])],
                ),
            ],
            VerificationOutput: [
                VerificationOutput(
                    all_supported=False,
                    verdicts=[
                        ClaimVerdict(
                            claim_index=0, supported=False, reason="Not in source."
                        )
                    ],
                ),
                VerificationOutput(
                    all_supported=True,
                    verdicts=[
                        ClaimVerdict(
                            claim_index=0, supported=True, reason="Supported now."
                        )
                    ],
                ),
            ],
        }
    )

    graph = build_qa_graph(retriever=retriever, llm=llm).compile()
    result = asyncio.run(graph.ainvoke(_base_qa_input("What did the Court direct?")))

    assert result["regen_count"] == 1
    assert result["verification_status"] == "verified"
    assert result["trace"].count("generate") == 2


# ============================================================================
# 4. LANGGRAPH SUMMARIZATION WORKFLOW TEST
# ============================================================================


def test_summary_graph_judgment_verified() -> None:
    llm = ScriptedLLM(
        {
            DocTypeClassification: [
                DocTypeClassification(
                    document_type="judgment", reason="Supreme Court order"
                )
            ],
            JudgmentSummaryOutput: [
                JudgmentSummaryOutput(
                    court="Supreme Court of India",
                    date="02.07.2026",
                    case_number="RCP (IB) 6/MB/2023",
                    parties=["Appellant", "Jammu and Kashmir Bank Limited"],
                    facts=["Default occurred (Page 6)."],
                    issues=["Validity of AI-hallucinated citations (Page 6)."],
                    petitioner_arguments=["Citations were fake (Page 6)."],
                    respondent_arguments=["Guarantee survived (Page 6)."],
                    reasoning=["Fake precedents void adjudication (Page 6)."],
                    holding="Orders based on fake precedents are unsustainable (Page 11).",
                    final_order="Orders set aside and restored to NCLT (Page 11).",
                    statutes_cited=["Section 7 IBC"],
                    precedents_cited=[],
                    markdown_summary="### Overview\nOrders set aside (Page 11).",
                    claims=[
                        SummaryClaim(
                            text="NCLT and NCLAT orders were set aside.",
                            pages=[11],
                        )
                    ],
                )
            ],
            SummaryVerificationOutput: [
                SummaryVerificationOutput(
                    all_supported=True,
                    verdicts=[
                        SummaryClaimVerdict(
                            claim_index=0, supported=True, reason="Supported by Page 11."
                        )
                    ],
                )
            ],
        }
    )

    graph = build_summary_graph(llm=llm).compile()
    result = asyncio.run(
        graph.ainvoke(
            {
                "document_id": "doc-1",
                "summary_type": "medium",
                "language": "en",
                "chunks": SAMPLE_CHUNKS,
                "regen_count": 0,
                "verification_feedback": "",
                "trace": [],
            }
        )
    )

    assert result["document_type"] == "judgment"
    assert result["verification_status"] == "verified"
    assert result["trace"] == [
        "detect_document_type",
        "summarize_sections",
        "reduce_summary",
        "verify_summary",
        "translate_if_needed",
        "finalize",
    ]
    entity_types = {e["entity_type"] for e in result["extracted_entities"]}
    assert "court" in entity_types
    assert "party" in entity_types