"""Stateful LangGraph legal question-answering pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import re
import time
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.pipelines.rag.llm_service import GeminiLegalLLM
from app.pipelines.rag.retriever import LegalRetriever


logger = get_logger("qa_graph")

MAX_RETRIEVAL_RETRIES = 2
MAX_REGEN_RETRIES = 1
MAX_HISTORY_TURNS = 6

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

NODE_PROGRESS_LABELS = {
    "router": "routing",
    "analyze_query": "analyzing",
    "retrieve": "retrieving",
    "rerank": "reranking",
    "grade_evidence": "grading",
    "rewrite_and_retry": "rewriting",
    "generate": "generating",
    "verify": "verifying",
    "refuse": "refusing",
    "finalize": "finalizing",
}


# ============================================================================
# STRUCTURED LLM OUTPUT SCHEMAS (Flat types for Gemini schema compatibility)
# ============================================================================


class QueryRewriteOutput(BaseModel):
    rewritten_query: str = Field(
        description="Self-contained search query optimized for Indian legal document retrieval."
    )


class EvidenceGradeOutput(BaseModel):
    sufficient: bool = Field(
        description="True if the retrieved chunks contain enough information to answer the question."
    )
    reason: str = Field(
        description="Short explanation of why the evidence is or is not sufficient."
    )
    missing_info: str = Field(
        description="What specific information is missing, or empty string if sufficient."
    )


class DraftClaim(BaseModel):
    text: str = Field(description="A single factual claim made in the answer.")
    source_ids: list[str] = Field(
        description="List of source IDs such as ['S1', 'S2'] that support this claim."
    )


class DraftAnswerOutput(BaseModel):
    cannot_answer: bool = Field(
        description="True if the provided sources do not contain the answer."
    )
    refusal_reason: str = Field(
        description="Reason for refusal if cannot_answer is True, else empty string."
    )
    answer: str = Field(
        description="Clear, grounded answer in the requested language citing page numbers."
    )
    claims: list[DraftClaim] = Field(
        description="Key factual claims in the answer mapped to supporting source IDs."
    )


class ClaimVerdict(BaseModel):
    claim_index: int = Field(description="Zero-based index of the claim being checked.")
    supported: bool = Field(
        description="True only if the claim is directly entailed by its cited sources."
    )
    reason: str = Field(description="Brief justification for the verdict.")


class VerificationOutput(BaseModel):
    all_supported: bool = Field(
        description="True if every checked claim is supported by its cited sources."
    )
    verdicts: list[ClaimVerdict] = Field(
        description="Verification verdict for each claim."
    )


# ============================================================================
# GRAPH STATE (Strictly JSON-serializable for SQLite checkpointing)
# ============================================================================


class QAState(TypedDict, total=False):
    question: str
    document_id: str
    session_id: str
    language: str
    top_k: int
    history: list[dict[str, str]]

    intent: str
    rewritten_query: str
    section_refs: list[str]
    page_refs: list[int]

    retrieved: list[dict[str, Any]]
    reranked: list[dict[str, Any]]

    grade: str
    grade_reason: str
    missing_info: str
    retry_count: int
    regen_count: int
    verification_feedback: str

    draft: dict[str, Any]
    verification: dict[str, Any]
    verification_status: str

    final_answer: str
    citations: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    was_refused: bool
    refusal_reason: str
    trace: list[str]


# ============================================================================
# HELPERS
# ============================================================================


_SECTION_PATTERN = re.compile(
    r"\b(?:section|sec\.?|article|art\.?|clause|para(?:graph)?|order|rule)\s*"
    r"(\d+[A-Za-z]?(?:\(\d+\))?)",
    re.IGNORECASE,
)
_PAGE_PATTERN = re.compile(r"\bpage\s*(\d+)\b", re.IGNORECASE)
_FOLLOWUP_PATTERN = re.compile(
    r"\b(it|its|they|them|their|that|this|those|these|above|previous|he|she|same|more detail|elaborate)\b",
    re.IGNORECASE,
)


def _detect_intent(question: str) -> str:
    q = question.strip().lower()

    out_of_scope_markers = (
        "write a poem",
        "weather today",
        "stock price",
        "cricket score",
        "recipe for",
        "joke about",
        "who is the prime minister",
    )
    if any(marker in q for marker in out_of_scope_markers):
        return "out_of_scope"

    if any(
        kw in q
        for kw in (
            "translate",
            "अनुवाद",
            "भाषांतर",
            "in hindi",
            "in marathi",
        )
    ):
        return "translate"

    if any(
        kw in q
        for kw in (
            "simplify",
            "plain english",
            "simple terms",
            "layman",
            "explain simply",
            "easy words",
        )
    ):
        return "simplify"

    if any(
        kw in q
        for kw in (
            "summarize",
            "summarise",
            "summary of",
            "overview of the document",
            "brief of this judgment",
            "brief of this contract",
        )
    ):
        return "summary_request"

    if _SECTION_PATTERN.search(question):
        return "section_lookup"

    return "factual_qa"


def _extract_references(question: str) -> tuple[list[str], list[int]]:
    section_refs = [m.group(0).strip() for m in _SECTION_PATTERN.finditer(question)]
    page_refs = [int(m.group(1)) for m in _PAGE_PATTERN.finditer(question)]
    return section_refs, page_refs


def _format_sources_for_prompt(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source_id = f"S{idx}"
        page = chunk.get("page_number", "unknown")
        section = chunk.get("section_title") or chunk.get("section_type") or "general"
        paragraphs = chunk.get("paragraph_numbers")
        para_str = (
            ", ".join(str(p) for p in paragraphs)
            if isinstance(paragraphs, list) and paragraphs
            else "N/A"
        )
        content = str(chunk.get("content", "")).strip()
        blocks.append(
            f"[{source_id}]\n"
            f"Page: {page}\n"
            f"Section: {section}\n"
            f"Paragraphs: {para_str}\n"
            f"Text:\n{content}"
        )
    return "\n\n".join(blocks)


def _format_history_for_prompt(history: list[dict[str, str]]) -> str:
    if not history:
        return "None"
    lines: list[str] = []
    for turn in history[-3:]:
        lines.append(f"User: {turn.get('question', '')}")
        lines.append(f"Assistant: {turn.get('answer', '')}")
    return "\n".join(lines)


def _source_id_to_chunk_map(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for idx, chunk in enumerate(chunks, start=1):
        sid = f"S{idx}"
        mapping[sid] = chunk
        mapping[sid.lower()] = chunk
        if chunk.get("chunk_id"):
            mapping[str(chunk["chunk_id"])] = chunk
    return mapping


def _get_checkpoint_db_path() -> str:
    settings = get_settings()
    data_dir = settings.project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "langgraph_checkpoints.sqlite")


# ============================================================================
# GRAPH BUILDER
# ============================================================================


def build_qa_graph(
    retriever: LegalRetriever,
    llm: GeminiLegalLLM,
) -> StateGraph:
    """Build the uncompiled LangGraph StateGraph for legal document QA."""

    async def router_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        intent = _detect_intent(state["question"])
        trace.append("router")

        logger.info(
            "qa_node_router",
            document_id=state["document_id"],
            intent=intent,
        )
        return {"intent": intent, "trace": trace}

    async def analyze_query_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        question = state["question"].strip()
        history = list(state.get("history") or [])
        section_refs, page_refs = _extract_references(question)

        rewritten = question
        economy_mode = get_settings().qa_economy_mode

        if economy_mode and history and (
            len(question.split()) <= 8 or _FOLLOWUP_PATTERN.search(question)
        ):
            last_q = history[-1].get("question", "").strip()
            if last_q:
                rewritten = f"{last_q} {question}"
        elif history and (
            len(question.split()) <= 8 or _FOLLOWUP_PATTERN.search(question)
        ):
            history_text = _format_history_for_prompt(history)
            prompt = (
                "Rewrite the user's latest follow-up question into a standalone search query "
                "for retrieving passages from an Indian legal document. Preserve exact section "
                "numbers, party names, case numbers, and dates.\n\n"
                f"CONVERSATION HISTORY:\n{history_text}\n\n"
                f"LATEST QUESTION:\n{question}\n"
            )
            try:
                out = await llm.generate_structured(
                    prompt,
                    QueryRewriteOutput,
                    temperature=0.0,
                )
                if out.rewritten_query.strip():
                    rewritten = out.rewritten_query.strip()
                    more_sections, more_pages = _extract_references(rewritten)
                    section_refs = list(dict.fromkeys(section_refs + more_sections))
                    page_refs = list(dict.fromkeys(page_refs + more_pages))
            except Exception as exc:
                logger.warning("query_rewrite_skipped", error=str(exc))

        trace.append("analyze_query")
        return {
            "rewritten_query": rewritten,
            "section_refs": section_refs,
            "page_refs": page_refs,
            "trace": trace,
        }

    async def retrieve_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        query = state.get("rewritten_query") or state["question"]
        top_k = max(int(state.get("top_k") or 5), 5)
        candidate_k = max(top_k * 2, 10)

        results = await asyncio.to_thread(
            retriever.retrieve_hybrid,
            state["document_id"],
            query,
            candidate_k,
            state.get("section_refs") or [],
            state.get("page_refs") or [],
        )

        trace.append("retrieve")
        return {"retrieved": results, "trace": trace}

    async def rerank_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        retrieved = list(state.get("retrieved") or [])
        top_k = int(state.get("top_k") or 5)
        query_terms = set(
            re.findall(
                r"[^\W_]+",
                (state.get("rewritten_query") or state["question"]).lower(),
                flags=re.UNICODE,
            )
        )
        section_refs = [s.lower() for s in (state.get("section_refs") or [])]

        scored: list[tuple[float, dict[str, Any]]] = []
        for rank, chunk in enumerate(retrieved):
            content_lower = str(chunk.get("content", "")).lower()
            base_score = float(chunk.get("rrf_score", chunk.get("score", 0.0)))

            overlap = 0.0
            if query_terms:
                chunk_terms = set(
                    re.findall(r"[^\W_]+", content_lower, flags=re.UNICODE)
                )
                overlap = len(query_terms & chunk_terms) / len(query_terms)

            ref_bonus = sum(0.05 for ref in section_refs if ref in content_lower)
            combined = base_score + (0.02 * overlap) + ref_bonus - (rank * 0.0001)
            scored.append((combined, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        reranked = [item[1] for item in scored[:top_k]]

        trace.append("rerank")
        return {"reranked": reranked, "trace": trace}

    async def grade_evidence_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        reranked = list(state.get("reranked") or [])

        if not reranked:
            trace.append("grade_evidence")
            return {
                "grade": "insufficient",
                "grade_reason": "No chunks were retrieved from the document.",
                "missing_info": "Relevant passages from the uploaded document.",
                "trace": trace,
            }

        if get_settings().qa_economy_mode:
            has_text = any(str(c.get("content", "")).strip() for c in reranked)
            trace.append("grade_evidence")
            return {
                "grade": "sufficient" if has_text else "insufficient",
                "grade_reason": (
                    "Retrieved hybrid passages passed deterministic sufficiency check."
                    if has_text
                    else "Retrieved chunks were empty."
                ),
                "missing_info": "",
                "trace": trace,
            }

        sources_text = _format_sources_for_prompt(reranked)
        prompt = (
            "You are grading whether the retrieved excerpts from an Indian legal document "
            "contain enough evidence to answer the user's question accurately.\n"
            "Be strict: mark sufficient=true if the excerpts directly answer or substantially "
            "address the question; mark sufficient=false if the needed facts, parties, sections, "
            "or holdings are absent.\n\n"
            f"USER QUESTION:\n{state['question']}\n\n"
            f"RETRIEVED EXCERPTS:\n{sources_text}\n"
        )

        try:
            grade_out = await llm.generate_structured(
                prompt,
                EvidenceGradeOutput,
                temperature=0.0,
            )
            grade = "sufficient" if grade_out.sufficient else "insufficient"
            grade_reason = grade_out.reason
            missing_info = grade_out.missing_info
        except Exception as exc:
            logger.warning("evidence_grading_fallback_sufficient", error=str(exc))
            grade = "sufficient"
            grade_reason = "Proceeding with retrieved evidence."
            missing_info = ""

        trace.append("grade_evidence")
        return {
            "grade": grade,
            "grade_reason": grade_reason,
            "missing_info": missing_info,
            "trace": trace,
        }

    async def rewrite_and_retry_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        retry_count = int(state.get("retry_count") or 0) + 1
        current_query = state.get("rewritten_query") or state["question"]
        missing = state.get("missing_info") or state.get("grade_reason") or ""

        prompt = (
            "The previous search query did not retrieve enough evidence from the Indian legal document.\n"
            "Generate an improved, alternative search query using legal synonyms, section titles, "
            "or broader phrasing to locate the missing facts.\n\n"
            f"ORIGINAL QUESTION: {state['question']}\n"
            f"PREVIOUS QUERY: {current_query}\n"
            f"MISSING EVIDENCE: {missing}\n"
        )

        new_query = current_query
        try:
            out = await llm.generate_structured(
                prompt,
                QueryRewriteOutput,
                temperature=0.2,
            )
            if out.rewritten_query.strip():
                new_query = out.rewritten_query.strip()
        except Exception as exc:
            logger.warning("rewrite_retry_failed", error=str(exc))

        trace.append("rewrite_and_retry")
        return {
            "rewritten_query": new_query,
            "retry_count": retry_count,
            "trace": trace,
        }

    async def generate_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        reranked = list(state.get("reranked") or [])
        language = state.get("language") or "en"
        language_name = LANGUAGE_NAMES.get(language, "English")
        intent = state.get("intent") or "factual_qa"
        history_text = _format_history_for_prompt(list(state.get("history") or []))
        sources_text = _format_sources_for_prompt(reranked)
        feedback = state.get("verification_feedback") or ""

        style_hint = ""
        if intent == "simplify":
            style_hint = (
                "Explain in simple, plain language suitable for a non-lawyer while keeping "
                "key legal terms accurate."
            )
        elif intent == "summary_request":
            style_hint = (
                "Provide a structured, concise summary of the relevant points supported by the excerpts."
            )
        elif intent == "section_lookup":
            style_hint = (
                "Quote or state the exact section/clause details as they appear in the excerpts."
            )

        feedback_block = (
            f"\nPREVIOUS DRAFT FEEDBACK (FIX THESE UNSUPPORTED CLAIMS):\n{feedback}\n"
            if feedback
            else ""
        )

        prompt = (
            "You are NyayaSetu, an Indian legal document understanding assistant.\n"
            "Answer the user's question using ONLY the provided numbered sources [S1], [S2], etc.\n"
            "STRICT RULES:\n"
            "1. Never invent facts, dates, party names, case citations, or orders not in the sources.\n"
            "2. If the sources cannot answer the question, set cannot_answer=true and explain why.\n"
            "3. Every claim in `claims` MUST cite one or more exact source IDs like ['S1'] or ['S1', 'S2'].\n"
            "4. Mention page numbers in the answer where helpful.\n"
            f"5. Write the `answer` in {language_name}.\n"
            f"{style_hint}\n"
            f"{feedback_block}\n"
            f"RECENT CONVERSATION:\n{history_text}\n\n"
            f"USER QUESTION:\n{state['question']}\n\n"
            f"DOCUMENT SOURCES:\n{sources_text}\n"
        )

        draft_out = await llm.generate_structured(
            prompt,
            DraftAnswerOutput,
            temperature=0.1,
        )

        trace.append("generate")
        return {
            "draft": draft_out.model_dump(),
            "trace": trace,
        }

    async def verify_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        draft = state.get("draft") or {}
        reranked = list(state.get("reranked") or [])
        regen_count = int(state.get("regen_count") or 0)

        if draft.get("cannot_answer"):
            trace.append("verify")
            return {
                "verification": {"all_supported": True, "verdicts": []},
                "verification_status": "refused",
                "trace": trace,
            }

        claims = list(draft.get("claims") or [])
        if not claims:
            trace.append("verify")
            return {
                "verification": {"all_supported": True, "verdicts": []},
                "verification_status": "verified",
                "trace": trace,
            }

        source_map = _source_id_to_chunk_map(reranked)

        if get_settings().qa_economy_mode:
            verdicts: list[dict[str, Any]] = []
            for idx, claim in enumerate(claims):
                cited_ids = claim.get("source_ids") or []
                valid_chunks = [source_map[sid] for sid in cited_ids if sid in source_map]
                supported = len(valid_chunks) > 0
                verdicts.append(
                    {
                        "claim_index": idx,
                        "supported": supported,
                        "reason": (
                            "Grounded in cited retrieved passage."
                            if supported
                            else "Cited source ID not found in retrieved passages."
                        ),
                    }
                )
            all_supported = all(v["supported"] for v in verdicts)
            trace.append("verify")
            return {
                "verification": {"all_supported": all_supported, "verdicts": verdicts},
                "verification_status": (
                    "verified" if all_supported else "partially_verified"
                ),
                "verification_feedback": "",
                "trace": trace,
            }

        claims_lines: list[str] = []
        missing_source_indices: set[int] = set()

        for idx, claim in enumerate(claims):
            cited_ids = claim.get("source_ids") or []
            valid_ids = [sid for sid in cited_ids if sid in source_map]
            if not valid_ids:
                missing_source_indices.add(idx)
            claims_lines.append(
                f"Claim {idx}: {claim.get('text', '')} | Cited: {cited_ids}"
            )

        sources_text = _format_sources_for_prompt(reranked)
        prompt = (
            "Verify each factual claim against ONLY the cited document sources.\n"
            "Mark a claim supported=true only if the cited source text directly supports it.\n"
            "Mark supported=false if the claim exaggerates, invents details, or cites the wrong source.\n\n"
            "CLAIMS TO VERIFY:\n" + "\n".join(claims_lines) + "\n\n"
            f"DOCUMENT SOURCES:\n{sources_text}\n"
        )

        try:
            ver_out = await llm.generate_structured(
                prompt,
                VerificationOutput,
                temperature=0.0,
            )
            verdicts = [v.model_dump() for v in ver_out.verdicts]
        except Exception as exc:
            logger.warning("claim_verification_skipped", error=str(exc))
            trace.append("verify")
            return {
                "verification": {"all_supported": True, "verdicts": []},
                "verification_status": "unverified",
                "trace": trace,
            }

        unsupported: list[dict[str, Any]] = []
        for verdict in verdicts:
            idx = int(verdict.get("claim_index", -1))
            if idx in missing_source_indices:
                verdict["supported"] = False
                verdict["reason"] = "Cited source ID was not present in retrieved chunks."
            if not verdict.get("supported", False):
                unsupported.append(verdict)

        all_supported = len(unsupported) == 0
        feedback_lines = [
            f"- Claim {u.get('claim_index')}: {u.get('reason')}"
            for u in unsupported
        ]

        trace.append("verify")

        if all_supported:
            return {
                "verification": {"all_supported": True, "verdicts": verdicts},
                "verification_status": "verified",
                "verification_feedback": "",
                "trace": trace,
            }

        if regen_count < MAX_REGEN_RETRIES:
            return {
                "verification": {"all_supported": False, "verdicts": verdicts},
                "verification_status": "needs_regeneration",
                "verification_feedback": "\n".join(feedback_lines),
                "regen_count": regen_count + 1,
                "trace": trace,
            }

        return {
            "verification": {"all_supported": False, "verdicts": verdicts},
            "verification_status": "partially_verified",
            "verification_feedback": "\n".join(feedback_lines),
            "trace": trace,
        }

    async def refuse_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        intent = state.get("intent") or "factual_qa"
        language = state.get("language") or "en"

        if intent == "out_of_scope":
            reason = (
                "This question is outside the scope of the uploaded legal document."
            )
        else:
            reason = (
                state.get("grade_reason")
                or "The uploaded document does not contain sufficient information to answer this question."
            )

        refusal_messages = {
            "en": (
                "I cannot reliably answer this question from the uploaded legal document. "
                f"Reason: {reason}"
            ),
            "hi": (
                "अपलोड किए गए कानूनी दस्तावेज़ के आधार पर मैं इस प्रश्न का विश्वसनीय उत्तर नहीं दे सकता। "
                f"कारण: {reason}"
            ),
            "mr": (
                "अपलोड केलेल्या कायदेशीर दस्तऐवजाच्या आधारे मी या प्रश्नाचे विश्वसनीय उत्तर देऊ शकत नाही. "
                f"कारण: {reason}"
            ),
        }

        final_answer = refusal_messages.get(language, refusal_messages["en"])
        history = list(state.get("history") or [])
        history.append({"question": state["question"], "answer": final_answer[:500]})
        history = history[-MAX_HISTORY_TURNS:]

        trace.append("refuse")
        return {
            "final_answer": final_answer,
            "citations": [],
            "claims": [],
            "was_refused": True,
            "refusal_reason": reason,
            "verification_status": "refused",
            "history": history,
            "trace": trace,
        }

    async def finalize_node(state: QAState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        draft = state.get("draft") or {}
        reranked = list(state.get("reranked") or [])
        verification = state.get("verification") or {}
        verification_status = state.get("verification_status") or "verified"

        if draft.get("cannot_answer"):
            reason = (
                draft.get("refusal_reason")
                or "The retrieved document sections do not contain this information."
            )
            final_answer = (
                draft.get("answer")
                or f"I could not find enough information in the uploaded document. ({reason})"
            )
            history = list(state.get("history") or [])
            history.append({"question": state["question"], "answer": final_answer[:500]})
            history = history[-MAX_HISTORY_TURNS:]
            trace.append("finalize")
            return {
                "final_answer": final_answer,
                "citations": reranked,
                "claims": [],
                "was_refused": True,
                "refusal_reason": reason,
                "verification_status": "refused",
                "history": history,
                "trace": trace,
            }

        source_map = _source_id_to_chunk_map(reranked)
        verdict_by_index = {
            int(v.get("claim_index", -1)): bool(v.get("supported", True))
            for v in (verification.get("verdicts") or [])
        }

        enriched_claims: list[dict[str, Any]] = []
        cited_chunk_keys: set[str] = set()

        for idx, claim in enumerate(draft.get("claims") or []):
            is_supported = verdict_by_index.get(idx, True)
            if verification_status == "partially_verified" and not is_supported:
                continue

            mapped_chunk_ids: list[str] = []
            mapped_pages: list[int] = []
            for sid in claim.get("source_ids") or []:
                chunk = source_map.get(sid)
                if chunk is not None:
                    cid = str(chunk.get("chunk_id") or chunk.get("chunk_index"))
                    mapped_chunk_ids.append(cid)
                    cited_chunk_keys.add(cid)
                    if chunk.get("page_number") is not None:
                        mapped_pages.append(int(chunk["page_number"]))

            enriched_claims.append(
                {
                    "text": claim.get("text", ""),
                    "source_ids": claim.get("source_ids") or [],
                    "chunk_ids": list(dict.fromkeys(mapped_chunk_ids)),
                    "pages": sorted(set(mapped_pages)),
                    "supported": is_supported,
                }
            )

        cited_chunks = [
            c
            for c in reranked
            if str(c.get("chunk_id") or c.get("chunk_index")) in cited_chunk_keys
        ]
        citations = cited_chunks if cited_chunks else reranked

        final_answer = str(draft.get("answer", "")).strip()
        if verification_status == "partially_verified":
            final_answer += (
                "\n\n*(Note: Some parts of this response could only be partially verified "
                "against the retrieved passages; please verify against the cited pages.)*"
            )

        history = list(state.get("history") or [])
        history.append({"question": state["question"], "answer": final_answer[:500]})
        history = history[-MAX_HISTORY_TURNS:]

        trace.append("finalize")
        return {
            "final_answer": final_answer,
            "citations": citations,
            "claims": enriched_claims,
            "was_refused": False,
            "refusal_reason": "",
            "verification_status": verification_status,
            "history": history,
            "trace": trace,
        }

    def route_after_router(state: QAState) -> str:
        if state.get("intent") == "out_of_scope":
            return "refuse"
        return "analyze_query"

    def route_after_grade(state: QAState) -> str:
        if state.get("grade") == "sufficient":
            return "generate"
        if int(state.get("retry_count") or 0) < MAX_RETRIEVAL_RETRIES:
            return "rewrite_and_retry"
        return "refuse"

    def route_after_verify(state: QAState) -> str:
        if state.get("verification_status") == "needs_regeneration":
            return "generate"
        return "finalize"

    builder = StateGraph(QAState)
    builder.add_node("router", router_node)
    builder.add_node("analyze_query", analyze_query_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("grade_evidence", grade_evidence_node)
    builder.add_node("rewrite_and_retry", rewrite_and_retry_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify", verify_node)
    builder.add_node("refuse", refuse_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"analyze_query": "analyze_query", "refuse": "refuse"},
    )
    builder.add_edge("analyze_query", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "grade_evidence")
    builder.add_conditional_edges(
        "grade_evidence",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite_and_retry": "rewrite_and_retry",
            "refuse": "refuse",
        },
    )
    builder.add_edge("rewrite_and_retry", "retrieve")
    builder.add_edge("generate", "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {"generate": "generate", "finalize": "finalize"},
    )
    builder.add_edge("refuse", END)
    builder.add_edge("finalize", END)

    return builder


# ============================================================================
# EXECUTION HELPERS (WITH PERSISTENT SQLITE CHECKPOINTING)
# ============================================================================


def _initial_turn_input(
    *,
    document_id: str,
    question: str,
    session_id: str,
    language: str,
    top_k: int,
) -> QAState:
    return {
        "question": question,
        "document_id": document_id,
        "session_id": session_id,
        "language": language,
        "top_k": top_k,
        "intent": "",
        "rewritten_query": question,
        "section_refs": [],
        "page_refs": [],
        "retrieved": [],
        "reranked": [],
        "grade": "",
        "grade_reason": "",
        "missing_info": "",
        "retry_count": 0,
        "regen_count": 0,
        "verification_feedback": "",
        "draft": {},
        "verification": {},
        "verification_status": "",
        "final_answer": "",
        "citations": [],
        "claims": [],
        "was_refused": False,
        "refusal_reason": "",
        "trace": [],
    }


async def run_qa_graph(
    *,
    document_id: str,
    question: str,
    session_id: str,
    language: str = "en",
    top_k: int = 5,
    retriever: LegalRetriever,
    llm: GeminiLegalLLM,
) -> dict[str, Any]:
    """Run the stateful LangGraph QA pipeline and return the final result."""
    start = time.perf_counter()
    builder = build_qa_graph(retriever=retriever, llm=llm)
    db_path = _get_checkpoint_db_path()
    config = {"configurable": {"thread_id": session_id}}
    turn_input = _initial_turn_input(
        document_id=document_id,
        question=question,
        session_id=session_id,
        language=language,
        top_k=top_k,
    )

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        final_state = await graph.ainvoke(turn_input, config=config)

    elapsed = round(time.perf_counter() - start, 4)

    logger.info(
        "qa_graph_complete",
        document_id=document_id,
        session_id=session_id,
        intent=final_state.get("intent"),
        verification_status=final_state.get("verification_status"),
        was_refused=final_state.get("was_refused", False),
        trace=final_state.get("trace", []),
        elapsed_seconds=elapsed,
    )

    return {
        "answer": final_state.get("final_answer", ""),
        "sources": final_state.get("citations", []),
        "claims": final_state.get("claims", []),
        "intent": final_state.get("intent", "factual_qa"),
        "verification_status": final_state.get("verification_status", "verified"),
        "was_refused": bool(final_state.get("was_refused", False)),
        "refusal_reason": final_state.get("refusal_reason") or None,
        "trace": final_state.get("trace", []),
        "elapsed_seconds": elapsed,
    }


async def stream_qa_graph(
    *,
    document_id: str,
    question: str,
    session_id: str,
    language: str = "en",
    top_k: int = 5,
    retriever: LegalRetriever,
    llm: GeminiLegalLLM,
) -> AsyncIterator[dict[str, Any]]:
    """
    Execute the LangGraph QA pipeline and yield node-level progress events,
    incremental answer text chunks, and the final structured payload.
    """
    start = time.perf_counter()
    builder = build_qa_graph(retriever=retriever, llm=llm)
    db_path = _get_checkpoint_db_path()
    config = {"configurable": {"thread_id": session_id}}
    turn_input = _initial_turn_input(
        document_id=document_id,
        question=question,
        session_id=session_id,
        language=language,
        top_k=top_k,
    )

    merged_state: dict[str, Any] = dict(turn_input)

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        async for update in graph.astream(
            turn_input,
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                if isinstance(node_output, dict):
                    merged_state.update(node_output)
                yield {
                    "type": "progress",
                    "node": node_name,
                    "status": NODE_PROGRESS_LABELS.get(node_name, node_name),
                }

    final_answer = str(merged_state.get("final_answer", ""))

    words = final_answer.split(" ")
    chunk_size = 8
    for i in range(0, len(words), chunk_size):
        piece = " ".join(words[i : i + chunk_size])
        if i + chunk_size < len(words):
            piece += " "
        if piece:
            yield {"type": "token", "text": piece}

    elapsed = round(time.perf_counter() - start, 4)

    yield {
        "type": "final",
        "answer": final_answer,
        "sources": merged_state.get("citations", []),
        "claims": merged_state.get("claims", []),
        "intent": merged_state.get("intent", "factual_qa"),
        "verification_status": merged_state.get("verification_status", "verified"),
        "was_refused": bool(merged_state.get("was_refused", False)),
        "refusal_reason": merged_state.get("refusal_reason") or None,
        "trace": merged_state.get("trace", []),
        "elapsed_seconds": elapsed,
    }