"""Stateful LangGraph legal document summarization pipeline.

Supports Indian judgments, contracts, legal notices, and general legal
documents via hierarchical batch summarization, structured output reduction,
claim verification, and optional Hindi/Marathi translation.
"""

from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.logging_config import get_logger
from app.pipelines.rag.llm_service import GeminiLegalLLM


logger = get_logger("summary_graph")

MAX_CHARS_PER_BATCH = 12000
MAX_SUMMARY_REGEN = 1

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}


# ============================================================================
# STRUCTURED OUTPUT SCHEMAS (Flat types for reliable Gemini JSON output)
# ============================================================================


class DocTypeClassification(BaseModel):
    document_type: Literal["judgment", "contract", "notice", "other"] = Field(
        description="Primary legal category of the document."
    )
    reason: str = Field(
        description="Short reason for the classification."
    )


class BatchSummaryOutput(BaseModel):
    batch_summary: str = Field(
        description="Concise factual summary of this batch with page references like (Page X)."
    )
    key_entities: list[str] = Field(
        description="Key parties, courts, statutes, sections, or precedents mentioned in this batch."
    )


class SummaryClaim(BaseModel):
    text: str = Field(description="A single key factual claim made in the summary.")
    pages: list[int] = Field(
        description="Page numbers supporting this claim."
    )


class JudgmentSummaryOutput(BaseModel):
    court: str = Field(description="Name of the court or tribunal, or 'Not specified'.")
    date: str = Field(description="Date of judgment/order, or 'Not specified'.")
    case_number: str = Field(description="Case/appeal number, or 'Not specified'.")
    parties: list[str] = Field(description="List of parties (appellant/petitioner, respondent, etc.).")
    facts: list[str] = Field(description="Key background facts with page references.")
    issues: list[str] = Field(description="Main legal issues before the court with page references.")
    petitioner_arguments: list[str] = Field(description="Arguments advanced by appellant/petitioner.")
    respondent_arguments: list[str] = Field(description="Arguments advanced by respondent.")
    reasoning: list[str] = Field(description="Court's legal analysis and reasoning with page references.")
    holding: str = Field(description="Core legal holding or ratio decidendi with page references.")
    final_order: str = Field(description="Operative final order/directions with page references.")
    statutes_cited: list[str] = Field(description="Acts, sections, articles, or rules cited.")
    precedents_cited: list[str] = Field(description="Case precedents cited in the judgment.")
    markdown_summary: str = Field(
        description="Well-structured Markdown summary matching the requested detail level with page citations."
    )
    claims: list[SummaryClaim] = Field(
        description="5-10 core factual claims from the summary mapped to supporting page numbers."
    )


class UnusualClause(BaseModel):
    clause_reference: str = Field(description="Clause number/heading and page number.")
    risk_explanation: str = Field(description="Why the clause is unusual, onerous, or one-sided.")


class ContractSummaryOutput(BaseModel):
    parties: list[str] = Field(description="Contracting parties and their roles.")
    effective_date: str = Field(description="Execution or effective date, or 'Not specified'.")
    term: str = Field(description="Duration/term and renewal terms with page references.")
    obligations: list[str] = Field(description="Key obligations of each party with page references.")
    payment_terms: str = Field(description="Consideration, pricing, invoicing, and payment terms.")
    termination: str = Field(description="Termination rights, notice periods, and consequences.")
    liability_and_indemnity: str = Field(description="Liability caps, indemnities, and warranties.")
    governing_law_and_jurisdiction: str = Field(description="Governing law, arbitration, and jurisdiction.")
    unusual_or_one_sided_clauses: list[UnusualClause] = Field(
        description="Any unusual, high-risk, or one-sided clauses found, or empty list."
    )
    markdown_summary: str = Field(
        description="Well-structured Markdown summary matching the requested detail level with page citations."
    )
    claims: list[SummaryClaim] = Field(
        description="5-10 core factual claims from the summary mapped to supporting page numbers."
    )


class GeneralLegalSummaryOutput(BaseModel):
    title: str = Field(description="Document title or subject.")
    date: str = Field(description="Document date, or 'Not specified'.")
    parties: list[str] = Field(description="Sender, recipient, or involved parties.")
    key_points: list[str] = Field(description="Main facts, demands, or clauses with page references.")
    deadlines_or_actions: list[str] = Field(description="Required actions, timelines, or deadlines.")
    statutes_or_clauses_cited: list[str] = Field(description="Statutes, sections, or clauses referenced.")
    markdown_summary: str = Field(
        description="Well-structured Markdown summary matching the requested detail level with page citations."
    )
    claims: list[SummaryClaim] = Field(
        description="5-8 core factual claims from the summary mapped to supporting page numbers."
    )


class SummaryClaimVerdict(BaseModel):
    claim_index: int = Field(description="Zero-based index of the claim.")
    supported: bool = Field(description="True only if supported by the source document text.")
    reason: str = Field(description="Short verification justification.")


class SummaryVerificationOutput(BaseModel):
    all_supported: bool = Field(description="True if every claim is supported by the document.")
    verdicts: list[SummaryClaimVerdict] = Field(description="Verdict for each claim.")


class TranslatedSummaryOutput(BaseModel):
    translated_markdown: str = Field(
        description="Accurate translation preserving Markdown headings, legal terms, and page citations."
    )


# ============================================================================
# GRAPH STATE
# ============================================================================


class SummaryState(TypedDict, total=False):
    document_id: str
    summary_type: str  # short, medium, detailed
    language: str  # en, hi, mr
    chunks: list[dict[str, Any]]

    document_type: str
    batch_summaries: list[dict[str, Any]]
    structured_summary: dict[str, Any]
    markdown_summary: str
    claims: list[dict[str, Any]]

    regen_count: int
    verification_feedback: str
    verification: dict[str, Any]
    verification_status: str

    final_markdown: str
    extracted_entities: list[dict[str, Any]]
    trace: list[str]


# ============================================================================
# HELPERS
# ============================================================================


def _batch_chunks(chunks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group chunks into context-safe batches preserving document order."""
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_chars = 0

    for chunk in chunks:
        text = str(chunk.get("content", ""))
        text_len = len(text) + 120
        if current_batch and (current_chars + text_len > MAX_CHARS_PER_BATCH):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(chunk)
        current_chars += text_len

    if current_batch:
        batches.append(current_batch)

    return batches


def _format_chunk_batch(batch: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for chunk in batch:
        page = chunk.get("page_number", "unknown")
        section = chunk.get("section_title") or chunk.get("section_type") or "general"
        idx = chunk.get("chunk_index", 0)
        content = str(chunk.get("content", "")).strip()
        blocks.append(
            f"[Chunk {idx} | Page {page} | Section: {section}]\n{content}"
        )
    return "\n\n".join(blocks)


def _detail_instruction(summary_type: str) -> str:
    if summary_type == "short":
        return (
            "Keep the Markdown summary concise (150-250 words) focusing on the core "
            "outcome, parties, and key obligations/holdings."
        )
    if summary_type == "detailed":
        return (
            "Provide a comprehensive, well-organized Markdown summary covering all "
            "major sections, arguments/clauses, dates, legal citations, and page references."
        )
    return (
        "Provide a clear, balanced Markdown summary (350-600 words) with bulleted "
        "headings and explicit page references."
    )


def _extract_entities_from_summary(
    doc_type: str,
    structured: dict[str, Any],
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add(entity_type: str, value: Any) -> None:
        text = str(value or "").strip()
        if not text or text.lower() in {"not specified", "none", "n/a", "unknown"}:
            return
        key = (entity_type, text.lower())
        if key in seen:
            return
        seen.add(key)
        entities.append({"entity_type": entity_type, "entity_value": text[:1900]})

    for party in structured.get("parties") or []:
        _add("party", party)

    if doc_type == "judgment":
        _add("court", structured.get("court"))
        _add("case_number", structured.get("case_number"))
        _add("date", structured.get("date"))
        for statute in structured.get("statutes_cited") or []:
            _add("act", statute)
        for precedent in structured.get("precedents_cited") or []:
            _add("precedent", precedent)
    elif doc_type == "contract":
        _add("date", structured.get("effective_date"))
        _add("jurisdiction", structured.get("governing_law_and_jurisdiction"))
    else:
        _add("date", structured.get("date"))
        for statute in structured.get("statutes_or_clauses_cited") or []:
            _add("act", statute)

    return entities


# ============================================================================
# GRAPH BUILDER
# ============================================================================


def build_summary_graph(llm: GeminiLegalLLM) -> StateGraph:
    """Build the LangGraph StateGraph for structured legal summarization."""

    async def detect_document_type_node(state: SummaryState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        chunks = list(state.get("chunks") or [])
        early_text = "\n\n".join(
            str(c.get("content", "")) for c in chunks[:3]
        )[:4500]

        prompt = (
            "Classify this Indian legal document into one of: "
            "'judgment' (court/tribunal judgment or order), "
            "'contract' (agreement, deed, lease, NDA, MoU), "
            "'notice' (legal notice, demand notice, show-cause notice), or 'other'.\n\n"
            f"DOCUMENT OPENING EXCERPT:\n{early_text}\n"
        )

        doc_type = "other"
        try:
            res = await llm.generate_structured(
                prompt,
                DocTypeClassification,
                temperature=0.0,
            )
            doc_type = res.document_type
        except Exception as exc:
            logger.warning("summary_doc_type_fallback", error=str(exc))
            lower = early_text.lower()
            if any(k in lower for k in ("appellant", "respondent", "hon'ble", "judgment", "tribunal", "court")):
                doc_type = "judgment"
            elif any(k in lower for k in ("agreement", "hereinafter", "indemnity", "termination", "whereas")):
                doc_type = "contract"
            elif "notice" in lower:
                doc_type = "notice"

        trace.append("detect_document_type")
        logger.info(
            "summary_doc_type_detected",
            document_id=state["document_id"],
            document_type=doc_type,
        )
        return {"document_type": doc_type, "trace": trace}

    async def summarize_sections_node(state: SummaryState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        chunks = list(state.get("chunks") or [])
        doc_type = state.get("document_type") or "other"
        batches = _batch_chunks(chunks)

        # For short documents that fit cleanly in a single batch, keep the raw
        # text directly as batch 1 to avoid unnecessary extra LLM calls.
        if len(batches) == 1:
            pages = [
                int(c["page_number"])
                for c in batches[0]
                if c.get("page_number") is not None
            ]
            batch_summaries = [
                {
                    "batch_index": 1,
                    "pages": sorted(set(pages)),
                    "summary": _format_chunk_batch(batches[0]),
                    "key_entities": [],
                }
            ]
            trace.append("summarize_sections")
            return {"batch_summaries": batch_summaries, "trace": trace}

        batch_summaries: list[dict[str, Any]] = []
        for idx, batch in enumerate(batches, start=1):
            batch_text = _format_chunk_batch(batch)
            pages = sorted(
                {
                    int(c["page_number"])
                    for c in batch
                    if c.get("page_number") is not None
                }
            )
            prompt = (
                f"You are summarizing Part {idx} of {len(batches)} of an Indian legal {doc_type}.\n"
                "Extract all legally material facts, parties, dates, section/clause numbers, "
                "arguments, obligations, orders, and unusual terms.\n"
                "Every point MUST preserve its page reference like (Page X). Do not invent anything.\n\n"
                f"TEXT:\n{batch_text}\n"
            )
            out = await llm.generate_structured(
                prompt,
                BatchSummaryOutput,
                temperature=0.1,
            )
            batch_summaries.append(
                {
                    "batch_index": idx,
                    "pages": pages,
                    "summary": out.batch_summary,
                    "key_entities": out.key_entities,
                }
            )

        trace.append("summarize_sections")
        return {"batch_summaries": batch_summaries, "trace": trace}

    async def reduce_summary_node(state: SummaryState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        doc_type = state.get("document_type") or "other"
        summary_type = state.get("summary_type") or "medium"
        batch_summaries = list(state.get("batch_summaries") or [])
        feedback = state.get("verification_feedback") or ""

        combined_source = "\n\n".join(
            f"=== PART {b['batch_index']} (Pages {b['pages']}) ===\n{b['summary']}"
            for b in batch_summaries
        )

        feedback_block = (
            f"\nPREVIOUS VERIFICATION FEEDBACK (CORRECT THESE UNSUPPORTED ITEMS):\n{feedback}\n"
            if feedback
            else ""
        )

        prompt = (
            f"You are NyayaSetu, an Indian legal document summarization engine.\n"
            f"Produce a strictly grounded, structured summary of this {doc_type} in English.\n"
            "RULES:\n"
            "1. Use ONLY facts, dates, party names, statutes, clauses, and orders present in the source text.\n"
            "2. If a field is absent from the document, write 'Not specified' or an empty list [].\n"
            "3. Attach explicit page references like (Page 3) to facts, holdings, clauses, and claims.\n"
            f"4. {_detail_instruction(summary_type)}\n"
            f"{feedback_block}\n"
            f"DOCUMENT SOURCE MATERIAL:\n{combined_source}\n"
        )

        if doc_type == "judgment":
            schema: type[BaseModel] = JudgmentSummaryOutput
        elif doc_type == "contract":
            schema = ContractSummaryOutput
        else:
            schema = GeneralLegalSummaryOutput

        reduced = await llm.generate_structured(
            prompt,
            schema,
            temperature=0.1,
        )
        data = reduced.model_dump()
        claims = list(data.pop("claims", []))
        markdown = str(data.pop("markdown_summary", "")).strip()

        trace.append("reduce_summary")
        return {
            "structured_summary": data,
            "markdown_summary": markdown,
            "claims": claims,
            "trace": trace,
        }

    async def verify_summary_node(state: SummaryState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        claims = list(state.get("claims") or [])
        chunks = list(state.get("chunks") or [])
        regen_count = int(state.get("regen_count") or 0)

        if not claims:
            trace.append("verify_summary")
            return {
                "verification": {"all_supported": True, "verdicts": []},
                "verification_status": "verified",
                "trace": trace,
            }

        # Collect chunks matching cited pages (or first chunks as fallback).
        cited_pages: set[int] = set()
        for claim in claims:
            for p in claim.get("pages") or []:
                try:
                    cited_pages.add(int(p))
                except (TypeError, ValueError):
                    pass

        relevant_chunks = [
            c
            for c in chunks
            if c.get("page_number") is not None and int(c["page_number"]) in cited_pages
        ]
        if not relevant_chunks:
            relevant_chunks = chunks[:8]

        evidence_text = _format_chunk_batch(relevant_chunks)[:14000]
        claims_block = "\n".join(
            f"Claim {idx} (Pages {c.get('pages', [])}): {c.get('text', '')}"
            for idx, c in enumerate(claims)
        )

        prompt = (
            "Verify each summary claim strictly against the provided legal document excerpts.\n"
            "Mark supported=true ONLY if directly supported by the excerpts; mark supported=false "
            "if inaccurate, exaggerated, or fabricated.\n\n"
            "CLAIMS:\n"
            f"{claims_block}\n\n"
            "DOCUMENT EXCERPTS:\n"
            f"{evidence_text}\n"
        )

        try:
            ver_out = await llm.generate_structured(
                prompt,
                SummaryVerificationOutput,
                temperature=0.0,
            )
            verdicts = [v.model_dump() for v in ver_out.verdicts]
        except Exception as exc:
            logger.warning("summary_verification_skipped", error=str(exc))
            trace.append("verify_summary")
            return {
                "verification": {"all_supported": True, "verdicts": []},
                "verification_status": "unverified",
                "trace": trace,
            }

        unsupported = [v for v in verdicts if not v.get("supported", False)]
        trace.append("verify_summary")

        if not unsupported:
            return {
                "verification": {"all_supported": True, "verdicts": verdicts},
                "verification_status": "verified",
                "verification_feedback": "",
                "trace": trace,
            }

        feedback = "\n".join(
            f"- Claim {u.get('claim_index')}: {u.get('reason')}"
            for u in unsupported
        )

        if regen_count < MAX_SUMMARY_REGEN:
            return {
                "verification": {"all_supported": False, "verdicts": verdicts},
                "verification_status": "needs_regeneration",
                "verification_feedback": feedback,
                "regen_count": regen_count + 1,
                "trace": trace,
            }

        return {
            "verification": {"all_supported": False, "verdicts": verdicts},
            "verification_status": "partially_verified",
            "verification_feedback": feedback,
            "trace": trace,
        }

    async def translate_if_needed_node(state: SummaryState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        language = state.get("language") or "en"
        markdown = state.get("markdown_summary") or ""

        if language == "en" or not markdown:
            trace.append("translate_if_needed")
            return {"final_markdown": markdown, "trace": trace}

        target_lang = LANGUAGE_NAMES.get(language, "English")
        prompt = (
            f"Translate the following Indian legal document summary into natural, accurate {target_lang}.\n"
            "Preserve all Markdown headings, bullet structure, proper names, section/act numbers, "
            "and page citations like (Page 3).\n\n"
            f"SUMMARY:\n{markdown}\n"
        )

        translated = markdown
        try:
            out = await llm.generate_structured(
                prompt,
                TranslatedSummaryOutput,
                temperature=0.1,
            )
            if out.translated_markdown.strip():
                translated = out.translated_markdown.strip()
        except Exception as exc:
            logger.warning(
                "summary_translation_failed_returning_english",
                language=language,
                error=str(exc),
            )

        trace.append("translate_if_needed")
        return {"final_markdown": translated, "trace": trace}

    async def finalize_node(state: SummaryState) -> dict[str, Any]:
        trace = list(state.get("trace") or [])
        doc_type = state.get("document_type") or "other"
        structured = dict(state.get("structured_summary") or {})
        final_markdown = state.get("final_markdown") or state.get("markdown_summary") or ""
        verification_status = state.get("verification_status") or "verified"

        if verification_status == "partially_verified":
            final_markdown += (
                "\n\n*(Note: Some claims in this summary were flagged as partially verified; "
                "please cross-check against the cited pages.)*"
            )

        entities = _extract_entities_from_summary(doc_type, structured)
        trace.append("finalize")

        return {
            "final_markdown": final_markdown,
            "extracted_entities": entities,
            "trace": trace,
        }

    def route_after_verify(state: SummaryState) -> str:
        if state.get("verification_status") == "needs_regeneration":
            return "reduce_summary"
        return "translate_if_needed"

    builder = StateGraph(SummaryState)
    builder.add_node("detect_document_type", detect_document_type_node)
    builder.add_node("summarize_sections", summarize_sections_node)
    builder.add_node("reduce_summary", reduce_summary_node)
    builder.add_node("verify_summary", verify_summary_node)
    builder.add_node("translate_if_needed", translate_if_needed_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "detect_document_type")
    builder.add_edge("detect_document_type", "summarize_sections")
    builder.add_edge("summarize_sections", "reduce_summary")
    builder.add_edge("reduce_summary", "verify_summary")
    builder.add_conditional_edges(
        "verify_summary",
        route_after_verify,
        {
            "reduce_summary": "reduce_summary",
            "translate_if_needed": "translate_if_needed",
        },
    )
    builder.add_edge("translate_if_needed", "finalize")
    builder.add_edge("finalize", END)

    return builder


async def run_summary_graph(
    *,
    document_id: str,
    chunks: list[dict[str, Any]],
    summary_type: str = "medium",
    language: str = "en",
    llm: GeminiLegalLLM | None = None,
) -> dict[str, Any]:
    """Execute the LangGraph summarization pipeline over document chunks."""
    start = time.perf_counter()
    llm_service = llm or GeminiLegalLLM()
    graph = build_summary_graph(llm_service).compile()

    initial_state: SummaryState = {
        "document_id": document_id,
        "summary_type": summary_type,
        "language": language,
        "chunks": chunks,
        "regen_count": 0,
        "verification_feedback": "",
        "trace": [],
    }

    final_state = await graph.ainvoke(initial_state)
    elapsed = round(time.perf_counter() - start, 4)

    logger.info(
        "summary_graph_complete",
        document_id=document_id,
        document_type=final_state.get("document_type"),
        summary_type=summary_type,
        language=language,
        verification_status=final_state.get("verification_status"),
        trace=final_state.get("trace", []),
        elapsed_seconds=elapsed,
    )

    return {
        "document_type": final_state.get("document_type", "other"),
        "structured_summary": final_state.get("structured_summary", {}),
        "markdown_summary": final_state.get("final_markdown", ""),
        "claims": final_state.get("claims", []),
        "verification_status": final_state.get("verification_status", "verified"),
        "verification": final_state.get("verification", {}),
        "extracted_entities": final_state.get("extracted_entities", []),
        "trace": final_state.get("trace", []),
        "model_used": llm_service.primary_model,
        "elapsed_seconds": elapsed,
    }