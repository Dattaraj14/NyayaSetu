"""Document upload, extraction, OCR, preprocessing, structure detection,
legal-aware chunking, and deletion routes for NyayaSetu.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import time
from typing import Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.database import get_db
from app.models.document import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentStatus,
    ExtractionMethod,
    PDFType,
)
from app.pipelines.extraction.pdf_extractor import PDFExtractor
from app.pipelines.ocr.ocr_engine import ALLOWED_OCR_LANGUAGES, OCRPipeline
from app.pipelines.preprocessing.preprocessor import LegalTextPreprocessor
from app.pipelines.rag.chunker import LegalChunker
from app.pipelines.rag.vector_store import LegalVectorStore
from app.pipelines.structure.structure_detector import LegalStructureDetector
from app.schemas.document import DocumentDetail, DocumentUploadResponse
from app.security.file_validator import (
    generate_safe_filename,
    validate_file_extension,
    validate_file_size,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

logger = get_logger("documents")
settings = get_settings()


# ============================================================================
# LEGAL STRUCTURE DETECTION
# ============================================================================


async def save_detected_structure(
    db: AsyncSession,
    document_id: UUID,
    pages: list[dict],
) -> dict:
    """Detect legal structure from processed pages and persist sections."""
    structure_start = time.perf_counter()

    detector = LegalStructureDetector()
    structure_result = await asyncio.to_thread(
        detector.detect,
        pages,
    )

    structure_elapsed = round(
        time.perf_counter() - structure_start,
        4,
    )

    if isinstance(structure_result, dict):
        sections = structure_result.get("sections", [])
        raw_paragraphs = structure_result.get("paragraphs", [])
        section_source = structure_result.get("section_source", "unknown")
    else:
        sections = getattr(structure_result, "sections", [])
        raw_paragraphs = getattr(structure_result, "paragraphs", [])
        section_source = getattr(structure_result, "section_source", "unknown")

    paragraphs: list[dict] = []

    for sequence, paragraph in enumerate(raw_paragraphs, start=1):
        if isinstance(paragraph, dict):
            paragraph_dict = dict(paragraph)
            paragraph_dict.setdefault("sequence_number", sequence)
            paragraph_dict.setdefault("number", paragraph_dict.get("paragraph_number"))
            paragraph_dict.setdefault("content", paragraph_dict.get("text", ""))
            paragraph_dict.setdefault("start_page", paragraph_dict.get("page_number", 1))
            paragraph_dict.setdefault("end_page", paragraph_dict.get("start_page", 1))
            paragraph_dict.setdefault("section_type", "general")
            paragraph_dict.setdefault("section_title", "General")
            paragraph_dict.setdefault("ocr_artifacts", [])
            paragraphs.append(paragraph_dict)
            continue

        number = getattr(
            paragraph,
            "number",
            getattr(paragraph, "paragraph_number", None),
        )
        content = getattr(
            paragraph,
            "content",
            getattr(paragraph, "text", ""),
        )
        start_page = getattr(paragraph, "start_page", 1)
        end_page = getattr(paragraph, "end_page", start_page)
        section_type = getattr(paragraph, "section_type", "general")
        section_title = getattr(
            paragraph,
            "section_title",
            getattr(paragraph, "title", "General"),
        )
        ocr_artifacts = getattr(paragraph, "ocr_artifacts", [])

        paragraphs.append(
            {
                "sequence_number": sequence,
                "paragraph_number": number,
                "source_paragraph_number": number,
                "raw_paragraph_number": number,
                "expected_paragraph_number": None,
                "paragraph_number_corrected": False,
                "ocr_artifact": bool(ocr_artifacts),
                "ocr_artifact_reason": ocr_artifacts[0] if ocr_artifacts else None,
                "number": number,
                "content": content,
                "text": content,
                "start_page": start_page,
                "end_page": end_page,
                "section_type": section_type,
                "section_title": section_title,
                "ocr_artifacts": ocr_artifacts,
                "order_index": getattr(paragraph, "order_index", sequence),
            }
        )

    logger.info(
        "legal_structure_detection_complete",
        document_id=str(document_id),
        sections=len(sections),
        paragraphs=len(paragraphs),
        section_source=section_source,
        elapsed_seconds=structure_elapsed,
    )

    await db.execute(
        delete(DocumentSection).where(DocumentSection.document_id == document_id)
    )

    for section in sections:
        if isinstance(section, dict):
            section_type = section.get("section_type", "general")
            section_title = section.get(
                "section_title", section.get("title", "General")
            )
            content = section.get("content", "")
            start_page = section.get("start_page")
            end_page = section.get("end_page", start_page)
            order_index = section.get("order_index", 0)
        else:
            section_type = getattr(section, "section_type", "general")
            section_title = getattr(
                section,
                "section_title",
                getattr(section, "title", "General"),
            )
            content = getattr(section, "content", "")
            start_page = getattr(section, "start_page", None)
            end_page = getattr(section, "end_page", start_page)
            order_index = getattr(section, "order_index", 0)

        db.add(
            DocumentSection(
                document_id=document_id,
                section_type=section_type,
                section_title=section_title,
                content=content,
                start_page=start_page,
                end_page=end_page,
                order_index=order_index,
            )
        )

    await db.commit()

    logger.info(
        "legal_structure_saved",
        document_id=str(document_id),
        sections=len(sections),
        paragraphs=len(paragraphs),
    )

    return {
        "sections": sections,
        "paragraphs": paragraphs,
        "section_source": section_source,
    }


# ============================================================================
# LEGAL CHUNKING
# ============================================================================


async def save_document_chunks(
    db: AsyncSession,
    document_id: UUID,
    paragraphs: list[dict],
    sections: list,
) -> list:
    """Create legal-aware chunks from detected paragraphs and sections."""
    logger.info(
        "legal_chunking_started",
        document_id=str(document_id),
        paragraphs=len(paragraphs),
        sections=len(sections),
    )

    chunker = LegalChunker(target_tokens=700, max_tokens=1000)
    chunk_start = time.perf_counter()

    chunks = await asyncio.to_thread(
        chunker.create_chunks,
        paragraphs,
        sections,
    )

    logger.info(
        "legal_chunking_complete",
        document_id=str(document_id),
        chunks=len(chunks),
        paragraphs=len(paragraphs),
        elapsed_seconds=round(time.perf_counter() - chunk_start, 4),
    )

    await db.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )

    for chunk in chunks:
        metadata = {
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "section_title": chunk.section_title,
            "paragraph_numbers": chunk.paragraph_numbers,
            "paragraph_sequences": chunk.paragraph_sequences,
            "ocr_artifacts": chunk.ocr_artifacts,
        }

        db.add(
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                page_number=chunk.start_page,
                section_type=chunk.section_type,
                language="en",
                metadata_json=metadata,
                vector_id=None,
            )
        )

    await db.commit()

    logger.info(
        "legal_chunks_saved",
        document_id=str(document_id),
        chunks=len(chunks),
    )

    return chunks


# ============================================================================
# PARALLEL OCR HELPER
# ============================================================================


def _run_parallel_ocr(
    ocr_pipeline: OCRPipeline,
    rendered_images: dict[int, bytes],
    language: str,
    max_workers: int,
) -> dict[int, dict]:
    """Run Tesseract OCR across rendered page images in parallel."""
    if not rendered_images:
        return {}

    def _ocr_one(item: tuple[int, bytes]) -> tuple[int, dict, float]:
        page_number, image_bytes = item
        start = time.perf_counter()
        res = ocr_pipeline.ocr_page(
            image_bytes,
            page_number=page_number,
            language=language,
        )
        return page_number, res, time.perf_counter() - start

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for page_number, ocr_res, elapsed in executor.map(
            _ocr_one,
            sorted(rendered_images.items()),
        ):
            ocr_res["elapsed_seconds"] = round(elapsed, 4)
            results[page_number] = ocr_res

    return results


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================


async def process_document(
    document_id: UUID,
    file_path: str,
    ocr_language: Optional[str] = None,
) -> None:
    """Complete document-processing pipeline."""
    async for db in get_db():
        try:
            pipeline_start = time.perf_counter()
            commit_batch = max(1, settings.ocr_commit_batch_size)

            # 1. Load document
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if document is None:
                logger.error(
                    "document_not_found_for_processing",
                    document_id=str(document_id),
                )
                return

            # 2. Mark extracting
            document.status = DocumentStatus.EXTRACTING
            document.status_message = "Detecting PDF type..."
            document.processing_progress = 5.0
            await db.commit()

            # 3. Detect PDF type
            detect_start = time.perf_counter()
            pdf_type, pdf_info = await asyncio.to_thread(
                PDFExtractor.detect_pdf_type,
                file_path,
            )

            logger.info(
                "pdf_type_detected",
                document_id=str(document_id),
                pdf_type=pdf_type,
                total_pages=pdf_info.get("total_pages"),
                text_pages=pdf_info.get("text_pages"),
                image_pages=pdf_info.get("image_pages"),
                elapsed_seconds=round(time.perf_counter() - detect_start, 4),
            )

            try:
                document.pdf_type = PDFType(pdf_type)
            except ValueError:
                document.pdf_type = None

            document.page_count = pdf_info.get("total_pages", 0)
            document.status_message = f"PDF detected as {pdf_type}"
            await db.commit()

            # 4. Remove old processing data
            await db.execute(
                delete(DocumentPage).where(DocumentPage.document_id == document_id)
            )
            await db.execute(
                delete(DocumentSection).where(
                    DocumentSection.document_id == document_id
                )
            )
            await db.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            await db.commit()

            # 5. Initialize preprocessing
            preprocessor = LegalTextPreprocessor()
            processed_pages: list[dict] = []

            # 6. TEXT PDF
            if pdf_type == "text":
                document.status_message = "Extracting text from PDF..."
                document.processing_progress = 10.0
                await db.commit()

                extraction_start = time.perf_counter()
                pages = await asyncio.to_thread(
                    PDFExtractor.extract_text,
                    file_path,
                )
                total_pages = len(pages)

                for index, page_data in enumerate(pages):
                    page_number = page_data["page_number"]
                    raw_text = page_data.get("raw_text", "")
                    processed_text = preprocessor.preprocess_page(raw_text)[
                        "processed_text"
                    ]

                    processed_pages.append(
                        {
                            "page_number": page_number,
                            "processed_text": processed_text,
                        }
                    )

                    db.add(
                        DocumentPage(
                            document_id=document_id,
                            page_number=page_number,
                            raw_text=raw_text,
                            processed_text=processed_text,
                            extraction_method=ExtractionMethod.TEXT,
                            ocr_confidence=None,
                            has_images=page_data.get("has_images", False),
                            is_low_confidence=False,
                        )
                    )

                    if (index + 1) % commit_batch == 0 or (index + 1) == total_pages:
                        progress = 10.0 + ((index + 1) / max(total_pages, 1)) * 65.0
                        document.processing_progress = round(progress, 2)
                        document.status_message = (
                            f"Processed page {page_number}/{total_pages}"
                        )
                        await db.commit()

                logger.info(
                    "extraction_complete",
                    document_id=str(document_id),
                    pdf_type=pdf_type,
                    pages=total_pages,
                    elapsed_seconds=round(time.perf_counter() - extraction_start, 4),
                )

            # 7. SCANNED / HYBRID PDF
            elif pdf_type in {"scanned", "hybrid"}:
                selected_language = ocr_language or settings.ocr_languages or "eng"
                render_dpi = settings.ocr_render_dpi
                workers = settings.ocr_workers or min(4, os.cpu_count() or 1)

                document.status = DocumentStatus.OCR_PROCESSING
                document.status_message = (
                    f"Running OCR ({selected_language}, {render_dpi} DPI)..."
                )
                document.processing_progress = 10.0
                await db.commit()

                extraction_start = time.perf_counter()

                page_plan = await asyncio.to_thread(
                    PDFExtractor.inspect_pages_for_ocr,
                    file_path,
                )
                total_pages = len(page_plan)
                pages_needing_ocr = [
                    p["page_number"] for p in page_plan if p["needs_ocr"]
                ]

                render_start = time.perf_counter()
                rendered_images = await asyncio.to_thread(
                    PDFExtractor.render_pages_as_images,
                    file_path,
                    pages_needing_ocr,
                    render_dpi,
                )
                render_elapsed = round(time.perf_counter() - render_start, 4)

                logger.info(
                    "pages_rendered_for_ocr",
                    document_id=str(document_id),
                    pages_rendered=len(rendered_images),
                    pages_skipped_text_layer=total_pages - len(rendered_images),
                    dpi=render_dpi,
                    elapsed_seconds=render_elapsed,
                )

                ocr_results: dict[int, dict] = {}
                ocr_wall_seconds = 0.0
                if rendered_images:
                    ocr_pipeline = OCRPipeline(language=selected_language)
                    ocr_start = time.perf_counter()
                    ocr_results = await asyncio.to_thread(
                        _run_parallel_ocr,
                        ocr_pipeline,
                        rendered_images,
                        selected_language,
                        workers,
                    )
                    ocr_wall_seconds = round(time.perf_counter() - ocr_start, 4)

                document.status = DocumentStatus.PREPROCESSING
                for index, page_item in enumerate(page_plan):
                    page_number = page_item["page_number"]

                    if page_item["needs_ocr"]:
                        ocr_res = ocr_results[page_number]
                        raw_text = ocr_res.get("text", "")
                        confidence: Optional[float] = float(
                            ocr_res.get("confidence", 0.0)
                        )
                        is_low_confidence = bool(
                            ocr_res.get("is_low_confidence", False)
                        )
                        extraction_method = ExtractionMethod.OCR

                        logger.info(
                            "page_ocr_complete",
                            document_id=str(document_id),
                            page_number=page_number,
                            confidence=round(confidence, 4),
                            elapsed_seconds=ocr_res.get("elapsed_seconds", 0.0),
                        )
                    else:
                        raw_text = page_item["raw_text"]
                        confidence = None
                        is_low_confidence = False
                        extraction_method = ExtractionMethod.TEXT

                    processed_text = preprocessor.preprocess_page(raw_text)[
                        "processed_text"
                    ]
                    processed_pages.append(
                        {
                            "page_number": page_number,
                            "processed_text": processed_text,
                        }
                    )

                    db.add(
                        DocumentPage(
                            document_id=document_id,
                            page_number=page_number,
                            raw_text=raw_text,
                            processed_text=processed_text,
                            extraction_method=extraction_method,
                            ocr_confidence=confidence,
                            has_images=page_item["has_images"],
                            is_low_confidence=is_low_confidence,
                        )
                    )

                    if (index + 1) % commit_batch == 0 or (index + 1) == total_pages:
                        progress = 10.0 + ((index + 1) / max(total_pages, 1)) * 65.0
                        document.processing_progress = round(progress, 2)
                        document.status_message = (
                            f"Completed page {page_number}/{total_pages}"
                        )
                        await db.commit()

                logger.info(
                    "extraction_complete",
                    document_id=str(document_id),
                    pdf_type=pdf_type,
                    pages=total_pages,
                    ocr_pages=len(rendered_images),
                    ocr_language=selected_language,
                    dpi=render_dpi,
                    workers=workers,
                    render_seconds_total=render_elapsed,
                    ocr_wall_seconds=ocr_wall_seconds,
                    elapsed_seconds=round(time.perf_counter() - extraction_start, 4),
                )

            else:
                raise ValueError(f"Unsupported or unknown PDF type: {pdf_type}")

            # 8. Structure detection
            document.status = DocumentStatus.ANALYZING
            document.status_message = "Detecting legal document structure..."
            document.processing_progress = 80.0
            await db.commit()

            structure_result = await save_detected_structure(
                db=db,
                document_id=document_id,
                pages=processed_pages,
            )

            # 9. Legal chunking
            document.status_message = "Creating legal document chunks..."
            document.processing_progress = 90.0
            await db.commit()

            chunks = await save_document_chunks(
                db=db,
                document_id=document_id,
                paragraphs=structure_result["paragraphs"],
                sections=structure_result["sections"],
            )

            # 10. Completed
            document.status = DocumentStatus.COMPLETED
            document.status_message = (
                "Extraction, preprocessing, legal structure detection, "
                "and chunking completed"
            )
            document.processing_progress = 100.0
            await db.commit()

            event_name = (
                "text_processing_complete"
                if pdf_type == "text"
                else "ocr_processing_complete"
            )
            logger.info(
                event_name,
                document_id=str(document_id),
                pages=len(processed_pages),
                sections=len(structure_result["sections"]),
                paragraphs=len(structure_result["paragraphs"]),
                chunks=len(chunks),
                elapsed_seconds=round(time.perf_counter() - pipeline_start, 4),
            )
            return

        except Exception as exc:
            logger.exception(
                "document_processing_failed",
                document_id=str(document_id),
                error=str(exc),
            )

            await db.rollback()

            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.status_message = f"Processing failed: {str(exc)}"
                document.processing_progress = 0.0
                await db.commit()
            return

        finally:
            await db.close()


# ============================================================================
# DOCUMENT UPLOAD
# ============================================================================


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ocr_language: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF document for processing."""
    if ocr_language is not None and ocr_language not in ALLOWED_OCR_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported ocr_language '{ocr_language}'. "
                f"Allowed values: {sorted(ALLOWED_OCR_LANGUAGES)}"
            ),
        )

    try:
        validate_file_extension(file.filename or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_content = await file.read()

    try:
        validate_file_size(len(file_content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    safe_filename = generate_safe_filename(file.filename or "document.pdf")
    upload_dir = settings.upload_dir
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, safe_filename)

    try:
        with open(file_path, "wb") as output_file:
            output_file.write(file_content)
    except OSError as exc:
        logger.exception("file_save_failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        ) from exc

    document = Document(
        filename=safe_filename,
        original_filename=file.filename or "document.pdf",
        file_path=file_path,
        file_size=len(file_content),
        mime_type=file.content_type or "application/pdf",
        page_count=0,
        status=DocumentStatus.UPLOADED,
        status_message="File uploaded successfully",
        processing_progress=0.0,
    )

    db.add(document)

    try:
        await db.commit()
        await db.refresh(document)
    except Exception as exc:
        await db.rollback()
        try:
            os.remove(file_path)
        except OSError:
            pass
        logger.exception("document_database_creation_failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="Failed to create document record.",
        ) from exc

    background_tasks.add_task(
        process_document,
        document.id,
        file_path,
        ocr_language,
    )

    logger.info(
        "document_uploaded",
        document_id=str(document.id),
        filename=file.filename,
        size=len(file_content),
        ocr_language=ocr_language or settings.ocr_languages,
    )

    return DocumentUploadResponse(
        id=document.id,
        filename=document.original_filename,
        status=document.status.value,
        message="Document uploaded and processing started.",
    )


# ============================================================================
# GET DOCUMENT
# ============================================================================


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get document details, pages, sections, and entities."""
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id)
        .options(
            selectinload(Document.pages),
            selectinload(Document.sections),
            selectinload(Document.entities),
        )
    )
    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    return document


# ============================================================================
# LIST DOCUMENTS
# ============================================================================


@router.get("/")
async def list_documents(
    db: AsyncSession = Depends(get_db),
):
    """List uploaded documents."""
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()

    return [
        {
            "id": str(document.id),
            "filename": document.original_filename,
            "status": document.status.value,
            "progress": document.processing_progress,
            "message": document.status_message,
            "created_at": document.created_at,
        }
        for document in documents
    ]


# ============================================================================
# DELETE DOCUMENT
# ============================================================================


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document along with its PDF file, FAISS vectors, and DB records."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = document.file_path

    # Remove persisted FAISS index + evict from in-memory LRU cache.
    try:
        store = LegalVectorStore(document_id=str(document_id), dimension=1)
        store.delete()
    except Exception as exc:
        logger.warning(
            "vector_store_cleanup_failed",
            document_id=str(document_id),
            error=str(exc),
        )

    # Remove uploaded file from disk.
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as exc:
            logger.warning(
                "uploaded_file_cleanup_failed",
                document_id=str(document_id),
                error=str(exc),
            )

    await db.delete(document)
    await db.commit()

    logger.info("document_deleted", document_id=str(document_id))
    return {
        "document_id": str(document_id),
        "status": "deleted",
        "message": "Document, vector index, and related records deleted.",
    }