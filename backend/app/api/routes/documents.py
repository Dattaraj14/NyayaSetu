"""Document upload, extraction, OCR, preprocessing, structure detection,
and legal-aware chunking routes for NyayaSetu.
"""

from __future__ import annotations

import asyncio
import os
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
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

from app.pipelines.ocr.ocr_engine import OCRPipeline

from app.pipelines.preprocessing.preprocessor import (
    LegalTextPreprocessor,
)

from app.pipelines.structure.structure_detector import (
    LegalStructureDetector,
)

from app.pipelines.rag.chunker import LegalChunker

from app.schemas.document import (
    DocumentDetail,
    DocumentUploadResponse,
)

from app.security.file_validator import (
    generate_safe_filename,
    validate_file_extension,
    validate_file_size,
)


# ============================================================================
# ROUTER
# ============================================================================

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
    """
    Detect legal structure from processed pages.

    Supports the current StructureDetector API whether
    detect() is implemented as an instance method or a
    class method.
    """

    # -------------------------------------------------------------
    # 1. Run structure detection
    # -------------------------------------------------------------

    detector = LegalStructureDetector()

    structure_result = await asyncio.to_thread(
        detector.detect,
        pages,
    )

    # -------------------------------------------------------------
    # 2. Normalize detector output
    #
    # The detector may return either:
    #
    #   LegalStructure object
    #
    # or:
    #
    #   dict
    # -------------------------------------------------------------

    if isinstance(
        structure_result,
        dict,
    ):

        sections = structure_result.get(
            "sections",
            [],
        )

        raw_paragraphs = structure_result.get(
            "paragraphs",
            [],
        )

        section_source = structure_result.get(
            "section_source",
            "unknown",
        )

    else:

        sections = getattr(
            structure_result,
            "sections",
            [],
        )

        raw_paragraphs = getattr(
            structure_result,
            "paragraphs",
            [],
        )

        section_source = getattr(
            structure_result,
            "section_source",
            "unknown",
        )

    # -------------------------------------------------------------
    # 3. Normalize paragraphs for LegalChunker
    # -------------------------------------------------------------

    paragraphs: list[dict] = []

    for sequence, paragraph in enumerate(
        raw_paragraphs,
        start=1,
    ):

        # Already a dictionary
        if isinstance(
            paragraph,
            dict,
        ):

            paragraph_dict = dict(
                paragraph
            )

            paragraph_dict.setdefault(
                "sequence_number",
                sequence,
            )

            paragraph_dict.setdefault(
                "number",
                paragraph_dict.get(
                    "paragraph_number"
                ),
            )

            paragraph_dict.setdefault(
                "content",
                paragraph_dict.get(
                    "text",
                    "",
                ),
            )

            paragraph_dict.setdefault(
                "start_page",
                paragraph_dict.get(
                    "page_number",
                    1,
                ),
            )

            paragraph_dict.setdefault(
                "end_page",
                paragraph_dict.get(
                    "start_page",
                    1,
                ),
            )

            paragraph_dict.setdefault(
                "section_type",
                "general",
            )

            paragraph_dict.setdefault(
                "section_title",
                "General",
            )

            paragraph_dict.setdefault(
                "ocr_artifacts",
                [],
            )

            paragraphs.append(
                paragraph_dict
            )

            continue

        # Object returned by the new detector
        number = getattr(
            paragraph,
            "number",
            getattr(
                paragraph,
                "paragraph_number",
                None,
            ),
        )

        content = getattr(
            paragraph,
            "content",
            getattr(
                paragraph,
                "text",
                "",
            ),
        )

        start_page = getattr(
            paragraph,
            "start_page",
            1,
        )

        end_page = getattr(
            paragraph,
            "end_page",
            start_page,
        )

        section_type = getattr(
            paragraph,
            "section_type",
            "general",
        )

        section_title = getattr(
            paragraph,
            "section_title",
            getattr(
                paragraph,
                "title",
                "General",
            ),
        )

        ocr_artifacts = getattr(
            paragraph,
            "ocr_artifacts",
            [],
        )

        paragraphs.append(
            {
                "sequence_number": sequence,
                "paragraph_number": number,
                "source_paragraph_number": number,
                "raw_paragraph_number": number,
                "expected_paragraph_number": None,
                "paragraph_number_corrected": False,
                "ocr_artifact": bool(
                    ocr_artifacts
                ),
                "ocr_artifact_reason": (
                    ocr_artifacts[0]
                    if ocr_artifacts
                    else None
                ),
                "number": number,
                "content": content,
                "text": content,
                "start_page": start_page,
                "end_page": end_page,
                "section_type": section_type,
                "section_title": section_title,
                "ocr_artifacts": ocr_artifacts,
                "order_index": getattr(
                    paragraph,
                    "order_index",
                    sequence,
                ),
            }
        )

    logger.info(
        "legal_structure_detection_complete",
        document_id=str(
            document_id
        ),
        sections=len(sections),
        paragraphs=len(paragraphs),
        section_source=section_source,
    )

    # -------------------------------------------------------------
    # 4. Remove old sections
    # -------------------------------------------------------------

    await db.execute(
        delete(
            DocumentSection
        ).where(
            DocumentSection.document_id
            == document_id
        )
    )

    # -------------------------------------------------------------
    # 5. Save sections
    # -------------------------------------------------------------

    for section in sections:

        if isinstance(
            section,
            dict,
        ):

            section_type = section.get(
                "section_type",
                "general",
            )

            section_title = section.get(
                "section_title",
                section.get(
                    "title",
                    "General",
                ),
            )

            content = section.get(
                "content",
                "",
            )

            start_page = section.get(
                "start_page",
            )

            end_page = section.get(
                "end_page",
                start_page,
            )

            order_index = section.get(
                "order_index",
                0,
            )

        else:

            section_type = getattr(
                section,
                "section_type",
                "general",
            )

            section_title = getattr(
                section,
                "section_title",
                getattr(
                    section,
                    "title",
                    "General",
                ),
            )

            content = getattr(
                section,
                "content",
                "",
            )

            start_page = getattr(
                section,
                "start_page",
                None,
            )

            end_page = getattr(
                section,
                "end_page",
                start_page,
            )

            order_index = getattr(
                section,
                "order_index",
                0,
            )

        section_record = DocumentSection(
            document_id=document_id,
            section_type=section_type,
            section_title=section_title,
            content=content,
            start_page=start_page,
            end_page=end_page,
            order_index=order_index,
        )

        db.add(
            section_record
        )

    await db.commit()

    logger.info(
        "legal_structure_saved",
        document_id=str(
            document_id
        ),
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
    """
    Create legal-aware chunks from detected paragraphs and sections.

    Target:
        ~700 estimated tokens

    Maximum:
        1000 estimated tokens

    Chunks preserve:
        - paragraph numbers
        - page information
        - section information
        - OCR artifacts
    """

    logger.info(
        "legal_chunking_started",
        document_id=str(document_id),
        paragraphs=len(paragraphs),
        sections=len(sections),
    )

    # ------------------------------------------------------------------
    # 1. Initialize chunker
    # ------------------------------------------------------------------

    chunker = LegalChunker(
        target_tokens=700,
        max_tokens=1000,
    )

    # ------------------------------------------------------------------
    # 2. Create chunks in worker thread
    # ------------------------------------------------------------------

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
    )

    # ------------------------------------------------------------------
    # 3. Remove old chunks
    # ------------------------------------------------------------------

    await db.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        )
    )

    # ------------------------------------------------------------------
    # 4. Persist new chunks
    # ------------------------------------------------------------------

    for chunk in chunks:

        metadata = {
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "section_title": chunk.section_title,
            "paragraph_numbers": chunk.paragraph_numbers,
            "paragraph_sequences": chunk.paragraph_sequences,
            "ocr_artifacts": chunk.ocr_artifacts,
        }

        chunk_record = DocumentChunk(
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

        db.add(chunk_record)

    await db.commit()

    logger.info(
        "legal_chunks_saved",
        document_id=str(document_id),
        chunks=len(chunks),
    )

    return chunks


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

async def process_document(
    document_id: UUID,
    file_path: str,
) -> None:
    """
    Complete document-processing pipeline.

    Pipeline:

        PDF
         ↓
        PDF type detection
         ↓
        Text extraction / OCR
         ↓
        Page preprocessing
         ↓
        Legal structure detection
         ↓
        Paragraph detection
         ↓
        Legal chunking
         ↓
        Database
    """

    async for db in get_db():

        try:

            # ==========================================================
            # 1. Load document
            # ==========================================================

            result = await db.execute(
                select(Document).where(
                    Document.id == document_id
                )
            )

            document = result.scalar_one_or_none()

            if document is None:

                logger.error(
                    "document_not_found_for_processing",
                    document_id=str(document_id),
                )

                return

            # ==========================================================
            # 2. Mark extracting
            # ==========================================================

            document.status = (
                DocumentStatus.EXTRACTING
            )

            document.status_message = (
                "Detecting PDF type..."
            )

            document.processing_progress = 5.0

            await db.commit()

            # ==========================================================
            # 3. Detect PDF type
            # ==========================================================

            pdf_type, pdf_info = (
                await asyncio.to_thread(
                    PDFExtractor.detect_pdf_type,
                    file_path,
                )
            )

            logger.info(
                "pdf_type_detected",
                document_id=str(document_id),
                pdf_type=pdf_type,
                total_pages=pdf_info.get(
                    "total_pages"
                ),
                text_pages=pdf_info.get(
                    "text_pages"
                ),
                image_pages=pdf_info.get(
                    "image_pages"
                ),
            )

            # ----------------------------------------------------------
            # Store PDF type
            # ----------------------------------------------------------

            try:

                document.pdf_type = PDFType(
                    pdf_type
                )

            except ValueError:

                document.pdf_type = None

            document.page_count = pdf_info.get(
                "total_pages",
                0,
            )

            document.status_message = (
                f"PDF detected as {pdf_type}"
            )

            await db.commit()

            # ==========================================================
            # 4. Remove old processing data
            # ==========================================================

            await db.execute(
                delete(DocumentPage).where(
                    DocumentPage.document_id
                    == document_id
                )
            )

            await db.execute(
                delete(DocumentSection).where(
                    DocumentSection.document_id
                    == document_id
                )
            )

            await db.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id
                    == document_id
                )
            )

            await db.commit()

            # ==========================================================
            # 5. Initialize preprocessing
            # ==========================================================

            preprocessor = LegalTextPreprocessor()

            processed_pages: list[dict] = []

            # ==========================================================
            # 6. TEXT PDF
            # ==========================================================

            if pdf_type == "text":

                document.status_message = (
                    "Extracting text from PDF..."
                )

                document.processing_progress = 10.0

                await db.commit()

                # ------------------------------------------------------
                # Extract text
                # ------------------------------------------------------

                pages = await asyncio.to_thread(
                    PDFExtractor.extract_text,
                    file_path,
                )

                total_pages = len(pages)

                # ------------------------------------------------------
                # Process pages
                # ------------------------------------------------------

                for index, page_data in enumerate(
                    pages
                ):

                    page_number = page_data[
                        "page_number"
                    ]

                    raw_text = page_data.get(
                        "raw_text",
                        "",
                    )

                    # --------------------------------------------------
                    # Preprocess
                    # --------------------------------------------------

                    processed_result = (
                        preprocessor.preprocess_page(
                            raw_text
                        )
                    )

                    processed_text = (
                        processed_result[
                            "processed_text"
                        ]
                    )

                    # --------------------------------------------------
                    # Keep processed page in memory
                    # --------------------------------------------------

                    processed_pages.append(
                        {
                            "page_number": page_number,
                            "processed_text": (
                                processed_text
                            ),
                        }
                    )

                    # --------------------------------------------------
                    # Save page
                    # --------------------------------------------------

                    page_record = DocumentPage(
                        document_id=document_id,
                        page_number=page_number,
                        raw_text=raw_text,
                        processed_text=(
                            processed_text
                        ),
                        extraction_method=(
                            ExtractionMethod.TEXT
                        ),
                        ocr_confidence=None,
                        has_images=page_data.get(
                            "has_images",
                            False,
                        ),
                        is_low_confidence=False,
                    )

                    db.add(page_record)

                    # --------------------------------------------------
                    # Progress
                    # 10% → 75%
                    # --------------------------------------------------

                    progress = (
                        10.0
                        + (
                            (index + 1)
                            / max(
                                total_pages,
                                1,
                            )
                        )
                        * 65.0
                    )

                    document.processing_progress = (
                        round(
                            progress,
                            2,
                        )
                    )

                    document.status_message = (
                        f"Processing page "
                        f"{page_number}/"
                        f"{total_pages}"
                    )

                    await db.commit()

                # ======================================================
                # Structure detection
                # ======================================================

                document.status = (
                    DocumentStatus.ANALYZING
                )

                document.status_message = (
                    "Detecting legal document "
                    "structure..."
                )

                document.processing_progress = 80.0

                await db.commit()

                structure_result = (
                    await save_detected_structure(
                        db=db,
                        document_id=document_id,
                        pages=processed_pages,
                    )
                )

                # ======================================================
                # Legal chunking
                # ======================================================

                document.status_message = (
                    "Creating legal document chunks..."
                )

                document.processing_progress = 90.0

                await db.commit()

                chunks = (
                    await save_document_chunks(
                        db=db,
                        document_id=document_id,
                        paragraphs=(
                            structure_result[
                                "paragraphs"
                            ]
                        ),
                        sections=(
                            structure_result[
                                "sections"
                            ]
                        ),
                    )
                )

                # ======================================================
                # Completed
                # ======================================================

                document.status = (
                    DocumentStatus.COMPLETED
                )

                document.status_message = (
                    "Text extraction, preprocessing, "
                    "legal structure detection, and "
                    "chunking completed"
                )

                document.processing_progress = 100.0

                await db.commit()

                logger.info(
                    "text_processing_complete",
                    document_id=str(
                        document_id
                    ),
                    pages=total_pages,
                    sections=len(
                        structure_result[
                            "sections"
                        ]
                    ),
                    paragraphs=len(
                        structure_result[
                            "paragraphs"
                        ]
                    ),
                    chunks=len(chunks),
                )

                return

            # ==========================================================
            # 7. SCANNED / HYBRID PDF
            # ==========================================================

            if pdf_type in {
                "scanned",
                "hybrid",
            }:

                document.status_message = (
                    "Initializing OCR engine..."
                )

                document.processing_progress = 10.0

                await db.commit()

                document.status = (
                    DocumentStatus.OCR_PROCESSING
                )

                ocr_pipeline = OCRPipeline()

                total_pages = pdf_info.get(
                    "total_pages",
                    0,
                )

                # ------------------------------------------------------
                # Process every page
                # ------------------------------------------------------

                for page_number in range(
                    1,
                    total_pages + 1,
                ):

                    # --------------------------------------------------
                    # Rendering
                    # --------------------------------------------------

                    document.status_message = (
                        f"Rendering page "
                        f"{page_number}/"
                        f"{total_pages}"
                    )

                    await db.commit()

                    image_bytes = (
                        await asyncio.to_thread(
                            PDFExtractor.extract_page_as_image,
                            file_path,
                            page_number,
                            300,
                        )
                    )

                    # --------------------------------------------------
                    # OCR
                    # --------------------------------------------------

                    document.status_message = (
                        f"Running OCR on page "
                        f"{page_number}/"
                        f"{total_pages}"
                    )

                    await db.commit()

                    ocr_result = (
                        await asyncio.to_thread(
                            ocr_pipeline.ocr_page,
                            image_bytes,
                            page_number,
                        )
                    )

                    raw_text = ocr_result.get(
                        "text",
                        "",
                    )

                    confidence = float(
                        ocr_result.get(
                            "confidence",
                            0.0,
                        )
                    )

                    is_low_confidence = bool(
                        ocr_result.get(
                            "is_low_confidence",
                            False,
                        )
                    )

                    # --------------------------------------------------
                    # Preprocessing
                    # --------------------------------------------------

                    document.status = (
                        DocumentStatus.PREPROCESSING
                    )

                    document.status_message = (
                        f"Preprocessing page "
                        f"{page_number}/"
                        f"{total_pages}"
                    )

                    await db.commit()

                    processed_result = (
                        preprocessor.preprocess_page(
                            raw_text
                        )
                    )

                    processed_text = (
                        processed_result[
                            "processed_text"
                        ]
                    )

                    # --------------------------------------------------
                    # Keep processed page
                    # --------------------------------------------------

                    processed_pages.append(
                        {
                            "page_number": page_number,
                            "processed_text": (
                                processed_text
                            ),
                        }
                    )

                    # --------------------------------------------------
                    # Save OCR page
                    # --------------------------------------------------

                    page_record = DocumentPage(
                        document_id=document_id,
                        page_number=page_number,
                        raw_text=raw_text,
                        processed_text=(
                            processed_text
                        ),
                        extraction_method=(
                            ExtractionMethod.OCR
                        ),
                        ocr_confidence=confidence,
                        has_images=True,
                        is_low_confidence=(
                            is_low_confidence
                        ),
                    )

                    db.add(page_record)

                    # --------------------------------------------------
                    # Progress
                    # 10% → 75%
                    # --------------------------------------------------

                    progress = (
                        10.0
                        + (
                            page_number
                            / max(
                                total_pages,
                                1,
                            )
                        )
                        * 65.0
                    )

                    document.processing_progress = (
                        round(
                            progress,
                            2,
                        )
                    )

                    document.status_message = (
                        f"Completed page "
                        f"{page_number}/"
                        f"{total_pages}"
                    )

                    await db.commit()

                # ======================================================
                # Structure detection
                # ======================================================

                document.status = (
                    DocumentStatus.ANALYZING
                )

                document.status_message = (
                    "Detecting legal document "
                    "structure..."
                )

                document.processing_progress = 80.0

                await db.commit()

                structure_result = (
                    await save_detected_structure(
                        db=db,
                        document_id=document_id,
                        pages=processed_pages,
                    )
                )

                # ======================================================
                # Legal chunking
                # ======================================================

                document.status_message = (
                    "Creating legal document chunks..."
                )

                document.processing_progress = 90.0

                await db.commit()

                chunks = (
                    await save_document_chunks(
                        db=db,
                        document_id=document_id,
                        paragraphs=(
                            structure_result[
                                "paragraphs"
                            ]
                        ),
                        sections=(
                            structure_result[
                                "sections"
                            ]
                        ),
                    )
                )

                # ======================================================
                # Completed
                # ======================================================

                document.status = (
                    DocumentStatus.COMPLETED
                )

                document.status_message = (
                    "OCR, preprocessing, legal "
                    "structure detection, and "
                    "chunking completed"
                )

                document.processing_progress = 100.0

                await db.commit()

                logger.info(
                    "ocr_processing_complete",
                    document_id=str(
                        document_id
                    ),
                    pages=total_pages,
                    sections=len(
                        structure_result[
                            "sections"
                        ]
                    ),
                    paragraphs=len(
                        structure_result[
                            "paragraphs"
                        ]
                    ),
                    chunks=len(chunks),
                )

                return

            # ==========================================================
            # 8. Unsupported PDF
            # ==========================================================

            raise ValueError(
                f"Unsupported or unknown PDF type: "
                f"{pdf_type}"
            )

        # ==============================================================
        # ERROR HANDLING
        # ==============================================================

        except Exception as exc:

            logger.exception(
                "document_processing_failed",
                document_id=str(
                    document_id
                ),
                error=str(exc),
            )

            await db.rollback()

            # ----------------------------------------------------------
            # Re-fetch document after rollback
            # ----------------------------------------------------------

            result = await db.execute(
                select(Document).where(
                    Document.id == document_id
                )
            )

            document = (
                result.scalar_one_or_none()
            )

            if document is not None:

                document.status = (
                    DocumentStatus.FAILED
                )

                document.status_message = (
                    f"Processing failed: "
                    f"{str(exc)}"
                )

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
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF document for processing."""

    # ------------------------------------------------------------------
    # 1. Validate extension
    # ------------------------------------------------------------------

    try:

        validate_file_extension(
            file.filename or ""
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # ------------------------------------------------------------------
    # 2. Read file
    # ------------------------------------------------------------------

    file_content = await file.read()

    # ------------------------------------------------------------------
    # 3. Validate size
    # ------------------------------------------------------------------

    try:

        validate_file_size(
            len(file_content)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # ------------------------------------------------------------------
    # 4. Generate safe filename
    # ------------------------------------------------------------------

    safe_filename = generate_safe_filename(
        file.filename or "document.pdf"
    )

    upload_dir = settings.upload_dir

    os.makedirs(
        upload_dir,
        exist_ok=True,
    )

    file_path = os.path.join(
        upload_dir,
        safe_filename,
    )

    # ------------------------------------------------------------------
    # 5. Save uploaded file
    # ------------------------------------------------------------------

    try:

        with open(
            file_path,
            "wb",
        ) as output_file:

            output_file.write(
                file_content
            )

    except OSError as exc:

        logger.exception(
            "file_save_failed",
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        )

    # ------------------------------------------------------------------
    # 6. Create document record
    # ------------------------------------------------------------------

    document = Document(
        filename=safe_filename,
        original_filename=(
            file.filename
            or "document.pdf"
        ),
        file_path=file_path,
        file_size=len(file_content),
        mime_type=(
            file.content_type
            or "application/pdf"
        ),
        page_count=0,
        status=DocumentStatus.UPLOADED,
        status_message=(
            "File uploaded successfully"
        ),
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

        logger.exception(
            "document_database_creation_failed",
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to create document record.",
        )

    # ------------------------------------------------------------------
    # 7. Start background processing
    # ------------------------------------------------------------------

    background_tasks.add_task(
        process_document,
        document.id,
        file_path,
    )

    logger.info(
        "document_uploaded",
        document_id=str(
            document.id
        ),
        filename=file.filename,
        size=len(file_content),
    )

    return DocumentUploadResponse(
        id=document.id,
        filename=document.original_filename,
        status=document.status.value,
        message=(
            "Document uploaded and processing started."
        ),
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
        .where(
            Document.id == document_id
        )
        .options(
            selectinload(Document.pages),
            selectinload(Document.sections),
            selectinload(Document.entities),
        )
    )

    document = (
        result.scalar_one_or_none()
    )

    if document is None:

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document


# ============================================================================
# LIST DOCUMENTS
# ============================================================================

@router.get(
    "/",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
):
    """List uploaded documents."""

    result = await db.execute(
        select(Document).order_by(
            Document.created_at.desc()
        )
    )

    documents = result.scalars().all()

    return [
        {
            "id": str(document.id),
            "filename": (
                document.original_filename
            ),
            "status": (
                document.status.value
            ),
            "progress": (
                document.processing_progress
            ),
            "message": (
                document.status_message
            ),
            "created_at": (
                document.created_at
            ),
        }
        for document in documents
    ]