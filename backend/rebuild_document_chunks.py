"""Rebuild stored sections and chunks for an existing document."""

from __future__ import annotations

import asyncio
import json
from uuid import UUID

from sqlalchemy import delete, select

from app.models.database import async_session_factory
from app.models.document import (
    DocumentChunk,
    DocumentPage,
    DocumentSection,
)
from app.pipelines.rag.chunker import LegalChunker
from app.pipelines.structure.structure_detector import (
    LegalStructureDetector,
)


DOCUMENT_ID = UUID(
    "026f44fe-430d-4522-985f-f12cff9de1b2"
)


async def main():
    print(
        f"Rebuilding document: {DOCUMENT_ID}"
    )
    print()

    # --------------------------------------------------------------
    # Load existing pages
    # --------------------------------------------------------------

    async with async_session_factory() as db:

        result = await db.execute(
            select(DocumentPage)
            .where(
                DocumentPage.document_id
                == DOCUMENT_ID
            )
            .order_by(
                DocumentPage.page_number
            )
        )

        pages_db = result.scalars().all()

    if not pages_db:
        print("No stored pages found.")
        return

    pages = [
        {
            "page_number": page.page_number,
            "processed_text": (
                page.processed_text
                or page.raw_text
                or ""
            ),
        }
        for page in pages_db
    ]

    print(
        f"Pages loaded: {len(pages)}"
    )

    # --------------------------------------------------------------
    # Run corrected structure detection
    # --------------------------------------------------------------

    detector = LegalStructureDetector()

    structure = detector.detect(
        pages
    )

    sections = structure["sections"]
    paragraphs = structure["paragraphs"]

    print(
        f"Sections detected: {len(sections)}"
    )

    print(
        f"Paragraphs detected: {len(paragraphs)}"
    )

    # --------------------------------------------------------------
    # Create corrected chunks
    # --------------------------------------------------------------

    chunker = LegalChunker(
        target_tokens=700,
        max_tokens=1000,
    )

    chunks = chunker.create_chunks(
        paragraphs,
        sections,
    )

    print(
        f"Chunks created: {len(chunks)}"
    )
    print()

    # --------------------------------------------------------------
    # Replace old database sections/chunks
    # --------------------------------------------------------------

    async with async_session_factory() as db:

        await db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id
                == DOCUMENT_ID
            )
        )

        await db.execute(
            delete(DocumentSection).where(
                DocumentSection.document_id
                == DOCUMENT_ID
            )
        )

        # ----------------------------------------------------------
        # Save sections
        # ----------------------------------------------------------

        for section in sections:

            db.add(
                DocumentSection(
                    document_id=DOCUMENT_ID,
                    section_type=section.section_type,
                    section_title=section.title,
                    content=section.content,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    order_index=section.order_index,
                )
            )

        # ----------------------------------------------------------
        # Save chunks
        # ----------------------------------------------------------

        for chunk in chunks:

            metadata = {
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "section_title": chunk.section_title,
                "paragraph_numbers": (
                    chunk.paragraph_numbers
                ),
                "paragraph_sequences": (
                    chunk.paragraph_sequences
                ),
                "ocr_artifacts": (
                    chunk.ocr_artifacts
                ),
            }

            db.add(
                DocumentChunk(
                    document_id=DOCUMENT_ID,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    page_number=chunk.start_page,
                    section_type=chunk.section_type,
                    language="en",
                    metadata_json=json.dumps(
                        metadata,
                        ensure_ascii=False,
                    ),
                    vector_id=None,
                )
            )

        await db.commit()

    print(
        "Corrected sections and chunks saved."
    )
    print()

    # --------------------------------------------------------------
    # Show final chunk summary
    # --------------------------------------------------------------

    for chunk in chunks:

        print(
            f"Chunk {chunk.chunk_index} | "
            f"Pages {chunk.start_page}-{chunk.end_page} | "
            f"Section {chunk.section_type} | "
            f"Tokens {chunk.token_count}"
        )

    print()
    print(
        "DOCUMENT CHUNK REBUILD COMPLETE"
    )


if __name__ == "__main__":
    asyncio.run(main())