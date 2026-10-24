"""Test legal document chunking using the latest processed document."""

import sqlite3

from app.pipelines.rag.chunker import LegalChunker
from app.pipelines.structure.structure_detector import (
    LegalStructureDetector,
)


DB_PATH = "nyayasetu.db"


def get_latest_document_id(
    connection: sqlite3.Connection,
) -> str | None:
    """Return the latest document that has stored pages."""

    row = connection.execute(
        """
        SELECT d.id
        FROM documents d
        INNER JOIN document_pages p
            ON p.document_id = d.id
        GROUP BY d.id
        ORDER BY MAX(p.created_at) DESC
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        return None

    return row[0]


def main() -> None:

    connection = sqlite3.connect(DB_PATH)

    # -------------------------------------------------------------
    # Find latest processed document automatically.
    # -------------------------------------------------------------

    document_id = get_latest_document_id(
        connection
    )

    if document_id is None:

        print(
            "ERROR: No document with stored pages was found."
        )

        connection.close()
        return

    print(
        f"Document ID: {document_id}"
    )

    # -------------------------------------------------------------
    # Load processed pages.
    # -------------------------------------------------------------

    rows = connection.execute(
        """
        SELECT page_number, processed_text
        FROM document_pages
        WHERE document_id = ?
        ORDER BY page_number
        """,
        (document_id,),
    ).fetchall()

    connection.close()

    pages = [
        {
            "page_number": page_number,
            "processed_text": (
                processed_text or ""
            ),
        }
        for page_number, processed_text in rows
    ]

    print(
        f"Pages loaded: {len(pages)}"
    )

    if not pages:

        print(
            "ERROR: Document exists but has no stored pages."
        )

        return

    # -------------------------------------------------------------
    # Structure detection.
    # -------------------------------------------------------------

    structure_result = (
        LegalStructureDetector.detect(
            pages
        )
    )

    sections = structure_result.get(
        "sections",
        [],
    )

    paragraphs = structure_result.get(
        "paragraphs",
        [],
    )

    print(
        f"Sections detected: {len(sections)}"
    )

    print(
        f"Paragraphs detected: {len(paragraphs)}"
    )

    if not paragraphs:

        print(
            "ERROR: No paragraphs were detected."
        )

        return

    # -------------------------------------------------------------
    # Chunking.
    # -------------------------------------------------------------

    chunker = LegalChunker(
        target_tokens=700,
        max_tokens=1000,
    )

    chunks = chunker.create_chunks(
        paragraphs=paragraphs,
        sections=sections,
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print()

    # -------------------------------------------------------------
    # Display chunks.
    # -------------------------------------------------------------

    print(
        "=== CHUNKS ==="
    )

    for chunk in chunks:

        print()

        print(
            f"Chunk {chunk.chunk_index}"
        )

        print(
            f"  Pages: "
            f"{chunk.start_page}-"
            f"{chunk.end_page}"
        )

        print(
            f"  Section: "
            f"{chunk.section_type}"
        )

        print(
            f"  Section title: "
            f"{chunk.section_title}"
        )

        print(
            f"  Tokens: "
            f"{chunk.token_count}"
        )

        print(
            f"  Paragraphs: "
            f"{chunk.paragraph_numbers}"
        )

        print(
            f"  Sequences: "
            f"{chunk.paragraph_sequences}"
        )

        print(
            f"  OCR artifacts: "
            f"{len(chunk.ocr_artifacts)}"
        )

        preview = (
            chunk.content
            .replace("\n", " ")
        )

        print(
            f"  Content: "
            f"{preview[:250]}..."
        )


if __name__ == "__main__":
    main()