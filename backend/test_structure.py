import sqlite3

from app.pipelines.structure.structure_detector import (
    LegalStructureDetector,
)


DOCUMENT_ID = "aad8cdcccc684f4dbf50aec3d0ec36a7"


def main():
    connection = sqlite3.connect("nyayasetu.db")

    rows = connection.execute(
        """
        SELECT page_number, processed_text
        FROM document_pages
        WHERE document_id = ?
        ORDER BY page_number
        """,
        (DOCUMENT_ID,),
    ).fetchall()

    connection.close()

    pages = [
        {
            "page_number": page_number,
            "processed_text": processed_text or "",
        }
        for page_number, processed_text in rows
    ]

    print(f"Pages loaded: {len(pages)}")

    result = LegalStructureDetector.detect(pages)

    sections = result["sections"]
    paragraphs = result["paragraphs"]

    print(f"Sections detected: {len(sections)}")
    print(f"Paragraphs detected: {len(paragraphs)}")
    print()

    print("=== DETECTED SECTIONS ===")

    for section in sections:
        print(
            f"[{section.order_index}] "
            f"{section.section_type.upper()} | "
            f"{section.title} | "
            f"pages "
            f"{section.start_page}-"
            f"{section.end_page}"
        )

    print()

    print("=== DETECTED PARAGRAPHS ===")

    for paragraph in paragraphs:
        corrected = paragraph.get(
            "paragraph_number_corrected",
            False,
        )

        artifact_count = len(
            paragraph.get(
                "ocr_artifacts",
                [],
            )
        )

        extra_info = ""

        if corrected:
            extra_info += " [NUMBER CORRECTED]"

        if artifact_count:
            extra_info += (
                f" [OCR ARTIFACTS={artifact_count}]"
            )

        print(
            f"Para {paragraph['paragraph_number']} "
            f"(pages "
            f"{paragraph['start_page']}-"
            f"{paragraph['end_page']}): "
            f"{paragraph['text'][:150]}"
            f"{extra_info}"
        )


if __name__ == "__main__":
    main()