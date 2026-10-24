import asyncio

from sqlalchemy import select
from uuid import UUID

from app.models.database import async_session_factory
from app.models.document import DocumentPage, DocumentChunk


DOCUMENT_ID = UUID("026f44fe-430d-4522-985f-f12cff9de1b2")


async def main():
    async with async_session_factory() as db:

        print("=" * 80)
        print("RAW EXTRACTED TEXT — PAGES 7 AND 8")
        print("=" * 80)

        result = await db.execute(
            select(DocumentPage)
            .where(
                DocumentPage.document_id == DOCUMENT_ID,
                DocumentPage.page_number.in_([7, 8]),
            )
            .order_by(DocumentPage.page_number)
        )

        pages = result.scalars().all()

        for page in pages:
            print()
            print(f"PAGE {page.page_number}")
            print("-" * 80)
            print(page.processed_text or page.raw_text or "[NO TEXT]")

        print()
        print("=" * 80)
        print("STORED CHUNKS FOR THIS DOCUMENT")
        print("=" * 80)

        result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == DOCUMENT_ID)
            .order_by(DocumentChunk.chunk_index)
        )

        chunks = result.scalars().all()

        for chunk in chunks:
            print()
            print(
                f"CHUNK {chunk.chunk_index} | "
                f"PAGE {chunk.page_number} | "
                f"SECTION {chunk.section_type} | "
                f"TOKENS {chunk.token_count}"
            )
            print("-" * 80)
            print(chunk.content)


if __name__ == "__main__":
    asyncio.run(main())