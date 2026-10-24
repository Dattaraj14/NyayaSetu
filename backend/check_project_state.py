import asyncio

from sqlalchemy import func, select

from app.models.database import async_session_factory
from app.models.document import Document, DocumentChunk


async def main():
    async with async_session_factory() as db:
        documents = (
            await db.execute(
                select(Document)
                .order_by(Document.created_at.desc())
            )
        ).scalars().all()

        print("DOCUMENT / CHUNK STATUS")
        print("=" * 80)

        for document in documents:
            chunk_count = (
                await db.execute(
                    select(func.count())
                    .select_from(DocumentChunk)
                    .where(DocumentChunk.document_id == document.id)
                )
            ).scalar_one()

            print(
                f"{document.id} | "
                f"{document.status.value} | "
                f"pages={document.page_count} | "
                f"chunks={chunk_count} | "
                f"{document.filename}"
            )

        print()
        print("CHUNK DETAILS")
        print("=" * 80)

        chunks = (
            await db.execute(
                select(DocumentChunk)
                .order_by(
                    DocumentChunk.document_id,
                    DocumentChunk.chunk_index,
                )
            )
        ).scalars().all()

        for chunk in chunks:
            print(
                f"Chunk {chunk.chunk_index} | "
                f"Document={chunk.document_id} | "
                f"Pages={chunk.page_number} | "
                f"Section={chunk.section_type} | "
                f"Tokens={chunk.token_count} | "
                f"Vector ID={chunk.vector_id}"
            )


if __name__ == "__main__":
    asyncio.run(main())