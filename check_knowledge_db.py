import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_PATH = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_PATH))

from sqlalchemy import select, func

from app.models.database import async_session_factory
from app.models.legal import KnowledgeDocument, KnowledgeChunk


async def main():
    async with async_session_factory() as db:

        document_count = (
            await db.execute(
                select(
                    func.count()
                ).select_from(KnowledgeDocument)
            )
        ).scalar_one()

        chunk_count = (
            await db.execute(
                select(
                    func.count()
                ).select_from(KnowledgeChunk)
            )
        ).scalar_one()

        print(f"KnowledgeDocuments: {document_count}")
        print(f"KnowledgeChunks:    {chunk_count}")


if __name__ == "__main__":
    asyncio.run(main())