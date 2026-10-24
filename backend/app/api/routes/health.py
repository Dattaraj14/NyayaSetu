"""Health check routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.config import get_settings
from app.models.database import get_db
from app.schemas.document import HealthResponse

router = APIRouter()
settings = get_settings()


@router.get("", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint to verify system status."""
    db_status = "connected"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        version=settings.app_version,
        database=db_status,
        llm="available" if settings.llm_provider else "unavailable",
        ocr="available" if settings.ocr_enabled else "disabled",
        embeddings="available"
    )
