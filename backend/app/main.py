import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    chat,
    documents,
    health,
    indexing,
    summaries,
)
from app.core.config import get_settings
from app.core.exceptions import NyayaSetuError
from app.core.logging_config import get_logger, setup_logging
from app.models.database import close_db, init_db
from app.pipelines.rag.embeddings import LegalEmbeddingModel


settings = get_settings()

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("Starting NyayaSetu backend...")

    await init_db()
    logger.info("Database initialized successfully.")

    preload_start = time.perf_counter()

    try:
        logger.info(
            "embedding_model_preload_started",
            model=settings.embedding_model,
            requested_device=settings.embedding_device,
        )

        embedding_model = await asyncio.to_thread(LegalEmbeddingModel)

        logger.info(
            "embedding_model_preload_complete",
            preloaded=True,
            model=settings.embedding_model,
            device=embedding_model.device,
            dimension=embedding_model.dimension,
            elapsed_seconds=round(time.perf_counter() - preload_start, 4),
        )

    except Exception as exc:
        logger.error(
            "embedding_model_preload_failed",
            preloaded=False,
            model=settings.embedding_model,
            requested_device=settings.embedding_device,
            elapsed_seconds=round(time.perf_counter() - preload_start, 4),
            error=str(exc),
            exc_info=True,
        )

    yield

    logger.info("Shutting down NyayaSetu backend...")
    await close_db()
    logger.info("Database connection closed.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "NyayaSetu - Indian Legal Document Simplification, "
        "Summarization and Document-Grounded Chat"
    ),
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

allowed_origins = [
    origin.strip()
    for origin in settings.allowed_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# EXCEPTION HANDLING
# ============================================================


@app.exception_handler(NyayaSetuError)
async def nyayasetu_exception_handler(
    request: Request,
    exc: NyayaSetuError,
):
    logger.error(
        "NyayaSetu error",
        error=str(exc),
        path=str(request.url),
    )
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled application error",
        path=str(request.url),
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


# ============================================================
# API ROUTES
# ============================================================

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(indexing.router, prefix="/api/documents", tags=["documents"])
app.include_router(summaries.router, prefix="/api/documents", tags=["summaries"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


# ============================================================
# ROOT
# ============================================================


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }