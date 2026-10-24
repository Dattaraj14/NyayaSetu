import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging_config import setup_logging, get_logger
from app.core.exceptions import NyayaSetuError
from app.models.database import init_db, close_db

from app.api.routes import (
    chat,
    documents,
    health,
    indexing,
)

from app.pipelines.rag.embeddings import LegalEmbeddingModel


settings = get_settings()

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    """

    logger.info("Starting NyayaSetu backend...")

    # ---------------------------------------------------------
    # Database
    # ---------------------------------------------------------

    await init_db()

    logger.info("Database initialized successfully.")

    # ---------------------------------------------------------
    # Preload BGE-M3
    # ---------------------------------------------------------

    try:
        logger.info(
            "Preloading embedding model during startup..."
        )

        await asyncio.to_thread(
            LegalEmbeddingModel
        )

        logger.info(
            "Embedding model preloaded successfully."
        )

    except Exception as exc:
        logger.exception(
            "Embedding model preload failed",
            error=str(exc),
        )

        # Do not prevent FastAPI from starting.
        # The embedding model can still be initialized
        # when it is first requested.

    yield

    # ---------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------

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

    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
        },
    )


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
        content={
            "detail": "Internal server error.",
        },
    )


# ============================================================
# API ROUTES
# ============================================================

app.include_router(
    health.router,
    prefix="/api",
    tags=["health"],
)


app.include_router(
    documents.router,
    prefix="/api",
    tags=["documents"],
)


app.include_router(
    indexing.router,
    prefix="/api/documents",
    tags=["documents"],
)


app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["chat"],
)


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