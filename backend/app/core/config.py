"""NyayaSetu Core Configuration."""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "NyayaSetu"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    database_url: str = "sqlite+aiosqlite:///./nyayasetu.db"
    database_sync_url: str = "sqlite:///./nyayasetu.db"

    # LLM
    llm_provider: str = "gemini"  # local, gemini, openai, groq
    llm_model: str = "gemini-3.8-flash"
    local_llm_model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    local_llm_n_gpu_layers: int = 35
    local_llm_n_ctx: int = 4096

    # API Keys
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # Vector DB
    vector_db_path: str = "data/vector_indexes"
    faiss_index_type: str = "flat"

    # OCR
    ocr_enabled: bool = True
    ocr_primary: str = "paddleocr"
    ocr_fallback: str = "tesseract"
    ocr_confidence_threshold: float = 0.7
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Document Processing
    max_file_size_mb: int = 50
    max_pages: int = 500
    upload_dir: str = "uploads"
    temp_dir: str = "temp"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"

    # Security
    secret_key: str = "change-this-to-a-random-secret-key-in-production"
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"
    rate_limit_per_minute: int = 60

    # Translation
    translation_provider: str = "llm"  # llm, indictrans2

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def upload_path(self) -> Path:
        path = self.project_root / self.upload_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def temp_path(self) -> Path:
        path = self.project_root / self.temp_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def vector_db_full_path(self) -> Path:
        path = self.project_root / self.vector_db_path
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
