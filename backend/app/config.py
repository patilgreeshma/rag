"""
FinRule AI – Financial Regulation Research Assistant
Centralised configuration.

All settings are read from environment variables (or a .env file).
Import `settings` from here; never read os.environ directly in other modules.
"""
from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Open India Law Dataset ────────────────────────────────────────────────
    oil_raw_dir: Path = Field(
        default=Path("data/raw/open_india_law"),
        description="Directory where the raw Open India Law Parquet file is stored.",
    )
    oil_parquet_file: str = Field(
        default="in_sebi_regulations.parquet",
        description="Parquet filename within oil_raw_dir.",
    )
    oil_parquet_url: str = Field(
        default="https://oss-data-in.vaquill.ai/v2026.08.1/in_sebi_regulations.parquet",
        description="Public URL to download in_sebi_regulations.parquet.",
    )

    # ── Processed Data Output ─────────────────────────────────────────────────
    sebi_processed_dir: Path = Field(
        default=Path("data/processed/sebi"),
        description="Directory for processed SEBI JSONL outputs.",
    )
    faiss_index_path: Path = Field(
        default=Path("data/processed/sebi/faiss_index"),
        description="Directory where the FAISS index and metadata are stored.",
    )

    # ── Embedding Model ───────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence Transformers model name for dense embeddings.",
    )
    embedding_dim: int = Field(
        default=384, description="Vector dimension (must match model output)."
    )
    embedding_batch_size: int = Field(
        default=64, description="Chunks per batch for Sentence Transformer encoding."
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(
        default=5, description="Final context chunks returned to LLM."
    )
    retrieval_threshold: float = Field(
        default=0.30,
        description="Minimum cosine similarity score to include a chunk in context. "
                    "Below this, the RAG returns an insufficient-context response.",
    )

    # ── LLM / Generation ─────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", description="Google Gemini API key.")
    llm_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model name for answer generation.",
    )

    # ── API ────────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # ── Derived Helpers ───────────────────────────────────────────────────────
    @property
    def oil_parquet_path(self) -> Path:
        """Full path to the SEBI Parquet file."""
        return self.oil_raw_dir / self.oil_parquet_file


# Singleton – import this from any module
settings = Settings()

# Ensure data directories exist when config is first loaded
settings.oil_raw_dir.mkdir(parents=True, exist_ok=True)
settings.sebi_processed_dir.mkdir(parents=True, exist_ok=True)
settings.faiss_index_path.mkdir(parents=True, exist_ok=True)
