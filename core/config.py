"""
core/config.py — Configuración centralizada con Pydantic Settings v2.

Lee variables desde .env. Todas las partes del proyecto importan
el singleton `settings` desde aquí.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Proveedor LLM activo ──────────────────────────────────────────────────
    llm_provider: Literal["lmstudio", "ollama"] = "lmstudio"

    # ── LM Studio ────────────────────────────────────────────────────────────
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    lm_studio_model: str = "qwen2.5-7b-instruct"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3:8b"

    # ── Parámetros LLM compartidos ────────────────────────────────────────────
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    embedding_dim: int = 768
    embedding_batch_size: int = 32

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_index_dir: Path = BASE_DIR / "artifacts" / "faiss"
    faiss_top_k: int = 10

    # ── Datos ─────────────────────────────────────────────────────────────────
    data_raw_dir: Path = BASE_DIR / "data" / "raw"
    data_processed_dir: Path = BASE_DIR / "data" / "processed"
    metadata_dir: Path = BASE_DIR / "data" / "metadata"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Evaluación ────────────────────────────────────────────────────────────
    gold_set_path: Path = BASE_DIR / "eval" / "datasets" / "gold_set.json"
    eval_output_dir: Path = BASE_DIR / "artifacts" / "eval_runs"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Helpers ───────────────────────────────────────────────────────────────
    def get_llm_base_url(self) -> str:
        return (
            self.lm_studio_base_url
            if self.llm_provider == "lmstudio"
            else self.ollama_base_url
        )

    def get_llm_api_key(self) -> str:
        return self.lm_studio_api_key if self.llm_provider == "lmstudio" else "ollama"

    def get_llm_model(self) -> str:
        return (
            self.lm_studio_model if self.llm_provider == "lmstudio" else self.ollama_model
        )


settings = Settings()
