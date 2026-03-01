"""
core/embeddings.py — Modelo de embeddings local (sentence-transformers).

Usa paraphrase-multilingual-mpnet-base-v2 (768 dims, soporta español).
El modelo se carga una sola vez (lazy singleton).
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Carga y cachea el modelo de embeddings (singleton)."""
    global _model
    if _model is None:
        logger.info("Cargando modelo de embeddings: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Modelo listo | dim=%d", settings.embedding_dim)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Genera embeddings L2-normalizados para una lista de textos.
    Returns: np.ndarray shape (N, embedding_dim), dtype float32.
    """
    model = get_embedding_model()
    return model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


def embed_query(query: str) -> np.ndarray:
    """
    Genera embedding para una sola consulta.
    Returns: np.ndarray shape (1, embedding_dim), float32 — compatible con faiss.search().
    """
    model = get_embedding_model()
    return (
        model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        .astype("float32")
        .reshape(1, -1)
    )
