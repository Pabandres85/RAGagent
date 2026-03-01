"""
rag/reranker.py — Re-ranking de fragmentos recuperados.

Reordena los resultados del retriever para mejorar la precisión.
Usa cross-encoder si está disponible; si no, retorna los resultados
ordenados por score del retriever (passthrough).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from rag.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-ranker de fragmentos normativos.

    Args:
        use_cross_encoder: Si True, carga cross-encoder/ms-marco-MiniLM-L-6-v2.
                           Si False (default), usa el score FAISS directamente.
    """

    def __init__(self, use_cross_encoder: bool = False):
        self.use_cross_encoder = use_cross_encoder
        self._model = None

        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
                logger.info("Cross-encoder cargado.")
            except Exception as exc:
                logger.warning("No se pudo cargar cross-encoder: %s. Usando passthrough.", exc)
                self.use_cross_encoder = False

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Re-rankea resultados por relevancia a la consulta.

        Returns:
            Lista ordenada por score descendente, recortada a top_k si se especifica.
        """
        if not results:
            return results

        if self.use_cross_encoder and self._model:
            pairs = [(query, r.chunk.text) for r in results]
            scores = self._model.predict(pairs)
            for result, score in zip(results, scores):
                result.score = float(score)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        return reranked[:top_k] if top_k else reranked
