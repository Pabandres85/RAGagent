"""
rag/retriever.py - Recuperacion semantica sobre indices FAISS.

Soporta recuperacion por modulo (agentes especialistas) y
recuperacion global sobre todos los modulos (baseline mono-agente).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss

from core.config import settings
from core.embeddings import embed_query
from core.metadata_store import ChunkMetadata, MetadataStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk: ChunkMetadata
    score: float


class Retriever:
    """
    Recuperador semantico sobre indices FAISS.

    Args:
        module: Si se especifica, usa el indice del modulo.
                Si es None, usa el indice global.
        index_dir: Directorio donde estan los .faiss generados por ingest.py.
    """

    def __init__(
        self,
        module: Optional[str] = None,
        index_dir: Path = settings.faiss_index_dir,
    ):
        self.module = module
        self.index_dir = Path(index_dir)
        self._store = MetadataStore()
        self._index: faiss.Index | None = None
        self._chunks: List[ChunkMetadata] = []

    def _load(self) -> None:
        if self._index is not None:
            return

        name = self.module if self.module else "global"
        path = self.index_dir / f"{name}.faiss"

        if not path.exists():
            raise FileNotFoundError(
                f"Indice FAISS no encontrado: {path}\n"
                "Ejecuta primero: python scripts/ingest.py"
            )

        self._index = faiss.read_index(str(path))
        self._chunks = (
            self._store.load(self.module) if self.module else self._store.load_all()
        )
        logger.info(
            "Indice FAISS '%s' cargado | vectores=%d | chunks=%d",
            name,
            self._index.ntotal,
            len(self._chunks),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = settings.faiss_top_k,
    ) -> List[RetrievalResult]:
        """
        Recupera los top-k fragmentos mas similares a la consulta.

        Returns:
            Lista de RetrievalResult ordenada por score descendente.
        """
        if not query.strip():
            raise ValueError("La consulta no puede estar vacia.")

        self._load()

        query_vec = embed_query(query)
        scores, indices = self._index.search(query_vec, top_k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append(RetrievalResult(chunk=self._chunks[idx], score=float(score)))

        return results
