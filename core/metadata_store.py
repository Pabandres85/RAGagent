"""
core/metadata_store.py — Persistencia de metadatos de chunks normativos.

Los metadatos se guardan como JSON por módulo en data/metadata/.
El orden de la lista de chunks corresponde 1:1 con las posiciones del índice FAISS.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from core.config import settings

logger = logging.getLogger(__name__)

MODULES = [
    "talento_humano",
    "infraestructura",
    "dotacion",
    "medicamentos_dispositivos",
    "procesos_prioritarios",
    "historia_clinica",
    "interdependencia",
]

MODULE_DISPLAY_NAMES = {
    "talento_humano": "Talento Humano",
    "infraestructura": "Infraestructura",
    "dotacion": "Dotación",
    "medicamentos_dispositivos": "Medicamentos y Dispositivos Médicos",
    "procesos_prioritarios": "Procesos Prioritarios",
    "historia_clinica": "Historia Clínica y Registros",
    "interdependencia": "Interdependencia de Servicios",
}


class ChunkMetadata(BaseModel):
    chunk_id: str
    source_file: str
    module: str
    service: Optional[str] = None
    numeral: Optional[str] = None
    page: Optional[int] = None
    resolution: str = "Resolución 3100 de 2019"
    vigencia: str = "Vigente"
    text: str


class MetadataStore:
    def __init__(self, metadata_dir: Path = settings.metadata_dir):
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save(self, module: str, chunks: List[ChunkMetadata]) -> None:
        path = self.metadata_dir / f"{module}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)
        logger.info("Guardados %d chunks | módulo=%s → %s", len(chunks), module, path)

    def load(self, module: str) -> List[ChunkMetadata]:
        path = self.metadata_dir / f"{module}.json"
        if not path.exists():
            logger.warning("Sin metadatos para módulo '%s'", module)
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ChunkMetadata(**item) for item in data]

    def load_all(self) -> List[ChunkMetadata]:
        all_chunks: List[ChunkMetadata] = []
        for module in MODULES:
            chunks = self.load(module)
            all_chunks.extend(chunks)
        logger.info("Total chunks: %d", len(all_chunks))
        return all_chunks

    def get_by_indices(self, module: str, indices: List[int]) -> List[ChunkMetadata]:
        chunks = self.load(module)
        return [chunks[i] for i in indices if 0 <= i < len(chunks)]
