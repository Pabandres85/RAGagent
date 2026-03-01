"""
Pipeline de ingesta del corpus normativo.

Fases:
1. Extraer texto del PDF con PyMuPDF.
2. Detectar el modulo activo por encabezados.
3. Dividir en chunks.
4. Generar embeddings.
5. Construir indices FAISS por modulo y global.
6. Persistir metadatos por modulo.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import fitz
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from core.embeddings import embed_texts
from core.metadata_store import ChunkMetadata, MODULES, MetadataStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODULE_PATTERNS: Dict[str, List[str]] = {
    "talento_humano": [
        "TALENTO HUMANO",
        "Talento Humano",
        "TALENTO\nHUMANO",
    ],
    "infraestructura": [
        "INFRAESTRUCTURA",
        "Infraestructura",
    ],
    "dotacion": [
        "DOTACION",
        "Dotacion",
    ],
    "medicamentos_dispositivos": [
        "MEDICAMENTOS Y DISPOSITIVOS",
        "Medicamentos y Dispositivos",
        "DISPOSITIVOS MEDICOS",
    ],
    "procesos_prioritarios": [
        "PROCESOS PRIORITARIOS",
        "Procesos Prioritarios",
    ],
    "historia_clinica": [
        "HISTORIA CLINICA",
        "Historia Clinica",
        "HISTORIA CLINICA Y REGISTROS",
    ],
    "interdependencia": [
        "INTERDEPENDENCIA",
        "Interdependencia",
        "INTERDEPENDENCIA DE SERVICIOS",
    ],
}

SERVICE_HEADER_RE = re.compile(
    r"(?:SERVICIO[:\s]+|^)([A-Z\u00C1\u00C9\u00CD\u00D3\u00DA\u00DC\u00D1][A-Z\u00C1\u00C9\u00CD\u00D3\u00DA\u00DC\u00D1\s\-]{3,60})$",
    re.MULTILINE,
)
NUMERAL_RE = re.compile(r"\b(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\b")


def detect_module(text: str, current_module: Optional[str] = None) -> Optional[str]:
    text_upper = text.upper()
    for module, patterns in MODULE_PATTERNS.items():
        for pattern in patterns:
            if pattern.upper() in text_upper:
                return module
    return current_module


def extract_service(text: str) -> Optional[str]:
    match = SERVICE_HEADER_RE.search(text)
    return match.group(1).strip().title() if match else None


def extract_numeral(text: str) -> Optional[str]:
    match = NUMERAL_RE.search(text)
    return match.group(1) if match else None


def build_chunk_id(source_file: str, module: str, index: int) -> str:
    raw = f"{source_file}|{module}|{index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with fitz.open(str(pdf_path)) as doc:
        logger.info("PDF: %s | paginas=%d", pdf_path.name, len(doc))
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((page_number, text))
    return pages


def build_chunks(
    pages: List[Tuple[int, str]],
    source_file: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> List[ChunkMetadata]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[ChunkMetadata] = []
    current_module: Optional[str] = None
    current_service: Optional[str] = None
    global_index = 0

    for page_number, page_text in tqdm(pages, desc="Procesando paginas"):
        current_module = detect_module(page_text, current_module)
        service = extract_service(page_text)
        if service:
            current_service = service

        if not current_module:
            continue

        for chunk_text in splitter.split_text(page_text):
            normalized = chunk_text.strip()
            if len(normalized) < 30:
                continue

            chunks.append(
                ChunkMetadata(
                    chunk_id=build_chunk_id(source_file, current_module, global_index),
                    source_file=source_file,
                    module=current_module,
                    service=current_service,
                    numeral=extract_numeral(normalized),
                    page=page_number,
                    text=normalized,
                )
            )
            global_index += 1

    logger.info("Total chunks extraidos: %d", len(chunks))
    return chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def run_ingestion(
    pdf_paths: List[Path],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> None:
    store = MetadataStore()
    settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[ChunkMetadata] = []
    for pdf_path in pdf_paths:
        logger.info("Procesando: %s", pdf_path)
        pages = extract_pages(pdf_path)
        all_chunks.extend(
            build_chunks(
                pages=pages,
                source_file=pdf_path.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    if not all_chunks:
        logger.error("No se generaron chunks. Verifica que el PDF tenga texto extraible.")
        return

    by_module: Dict[str, List[ChunkMetadata]] = {module: [] for module in MODULES}
    for chunk in all_chunks:
        if chunk.module in by_module:
            by_module[chunk.module].append(chunk)

    for module in MODULES:
        module_chunks = by_module[module]
        if not module_chunks:
            logger.warning("Modulo '%s' sin chunks; se omite.", module)
            continue

        logger.info(
            "Generando embeddings para modulo '%s' (%d chunks)...",
            module,
            len(module_chunks),
        )
        module_embeddings = embed_texts([chunk.text for chunk in module_chunks])
        module_index = build_faiss_index(module_embeddings)
        module_index_path = settings.faiss_index_dir / f"{module}.faiss"
        faiss.write_index(module_index, str(module_index_path))
        store.save(module, module_chunks)
        logger.info(
            "Indice guardado: %s | vectores=%d",
            module_index_path,
            module_index.ntotal,
        )

    logger.info("Construyendo indice global...")
    global_embeddings = embed_texts([chunk.text for chunk in all_chunks])
    global_index = build_faiss_index(global_embeddings)
    global_index_path = settings.faiss_index_dir / "global.faiss"
    faiss.write_index(global_index, str(global_index_path))
    logger.info(
        "Indice global guardado: %s | vectores=%d",
        global_index_path,
        global_index.ntotal,
    )

    logger.info("Ingesta completada.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingesta del corpus normativo de la Resolucion 3100 de 2019."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Ruta a un PDF especifico. Si no se pasa, usa todos los PDFs en data/raw.",
    )
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    pdf_paths = [args.pdf] if args.pdf else sorted(settings.data_raw_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.error("No se encontraron PDFs en %s", settings.data_raw_dir)
        sys.exit(1)

    logger.info("PDFs a procesar: %s", [path.name for path in pdf_paths])
    run_ingestion(
        pdf_paths=pdf_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
