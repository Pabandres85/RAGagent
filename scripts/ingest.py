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
import difflib
import hashlib
import logging
import re
import sys
import unicodedata
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

# El PDF de la Resolucion 3100 es un manual OCR: capitulos 1-10 (articulado, inscripcion,
# autoevaluacion, visitas, novedades) y el capitulo 11 "ESTANDARES Y CRITERIOS DE
# HABILITACION", que es el nucleo normativo. Dentro del cap. 11, cada servicio
# (11.2.1 CONSULTA EXTERNA GENERAL, ...) repite los siete estandares, cada uno
# introducido por una linea propia "Estandar de <modulo>". Las menciones sueltas del
# nombre de un estandar dentro del texto NO deben cambiar el modulo.
CHAPTER_11_RE = re.compile(
    r"^\s*11\.\s*ESTANDARES\s+Y\s+CRITERIOS\s+DE\s+HABILITACI", re.IGNORECASE
)
STANDARD_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:Est\w{1,5}ndar(?:es)?|EST\w{1,5}NDAR(?:ES)?)\s+(?:de|DE)\s+(.{3,70})$"
)
# El OCR deja minusculas sueltas dentro de titulos en mayusculas ("VACUNACiON"), por eso
# tras la palabra inicial SERVICIO/TRANSPORTE (en mayuscula) se aceptan ambas.
SERVICE_HEADING_RE = re.compile(
    r"^\s*(11[.,]\d+[.,]\d+\.?)\s*((?:SERVICIO|TRANSPORTE)[A-Za-z \-,/]*|[A-Z][A-Z ,\-/]{6,})$"
)
# Algunos titulos pierden el numeral en el OCR ("SERVICIO DE CUIDADO INTENSIVO ADULTOS",
# pag. 182). Se aceptan solo si la linea es un titulo completo en mayusculas (el OCR deja
# 'i'/'l' sueltas) con al menos tres palabras; el texto corrido de los criterios no cumple.
UNNUMBERED_SERVICE_RE = re.compile(r"^\s*(SERVICIO\s+[A-Z][A-Za-z ]{8,60})$")

# "11.4 GRUPO INTERNACION" / "11.1. ESTANDARES Y CRITERIOS APLICABLES A TODOS LOS SERVICIOS":
# encabezados de grupo. Cortan el contexto (no pertenecen al estandar anterior).
GROUP_HEADING_RE = re.compile(
    r"^\s*(11\.\d+)\.?\s+(GRUPO\b.*|ESTANDARES\s+Y\s+CRITERIOS\s+APLICABLES.*)$", re.IGNORECASE
)
ALL_SERVICES_LABEL = "11.1 Aplicable A Todos Los Servicios"

# Lineas del encabezado repetido de cada pagina (el OCR las degrada y PAGE_HEADER_RE no
# siempre las captura). Solo se descartan lineas cortas que encajan en estos patrones.
HEADER_NOISE_RES = [
    re.compile(r"RESOLUC.{0,4}N\s+N.{0,4}MERO", re.IGNORECASE),
    re.compile(r"^\W*DE\s+2019\W*$", re.IGNORECASE),
    re.compile(r"Continua.{1,4}n\b.{0,14}resol", re.IGNORECASE),
    re.compile(r"prestadores\s+de\s+.{0,8}ic.{0,3}os\s+de\s+salud\s+y\s+de\s+habilitaci", re.IGNORECASE),
    re.compile(r"^\s*de\s+Prestadores\s+y\s+Hab", re.IGNORECASE),
    re.compile(r"adopt.\s+el\s+Manual\s+de\s+Inscri", re.IGNORECASE),
    re.compile(r"^.{0,12}P.gina\s*\d+\s*de\s*\d+", re.IGNORECASE),
    re.compile(r"^\W{0,3}[A-Za-z]?0{2,}3100\W*$"),
    re.compile(r"^\W{0,3}\d{0,2}\W{0,3}N\w{1,2}\W{0,3}\w{0,2}\W{0,3}20\d\w{0,2}\W*$", re.IGNORECASE),
]

# Primeras palabras de cada estandar (sin acentos, minusculas) -> modulo.
STANDARD_KEYS: Dict[str, str] = {
    "talento humano": "talento_humano",
    "infraestructura": "infraestructura",
    "dotacion": "dotacion",
    "medicamentos": "medicamentos_dispositivos",
    "procesos prioritarios": "procesos_prioritarios",
    "historia clinica": "historia_clinica",
    "interdependencia": "interdependencia",
}
NUMERAL_RE = re.compile(r"\b(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\b")

# Elimina la linea "Pagina N de NNN" del encabezado repetitivo del PDF.
# El PDF de la Resolucion 3100 incluye en cada pagina una cabecera con el
# numero de pagina (ej. "Pagina 72 de 230") que contamina los embeddings con
# texto variable (el numero cambia en cada pagina).
# No se intenta eliminar las otras lineas del encabezado (RESOLUCION NUMERO,
# fecha, etc.) porque PyMuPDF puede extraerlas en orden variable segun la
# pagina; intentar consumirlas con contexto elimina encabezados de modulo.
PAGE_HEADER_RE = re.compile(
    r"[Pp].{0,5}gina\s*\d+\s*de\s*\d+[^\n]*\n",
    re.IGNORECASE,
)


def clean_page_text(text: str) -> str:
    """Elimina el encabezado repetitivo de cada pagina del PDF."""
    return PAGE_HEADER_RE.sub("", text)


def _strip_accents(s: str) -> str:
    """Elimina diacriticos para comparacion robusta (DOTACION == DOTACION)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def match_standard_heading(line: str) -> Optional[str]:
    """
    Devuelve el modulo si la linea es un encabezado de estandar ("Estandar de talento
    humano", con numeral opcional); None si no. Exige linea completa, sin punto final
    (las referencias cruzadas del texto terminan en '.') y tolera ruido de OCR.
    """
    plain = _strip_accents(line).rstrip()
    if plain.endswith("."):
        return None
    match = STANDARD_HEADING_RE.match(plain)
    if not match:
        return None
    tail = match.group(1).strip().lower()
    for key, module in STANDARD_KEYS.items():
        similarity = difflib.SequenceMatcher(None, tail[: len(key)], key).ratio()
        if similarity >= 0.8:
            return module
    return None


# En las primeras lineas de cada pagina el OCR degrada la fecha ("25 NOV201'9", "? 5 NOIJ 201~")
# y el codigo de la resolucion ("COO310{)"). Se descartan solo si la linea es muy corta.
TOP_OF_PAGE_LINES = 10
_DATE_RE = re.compile(r"20?1")
_CODE_RE = re.compile(r"^[A-Za-z]{0,2}[0Oo]{2,}3[1lI][0Oo]$")


def is_top_of_page_noise(line: str) -> bool:
    plain = _strip_accents(line).strip()
    if not 0 < len(plain) <= 22:
        return False
    alnum = re.sub(r"[^A-Za-z0-9]", "", plain)
    if _CODE_RE.match(alnum):
        return True
    # fecha deformada: poco texto, una secuencia tipo 2019 y letras de 'NOV' (sin criterios numerados)
    return bool(_DATE_RE.search(plain)) and bool(re.search(r"N.{0,2}[OV]", plain, re.IGNORECASE)) and not re.match(r"^\d+\.\d", plain)


def is_header_noise(line: str) -> bool:
    plain = _strip_accents(line).strip()
    return 0 < len(plain) < 140 and any(rx.search(plain) for rx in HEADER_NOISE_RES)


def match_group_heading(line: str) -> Optional[str]:
    """Devuelve 'ALL' para 11.1 (aplicable a todos los servicios) o 'GROUP' para 11.x GRUPO ..."""
    match = GROUP_HEADING_RE.match(_strip_accents(line).rstrip())
    if not match:
        return None
    return "GROUP" if match.group(2).upper().startswith("GRUPO") else "ALL"


def match_service_heading(line: str) -> Optional[str]:
    """Devuelve el titulo del servicio si la linea es un encabezado 11.x.y SERVICIO ..."""
    plain = _strip_accents(line).rstrip()
    match = SERVICE_HEADING_RE.match(plain)
    if match:
        number = match.group(1).rstrip(".").replace(",", ".")
        return f"{number} {match.group(2).strip()}".title()
    match = UNNUMBERED_SERVICE_RE.match(plain)
    if match and len(match.group(1).split()) >= 4 and not re.search(r"[a-z]{4,}", match.group(1)):
        return match.group(1).strip().title()
    return None


def find_corpus_start_page(pages: List[Tuple[int, str]]) -> Optional[int]:
    """Primera pagina que contiene el encabezado '11. ESTANDARES Y CRITERIOS DE HABILITACION'."""
    for page_number, text in pages:
        for line in text.split("\n"):
            if CHAPTER_11_RE.match(_strip_accents(line)):
                return page_number
    return None


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
    start_page: Optional[int] = None,
) -> List[ChunkMetadata]:
    """
    Politica de corpus: solo el capitulo 11 (estandares y criterios de habilitacion).
    El modulo cambia unicamente con una linea de encabezado de estandar; el servicio, con
    un encabezado 11.x.y SERVICIO ... Cada chunk conserva la pagina real de su texto.
    """
    explicit_start = start_page is not None
    if start_page is None:
        start_page = find_corpus_start_page(pages)
    if start_page is None:
        raise RuntimeError(
            f"No se encontro el encabezado del capitulo 11 en {source_file}. "
            "Indica la pagina inicial con --start-page."
        )
    logger.info("Corpus normativo desde la pagina %d (capitulo 11).", start_page)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[ChunkMetadata] = []
    current_module: Optional[str] = None
    current_service: Optional[str] = None
    global_index = 0

    def flush(buffer: List[str], page_number: int) -> None:
        nonlocal global_index
        if not buffer or current_module is None:
            buffer.clear()
            return
        for chunk_text in splitter.split_text("\n".join(buffer)):
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
        buffer.clear()

    # Si la pagina inicial se indico a mano no se exige encontrar el titulo del capitulo.
    seen_chapter_heading = explicit_start
    for page_number, page_text in tqdm(pages, desc="Procesando paginas"):
        if page_number < start_page:
            continue

        buffer: List[str] = []
        for line_number, line in enumerate(clean_page_text(page_text).split("\n")):
            plain = _strip_accents(line)
            if not seen_chapter_heading:
                # En la pagina inicial, lo anterior al titulo del cap. 11 es el final
                # del capitulo 10 (tramites): se descarta.
                if CHAPTER_11_RE.match(plain):
                    seen_chapter_heading = True
                else:
                    continue

            if is_header_noise(line) or (line_number < TOP_OF_PAGE_LINES and is_top_of_page_noise(line)):
                continue

            group = match_group_heading(line)
            if group:
                flush(buffer, page_number)
                if group == "ALL":
                    current_service = ALL_SERVICES_LABEL
                else:
                    current_service = None
                current_module = None  # hasta el primer encabezado de estandar/servicio
                buffer.append(line)
                continue

            service = match_service_heading(line)
            if service:
                flush(buffer, page_number)
                current_service = service
                # La descripcion del servicio (antes del primer "Estandar de ...") no pertenece a
                # ningun estandar: sin modulo hasta un encabezado explicito. Un numeral por si
                # solo no demuestra a que estandar pertenece, tampoco en los servicios cuyo
                # primer encabezado se perdio en el OCR.
                current_module = None
                buffer.append(line)
                continue

            module = match_standard_heading(line)
            if module:
                flush(buffer, page_number)
                current_module = module
                buffer.append(line)
                continue

            buffer.append(line)
        flush(buffer, page_number)

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
    start_page: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> None:
    """out_dir: escribe indices y metadatos en <out_dir>/faiss y <out_dir>/metadata (staging)."""
    faiss_dir = (out_dir / "faiss") if out_dir else settings.faiss_index_dir
    store = MetadataStore(out_dir / "metadata") if out_dir else MetadataStore()
    faiss_dir.mkdir(parents=True, exist_ok=True)
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
                start_page=start_page,
            )
        )

    if not all_chunks:
        logger.error("No se generaron chunks. Verifica que el PDF tenga texto extraible.")
        return

    by_module: Dict[str, List[ChunkMetadata]] = {module: [] for module in MODULES}
    for chunk in all_chunks:
        if chunk.module in by_module:
            by_module[chunk.module].append(chunk)

    module_embedding_blocks: List[np.ndarray] = []
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
        module_embedding_blocks.append(module_embeddings)
        module_index = build_faiss_index(module_embeddings)
        module_index_path = faiss_dir / f"{module}.faiss"
        faiss.write_index(module_index, str(module_index_path))
        store.save(module, module_chunks)
        logger.info(
            "Indice guardado: %s | vectores=%d",
            module_index_path,
            module_index.ntotal,
        )

    # El indice global DEBE seguir el mismo orden que MetadataStore.load_all()
    # (modulo por modulo, en el orden de MODULES). Si se construyera en orden de
    # pagina, la posicion FAISS no coincidiria con la posicion del metadato.
    logger.info("Construyendo indice global (orden por modulo)...")
    global_embeddings = np.vstack(module_embedding_blocks)
    global_index = build_faiss_index(global_embeddings)
    global_index_path = faiss_dir / "global.faiss"
    faiss.write_index(global_index, str(global_index_path))
    logger.info(
        "Indice global guardado: %s | vectores=%d",
        global_index_path,
        global_index.ntotal,
    )

    logger.info("Ingesta completada.")


def rebuild_global_index() -> int:
    """
    Reconstruye global.faiss concatenando los vectores de los indices por modulo
    (sin recalcular embeddings), en el orden de MODULES = orden de load_all().

    Falla ANTES de escribir si falta algun indice o si algun conteo no coincide
    con sus metadatos, para no reemplazar un indice correcto por uno incompleto.
    Devuelve el numero de vectores escritos.
    """
    store = MetadataStore()
    blocks: List[np.ndarray] = []
    for module in MODULES:
        path = settings.faiss_index_dir / f"{module}.faiss"
        if not path.exists():
            raise FileNotFoundError(f"Falta el indice del modulo '{module}': {path}")
        index = faiss.read_index(str(path))
        n_meta = len(store.load(module))
        if index.ntotal != n_meta:
            raise ValueError(
                f"Modulo '{module}': {index.ntotal} vectores != {n_meta} metadatos."
            )
        blocks.append(index.reconstruct_n(0, index.ntotal))

    vectors = np.vstack(blocks).astype("float32")
    total_meta = len(store.load_all())
    if vectors.shape[0] != total_meta:
        raise ValueError(f"Global: {vectors.shape[0]} vectores != {total_meta} metadatos.")

    global_index = build_faiss_index(vectors)
    target = settings.faiss_index_dir / "global.faiss"
    tmp = target.with_suffix(".faiss.tmp")
    faiss.write_index(global_index, str(tmp))
    tmp.replace(target)
    logger.info("global.faiss reconstruido | vectores=%d", global_index.ntotal)
    return global_index.ntotal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingesta del corpus normativo de la Resolucion 3100 de 2019."
    )
    parser.add_argument(
        "--rebuild-global",
        action="store_true",
        help="Solo reconstruye global.faiss desde los indices por modulo (sin re-embeber).",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Ruta a un PDF especifico. Si no se pasa, usa todos los PDFs en data/raw.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="Pagina inicial del corpus (por defecto se detecta el capitulo 11).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Escribe indices y metadatos en <out-dir>/faiss y <out-dir>/metadata (no toca los vigentes).",
    )
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    if args.rebuild_global:
        rebuild_global_index()
        return

    pdf_paths = [args.pdf] if args.pdf else sorted(settings.data_raw_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.error("No se encontraron PDFs en %s", settings.data_raw_dir)
        sys.exit(1)

    logger.info("PDFs a procesar: %s", [path.name for path in pdf_paths])
    run_ingestion(
        pdf_paths=pdf_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        start_page=args.start_page,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
