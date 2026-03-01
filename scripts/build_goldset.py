"""
scripts/build_goldset.py — Generador del conjunto de evaluación (gold set).

Estrategia:
  1. Carga los metadatos de chunks ya ingestados (requiere haber ejecutado ingest.py).
  2. Muestrea N_PER_MODULE chunks por módulo (filtrando por longitud mínima).
  3. Para cada chunk, pide al LLM que genere un par (pregunta, respuesta de referencia)
     anclado al texto del fragmento normativo.
  4. Guarda el resultado en eval/datasets/gold_set.json.

Reanudación: si el archivo de salida ya existe, omite las preguntas ya generadas
para no repetir llamadas al LLM.

Uso:
    python scripts/build_goldset.py
    python scripts/build_goldset.py --n-per-module 10   # prueba rapida
    python scripts/build_goldset.py --module talento_humano  # solo un modulo
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from core.llm_client import chat_completion
from core.metadata_store import ChunkMetadata, MetadataStore, MODULE_DISPLAY_NAMES, MODULES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Parametros por defecto ────────────────────────────────────────────────────
N_PER_MODULE = 30          # preguntas por modulo -> 210 total
MIN_CHUNK_LEN = 80         # caracteres minimos para que el chunk sea util
QA_TEMPERATURE = 0.3       # algo de variedad en las preguntas generadas
QA_MAX_TOKENS = 512        # suficiente para pregunta + respuesta concisa

# ── Prompt para generacion de Q&A ────────────────────────────────────────────
QA_SYSTEM_PROMPT = (
    "Eres un experto en habilitacion de servicios de salud en Colombia "
    "(Resolucion 3100 de 2019). Tu tarea es generar preguntas de evaluacion "
    "para un sistema RAG normativo.\n\n"
    "Dado un fragmento normativo, debes generar:\n"
    "1. Una PREGUNTA concreta que un evaluador de habilitacion podria hacer sobre "
    "ese requisito (usa lenguaje tecnico real, no generico).\n"
    "2. Una RESPUESTA DE REFERENCIA breve y directa, basada exclusivamente en el "
    "fragmento proporcionado.\n\n"
    "REGLAS:\n"
    "- La pregunta debe ser especifica y responderse con algo del fragmento.\n"
    "- La respuesta debe citar el requisito de forma precisa.\n"
    "- Responde UNICAMENTE con este JSON (sin texto adicional):\n"
    '{"question": "...", "answer": "..."}'
)

QA_USER_TEMPLATE = (
    "MODULO: {module_name}\n"
    "NUMERAL: {numeral}\n"
    "PAGINA: {page}\n\n"
    "FRAGMENTO NORMATIVO:\n{text}\n\n"
    "Genera la pregunta y respuesta de referencia para este fragmento."
)


def sample_chunks(
    store: MetadataStore,
    module: str,
    n: int,
    min_len: int,
    seed: int = 42,
) -> List[ChunkMetadata]:
    """Carga y muestrea chunks de un modulo con longitud minima."""
    chunks = store.load(module)
    if not chunks:
        logger.warning("Modulo '%s' sin metadatos. Ya corriste ingest.py?", module)
        return []

    preferred = [
        c
        for c in chunks
        if len(c.text) >= min_len and c.numeral is not None and (c.page or 0) >= 36
    ]
    eligible = preferred
    if not eligible:
        eligible = [c for c in chunks if len(c.text) >= min_len and c.numeral is not None]
    if not eligible:
        eligible = [c for c in chunks if len(c.text) >= min_len]

    if not eligible:
        logger.warning(
            "Modulo '%s': ningun chunk supera longitud minima %d.", module, min_len
        )
        return []

    rng = random.Random(seed)
    sample = rng.sample(eligible, min(n, len(eligible)))
    logger.info(
        "Modulo '%s': %d chunks elegibles -> %d muestreados.",
        module, len(eligible), len(sample),
    )
    return sample


def generate_qa(chunk: ChunkMetadata) -> Optional[dict]:
    """
    Llama al LLM para generar un par (pregunta, respuesta) a partir del chunk.
    Retorna None si la respuesta no es JSON valido.
    """
    module_name = MODULE_DISPLAY_NAMES.get(chunk.module, chunk.module)
    user_msg = QA_USER_TEMPLATE.format(
        module_name=module_name,
        numeral=chunk.numeral or "N/A",
        page=chunk.page or "N/A",
        text=chunk.text[:800],  # limitar por tokens
    )

    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = chat_completion(messages, temperature=QA_TEMPERATURE, max_tokens=QA_MAX_TOKENS)
    except Exception as exc:
        logger.error("Error llamando al LLM: %s", exc)
        return None

    # Extraer JSON de la respuesta
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        logger.warning("Respuesta sin JSON: %s", raw[:120])
        return None

    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("JSON malformado: %s | raw=%s", exc, raw[:120])
        return None

    if "question" not in data or "answer" not in data:
        logger.warning("JSON sin campos requeridos: %s", data)
        return None

    return {
        "question": str(data["question"]).strip(),
        "answer": str(data["answer"]).strip(),
        "module": chunk.module,
        "chunk_id": chunk.chunk_id,
        "numeral": chunk.numeral,
        "page": chunk.page,
        "source_file": chunk.source_file,
    }


def load_existing(output_path: Path) -> list[dict]:
    """Carga el gold set parcial si existe (para reanudar)."""
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save(output_path: Path, entries: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def build_goldset(
    modules: List[str],
    n_per_module: int,
    output_path: Path,
    min_chunk_len: int = MIN_CHUNK_LEN,
    seed: int = 42,
) -> None:
    store = MetadataStore()
    existing = load_existing(output_path)
    existing_chunk_ids = {e["chunk_id"] for e in existing}
    entries = list(existing)

    logger.info(
        "Gold set existente: %d entradas. Generando para modulos: %s",
        len(existing), modules,
    )

    for module in modules:
        module_name = MODULE_DISPLAY_NAMES.get(module, module)
        chunks = sample_chunks(store, module, n_per_module, min_chunk_len, seed=seed)
        if not chunks:
            continue

        module_new = 0
        for chunk in chunks:
            # Saltar chunks ya procesados (reanudacion)
            if chunk.chunk_id in existing_chunk_ids:
                continue

            logger.info(
                "  [%s] chunk_id=%s | numeral=%s | pagina=%s",
                module, chunk.chunk_id, chunk.numeral, chunk.page,
            )

            qa = generate_qa(chunk)
            if qa is None:
                logger.warning("  -> Omitido (respuesta invalida).")
                continue

            entries.append(qa)
            existing_chunk_ids.add(chunk.chunk_id)
            module_new += 1

            # Guardar incrementalmente cada 5 entradas nuevas
            if module_new % 5 == 0:
                save(output_path, entries)
                logger.info("  Progreso guardado: %d entradas totales.", len(entries))

        logger.info(
            "Modulo '%s' completado: +%d nuevas (total acumulado: %d).",
            module_name, module_new, len(entries),
        )

    save(output_path, entries)
    logger.info("=" * 60)
    logger.info("Gold set final: %d entradas -> %s", len(entries), output_path)
    _print_summary(entries)


def _print_summary(entries: list[dict]) -> None:
    from collections import Counter
    counts = Counter(e["module"] for e in entries)
    logger.info("Distribucion por modulo:")
    for module in MODULES:
        name = MODULE_DISPLAY_NAMES.get(module, module)
        logger.info("  %-45s %3d preguntas", name, counts.get(module, 0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el gold set de evaluacion (210 Q&A desde el corpus ingestado)."
    )
    parser.add_argument(
        "--n-per-module",
        type=int,
        default=N_PER_MODULE,
        help=f"Numero de preguntas por modulo (default: {N_PER_MODULE}).",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        choices=MODULES,
        help="Genera solo para un modulo especifico.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.gold_set_path,
        help="Ruta de salida del gold set JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible.",
    )
    parser.add_argument(
        "--min-chunk-len",
        type=int,
        default=MIN_CHUNK_LEN,
        help=f"Longitud minima de chunk en caracteres (default: {MIN_CHUNK_LEN}).",
    )
    args = parser.parse_args()

    target_modules = [args.module] if args.module else MODULES

    build_goldset(
        modules=target_modules,
        n_per_module=args.n_per_module,
        output_path=args.output,
        min_chunk_len=args.min_chunk_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
