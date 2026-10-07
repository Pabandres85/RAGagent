"""
scripts/find_chunk.py - Busca fragmentos del corpus candidato (staging_v2) para la auditoria.

Sirve para obtener el `evidence_chunk_id_v2` cuando el mejor candidato de la hoja no es el
fragmento correcto. Solo lee; no modifica nada.

Ejemplos:
    python scripts/find_chunk.py --page 182
    python scripts/find_chunk.py --page 182 --module dotacion
    python scripts/find_chunk.py --numeral 43.1 --service "Cuidado Intermedio"
    python scripts/find_chunk.py --text "tubos endotraqueales"
    python scripts/find_chunk.py --id 3f2a9c1b7d44          # un fragmento completo
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from core.metadata_store import MODULES


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def load_chunks(staging_dir: Path) -> list[dict]:
    import json

    chunks: list[dict] = []
    for module in MODULES:
        path = staging_dir / "metadata" / f"{module}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                chunks.extend(json.load(handle))
    return chunks


def search_chunks(
    chunks: list[dict],
    page: int | None = None,
    numeral: str | None = None,
    module: str | None = None,
    service: str | None = None,
    text: str | None = None,
    chunk_id: str | None = None,
) -> list[dict]:
    """Filtra (AND) por pagina exacta, numeral contenido, modulo, servicio y texto (sin acentos)."""
    result = []
    for chunk in chunks:
        if chunk_id and chunk["chunk_id"] != chunk_id:
            continue
        if page is not None and chunk["page"] != page:
            continue
        if module and chunk["module"] != module:
            continue
        if service and _norm(service) not in _norm(chunk.get("service") or ""):
            continue
        if numeral and numeral not in (chunk.get("numeral") or "") and numeral not in chunk["text"]:
            continue
        if text and _norm(text) not in _norm(chunk["text"]):
            continue
        result.append(chunk)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca fragmentos del corpus candidato.")
    parser.add_argument("--staging", type=Path, default=settings.faiss_index_dir.parent / "staging_v2")
    parser.add_argument("--page", type=int)
    parser.add_argument("--numeral", help="numeral (p. ej. 43.1) en el metadato o en el texto")
    parser.add_argument("--module", choices=MODULES)
    parser.add_argument("--service", help="fragmento del nombre del servicio")
    parser.add_argument("--text", help="texto contenido (sin distinguir acentos)")
    parser.add_argument("--id", dest="chunk_id", help="muestra un fragmento completo por su chunk_id")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--width", type=int, default=260, help="caracteres de texto por fragmento")
    args = parser.parse_args()

    found = search_chunks(
        load_chunks(args.staging), args.page, args.numeral, args.module, args.service, args.text, args.chunk_id
    )
    if not found:
        print("Sin resultados.")
        return
    width = 100000 if args.chunk_id else args.width
    for chunk in found[: args.limit]:
        body = chunk["text"].replace("\n", " / ")
        print(f"{chunk['chunk_id']} | {chunk['module']} | pag {chunk['page']} | num {chunk.get('numeral')} | {chunk.get('service')}")
        print(f"    {body[:width]}{'...' if len(body) > width else ''}")
    if len(found) > args.limit:
        print(f"... {len(found) - args.limit} resultados mas (usa --limit o filtra mas)")


if __name__ == "__main__":
    main()
