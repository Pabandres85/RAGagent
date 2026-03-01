"""
Limpia el gold set para dejar solo preguntas mas confiables por modulo.

Reglas iniciales:
1. Excluir entradas sin numeral.
2. Excluir entradas previas a la pagina 36 (zona administrativa/introductoria).
3. Crear un respaldo del archivo original antes de sobrescribir.
"""
from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_SET_PATH = BASE_DIR / "eval" / "datasets" / "gold_set.json"
BACKUP_PATH = BASE_DIR / "eval" / "datasets" / "gold_set.backup.json"
MIN_PAGE = 36


def clean_goldset() -> None:
    if not GOLD_SET_PATH.exists():
        print("No se encontro el gold_set.json")
        return

    with open(GOLD_SET_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    kept: list[dict] = []
    removed = 0

    for item in data:
        page = item.get("page") or 0
        numeral = item.get("numeral")

        if page >= MIN_PAGE and numeral is not None:
            kept.append(item)
        else:
            removed += 1

    with open(BACKUP_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    with open(GOLD_SET_PATH, "w", encoding="utf-8") as handle:
        json.dump(kept, handle, ensure_ascii=False, indent=2)

    print(f"Total original: {len(data)}")
    print(f"Total conservado: {len(kept)}")
    print(f"Total removido: {removed}")
    print(f"Respaldo: {BACKUP_PATH}")


if __name__ == "__main__":
    clean_goldset()
