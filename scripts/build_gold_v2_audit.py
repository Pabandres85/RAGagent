"""
scripts/build_gold_v2_audit.py - Hoja de auditoria para el gold set v2.

Cruza el gold set v1 (122 preguntas) con el corpus candidato (staging_v2) y genera,
por cada pregunta, la evidencia que necesita un revisor humano para decidir si se
conserva, se corrige (modulo, pregunta, respuesta y/o evidencia) o se retira. NO modifica
el gold v1 ni los artefactos vigentes: escribe archivos nuevos.

Uso:
    python scripts/build_gold_v2_audit.py
    python scripts/build_gold_v2_audit.py --staging artifacts/staging_v2

Salidas (eval/datasets/):
    gold_v2_audit.json   registro completo (para la UI de auditoria o scripts)
    gold_v2_audit.csv    version tabular para revisar en Excel (UTF-8 con BOM)

Prioridades: 1 = revision obligatoria (cambia el modulo, sin correspondencia, fuera del
corpus, contradiccion con el nombre de modulo escrito en la pregunta, general);
2 = recomendada (respuesta poco respaldada, numerales ausentes, emparejamiento ambiguo);
3 = sin banderas (basta una muestra).

ESQUEMA DE DECISION (v2). Una sola decision por registro:
    conservar | corregir | retirar
  * corregir admite correcciones COMBINADAS: new_module, rewritten_question,
    rewritten_answer y/o evidence_chunk_id_v2 (al menos una). Ej.: reetiquetar Y reescribir.
  * review_level: preaudit (propuesta tecnica) | author (revision del autor) | expert
    (revision experta). author/expert exigen reviewer Y reviewed_at (fecha ISO); el campo
    por si solo NO constituye validacion. Solo author/expert cuentan para activar el gold v2.
  * retire_category (retirar): fuera_de_alcance | sin_evidencia | pregunta_defectuosa |
    referencia_incorrecta | otro  (distingue una decision de alcance de "no existe").
  * history: historial de cambios y de la migracion desde el esquema v1.
Los campos de revision quedan en blanco: los completa el revisor.

FUENTE DE VERDAD: el JSON. El CSV es solo una vista para revisar en Excel y se regenera
desde el JSON. Flujo de trabajo:
    1. python scripts/build_gold_v2_audit.py                  (genera la hoja)
    2. editar las columnas de decision en el CSV
    3. python scripts/build_gold_v2_audit.py --import-csv eval/datasets/gold_v2_audit.csv
       (valida y guarda las decisiones en el JSON, por audit_id)
Reglas de edicion del CSV:
  * Celda vacia = conservar el valor guardado (proteccion contra borrados accidentales).
  * Escribe <borrar> en una celda para limpiarla sin cambiar la decision.
  * CAMBIAR la decision (p. ej. de una propuesta `corregir` a `retirar`) es una revision
    NUEVA y completa: las celdas vacias se limpian (no se heredan de la propuesta anterior),
    debes registrar tu propio reviewer / review_level / reviewed_at y la decision previa
    queda en `history`.
Regenerar la hoja NUNCA borra decisiones: se fusionan por audit_id, se hace copia de
seguridad y el generador se niega a sobrescribir un CSV con decisiones sin importar.

`answer_lexical_overlap_v2` es solo solapamiento de palabras entre la respuesta de
referencia y el fragmento nuevo: una senal para el revisor, NO una validacion de que la
respuesta sea normativamente correcta.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from core.metadata_store import MODULES

CORPUS_START_PAGE = 59
MATCH_MIN = 0.5            # cobertura minima del chunk viejo por el nuevo para aceptar el emparejamiento
SUPPORT_MIN = 0.5          # fraccion minima de la respuesta que debe aparecer en el fragmento
SCHEMA_VERSION = 2
DECISIONS = ["conservar", "corregir", "retirar"]
LEGACY_DECISIONS = {"reetiquetar", "reescribir"}          # esquema v1: se migran a "corregir"
REVIEW_LEVELS = ["preaudit", "author", "expert"]
HUMAN_LEVELS = {"author", "expert"}
# Marcador explicito para LIMPIAR una celda del CSV sin cambiar la decision (una celda vacia
# nunca borra: conserva el valor guardado).
CLEAR_TOKEN = "<borrar>"
# Estados con un candidato fiable en el corpus nuevo: solo ahi el candidato mostrado sirve de evidencia.
RELIABLE_STATUSES = ("modulo_cambia", "modulo_coincide")
RETIRE_CATEGORIES = ["fuera_de_alcance", "sin_evidencia", "pregunta_defectuosa", "referencia_incorrecta", "otro"]
CORRECTION_FIELDS = ["new_module", "rewritten_question", "rewritten_answer", "evidence_chunk_id_v2"]
REVIEW_FIELDS = [
    "decision", "decision_reason", "reviewer", "review_level", "reviewed_at",
    "new_module", "rewritten_question", "rewritten_answer", "evidence_chunk_id_v2", "retire_category",
]

# Nombres con los que una pregunta puede referirse a un estandar. Solo cuentan cuando se
# presentan COMO modulo/estandar/requisito ("del modulo de Dotacion"); mencionar el tema
# ("dispositivos medicos") no es nombrar el modulo.
MODULE_NAME_PATTERNS = {
    "talento_humano": r"talento humano",
    "infraestructura": r"infraestructura",
    "dotacion": r"dotaci[oó]n",
    "medicamentos_dispositivos": r"medicamentos(?: y dispositivos(?: m[eé]dicos)?(?: e insumos)?)?|dispositivos m[eé]dicos(?: e insumos)?",
    "procesos_prioritarios": r"procesos prioritarios",
    "historia_clinica": r"historia cl[ií]nica(?: y registros)?",
    "interdependencia": r"interdependencia",
}
_MODULE_INTRO = r"(?:m[oó]dulo|est[aá]ndar|requisito)s?\s+(?:de\s+|del\s+)?(?:la\s+|los\s+)?"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _tokens(text: str, min_len: int = 3) -> set[str]:
    return set(re.findall(rf"[a-z0-9]{{{min_len},}}", _norm(text)))


def _load_chunks(directory: Path) -> list[dict]:
    chunks: list[dict] = []
    for module in MODULES:
        path = directory / f"{module}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                chunks.extend(json.load(handle))
    return chunks


def _coverage(reference: set[str], candidate: set[str]) -> float:
    return len(reference & candidate) / max(len(reference), 1)


def _best_two(old_text: str, candidates: list[dict]) -> list[tuple[dict, float]]:
    old_tokens = _tokens(old_text)
    scored = [(c, _coverage(old_tokens, _tokens(c["text"]))) for c in candidates]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:2]


def _names_in_question(question: str) -> list[str]:
    return [
        m for m, rx in MODULE_NAME_PATTERNS.items()
        if re.search(_MODULE_INTRO + "(?:" + rx + ")", question, re.IGNORECASE)
    ]


def _answer_lexical_overlap(answer: str, chunk_text: str) -> float:
    """Fraccion de las palabras (>=4 letras) de la respuesta que aparecen en el fragmento."""
    answer_tokens = _tokens(answer, min_len=4)
    return _coverage(answer_tokens, _tokens(chunk_text, min_len=4)) if answer_tokens else 0.0


def _numerals_missing(question: str, answer: str, chunk_text: str) -> list[str]:
    """Numerales citados en pregunta/respuesta (p. ej. 25.1) que no aparecen en el fragmento."""
    cited = set(re.findall(r"\b\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\b", f"{question} {answer}"))
    return sorted(n for n in cited if n not in chunk_text)


# TODO lo que el revisor ve para decidir: si cualquiera cambia, la decision previa se invalida.
EVIDENCE_FIELDS = [
    "question", "reference_answer", "module_v1", "module_proposed", "status",
    "module_named_in_question",
    "chunk_id_v1", "page_v1", "numeral_v1",
    "chunk_id_v2", "page_v2", "service_v2", "numeral_v2",
    "match_score", "second_best_chunk_id_v2", "second_best_module_v2", "second_best_page_v2",
    "second_best_service_v2", "second_best_score", "match_margin",
    "answer_lexical_overlap_v2", "numerals_missing_in_v2",
]


def evidence_hash(record: dict) -> str:
    """Huella de todo lo que el revisor vio: si cambia, la decision previa ya no aplica."""
    payload = {f: record.get(f) for f in EVIDENCE_FIELDS}
    for text_field in ("chunk_text_v1", "chunk_text_v2", "second_best_chunk_text_v2"):
        payload[text_field] = hashlib.sha256((record.get(text_field) or "").encode("utf-8")).hexdigest()
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_audit(staging_dir: Path) -> list[dict]:
    with open(settings.gold_set_path, "r", encoding="utf-8") as handle:
        gold = json.load(handle)

    old_by_id = {c["chunk_id"]: c for c in _load_chunks(settings.metadata_dir)}
    new_chunks = _load_chunks(staging_dir / "metadata")
    if not new_chunks:
        raise FileNotFoundError(f"No hay metadatos del corpus candidato en {staging_dir / 'metadata'}")
    new_by_page: dict[int, list[dict]] = {}
    for chunk in new_chunks:
        new_by_page.setdefault(chunk["page"], []).append(chunk)

    records: list[dict] = []
    for index, item in enumerate(gold, start=1):
        question, answer, module_v1 = item["question"], item.get("answer", ""), item["module"]
        old = old_by_id.get(item.get("chunk_id")) if item.get("chunk_id") else None
        flags: list[str] = []

        record = {
            "audit_id": f"v1-{index:03d}",
            "v1_index": index,
            "question": question,
            "reference_answer": answer,
            "module_v1": module_v1,
            "module_proposed": None,
            "module_named_in_question": _names_in_question(question),
            "status": None,
            "flags": flags,
            "priority": 3,
            "chunk_id_v1": item.get("chunk_id"),
            "page_v1": old["page"] if old else item.get("page"),
            "numeral_v1": item.get("numeral"),
            "chunk_text_v1": old["text"] if old else None,
            "chunk_id_v2": None,
            "page_v2": None,
            "service_v2": None,
            "numeral_v2": None,
            "chunk_text_v2": None,
            "match_score": None,
            "second_best_chunk_id_v2": None,
            "second_best_module_v2": None,
            "second_best_page_v2": None,
            "second_best_service_v2": None,
            "second_best_chunk_text_v2": None,
            "second_best_score": None,
            "match_margin": None,
            "answer_lexical_overlap_v2": None,
            "numerals_missing_in_v2": [],
            "evidence_hash": None,
            "stale_review": None,      # decision previa cuya evidencia cambio (requiere nueva revision)
            "schema_version": SCHEMA_VERSION,
            "history": [],
            "decision": "",            # conservar | corregir | retirar
            "decision_reason": "",
            "reviewer": "",
            "review_level": "",        # preaudit | author | expert
            "reviewed_at": "",         # fecha ISO; obligatoria para author/expert
            "new_module": "",
            "rewritten_question": "",
            "rewritten_answer": "",
            "evidence_chunk_id_v2": "",
            "retire_category": "",
        }

        if module_v1 == "general":
            record["status"] = "general_fuera_de_alcance"
            flags.append("general: fuera del corpus del cap. 11 (referencia generada con indice defectuoso)")
            record["priority"] = 1
            records.append(record)
            continue
        if old is None:
            record["status"] = "sin_chunk_v1"
            flags.append("sin chunk_id localizable en el corpus v1")
            record["priority"] = 1
            records.append(record)
            continue
        if old["page"] < CORPUS_START_PAGE:
            record["status"] = "fuera_de_corpus"
            flags.append(f"fuente en pag. {old['page']} (< {CORPUS_START_PAGE}): sale del corpus")
            record["priority"] = 1
            records.append(record)
            continue

        ranked = _best_two(old["text"], new_by_page.get(old["page"], []))
        if not ranked or ranked[0][1] < MATCH_MIN:
            record["status"] = "sin_correspondencia"
            flags.append("sin fragmento equivalente en el corpus nuevo (mismo numero de pagina)")
            record["priority"] = 1
            if ranked:
                best, score = ranked[0]
                record.update(
                    chunk_id_v2=best["chunk_id"], page_v2=best["page"], service_v2=best["service"],
                    numeral_v2=best["numeral"], chunk_text_v2=best["text"], match_score=round(score, 3),
                    module_proposed=best["module"],
                )
            records.append(record)
            continue

        best, score = ranked[0]
        second, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
        support = _answer_lexical_overlap(answer, best["text"])
        missing = _numerals_missing(question, answer, best["text"])
        named = record["module_named_in_question"]

        record.update(
            module_proposed=best["module"],
            chunk_id_v2=best["chunk_id"], page_v2=best["page"], service_v2=best["service"],
            numeral_v2=best["numeral"], chunk_text_v2=best["text"],
            match_score=round(score, 3),
            second_best_chunk_id_v2=second["chunk_id"] if second else None,
            second_best_module_v2=second["module"] if second else None,
            second_best_page_v2=second["page"] if second else None,
            second_best_service_v2=second["service"] if second else None,
            second_best_chunk_text_v2=second["text"] if second else None,
            second_best_score=round(second_score, 3) if second else None,
            match_margin=round(score - second_score, 3),
            answer_lexical_overlap_v2=round(support, 3),
            numerals_missing_in_v2=missing,
        )

        changed = best["module"] != module_v1
        record["status"] = "modulo_cambia" if changed else "modulo_coincide"
        if changed:
            flags.append(f"modulo cambia: {module_v1} -> {best['module']}")
            record["priority"] = 1
        if named and best["module"] not in named:
            flags.append(f"la pregunta nombra {named} pero el modulo propuesto es {best['module']} (posible contradiccion)")
            record["priority"] = 1
        if named and module_v1 not in named:
            flags.append(f"la pregunta nombra {named}, distinto de la etiqueta v1 ({module_v1})")
        if support < SUPPORT_MIN:
            flags.append(f"solapamiento de palabras de la respuesta con el fragmento nuevo bajo ({support:.0%}); senal, no validacion")
            record["priority"] = min(record["priority"], 2)
        if missing:
            flags.append(f"numerales citados ausentes en el fragmento nuevo: {missing}")
            record["priority"] = min(record["priority"], 2)
        if second is not None and (score - second_score) < 0.1:
            flags.append(f"emparejamiento ambiguo (margen {score - second_score:.2f} con el 2.o candidato)")
            record["priority"] = min(record["priority"], 2)
        records.append(record)

    for record in records:
        record["evidence_hash"] = evidence_hash(record)
    records.sort(key=lambda r: (r["priority"], r["v1_index"]))
    return records


def _has_review(record: dict) -> bool:
    return any(str(record.get(field, "")).strip() for field in REVIEW_FIELDS)


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(path, target)
    return target


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ migracion v1 -> v2
def migrate_record(record: dict) -> bool:
    """
    Lleva un registro al esquema v2. Las decisiones existentes se CONSERVAN COMO PROPUESTAS
    `preaudit` (nunca se convierten en decisiones del autor o de un experto) y la decision
    original queda en `history`. Devuelve True si hubo cambios.
    """
    if record.get("schema_version") == SCHEMA_VERSION:
        return False
    legacy = {f: record.get(f, "") for f in ("decision", "decision_reason", "reviewer", "rewritten_question")}
    for field in REVIEW_FIELDS:
        record.setdefault(field, "")
    record.setdefault("history", [])

    decision = str(record.get("decision", "")).strip().lower()
    if decision:
        if decision == "reetiquetar":
            record["decision"] = "corregir"
            record["new_module"] = record.get("new_module") or record.get("module_proposed") or ""
        elif decision == "reescribir":
            record["decision"] = "corregir"
        record["review_level"] = "preaudit"
        record["reviewed_at"] = record.get("reviewed_at", "")
        record["history"].append({
            "event": "migracion_a_v2", "at": _now(), "legacy": legacy,
            "note": "decision previa conservada como propuesta preaudit; no equivale a revision del autor ni de un experto",
        })
    record["schema_version"] = SCHEMA_VERSION
    return True


def load_records(json_path: Path) -> list[dict]:
    """Lee el JSON y lo lleva al esquema v2 en memoria (no escribe)."""
    with open(json_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    for record in records:
        migrate_record(record)
    return records


def migrate_file(json_path: Path) -> int:
    with open(json_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    changed = sum(1 for record in records if migrate_record(record))
    if changed:
        _backup(json_path)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
    return changed


# ------------------------------------------------------------------ fusion al regenerar
def merge_existing_decisions(records: list[dict], json_path: Path) -> int:
    """
    Conserva las decisiones ya registradas en el JSON (por audit_id) SOLO si la evidencia que
    el revisor vio no cambio (huella `evidence_hash`). Si cambio, la decision previa se
    guarda en `stale_review`, los campos de revision quedan vacios, el registro sube a P1 y
    requiere nueva revision. Devuelve cuantas decisiones se conservaron.
    """
    if not json_path.exists():
        return 0
    previous = {r["audit_id"]: r for r in load_records(json_path)}
    current_ids = {r["audit_id"] for r in records}

    orphans = [a for a, r in previous.items() if _has_review(r) and a not in current_ids]
    if orphans:
        raise RuntimeError(f"Hay decisiones de registros que ya no existen: {orphans}. No se sobrescribe.")

    kept = 0
    for record in records:
        old = previous.get(record["audit_id"])
        if old is None:
            continue
        if old.get("history"):
            record["history"] = list(old["history"])
        if not _has_review(old):
            if old.get("stale_review"):  # no perder una decision previa ya marcada como obsoleta
                record["stale_review"] = old["stale_review"]
            continue
        if old["question"] != record["question"]:
            raise RuntimeError(
                f"El gold v1 cambio bajo una decision existente ({record['audit_id']}). "
                "Revisa antes de regenerar; no se sobrescribe."
            )
        if old.get("evidence_hash") != record["evidence_hash"]:
            record["stale_review"] = {f: old.get(f, "") for f in REVIEW_FIELDS}
            record["history"].append({"event": "decision_obsoleta", "at": _now(), "previous": record["stale_review"]})
            record["flags"].append("la evidencia cambio desde la decision previa; ver stale_review y revisar de nuevo")
            record["priority"] = 1
            continue
        for field in REVIEW_FIELDS:
            record[field] = old.get(field, "")
        kept += 1

    # Invariante: una decision obsoleta sin decision nueva SIEMPRE esta en P1 y senalada,
    # tambien cuando el historial obsoleto se arrastra de una regeneracion anterior.
    for record in records:
        if record.get("stale_review") and not record.get("decision"):
            record["priority"] = 1
            if not any("decision previa" in f for f in record["flags"]):
                record["flags"].append("decision previa obsoleta pendiente de nueva revision (ver stale_review)")
    return kept


# ------------------------------------------------------------------ CSV
def _read_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _normalize_for_compare(field: str, value: str) -> str:
    value = (value or "").strip()
    if field == "decision" and value.lower() in LEGACY_DECISIONS:
        return "corregir"  # un CSV del esquema v1 usaba reetiquetar/reescribir
    return value


def _csv_has_unimported_edits(csv_path: Path, json_records: list[dict]) -> list[str]:
    """audit_id cuyo CSV tiene valores no vacios que el JSON no contiene."""
    if not csv_path.exists():
        return []
    by_id = {r["audit_id"]: r for r in json_records}
    pending = []
    for row in _read_csv_rows(csv_path):
        record = by_id.get(row.get("audit_id"))
        if record is None:
            continue
        # Solo cuenta si el CSV trae un valor que el JSON no tiene (un CSV desactualizado,
        # con campos vacios, no es una edicion pendiente).
        for field in REVIEW_FIELDS:
            csv_value = _normalize_for_compare(field, row.get(field))
            if csv_value and csv_value != str(record.get(field, "")).strip():
                pending.append(row["audit_id"])
                break
    return pending


CSV_COLUMNS = [
    "audit_id", "priority", "status", "module_v1", "module_proposed", "flags",
    "question", "reference_answer", "page_v1", "page_v2", "service_v2", "numeral_v2",
    "match_score", "second_best_score", "match_margin", "answer_lexical_overlap_v2",
    "chunk_id_v1", "chunk_id_v2", "chunk_text_v2",
    "second_best_chunk_id_v2", "second_best_module_v2", "second_best_page_v2",
    "second_best_service_v2", "second_best_chunk_text_v2",
    "evidence_hash", "stale_review",
    "decision", "decision_reason", "new_module", "rewritten_question", "rewritten_answer",
    "evidence_chunk_id_v2", "retire_category", "review_level", "reviewer", "reviewed_at",
]


def _write_csv(records: list[dict], csv_path: Path) -> None:
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["flags"] = " | ".join(record["flags"])
            row["stale_review"] = json.dumps(record["stale_review"], ensure_ascii=False) if record.get("stale_review") else ""
            for key in ("chunk_text_v2", "second_best_chunk_text_v2"):
                row[key] = (record.get(key) or "")[:500].replace("\n", " / ")
            writer.writerow(row)


def write_outputs(records: list[dict], out_dir: Path) -> tuple[Path, Path, int]:
    """Escribe JSON (fuente de verdad) y CSV (vista) sin perder decisiones existentes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gold_v2_audit.json"
    csv_path = out_dir / "gold_v2_audit.csv"

    kept = merge_existing_decisions(records, json_path)

    # Si el CSV tiene decisiones editadas que no estan en el JSON, no se pisa.
    existing_json = load_records(json_path) if json_path.exists() else []
    pending = _csv_has_unimported_edits(csv_path, existing_json or records)
    if pending:
        raise RuntimeError(
            f"El CSV tiene decisiones sin importar ({len(pending)} registros, p. ej. {pending[:3]}). "
            "Ejecuta primero --import-csv; no se sobrescribe."
        )

    _backup(json_path)
    _backup(csv_path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    _write_csv(records, csv_path)
    return json_path, csv_path, kept


# ------------------------------------------------------------------ validacion e importacion
def corpus_by_id(staging_dir: Path) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in _load_chunks(staging_dir / "metadata")}


def validate_review(record: dict, state: dict, corpus: dict[str, dict]) -> list[str]:
    """Reglas de una decision (`state` = valores finales de los REVIEW_FIELDS). Devuelve errores."""
    aid = record["audit_id"]
    errors: list[str] = []
    decision = state["decision"]

    if decision in LEGACY_DECISIONS:
        return [f"{aid}: '{decision}' ya no existe; usa 'corregir' con new_module / rewritten_question ..."]
    if decision and decision not in DECISIONS:
        return [f"{aid}: decision '{decision}' invalida (usa {', '.join(DECISIONS)})"]
    if not decision:
        if state["review_level"] and state["review_level"] not in REVIEW_LEVELS:
            errors.append(f"{aid}: review_level '{state['review_level']}' invalido (usa {', '.join(REVIEW_LEVELS)})")
        return errors

    level = state["review_level"]
    if level not in REVIEW_LEVELS:
        errors.append(f"{aid}: falta review_level (usa {', '.join(REVIEW_LEVELS)})")
    if not state["reviewer"]:
        errors.append(f"{aid}: falta el revisor")
    if level in HUMAN_LEVELS:
        stamp = state["reviewed_at"]
        if not stamp:
            errors.append(f"{aid}: review_level={level} exige reviewed_at (fecha ISO)")
        else:
            try:
                when = datetime.fromisoformat(stamp)
                if when > datetime.now():
                    errors.append(f"{aid}: reviewed_at '{stamp}' esta en el futuro")
            except ValueError:
                errors.append(f"{aid}: reviewed_at '{stamp}' no es una fecha ISO (AAAA-MM-DD)")

    if decision == "corregir":
        if not any(state[f] for f in CORRECTION_FIELDS):
            errors.append(f"{aid}: 'corregir' requiere al menos uno de {', '.join(CORRECTION_FIELDS)}")
        if state["new_module"] and state["new_module"] not in MODULES:
            errors.append(f"{aid}: new_module '{state['new_module']}' no es un modulo valido")
    elif any(state[f] for f in CORRECTION_FIELDS):
        errors.append(f"{aid}: solo 'corregir' admite {', '.join(CORRECTION_FIELDS)}")

    # Modulo con el que quedaria la pregunta: new_module si se indica; si no, el modulo v1.
    final_module = state["new_module"] or record.get("module_v1", "")

    # Sin evidencia explicita, la evidencia es el candidato que el revisor ve en la hoja
    # (solo si el emparejamiento es confiable: estados modulo_cambia / modulo_coincide).
    if (
        decision == "corregir"
        and not state["evidence_chunk_id_v2"]
        and record.get("status") in RELIABLE_STATUSES
        and record.get("module_proposed")
        and final_module != record["module_proposed"]
    ):
        errors.append(
            f"{aid}: el candidato mostrado pertenece a '{record['module_proposed']}', incompatible con el modulo "
            f"final '{final_module}': aporta evidence_chunk_id_v2 de un fragmento de '{final_module}' o ajusta new_module"
        )

    if decision == "corregir" and record.get("status") not in RELIABLE_STATUSES and not state["evidence_chunk_id_v2"]:
        errors.append(
            f"{aid}: sin candidato fiable (estado '{record.get('status')}'); 'corregir' exige evidence_chunk_id_v2 "
            "de un fragmento del corpus candidato que respalde la pregunta"
        )

    if state["evidence_chunk_id_v2"]:
        chunk = corpus.get(state["evidence_chunk_id_v2"])
        if chunk is None:
            errors.append(f"{aid}: evidence_chunk_id_v2 '{state['evidence_chunk_id_v2']}' no existe en el corpus candidato")
        elif chunk["module"] != final_module:
            errors.append(
                f"{aid}: la evidencia pertenece a '{chunk['module']}', incompatible con el modulo final "
                f"'{final_module}' (usa new_module o elige otra evidencia)"
            )

    if decision in ("conservar", "corregir"):
        named = record.get("module_named_in_question") or []
        if named and final_module not in named and not state["rewritten_question"]:
            errors.append(
                f"{aid}: la pregunta nombra {named} pero el modulo final es '{final_module}': reescribe la pregunta "
                "(rewritten_question) para evitar la contradiccion"
            )
    if decision == "conservar" and record.get("status") != "modulo_coincide":
        errors.append(
            f"{aid}: solo se puede 'conservar' un registro con estado 'modulo_coincide' (evidencia compatible y dentro "
            f"del corpus); este esta en '{record.get('status')}'. Para mantener la etiqueta usa 'corregir' con "
            "new_module y evidence_chunk_id_v2 de un fragmento valido; si no hay evidencia, 'retirar'"
        )
    if decision == "retirar":
        if not state["decision_reason"]:
            errors.append(f"{aid}: 'retirar' requiere decision_reason")
        if state["retire_category"] and state["retire_category"] not in RETIRE_CATEGORIES:
            errors.append(f"{aid}: retire_category '{state['retire_category']}' invalida (usa {', '.join(RETIRE_CATEGORIES)})")
        if level in HUMAN_LEVELS and not state["retire_category"]:
            errors.append(f"{aid}: una baja con review_level={level} exige retire_category")
    elif state["retire_category"]:
        errors.append(f"{aid}: retire_category solo aplica a 'retirar'")
    return errors


def import_csv(csv_path: Path, json_path: Path, corpus: dict[str, dict] | None = None) -> int:
    """
    Importa las columnas de revision del CSV al JSON, por audit_id. Valida todo antes de
    escribir: si hay errores no se guarda nada. Reglas:
      - ids duplicados en el CSV -> rechazo
      - la huella de evidencia del CSV debe coincidir con la del JSON (CSV desactualizado -> rechazo)
      - un campo VACIO en el CSV nunca borra una decision ya guardada en el JSON; para limpiar
        una celda se escribe <borrar>; al CAMBIAR la decision, lo vacio se limpia (revision nueva)
      - cada decision se valida con `validate_review` (esquema v2)
    Cada cambio queda en `history`. Devuelve cuantos registros se actualizaron.
    """
    records = load_records(json_path)
    by_id = {r["audit_id"]: r for r in records}
    rows = _read_csv_rows(csv_path)
    if corpus is None:
        corpus = corpus_by_id(settings.faiss_index_dir.parent / "staging_v2")

    errors: list[str] = []
    seen: set[str] = set()
    updates: dict[str, dict] = {}
    record_cleared: dict[str, list[str]] = {}
    for row in rows:
        audit_id = row.get("audit_id", "")
        if audit_id in seen:
            errors.append(f"{audit_id}: aparece duplicado en el CSV")
            continue
        seen.add(audit_id)
        record = by_id.get(audit_id)
        if record is None:
            errors.append(f"{audit_id}: no existe en el JSON")
            continue
        if (row.get("evidence_hash") or "") != (record.get("evidence_hash") or ""):
            errors.append(f"{audit_id}: el CSV esta desactualizado respecto a la evidencia (huella distinta); regenera el CSV")
            continue

        stored = {f: str(record.get(f, "")).strip() for f in REVIEW_FIELDS}
        csv_decision = (row.get("decision") or "").strip().lower()
        # Cambiar de decision = nueva revision completa: lo vacio NO se hereda de la anterior.
        decision_changes = bool(csv_decision) and csv_decision != stored["decision"].lower()

        merged = {}
        cleared = []
        for field in REVIEW_FIELDS:
            raw = (row.get(field) or "").strip()  # sin normalizar: un valor v1 se rechaza, no se convierte
            if raw == CLEAR_TOKEN:
                merged[field] = ""
                cleared.append(field)
            elif raw:
                merged[field] = raw
            elif decision_changes and field != "decision":
                merged[field] = ""
                if stored[field]:
                    cleared.append(field)
            else:
                merged[field] = stored[field]  # vacio no borra
        merged["decision"] = merged["decision"].lower()
        merged["review_level"] = merged["review_level"].lower()

        stored_level = stored["review_level"].lower()
        promoting = (
            bool(stored["decision"])
            and merged["review_level"] in HUMAN_LEVELS
            and stored_level not in HUMAN_LEVELS
        )
        raw_reviewer = (row.get("reviewer") or "").strip()
        if stored["decision"] and stored_level == "preaudit" and (promoting or decision_changes):
            # Promover o cambiar una PROPUESTA preaudit: el revisor debe escribirse de forma
            # explicita y ser distinto del de la propuesta; la atribucion nunca se hereda.
            if not raw_reviewer or raw_reviewer == CLEAR_TOKEN or raw_reviewer.lower() == stored["reviewer"].lower():
                errors.append(
                    f"{audit_id}: la decision anterior es una propuesta preaudit de '{stored['reviewer']}'. Para "
                    "confirmarla, promoverla o cambiarla escribe TU nombre en reviewer (distinto del de la propuesta): "
                    "la atribucion no se hereda"
                )
        elif decision_changes and stored["decision"]:
            # Cambiar una decision humana: la terna reviewer/review_level/reviewed_at debe renovarse.
            identity = ("reviewer", "review_level", "reviewed_at")
            if all(merged[f] == stored[f] for f in identity):
                errors.append(
                    f"{audit_id}: cambiaste la decision de '{stored['decision']}' a '{merged['decision']}' pero "
                    "reviewer / review_level / reviewed_at son los de la decision anterior: registra los tuyos"
                )
        record_cleared[audit_id] = cleared
        errors.extend(validate_review(record, merged, corpus))
        updates[audit_id] = merged
    if errors:
        raise ValueError("Importacion rechazada, no se guardo nada:\n  " + "\n  ".join(errors))

    changed = 0
    for audit_id, merged in updates.items():
        record = by_id[audit_id]
        before = {f: str(record.get(f, "")).strip() for f in REVIEW_FIELDS}
        if before != merged:
            event = {"event": "import", "at": _now(), "previous": before}
            if record_cleared.get(audit_id):
                event["cleared_fields"] = record_cleared[audit_id]
            if before["decision"] and merged["decision"] != before["decision"]:
                event["decision_changed"] = [before["decision"], merged["decision"]]
            record["history"].append(event)
            record.update(merged)
            if record.get("decision"):
                record["stale_review"] = None  # solo una decision nueva y valida cierra el historial obsoleto
            changed += 1
    _backup(json_path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera o actualiza la hoja de auditoria del gold set v2.")
    parser.add_argument("--staging", type=Path, default=settings.faiss_index_dir.parent / "staging_v2")
    parser.add_argument("--out-dir", type=Path, default=settings.gold_set_path.parent)
    parser.add_argument(
        "--import-csv", type=Path, default=None,
        help="Importa las decisiones del CSV al JSON (fuente de verdad) y termina.",
    )
    parser.add_argument(
        "--migrate", action="store_true",
        help="Migra el JSON al esquema v2 (las decisiones previas quedan como propuestas preaudit) y termina.",
    )
    args = parser.parse_args()
    json_path = args.out_dir / "gold_v2_audit.json"

    if args.migrate:
        print(f"Migracion a esquema v{SCHEMA_VERSION}: {migrate_file(json_path)} registros actualizados")
        return

    if args.import_csv:
        changed = import_csv(args.import_csv, json_path, corpus_by_id(args.staging))
        print(f"Importacion OK: {changed} registros actualizados en gold_v2_audit.json")
        return

    records = build_audit(args.staging)
    json_path, csv_path, kept = write_outputs(records, args.out_dir)

    print(f"Registros: {len(records)}  ->  {json_path.name}, {csv_path.name}  (decisiones conservadas: {kept})")
    print("Por estado:", dict(Counter(r["status"] for r in records)))
    print("P1 obligatoria (cambio de modulo, excepciones, contradicciones, general):", sum(1 for r in records if r["priority"] == 1))
    print("P2 recomendada (solapamiento bajo / numerales ausentes / ambiguo):", sum(1 for r in records if r["priority"] == 2))
    print("P3 sin banderas (revision de muestra):", sum(1 for r in records if r["priority"] == 3))
    stale = sum(1 for r in records if r.get("stale_review"))
    if stale:
        print(f"ATENCION: {stale} decisiones previas quedaron OBSOLETAS (la evidencia cambio); ver stale_review.")
    levels = Counter(r["review_level"] or "(sin decision)" for r in records)
    print("Decisiones por nivel de revision:", dict(levels), "(solo author/expert cuentan para activar el gold v2)")
    print("Decisiones posibles:", ", ".join(DECISIONS))
    print("Revisado: edita el CSV y luego ejecuta --import-csv (el JSON es la fuente de verdad).")


if __name__ == "__main__":
    main()
