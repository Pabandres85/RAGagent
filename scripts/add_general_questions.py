"""
scripts/add_general_questions.py — Agrega preguntas generales/admin al gold set.

Estrategia:
  1. Define ~41 preguntas pre-elaboradas clasificadas por módulo.
  2. Para cada pregunta busca en el índice global los fragmentos más relevantes.
  3. Llama al LLM para generar una respuesta de referencia basada en esos fragmentos.
  4. Agrega al gold set con module="general" (admin/transversal) o el módulo real
     para preguntas de definición por estándar y talento humano específico.

Nota: las preguntas generales/admin corresponden a capítulos 1-10 de la resolución,
que pueden no estar completamente indexados. El LLM indicará "La información no se
encuentra en los fragmentos recuperados" cuando sea el caso — eso es también
un resultado válido que documenta la cobertura del sistema.

Las respuestas se pueden corregir manualmente en ui/pages/3_Auditar_Goldset.py

Uso:
    python scripts/add_general_questions.py
    python scripts/add_general_questions.py --dry-run    # muestra preguntas sin llamar LLM
    python scripts/add_general_questions.py --force      # regenera aunque ya existan
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from core.llm_client import chat_completion
from rag.citations import build_context
from rag.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Prompt para generar respuesta de referencia desde pregunta + contexto ─────
REF_ANSWER_SYSTEM_PROMPT = (
    "Eres un experto en habilitación de servicios de salud en Colombia "
    "(Resolución 3100 de 2019). Tu tarea es generar una RESPUESTA DE REFERENCIA "
    "para una pregunta de evaluación de un sistema RAG normativo.\n\n"
    "REGLAS:\n"
    "- Basa tu respuesta ÚNICAMENTE en los fragmentos normativos proporcionados.\n"
    "- La respuesta debe ser concisa y directa (2-6 oraciones).\n"
    "- Cita el numeral o página cuando esté disponible en los fragmentos.\n"
    "- Si los fragmentos NO contienen información suficiente para responder, "
    "indica: 'La información solicitada no se encuentra en los fragmentos recuperados.'\n"
    "- No uses conocimiento previo fuera de los fragmentos.\n\n"
    "Responde ÚNICAMENTE con este JSON (sin texto adicional):\n"
    '{"answer": "..."}'
)

REF_ANSWER_USER_TEMPLATE = (
    "PREGUNTA: {question}\n\n"
    "FRAGMENTOS RECUPERADOS:\n{context}\n\n"
    "Genera la respuesta de referencia."
)

# ── Preguntas a agregar ───────────────────────────────────────────────────────
# Formato: (pregunta, módulo)
# módulo = "general"            → pregunta admin/transversal (excluida del routing accuracy)
# módulo = nombre de especialista → pregunta de definición o contenido específico

QUESTIONS: list[tuple[str, str]] = [
    # ── Condiciones técnico-administrativas ──────────────────────────────────
    (
        "¿Cuáles son los requerimientos de las condiciones técnico administrativas "
        "que debo cumplir para habilitar una IPS?",
        "general",
    ),
    (
        "¿Cuáles son los requerimientos de las condiciones técnico administrativas "
        "que debo cumplir para habilitar un profesional independiente?",
        "general",
    ),
    # ── Modalidades, estándares y grupos ─────────────────────────────────────
    (
        "¿Cuántas y cuáles son las modalidades en las que es posible habilitar "
        "un servicio de salud?",
        "general",
    ),
    (
        "¿Cuántos y cuáles son los estándares que se deben validar para aperturar "
        "un servicio de salud?",
        "general",
    ),
    (
        "¿Cuántos y cuáles son los grupos en los que se puede habilitar "
        "un servicio de salud?",
        "general",
    ),
    # ── Definiciones generales ───────────────────────────────────────────────
    (
        "¿Cuáles son las definiciones generales para tener en cuenta en la "
        "habilitación de un servicio de salud?",
        "general",
    ),
    # ── Definiciones por estándar (asignadas al módulo correspondiente) ───────
    (
        "¿Cuáles son las definiciones a tener en cuenta para evaluar "
        "el estándar de talento humano?",
        "talento_humano",
    ),
    (
        "¿Cuáles son las definiciones a tener en cuenta para evaluar "
        "el estándar de infraestructura?",
        "infraestructura",
    ),
    (
        "¿Cuáles son las definiciones a tener en cuenta para evaluar "
        "el estándar de dotación?",
        "dotacion",
    ),
    (
        "¿Cuáles son las definiciones a tener en cuenta para evaluar "
        "el estándar de medicamentos y dispositivos médicos?",
        "medicamentos_dispositivos",
    ),
    (
        "¿Cuáles son las definiciones a tener en cuenta para evaluar "
        "el estándar de procesos prioritarios?",
        "procesos_prioritarios",
    ),
    (
        "¿Cuáles son las definiciones a tener en cuenta para evaluar "
        "el estándar de historia clínica y registros?",
        "historia_clinica",
    ),
    (
        "¿Cuáles son las definiciones a tener en cuenta para evaluar "
        "el estándar de interdependencia de servicios?",
        "interdependencia",
    ),
    # ── Condiciones especiales ────────────────────────────────────────────────
    (
        "¿Cuáles son las condiciones para la prestación de servicios de salud "
        "con apoyo de organismos de cooperación?",
        "general",
    ),
    (
        "¿Cuáles son las condiciones para la prestación de servicios de salud "
        "en situaciones de emergencia?",
        "general",
    ),
    # ── Tipos de prestadores ─────────────────────────────────────────────────
    (
        "¿Cuáles son los prestadores de servicios de salud que se pueden "
        "habilitar bajo la Resolución 3100 de 2019?",
        "general",
    ),
    (
        "¿Qué es una IPS según la Resolución 3100 de 2019?",
        "general",
    ),
    (
        "¿Qué es un profesional independiente de salud según "
        "la Resolución 3100 de 2019?",
        "general",
    ),
    (
        "¿Qué es una entidad con objeto social diferente según "
        "la Resolución 3100 de 2019?",
        "general",
    ),
    (
        "¿Qué es el transporte especial de pacientes según "
        "la Resolución 3100 de 2019?",
        "general",
    ),
    # ── Condiciones y clasificación ───────────────────────────────────────────
    (
        "¿Cuáles son las condiciones generales para habilitar un servicio de salud?",
        "general",
    ),
    (
        "¿De acuerdo a su naturaleza jurídica, cómo se clasifican los prestadores "
        "de servicios de salud?",
        "general",
    ),
    (
        "¿Qué se debe tener en cuenta en la capacidad relacionada con la "
        "suficiencia patrimonial de un prestador de servicios de salud?",
        "general",
    ),
    # ── Pasos / procedimientos ────────────────────────────────────────────────
    (
        "¿Cuáles son los pasos para la inscripción de los prestadores de "
        "servicios de salud en el REPS?",
        "general",
    ),
    (
        "¿Cuáles son los pasos para la habilitación de servicios de salud?",
        "general",
    ),
    # ── Distintivos ──────────────────────────────────────────────────────────
    (
        "¿Qué son los distintivos de habilitación y cuáles son las obligaciones "
        "en materia de distintivos de habilitación?",
        "general",
    ),
    # ── Visitas de habilitación ───────────────────────────────────────────────
    (
        "¿Cuáles son las generalidades de las visitas de habilitación?",
        "general",
    ),
    (
        "¿Cuáles son los casos en los que se realiza visita previa de habilitación?",
        "general",
    ),
    # ── Novedades REPS ────────────────────────────────────────────────────────
    (
        "¿Cuáles son los requisitos para presentar novedades relacionadas con "
        "cierre del prestador de servicios de salud?",
        "general",
    ),
    (
        "¿Cuáles son los requisitos para presentar novedades relacionadas con "
        "la disolución y liquidación de las entidades de salud?",
        "general",
    ),
    (
        "¿Cuáles son los requisitos para presentar novedades relacionadas con "
        "cambio de domicilio o dirección de una IPS o servicio de salud?",
        "general",
    ),
    (
        "¿Cuáles son los requisitos para presentar novedades relacionadas con "
        "un cambio de nomenclatura?",
        "general",
    ),
    (
        "¿Cuáles son los requisitos para presentar novedades relacionadas con "
        "el cambio de representante legal?",
        "general",
    ),
    (
        "¿Cuáles son los requisitos para presentar novedades relacionadas con "
        "el cambio de razón social?",
        "general",
    ),
    (
        "¿Cuáles son los requisitos para presentar novedades relacionadas con "
        "cambio de datos de contacto?",
        "general",
    ),
    (
        "¿Cuáles son los requisitos para presentar novedades relacionadas con "
        "apertura de nueva sede?",
        "general",
    ),
    (
        "¿Cuáles son los requisitos para presentar novedades relacionadas con "
        "cierre de sedes?",
        "general",
    ),
    # ── Talento humano específico ─────────────────────────────────────────────
    (
        "¿Cuáles son los requisitos mínimos para contratar talento humano en "
        "una IPS o servicio de salud?",
        "talento_humano",
    ),
    (
        "¿El recurso humano de qué servicios debe tener el certificado o acciones "
        "de formación relacionadas con gestión del duelo?",
        "talento_humano",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gold_set(path: Path) -> list[dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return []


def save_gold_set(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def generate_reference_answer(
    question: str,
    retriever: Retriever,
    top_k: int = 6,
) -> Optional[str]:
    """Recupera fragmentos del índice global y llama al LLM para generar la respuesta."""
    results = retriever.retrieve(question, top_k=top_k)
    if not results:
        logger.warning("Sin resultados FAISS para: %s", question[:60])
        return None

    context = build_context(results, max_chunks=top_k)

    messages = [
        {"role": "system", "content": REF_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REF_ANSWER_USER_TEMPLATE.format(
                question=question,
                context=context,
            ),
        },
    ]

    try:
        raw = chat_completion(messages, temperature=0.1, max_tokens=512)
    except Exception as exc:
        logger.error("Error LLM: %s", exc)
        return None

    # Extraer JSON de la respuesta
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        logger.warning("Respuesta sin JSON válido: %s", raw[:100])
        return None

    try:
        data = json.loads(raw[start : end + 1])
        answer = str(data.get("answer", "")).strip()
        return answer if answer else None
    except json.JSONDecodeError:
        logger.warning("JSON malformado: %s", raw[:100])
        return None


# ── Lógica principal ──────────────────────────────────────────────────────────

def add_general_questions(
    output_path: Path,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    existing = load_gold_set(output_path)
    existing_questions = {e["question"].strip().lower() for e in existing}

    pending = [
        (q, m) for q, m in QUESTIONS
        if q.strip().lower() not in existing_questions
    ]

    if not pending:
        logger.info("Todas las preguntas ya están en el gold set. Nada que agregar.")
        return

    if force:
        # Con --force, regenera todas aunque ya existan
        pending = list(QUESTIONS)
        logger.info("--force activo: regenerando todas las %d preguntas.", len(pending))
    else:
        logger.info(
            "Preguntas nuevas a agregar: %d de %d totales.",
            len(pending), len(QUESTIONS),
        )

    if dry_run:
        logger.info("-- DRY RUN (no se llama al LLM ni se guarda nada) --")
        for i, (q, m) in enumerate(pending, 1):
            print(f"[{i:02d}] module={m:30s} | {q[:80]}")
        return

    retriever = Retriever(module=None)  # índice global

    entries = list(existing) if not force else []
    new_count = 0

    for i, (question, module) in enumerate(pending, 1):
        logger.info(
            "[%d/%d] module=%-30s | %s",
            i, len(pending), module, question[:70],
        )

        answer = generate_reference_answer(question, retriever)

        if answer is None:
            answer = "[Respuesta no generada — revisar manualmente en la UI de auditoría]"
            logger.warning("  -> Usando placeholder para esta pregunta.")
        else:
            logger.info("  -> OK (%d chars)", len(answer))

        entry = {
            "question": question,
            "answer": answer,
            "module": module,
            "chunk_id": None,
            "numeral": None,
            "page": None,
            "source_file": "resolucion-3100-de-2019.pdf",
        }
        entries.append(entry)
        new_count += 1

        # Guardar incrementalmente cada 5 entradas
        if new_count % 5 == 0:
            save_gold_set(output_path, entries)
            logger.info("  Progreso guardado: %d entradas totales.", len(entries))

    save_gold_set(output_path, entries)
    logger.info("=" * 60)
    logger.info(
        "Completado: +%d nuevas preguntas agregadas. Total gold set: %d",
        new_count, len(entries),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Agrega preguntas generales/admin al gold set con respuestas "
            "generadas por el LLM (índice global)."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.gold_set_path,
        help="Ruta del gold set JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra las preguntas pendientes sin llamar al LLM ni guardar.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenera respuestas aunque las preguntas ya existan en el gold set.",
    )
    args = parser.parse_args()

    add_general_questions(
        output_path=args.output,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
