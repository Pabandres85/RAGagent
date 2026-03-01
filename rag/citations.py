"""
rag/citations.py - Formateo de citas normativas.

Genera citas trazables con numeral, pagina y vigencia para
incluir en respuestas del asistente normativo.
"""
from __future__ import annotations

from typing import List

from core.metadata_store import MODULE_DISPLAY_NAMES
from rag.retriever import RetrievalResult


def format_citation(result: RetrievalResult) -> str:
    """
    Formatea un resultado como cita normativa legible.

    Ejemplo: "Resolucion 3100 de 2019 | Estandar: Talento Humano | Numeral 1.1 | Pagina 45 | Vigente"
    """
    c = result.chunk
    parts = [c.resolution]

    if c.module:
        parts.append(f"Estandar: {MODULE_DISPLAY_NAMES.get(c.module, c.module)}")
    if c.service:
        parts.append(f"Servicio: {c.service}")
    if c.numeral:
        parts.append(f"Numeral {c.numeral}")
    if c.page:
        parts.append(f"Pagina {c.page}")
    parts.append(f"Vigencia: {c.vigencia}")

    return " | ".join(parts)


def build_context(results: List[RetrievalResult], max_chunks: int = 5) -> str:
    """
    Construye el bloque de contexto para el prompt del LLM.

    Cada fragmento va precedido de su cita para que el modelo
    pueda referenciarla en su respuesta.

    Returns:
        Texto formateado listo para insertar en el prompt del agente.
    """
    lines = []
    for i, result in enumerate(results[:max_chunks], start=1):
        citation = format_citation(result)
        lines.append(f"[FRAGMENTO {i}] Fuente: ({citation})")
        lines.append(result.chunk.text.strip())
        lines.append("")
    return "\n".join(lines)
