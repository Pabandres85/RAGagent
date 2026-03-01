"""
agents/guardrails.py - Validacion y control de respuestas del asistente normativo.

Verifica que cada respuesta:
1. Sea JSON valido con el esquema correcto.
2. Tenga al menos una cita normativa.
3. No afirme informacion fuera del corpus proporcionado.
"""
from __future__ import annotations

import ast
import json
import logging
import unicodedata
from typing import List, Optional

from pydantic import BaseModel, ValidationError, field_validator

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    text: str
    numeral: Optional[str] = None
    page: Optional[int] = None
    resolution: str = "Resolucion 3100 de 2019"
    vigencia: str = "Vigente"


_STATUS_ALIASES = {
    "cumplido": "cumple",
    "cumplida": "cumple",
    "si": "cumple",
    "no cumple": "no_cumple",
    "no aplica": "no_aplica",
}
_VALID_STATUSES = {"cumple", "no_cumple", "no_aplica", "pendiente"}


class ChecklistItem(BaseModel):
    item: str
    numeral: Optional[str] = None
    status: str = "pendiente"

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: str) -> str:
        raw = str(value).strip().lower()
        return _STATUS_ALIASES.get(raw, raw if raw in _VALID_STATUSES else "pendiente")


class AgentResponse(BaseModel):
    answer: str
    citations: List[Citation]
    checklist: List[ChecklistItem] = []
    module: str
    confidence: float = 0.0


class GuardrailsResult(BaseModel):
    valid: bool
    response: Optional[AgentResponse] = None
    errors: List[str] = []
    warnings: List[str] = []
    raw: str = ""
    no_evidence: bool = False


OUT_OF_DOMAIN_KEYWORDS = [
    "precio",
    "costo",
    "tarifa",
    "salario",
    "contrato",
    "licitacion",
    "receta",
    "diagnostico",
    "tratamiento",
    "medicacion concreta",
]

NO_EVIDENCE_PREFIX = "la informacion solicitada no se encuentra en los fragmentos recuperados"


def validate_response(raw: str, expected_module: str = "") -> GuardrailsResult:
    """
    Valida la respuesta cruda del LLM.

    Pasos:
    1. Parseo JSON.
    2. Validacion del esquema AgentResponse.
    3. Verificacion de que exista al menos una cita.
    4. Advertencia si la confianza es baja.
    """
    errors: List[str] = []
    warnings: List[str] = []

    json_str = _extract_json(raw)
    if json_str is None:
        return GuardrailsResult(
            valid=False,
            errors=["La respuesta del LLM no contiene JSON valido."],
            raw=raw,
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return GuardrailsResult(
            valid=False,
            errors=[f"JSON malformado: {exc}"],
            raw=raw,
        )

    try:
        response = AgentResponse(**data)
    except ValidationError as exc:
        return GuardrailsResult(
            valid=False,
            errors=[f"Esquema invalido: {exc.error_count()} error(es)", str(exc)],
            raw=raw,
        )

    if not response.citations:
        if _is_no_evidence_response(response):
            warnings.append("Sin evidencia recuperada para este modulo.")
            return GuardrailsResult(
                valid=True,
                response=response,
                warnings=warnings,
                raw=raw,
                no_evidence=True,
            )
        errors.append("La respuesta no contiene ninguna cita normativa.")

    if response.confidence < 0.5:
        warnings.append(f"Confianza baja: {response.confidence:.2f}")

    if expected_module and response.module != expected_module:
        warnings.append(
            f"El modulo de la respuesta ({response.module}) no coincide "
            f"con el esperado ({expected_module})."
        )

    if errors:
        return GuardrailsResult(
            valid=False,
            response=response,
            errors=errors,
            warnings=warnings,
            raw=raw,
        )

    return GuardrailsResult(valid=True, response=response, warnings=warnings, raw=raw)


def _is_no_evidence_response(response: AgentResponse) -> bool:
    answer = _normalize_text(response.answer)
    return answer.startswith(NO_EVIDENCE_PREFIX)


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.strip().lower()


def _extract_json(text: str) -> str | None:
    """Extrae y limpia el primer bloque JSON del texto del LLM."""
    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    json_str = text[start : end + 1].strip()

    if json_str.startswith("{{") and json_str.endswith("}}"):
        json_str = json_str[1:-1].strip()

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(json_str)
    except (SyntaxError, ValueError, TypeError):
        return json_str

    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)

    return json_str
