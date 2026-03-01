"""
Orquestador central del sistema multi-agente.

Mejoras aplicadas:
1. Ruteo hibrido: combina similitud semantica con senales lexicas fuertes.
2. Soporte transversal: cuando la consulta toca varios modulos cercanos,
   consulta a 2 o 3 especialistas y sintetiza una respuesta unica.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from agents.guardrails import AgentResponse, ChecklistItem, Citation, GuardrailsResult
from agents.prompts import MODULE_DESCRIPTIONS
from agents.specialists import (
    DotacionAgent,
    HistoriaClinicaAgent,
    InfraestructuraAgent,
    InterdependenciaAgent,
    MedicamentosDispositivosAgent,
    ProcesosPrioritariosAgent,
    TalentoHumanoAgent,
)
from core.embeddings import embed_query, embed_texts
from core.metadata_store import MODULES

logger = logging.getLogger(__name__)

ROUTING_KEYWORDS = {
    "talento_humano": [
        "talento humano",
        "personal",
        "profesional",
        "profesionales",
        "titulo",
        "titulos",
        "certificacion",
        "certificaciones",
        "rethus",
        "enfermeria",
        "medico",
        "medicos",
    ],
    "infraestructura": [
        "infraestructura",
        "area",
        "areas",
        "espacio",
        "espacios",
        "instalacion",
        "instalaciones",
        "ambiente",
        "ambientes",
        "metros",
        "m2",
        "pared",
        "paredes",
        "techo",
        "pisos",
        "ventilacion",
        "iluminacion",
        "quirurgico",
        "quirofano",
        "sala",
        "salas",
    ],
    "dotacion": [
        "dotacion",
        "equipo",
        "equipos",
        "equipo biomedico",
        "equipos biomedicos",
        "mobiliario",
        "instrumental",
        "camilla",
        "monitor",
        "desfibrilador",
        "carro de paro",
        "metrologia",
        "calibracion",
        "mantenimiento preventivo",
        "mantenimiento correctivo",
    ],
    "medicamentos_dispositivos": [
        "medicamento",
        "medicamentos",
        "farmaco",
        "farmacos",
        "farmacia",
        "farmaceutico",
        "farmaceutica",
        "dispensacion",
        "insumo",
        "insumos",
        "reactivo",
        "reactivos",
        "cadena de frio",
        "hemoderivados",
        "hemocomponente",
        "hemocomponentes",
        "sangre",
        "implante",
        "implantes",
        "dispositivo medico",
        "dispositivos medicos",
        "tecnovigilancia",
        "farmacovigilancia",
        "estupefacientes",
        "biologico",
        "biologicos",
        "vencimiento",
        "botiquin",
    ],
    "procesos_prioritarios": [
        "triage",
        "urgencias",
        "protocolo",
        "protocolos",
        "sepsis",
        "parto",
        "recien nacido",
        "transfusion",
        "referencia",
        "contra referencia",
        "proceso",
        "procesos",
    ],
    "historia_clinica": [
        "historia clinica",
        "registro",
        "registros",
        "evolucion",
        "consentimiento",
        "custodia",
        "epicrisis",
        "anotacion",
        "anotaciones",
    ],
    "interdependencia": [
        "interdependencia",
        "apoyo diagnostico",
        "servicio de apoyo",
        "servicios de apoyo",
        "referido a otro servicio",
        "complementario",
        "complementarios",
    ],
}

INTEGRAL_QUERY_PATTERNS = [
    "como habilito",
    "como habilitar",
    "como se habilita",
    "como abrir",
    "que debo tener",
    "que debo considerar",
    "que necesito",
    "que se requiere",
    "que requisitos debo cumplir",
    "tener en cuenta",
    "habilitar",
    "habilitacion",
    "abrir",
    "montar",
]

SERVICE_ANCHORS = [
    "servicio",
    "consultorio",
    "consulta externa",
    "cirugia general",
    "cirugia",
    "odontologico",
    "odontologia",
    "urgencias",
    "uci",
    "quirurgico",
    "hospitalizacion",
]


@dataclass
class RoutingDecision:
    module: str
    confidence: float
    reasoning: str
    modules: list[str]
    scores: dict[str, float]
    is_transversal: bool = False


class Orchestrator:
    """
    Orquestador central del sistema multi-agente RAG.

    Usa un ruteo hibrido:
    - embedding similarity entre consulta y descripciones de modulos
    - bonus por coincidencias lexicas de alta senal

    Si detecta transversalidad, consulta a varios especialistas y sintetiza
    la salida final sin perder trazabilidad.
    """

    def __init__(self, use_reranker: bool = False):
        self._agents = {
            "talento_humano": TalentoHumanoAgent(use_reranker),
            "infraestructura": InfraestructuraAgent(use_reranker),
            "dotacion": DotacionAgent(use_reranker),
            "medicamentos_dispositivos": MedicamentosDispositivosAgent(use_reranker),
            "procesos_prioritarios": ProcesosPrioritariosAgent(use_reranker),
            "historia_clinica": HistoriaClinicaAgent(use_reranker),
            "interdependencia": InterdependenciaAgent(use_reranker),
        }

        logger.info("Precomputando embeddings de modulos para ruteo...")
        descriptions = [MODULE_DESCRIPTIONS[module] for module in MODULES]
        self._module_embeddings = embed_texts(descriptions)
        logger.info("Orchestrator listo.")

    def _lexical_bonus(self, question: str) -> dict[str, float]:
        text = question.lower()
        bonus = {module: 0.0 for module in MODULES}

        for module, keywords in ROUTING_KEYWORDS.items():
            hits = 0
            for keyword in keywords:
                if keyword in text:
                    hits += 1
            if hits == 0:
                continue

            # Bonus escalonado: una senal fuerte ya debe impactar;
            # multiples senales refuerzan pero sin dominar por completo.
            bonus[module] = min(0.24, 0.12 + (hits - 1) * 0.04)

        return bonus

    def _is_integral_service_query(self, question: str) -> bool:
        text = question.lower()
        has_integral_pattern = any(pattern in text for pattern in INTEGRAL_QUERY_PATTERNS)
        has_service_anchor = any(anchor in text for anchor in SERVICE_ANCHORS)
        return has_integral_pattern and has_service_anchor

    def _integral_default_modules(self, question: str) -> list[str]:
        text = question.lower()
        selected = ["infraestructura", "dotacion", "talento_humano"]

        if any(term in text for term in ["urgencias", "triage", "parto", "uci"]):
            selected.append("procesos_prioritarios")

        if any(term in text for term in ["medicamento", "farmacia", "estupefaciente"]):
            selected.append("medicamentos_dispositivos")

        if any(term in text for term in ["historia clinica", "consentimiento", "registro"]):
            selected.append("historia_clinica")

        # Preservar orden y evitar duplicados.
        deduped: list[str] = []
        for module in selected:
            if module not in deduped:
                deduped.append(module)
        return deduped

    def route(self, question: str) -> RoutingDecision:
        query_vec = embed_query(question)
        semantic_scores = (self._module_embeddings @ query_vec.T).flatten()
        lexical_bonus = self._lexical_bonus(question)

        blended_scores = []
        for index, module in enumerate(MODULES):
            blended = float(semantic_scores[index]) + lexical_bonus[module]
            blended_scores.append((module, blended))

        ranked = sorted(blended_scores, key=lambda item: item[1], reverse=True)
        best_module, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0

        selected_modules = [best_module]
        is_transversal = False

        if self._is_integral_service_query(question):
            selected_modules = self._integral_default_modules(question)
            if best_module not in selected_modules:
                selected_modules.append(best_module)

            # Sumar un modulo adicional si hay una senal fuerte fuera del set base.
            for module, score in ranked[1:4]:
                if module not in selected_modules and (best_score - score) <= 0.10:
                    selected_modules.append(module)
                    break

            is_transversal = len(selected_modules) > 1

        # Consulta transversal si:
        # - hay "servicio" o "requisitos" amplios, y
        # - hay modulos cercanos en score.
        lower_question = question.lower()
        broad_query = ("servicio" in lower_question) or ("requisito" in lower_question)
        close_threshold = 0.08

        if (not is_transversal) and broad_query and (best_score - second_score) <= close_threshold:
            cutoff = best_score - close_threshold
            selected_modules = [
                module
                for module, score in ranked[:3]
                if score >= cutoff
            ]
            is_transversal = len(selected_modules) > 1

        logger.info(
            "Ruteo | modulo=%s | score=%.3f | transversal=%s | seleccion=%s | pregunta='%s'",
            best_module,
            best_score,
            is_transversal,
            selected_modules,
            question[:80],
        )

        if is_transversal:
            if self._is_integral_service_query(question):
                reasoning = (
                    "Consulta integral por servicio detectada. Se priorizan estandares base "
                    f"para habilitacion: {', '.join(selected_modules)}."
                )
            else:
                reasoning = (
                    "Consulta transversal detectada. Se seleccionan modulos cercanos por "
                    f"score hibrido: {', '.join(selected_modules)}."
                )
        else:
            reasoning = (
                f"Score hibrido mas alto en modulo '{best_module}' "
                f"({best_score:.3f})."
            )

        return RoutingDecision(
            module=best_module,
            confidence=best_score,
            reasoning=reasoning,
            modules=selected_modules,
            scores={module: score for module, score in ranked},
            is_transversal=is_transversal,
        )

    def _merge_results(
        self,
        question: str,
        routing: RoutingDecision,
        module_results: list[tuple[str, GuardrailsResult]],
    ) -> GuardrailsResult:
        valid_results = [
            (module, result)
            for module, result in module_results
            if result.valid and result.response is not None and not result.no_evidence
        ]
        no_evidence_modules = [
            module
            for module, result in module_results
            if result.response is not None and result.no_evidence
        ]

        warnings: list[str] = []
        errors: list[str] = []

        for module, result in module_results:
            for warning in result.warnings:
                if not warning.startswith("Confianza baja:") and not warning.startswith("Sin evidencia recuperada"):
                    warnings.append(f"[{module}] {warning}")
            for error in result.errors:
                errors.append(f"[{module}] {error}")

        if no_evidence_modules:
            warnings.append(
                "Sin evidencia directa en: " + ", ".join(no_evidence_modules) + "."
            )

        if not valid_results:
            if no_evidence_modules:
                fallback_response = AgentResponse(
                    answer=(
                        "No se encontro evidencia normativa suficiente en los modulos consultados "
                        "para construir una respuesta fundamentada. Reformula la pregunta de forma "
                        "mas especifica o consulta un requisito puntual por estandar."
                    ),
                    citations=[],
                    checklist=[],
                    module="transversal" if routing.is_transversal else routing.module,
                    confidence=routing.confidence,
                )
                return GuardrailsResult(
                    valid=False,
                    response=fallback_response,
                    errors=errors,
                    warnings=warnings,
                )

            return GuardrailsResult(
                valid=False,
                errors=errors or ["Ningun especialista produjo una respuesta valida."],
                warnings=warnings,
            )

        if len(valid_results) == 1:
            module, result = valid_results[0]
            warnings.extend(
                [
                    f"Consulta transversal parcial: solo hubo respuesta util desde '{module}'."
                ]
                if routing.is_transversal
                else []
            )
            return GuardrailsResult(
                valid=result.valid,
                response=result.response,
                errors=errors,
                warnings=warnings,
            )

        answer_sections: list[str] = []
        citations: list[Citation] = []
        checklist: list[ChecklistItem] = []
        seen_citations: set[tuple[str, str, str | None, int | None]] = set()
        seen_checklist: set[tuple[str, str | None, str]] = set()

        for module, result in valid_results:
            response = result.response
            assert response is not None

            answer_sections.append(
                f"[{module}] {response.answer}"
            )

            for citation in response.citations:
                key = (module, citation.text, citation.numeral, citation.page)
                if key in seen_citations:
                    continue
                seen_citations.add(key)
                citations.append(
                    Citation(
                        text=citation.text,
                        numeral=citation.numeral,
                        page=citation.page,
                        resolution=citation.resolution,
                        vigencia=citation.vigencia,
                    )
                )

            for item in response.checklist:
                checklist_text = f"[{module}] {item.item}"
                key = (checklist_text, item.numeral, item.status)
                if key in seen_checklist:
                    continue
                seen_checklist.add(key)
                checklist.append(
                    ChecklistItem(
                        item=checklist_text,
                        numeral=item.numeral,
                        status=item.status,
                    )
                )

        merged_response = AgentResponse(
            answer="\n\n".join(answer_sections),
            citations=citations,
            checklist=checklist,
            module="transversal",
            confidence=routing.confidence,
        )

        return GuardrailsResult(
            valid=True,
            response=merged_response,
            errors=errors,
            warnings=warnings,
        )

    def answer(self, question: str) -> dict:
        t_start = time.time()
        routing = self.route(question)
        t_routed = time.time()

        module_results: list[tuple[str, GuardrailsResult]] = []
        for module in routing.modules:
            agent = self._agents[module]
            result = agent.answer(question)
            module_results.append((module, result))
        t_agents = time.time()

        final_result = self._merge_results(question, routing, module_results)
        t_end = time.time()

        return {
            "routing": {
                "module": routing.module,
                "modules": routing.modules,
                "confidence": routing.confidence,
                "reasoning": routing.reasoning,
                "is_transversal": routing.is_transversal,
                "scores": routing.scores,
            },
            "valid": final_result.valid,
            "response": final_result.response.model_dump() if final_result.response else None,
            "errors": final_result.errors,
            "warnings": final_result.warnings,
            "timings": {
                "routing_ms": round((t_routed - t_start) * 1000),
                "agents_ms": round((t_agents - t_routed) * 1000),
                "total_ms": round((t_end - t_start) * 1000),
            },
        }
