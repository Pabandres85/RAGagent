"""
agents/orchestrator.py — Orquestador central del sistema multi-agente.

Determina qué agente especialista debe responder una consulta usando
similitud coseno entre el embedding de la consulta y los embeddings
precomputados de las descripciones de cada módulo.
No requiere llamada al LLM para el ruteo — es rápido y determinista.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from agents.guardrails import GuardrailsResult
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


@dataclass
class RoutingDecision:
    module: str
    confidence: float
    reasoning: str


class Orchestrator:
    """
    Orquestador central del sistema multi-agente RAG.

    Al inicializarse, precomputa los embeddings de las descripciones
    de cada módulo para usar como vectores de referencia para el ruteo.
    """

    def __init__(self, use_reranker: bool = False):
        # Instanciar los 7 agentes especialistas
        self._agents = {
            "talento_humano": TalentoHumanoAgent(use_reranker),
            "infraestructura": InfraestructuraAgent(use_reranker),
            "dotacion": DotacionAgent(use_reranker),
            "medicamentos_dispositivos": MedicamentosDispositivosAgent(use_reranker),
            "procesos_prioritarios": ProcesosPrioritariosAgent(use_reranker),
            "historia_clinica": HistoriaClinicaAgent(use_reranker),
            "interdependencia": InterdependenciaAgent(use_reranker),
        }

        # Precomputar embeddings de descripciones de módulos para ruteo
        logger.info("Precomputando embeddings de módulos para ruteo...")
        descriptions = [MODULE_DESCRIPTIONS[m] for m in MODULES]
        self._module_embeddings = embed_texts(descriptions)  # shape (7, dim)
        logger.info("Orchestrator listo.")

    def route(self, question: str) -> RoutingDecision:
        """
        Determina el módulo más relevante para la pregunta.

        Usa similitud coseno entre el embedding de la consulta
        y los embeddings de las descripciones de módulos.
        """
        query_vec = embed_query(question)  # shape (1, dim)
        scores = (self._module_embeddings @ query_vec.T).flatten()  # shape (7,)
        best_idx = int(np.argmax(scores))
        best_module = MODULES[best_idx]
        confidence = float(scores[best_idx])

        logger.info(
            "Ruteo | módulo=%s | confianza=%.3f | pregunta='%s'",
            best_module, confidence, question[:60],
        )
        return RoutingDecision(
            module=best_module,
            confidence=confidence,
            reasoning=f"Similitud coseno más alta con módulo '{best_module}' ({confidence:.3f})",
        )

    def answer(self, question: str) -> Dict:
        """
        Procesa la pregunta de principio a fin:
          1. Ruteo al agente especialista correcto.
          2. Delegación de la respuesta al agente.
          3. Retorno del resultado con metadatos de ruteo.
        """
        routing = self.route(question)
        agent = self._agents[routing.module]
        result: GuardrailsResult = agent.answer(question)

        return {
            "routing": {
                "module": routing.module,
                "confidence": routing.confidence,
                "reasoning": routing.reasoning,
            },
            "valid": result.valid,
            "response": result.response.model_dump() if result.response else None,
            "errors": result.errors,
            "warnings": result.warnings,
        }
