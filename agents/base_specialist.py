"""
agents/base_specialist.py — Clase base para los 7 agentes especialistas.

Cada especialista hereda de BaseSpecialist e implementa solo MODULE y
MODULE_NAME. Toda la lógica RAG (recuperación, re-ranking, prompt,
validación de guardrails) vive aquí.
"""
from __future__ import annotations

import logging
from typing import Optional

from core.llm_client import chat_completion
from core.config import settings
from agents.guardrails import AgentResponse, GuardrailsResult, validate_response
from agents.prompts import SPECIALIST_SYSTEM_TEMPLATE, SPECIALIST_USER_TEMPLATE
from rag.citations import build_context
from rag.reranker import Reranker
from rag.retriever import Retriever

logger = logging.getLogger(__name__)


class BaseSpecialist:
    """
    Agente especialista RAG para un módulo de la Resolución 3100.

    Subclases definen:
        MODULE      (str): clave del módulo, ej. "talento_humano"
        MODULE_NAME (str): nombre legible, ej. "Talento Humano"
    """

    MODULE: str = ""
    MODULE_NAME: str = ""

    def __init__(self, use_reranker: bool = False):
        if not self.MODULE:
            raise ValueError("La subclase debe definir MODULE.")

        self._retriever = Retriever(module=self.MODULE)
        self._reranker = Reranker(use_cross_encoder=use_reranker)
        self._system_prompt = SPECIALIST_SYSTEM_TEMPLATE.format(
            module_name=self.MODULE_NAME,
            module_key=self.MODULE,
        )

    def answer(
        self,
        question: str,
        top_k: int = settings.faiss_top_k,
        max_context_chunks: int = 5,
    ) -> GuardrailsResult:
        """
        Responde una pregunta normativa sobre el módulo del agente.

        Pipeline:
          1. Recuperación semántica (FAISS)
          2. Re-ranking (opcional)
          3. Construcción del contexto citado
          4. Generación LLM
          5. Validación guardrails

        Returns:
            GuardrailsResult con valid=True y response si todo OK.
        """
        logger.info("[%s] Pregunta: %s", self.MODULE, question[:80])

        # ── 1. Recuperar ─────────────────────────────────────────────────────
        results = self._retriever.retrieve(question, top_k=top_k)
        if not results:
            return GuardrailsResult(
                valid=False,
                errors=[f"No se encontraron fragmentos para la consulta en módulo '{self.MODULE}'."],
            )

        # ── 2. Re-rankear ─────────────────────────────────────────────────────
        results = self._reranker.rerank(question, results, top_k=max_context_chunks)

        # ── 3. Contexto ───────────────────────────────────────────────────────
        context = build_context(results, max_chunks=max_context_chunks)

        # ── 4. Generar ────────────────────────────────────────────────────────
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": SPECIALIST_USER_TEMPLATE.format(
                    context=context, question=question
                ),
            },
        ]
        raw = chat_completion(messages)

        # ── 5. Validar ────────────────────────────────────────────────────────
        result = validate_response(raw, expected_module=self.MODULE)
        if not result.valid:
            logger.warning("[%s] Guardrail FAIL: %s", self.MODULE, result.errors)
        else:
            logger.info("[%s] Respuesta válida | citas=%d", self.MODULE, len(result.response.citations))

        return result
