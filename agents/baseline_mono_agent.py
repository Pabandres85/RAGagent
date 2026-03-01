"""
agents/baseline_mono_agent.py — Agente mono para comparación (baseline).

Usa un único índice FAISS global (todos los módulos) sin ruteo.
Se compara contra el sistema multi-agente en la evaluación (Fase 4).
"""
from __future__ import annotations

import logging

from agents.guardrails import GuardrailsResult, validate_response
from agents.prompts import MONO_AGENT_SYSTEM_PROMPT, SPECIALIST_USER_TEMPLATE
from core.config import settings
from core.llm_client import chat_completion
from rag.citations import build_context
from rag.reranker import Reranker
from rag.retriever import Retriever

logger = logging.getLogger(__name__)


class MonoAgent:
    """
    Agente mono-agente baseline.
    Recupera del índice global (todos los módulos) sin especialización.
    """

    def __init__(self, use_reranker: bool = False):
        self._retriever = Retriever(module=None)  # índice global
        self._reranker = Reranker(use_cross_encoder=use_reranker)

    def answer(
        self,
        question: str,
        top_k: int = settings.faiss_top_k,
        max_context_chunks: int = 5,
    ) -> GuardrailsResult:
        logger.info("[MonoAgent] Pregunta: %s", question[:80])

        results = self._retriever.retrieve(question, top_k=top_k)
        if not results:
            return GuardrailsResult(
                valid=False,
                errors=["No se encontraron fragmentos para la consulta."],
            )

        results = self._reranker.rerank(question, results, top_k=max_context_chunks)
        context = build_context(results, max_chunks=max_context_chunks)

        messages = [
            {"role": "system", "content": MONO_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SPECIALIST_USER_TEMPLATE.format(
                    context=context, question=question
                ),
            },
        ]
        raw = chat_completion(messages)
        result = validate_response(raw, expected_module="global")

        if not result.valid:
            logger.warning("[MonoAgent] Guardrail FAIL: %s", result.errors)

        return result
