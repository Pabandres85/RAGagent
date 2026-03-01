"""
core/llm_client.py — Fábrica del cliente LLM.

Retorna un cliente OpenAI-compatible para LM Studio u Ollama.
El resto del proyecto solo llama a chat_completion() sin saber qué backend usa.
"""
from __future__ import annotations

import logging
from typing import Generator

from openai import OpenAI

from core.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    """Retorna el cliente OpenAI-compatible (singleton) para el proveedor activo."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=settings.get_llm_base_url(),
            api_key=settings.get_llm_api_key(),
        )
        logger.info(
            "LLM client | provider=%s  model=%s  url=%s",
            settings.llm_provider,
            settings.get_llm_model(),
            settings.get_llm_base_url(),
        )
    return _client


def chat_completion(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
) -> str | Generator:
    """
    Wrapper de chat completions.

    Args:
        messages:    Lista [{"role": ..., "content": ...}]
        temperature: Sobreescribe settings.llm_temperature si se pasa.
        max_tokens:  Sobreescribe settings.llm_max_tokens si se pasa.
        stream:      Si True retorna el generador (el llamador lo maneja).

    Returns:
        Contenido de texto de la respuesta, o generador si stream=True.
    """
    client = get_llm_client()
    response = client.chat.completions.create(
        model=settings.get_llm_model(),
        messages=messages,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
        stream=stream,
    )
    if stream:
        return response
    content = response.choices[0].message.content
    logger.debug("LLM resp | chars=%d", len(content or ""))
    return content


def ping_llm() -> bool:
    """Verifica que el servidor LLM responde. Usado en health check de la API."""
    try:
        client = get_llm_client()
        client.models.list()
        return True
    except Exception as exc:
        logger.error("LLM ping FAILED | %s", exc)
        return False
