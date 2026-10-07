"""
core/llm_client.py — Fábrica del cliente LLM.

Retorna un cliente OpenAI-compatible para LM Studio u Ollama.
El resto del proyecto solo llama a chat_completion() sin saber qué backend usa.
"""
from __future__ import annotations

import logging
import time
from typing import Generator

from openai import APIConnectionError, APIStatusError, OpenAI

from core.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

_MAX_RETRIES = 4


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    text = str(exc).lower()
    return any(k in text for k in ("unloaded", "loading", "not loaded", "503", "overloaded"))


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
    response = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=settings.get_llm_model(),
                messages=messages,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
                stream=stream,
            )
            break
        except (APIConnectionError, APIStatusError) as exc:
            # LM Studio puede descargar el modelo por inactividad ("Model unloaded")
            # y recargarlo en la siguiente peticion: reintentar con espera.
            if attempt == _MAX_RETRIES or not _is_transient(exc):
                raise
            wait = 5 * attempt
            logger.warning("LLM error transitorio (%s). Reintento %d/%d en %ds.", exc, attempt, _MAX_RETRIES, wait)
            time.sleep(wait)
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
