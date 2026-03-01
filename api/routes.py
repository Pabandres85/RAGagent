"""
Endpoints principales del asistente normativo.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.orchestrator import Orchestrator
from core.config import settings
from core.llm_client import ping_llm

logger = logging.getLogger(__name__)
router = APIRouter()

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        logger.info("Inicializando orchestrator...")
        _orchestrator = Orchestrator()
    return _orchestrator


class QueryRequest(BaseModel):
    question: str
    module: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_ok: bool


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        llm_ok=ping_llm(),
    )


@router.post("/query")
def query(request: QueryRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")

    try:
        result = get_orchestrator().answer(request.question)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Faltan indices FAISS. Ejecuta scripts/ingest.py primero. ({exc})",
        ) from exc
    except Exception as exc:
        logger.exception("Fallo procesando query")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result
