"""
Punto de entrada principal del sistema.
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sistema multi-agente RAG para la Resolucion 3100 de 2019."
    )
    parser.add_argument("--ingest", action="store_true", help="Ejecuta la ingesta del corpus.")
    parser.add_argument("--ping", action="store_true", help="Prueba conectividad con el LLM.")
    parser.add_argument("--host", default=None, help="Host para la API.")
    parser.add_argument("--port", type=int, default=None, help="Puerto para la API.")
    args = parser.parse_args()

    if args.ingest:
        from scripts.ingest import main as ingest_main

        ingest_main()
        return

    if args.ping:
        from core.config import settings
        from core.llm_client import ping_llm

        ok = ping_llm()
        print(f"LLM ping [{settings.llm_provider}]: {'OK' if ok else 'FALLO'}")
        sys.exit(0 if ok else 1)

    import uvicorn
    from core.config import settings

    uvicorn.run(
        "api.app:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
