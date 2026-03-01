# Sistema Multi-Agente RAG Local - Habilitacion en Salud

Asistente normativo local para consultas sobre la Resolucion 3100 de 2019 del Sistema Unico de Habilitacion en Colombia.

El proyecto implementa una arquitectura RAG multi-agente con:

- Un orquestador central.
- Siete agentes especialistas, uno por modulo normativo.
- Indices FAISS por modulo y un indice global para baseline mono-agente.
- Backend LLM local compatible con LM Studio y Ollama.

## Alcance

El sistema responde preguntas normativas y devuelve:

- Respuesta sustentada en fragmentos recuperados.
- Citas con numeral, pagina y vigencia.
- Checklist de verificacion cuando aplica.

## Modulos cubiertos

- `talento_humano`
- `infraestructura`
- `dotacion`
- `medicamentos_dispositivos`
- `procesos_prioritarios`
- `historia_clinica`
- `interdependencia`

## Arquitectura

```text
Consulta del usuario
    |
    v
Orchestrator
    |
    +--> Agente especialista
             |
             +--> Retriever FAISS por modulo
             +--> Reranker opcional
             +--> LLM local (LM Studio u Ollama)
             +--> Guardrails

Baseline alterno:
Consulta -> MonoAgent -> Indice global FAISS -> LLM local
```

## Requisitos

- Python 3.11 recomendado
- Docker Desktop (opcional para despliegue)
- LM Studio o Ollama

## Configuracion

1. Copiar `\.env.example` a `\.env`.
2. Ajustar `LLM_PROVIDER`.
3. Configurar la URL, API key y modelo del backend LLM elegido.

Ejemplo con LM Studio:

```env
LLM_PROVIDER=lmstudio
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=tu_token_local
LM_STUDIO_MODEL=qwen2.5-7b-instruct-1m
```

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Flujo de uso

1. Colocar los PDFs normativos en `data/raw/`.
2. Ejecutar la ingesta:

```bash
python scripts/ingest.py
```

3. Verificar conectividad con el LLM:

```bash
python main.py --ping
```

4. Levantar la API:

```bash
python main.py
```

5. Probar la UI:

```bash
streamlit run ui/app.py
```

## Evaluacion

El proyecto incluye:

- Generacion de gold set: `scripts/build_goldset.py`
- Metricas base: `eval/metrics.py`
- Runner offline: `eval/run_eval.py`

Ejemplo:

```bash
python scripts/build_goldset.py --n-per-module 5
python eval/run_eval.py
```

## Estructura

```text
proyecto_grado_rag/
|-- agents/
|-- api/
|-- artifacts/
|-- core/
|-- data/
|-- docs/
|-- eval/
|-- notebooks/
|-- rag/
|-- scripts/
|-- tests/
|-- ui/
|-- main.py
|-- requirements.txt
|-- .env.example
```

## Notas

- `artifacts/`, `data/raw/`, `data/metadata/` y los entornos virtuales no deben subirse al repositorio.
- El corpus normativo es publico, pero no se incluye en el repo.
- El sistema esta orientado a uso academico y validacion de tesis.
