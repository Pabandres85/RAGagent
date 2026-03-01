# Arquitectura del Sistema — RAG Multi-Agente Resolución 3100

> Última actualización: 2026-03-01

---

## 1. Visión General

El sistema es un asistente normativo basado en Retrieval-Augmented Generation (RAG) con arquitectura multi-agente, diseñado para responder preguntas sobre los estándares de habilitación de servicios de salud en Colombia (Resolución 3100 de 2019).

```
Usuario / UI Streamlit
        │
        ▼
  FastAPI  /api/v1/query
        │
        ▼
  Orchestrator  (ruteo híbrido)
   ├─ TalentoHumanoAgent
   ├─ InfraestructuraAgent
   ├─ DotacionAgent
   ├─ MedicamentosDispositivosAgent
   ├─ ProcesosPrioritariosAgent
   ├─ HistoriaClinicaAgent
   └─ InterdependenciaAgent
        │
        ▼
  GuardrailsResult → AgentResponse (JSON schema)
        │
        ▼
  Respuesta estructurada: answer + citations + checklist
```

---

## 2. Componentes

### 2.1 Capa de Datos (`core/`, `rag/`)

| Componente | Archivo | Responsabilidad |
|-----------|---------|-----------------|
| Config | `core/config.py` | Pydantic Settings v2; parámetros LLM, embeddings, FAISS |
| Embeddings | `core/embeddings.py` | Singleton `SentenceTransformer`; `embed_query()` / `embed_texts()` |
| Metadata Store | `core/metadata_store.py` | JSON persistence de `ChunkMetadata` por módulo; invariante orden=índice FAISS |
| LLM Client | `core/llm_client.py` | Factory OpenAI-compatible; soporta lmstudio y ollama; `ping_llm()` |
| Retriever | `rag/retriever.py` | `IndexFlatIP` FAISS L2-normalizado; `retrieve(query, module, top_k)` |
| Reranker | `rag/reranker.py` | Passthrough por defecto; opcionalmente cross-encoder |
| Citations | `rag/citations.py` | `format_citation()`: texto + numeral + página → string; `build_context()` |

### 2.2 Capa de Agentes (`agents/`)

| Componente | Archivo | Responsabilidad |
|-----------|---------|-----------------|
| Prompts | `agents/prompts.py` | `MODULE_DESCRIPTIONS`: descripciones semánticas de cada módulo para ruteo |
| Guardrails | `agents/guardrails.py` | Schema Pydantic: `AgentResponse`, `Citation`, `ChecklistItem`; `_extract_json()` |
| Base Specialist | `agents/base_specialist.py` | Template method: retrieve → rerank → build_context → LLM → validate |
| Specialists (×7) | `agents/specialists/` | Un agente por módulo; hereda `BaseSpecialistAgent` |
| Orchestrator | `agents/orchestrator.py` | Ruteo híbrido; detección transversal; merge de resultados; timings |
| Baseline | `agents/baseline_mono_agent.py` | Índice global; control experimental |

### 2.3 Capa de API (`api/`)

| Componente | Archivo | Responsabilidad |
|-----------|---------|-----------------|
| App | `api/app.py` | FastAPI + CORS middleware |
| Routes | `api/routes.py` | `GET /health`, `POST /query`; lazy singleton del Orchestrator |

### 2.4 Interfaz de Usuario (`ui/`)

| Componente | Archivo | Responsabilidad |
|-----------|---------|-----------------|
| App principal | `ui/app.py` | Streamlit; consulta, resultados, skeleton loader, timings |
| Evaluación | `ui/pages/2_Evaluacion.py` | Métricas del gold set; routing accuracy `_specific` |
| Auditoría | `ui/pages/3_Auditar_Goldset.py` | UI interactiva para auditar / corregir / eliminar entradas del gold set |

### 2.5 Scripts de Ingesta y Evaluación

| Script | Responsabilidad |
|--------|-----------------|
| `scripts/ingest.py` | PDF → chunks → embeddings → FAISS×8 (7 módulos + 1 global) |
| `scripts/build_goldset.py` | Muestrea chunks → LLM genera pares pregunta/respuesta → `gold_set.json` |
| `scripts/add_general_questions.py` | Agrega preguntas con `module="general"` al gold set |
| `eval/run_eval.py` | Runner offline; checkpoints; métricas multi-agente y mono-agente |
| `eval/metrics.py` | `recall_at_k`, `mrr`, `exact_match`, `f1_score`, `routing_accuracy` |

---

## 3. Flujo de una Consulta

```
1. Usuario envía pregunta via UI o POST /api/v1/query
        │
2. Orchestrator.route(question)
   ├─ embed_query(question) → vector 768 dims
   ├─ coseno con MODULE_DESCRIPTIONS × 7
   ├─ bonus léxico por ROUTING_KEYWORDS
   ├─ detección de consulta integral (_is_integral_service_query)
   └─ RoutingDecision: module, modules[], is_transversal, scores
        │
3. Para cada módulo en routing.modules:
   └─ BaseSpecialistAgent.answer(question)
      ├─ retriever.retrieve(question, module, top_k=10)
      │     └─ embed_query → FAISS.search → [chunk_ids] → [ChunkMetadata]
      ├─ reranker.rerank(chunks)   # passthrough por defecto
      ├─ citations.build_context(chunks)  → string con numerales y texto
      ├─ llm_client.chat(system_prompt, user_prompt+context)
      │     └─ OpenAI-compatible API → JSON string
      └─ guardrails._extract_json(response)
            └─ AgentResponse(answer, citations, checklist, module, confidence)
        │
4. Orchestrator._merge_results(module_results)
   ├─ Filtra resultados válidos y con evidencia
   ├─ Si 1 resultado válido: retorna directamente
   └─ Si N>1: concatena answer_sections, dedup citations + checklist
        │
5. Retorno dict:
   {routing, valid, response, errors, warnings, timings}
```

---

## 4. Índices FAISS y Corpus

| Índice | Módulo | Chunks |
|--------|--------|--------|
| `faiss_talento_humano.index` | talento_humano | ~85 |
| `faiss_infraestructura.index` | infraestructura | ~210 |
| `faiss_dotacion.index` | dotacion | ~220 |
| `faiss_medicamentos_dispositivos.index` | medicamentos_dispositivos | ~195 |
| `faiss_procesos_prioritarios.index` | procesos_prioritarios | ~180 |
| `faiss_historia_clinica.index` | historia_clinica | ~140 |
| `faiss_interdependencia.index` | interdependencia | ~20 |
| `faiss_global.index` | global (baseline) | 1.249 |

- **PDF fuente**: Resolución 3100 de 2019, Ministerio de Salud y Protección Social de Colombia
- **Extractor**: PyMuPDF (`fitz`)
- **Segmentación**: por encabezados de módulo (detección de `_strip_accents()` sobre headers) + chunking por caracteres con solapamiento
- **Parámetros**: `CHUNK_SIZE=768`, `CHUNK_OVERLAP=128`

---

## 5. Schema de Respuesta (Guardrails)

```python
class Citation(BaseModel):
    text: str           # fragmento normativo citado
    numeral: str | None # e.g. "17.3.1"
    page: int | None    # página del PDF
    resolution: str     # e.g. "Resolución 3100 de 2019"
    vigencia: str | None

class ChecklistItem(BaseModel):
    item: str           # descripción del requisito
    numeral: str | None
    status: Literal["cumple", "no_cumple", "no_aplica", "pendiente"]

class AgentResponse(BaseModel):
    answer: str         # respuesta en lenguaje natural
    citations: list[Citation]    # ≥1 obligatorio
    checklist: list[ChecklistItem]
    module: str
    confidence: float
```

Los `field_validator(mode="before")` en `Citation.page` y `Citation/ChecklistItem.numeral` toleran listas del LLM, tomando el primer elemento o uniéndolos con `", "`.

---

## 6. Ruteo Detallado

### 6.1 Scores híbridos

```
score_final[m] = coseno(embed(query), embed(MODULE_DESCRIPTIONS[m]))
                 + lexical_bonus[m]

lexical_bonus[m] = min(0.24, 0.12 + (hits - 1) * 0.04)
                   donde hits = # keywords de ROUTING_KEYWORDS[m] presentes en query
```

### 6.2 Detección de consulta transversal

**Caso A — Consulta integral de servicio**:
- Pattern: contiene frase de `INTEGRAL_QUERY_PATTERNS` ("cómo habilito", "qué requisitos") Y anchor de `SERVICE_ANCHORS` ("servicio", "quirófano", etc.)
- Acción: seleccionar módulos base [infraestructura, dotacion, talento_humano] + módulos adicionales según vocabulario de la consulta

**Caso B — Consulta amplia con scores cercanos**:
- Pattern: contiene "servicio" o "requisito" Y diferencia top-1 vs top-2 ≤ 0.08
- Acción: seleccionar top-3 módulos con score ≥ (best_score − 0.08)

---

## 7. Configuración

Archivo `.env` (no incluido en el repo):

```env
LLM_PROVIDER=lmstudio        # o "ollama"
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=qwen2.5-7b-instruct-1m
LM_STUDIO_API_KEY=sk-lm-...
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3:8b
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
FAISS_TOP_K=10
LOG_LEVEL=INFO
```

---

## 8. Árbol de Archivos Clave

```
proyecto_grado_rag/
├── api/
│   ├── app.py              # FastAPI app + CORS
│   └── routes.py           # /health + /query
├── agents/
│   ├── base_specialist.py
│   ├── baseline_mono_agent.py
│   ├── guardrails.py
│   ├── orchestrator.py
│   ├── prompts.py
│   └── specialists/
│       ├── __init__.py
│       ├── talento_humano.py
│       ├── infraestructura.py
│       ├── dotacion.py
│       ├── medicamentos_dispositivos.py
│       ├── procesos_prioritarios.py
│       ├── historia_clinica.py
│       └── interdependencia.py
├── core/
│   ├── config.py
│   ├── embeddings.py
│   ├── llm_client.py
│   └── metadata_store.py
├── rag/
│   ├── citations.py
│   ├── reranker.py
│   └── retriever.py
├── scripts/
│   ├── ingest.py
│   ├── build_goldset.py
│   └── add_general_questions.py
├── eval/
│   ├── datasets/
│   │   └── gold_set.json   # 122 entradas (post-auditoría)
│   ├── metrics.py
│   └── run_eval.py
├── ui/
│   ├── app.py
│   └── pages/
│       ├── 2_Evaluacion.py
│       └── 3_Auditar_Goldset.py
├── docs/
│   ├── architecture.md     # este archivo
│   └── decisions.md
├── main.py                 # CLI: --ingest, --ping, default uvicorn
├── ESTADO_PROYECTO.md      # tablero operacional
└── .env                    # configuración local (no en repo)
```

---

*Documento generado para el repositorio de tesis. Mantener sincronizado con `ESTADO_PROYECTO.md`.*
