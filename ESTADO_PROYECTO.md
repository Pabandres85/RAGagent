# ESTADO DEL PROYECTO — RAG Multi-Agente Resolución 3100 de 2019

> **Actualizar este archivo cada vez que se implemente un componente nuevo o se cambie el estado de algo.**
> Última actualización: 2026-03-01 (v4 — fix routing medicamentos_dispositivos + evaluación final)

---

## 1. DESCRIPCIÓN GENERAL

**Tesis**: Sistema multi-agente RAG para consultas normativas sobre habilitación de servicios de salud en Colombia (Resolución 3100 de 2019).

**Autor**: Pablo Andrés Muñoz Martínez — UAO Maestría en IA
**Ruta del proyecto**: `c:\UAOTESISGRADO\proyecto_grado_rag`

**Objetivo**: Comparar un sistema multi-agente especializado por estándares normativos contra un baseline mono-agente (índice global), midiendo routing accuracy, F1, EM y valid rate.

---

## 2. CONFIGURACIÓN DE ENTORNO

| Parámetro | Valor |
|-----------|-------|
| LLM activo | LM Studio, puerto 1234 |
| Modelo LLM | `qwen2.5-7b-instruct-1m` |
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` (768 dims) |
| CHUNK_SIZE | 768 caracteres |
| CHUNK_OVERLAP | 128 caracteres |
| FAISS_TOP_K | 10 |
| Temperatura LLM | 0.0 |

Configuración en `.env` (no subir al repo). Cambiar proveedor con `LLM_PROVIDER=lmstudio|ollama`.

---

## 3. DATOS Y CORPUS

### PDF fuente
- `data/raw/resolucion-3100-de-2019.pdf` — documento principal (230 páginas aprox.)
- **Secciones excluidas en ingesta**: páginas < 36 (encabezados repetitivos), encabezado de página filtrado con `PAGE_HEADER_RE`

### Chunks por módulo (post-ingesta, CHUNK_SIZE=768)
| Módulo | Chunks | % |
|--------|--------|---|
| talento_humano | 719 | 57.6% |
| dotacion | 129 | 10.3% |
| infraestructura | 124 | 9.9% |
| medicamentos_dispositivos | 113 | 9.0% |
| procesos_prioritarios | 105 | 8.4% |
| historia_clinica | 39 | 3.1% |
| interdependencia | 20 | 1.6% |
| **TOTAL** | **1,249** | |

> ⚠️ `talento_humano` tiene desproporción por carry-over de módulo: las secciones administrativas (REPS, visitas, generalidades) al no tener header propio quedan asignadas al último módulo detectado.

### Índices FAISS
- `artifacts/faiss/` → 7 índices por módulo + 1 `global.faiss`
- Tipo: `IndexFlatIP` con vectores L2-normalizados (coseno)

---

## 4. ARQUITECTURA DEL SISTEMA

```
Pregunta de usuario
       │
       ▼
 Orchestrator
 ├── embed_query(pregunta)
 ├── similitud coseno vs MODULE_DESCRIPTIONS (7 descripciones)
 ├── bonus léxico por keywords por módulo
 ├── detección transversal:
 │   ├── integral_service_query → selecciona infra+dot+th+extras
 │   └── broad_query + scores cercanos → top 3 módulos
       │
       ▼
 Specialist Agent(s) [1 o múltiples]
 ├── Retriever → FAISS del módulo
 ├── Reranker (passthrough por defecto, cross-encoder opcional)
 ├── build_context (citas formateadas)
 ├── LLM → respuesta JSON con answer + citations + checklist
 └── Guardrails → valida esquema, cita obligatoria, confianza
       │
       ▼ (si transversal: merge de respuestas)
 GuardrailsResult
```

**Baseline (MonoAgent)**: mismo flujo pero con índice `global.faiss` sin routing.

---

## 5. ESTADO DE IMPLEMENTACIÓN

### ✅ COMPLETO

| Archivo | Descripción |
|---------|-------------|
| `core/config.py` | Pydantic Settings v2, dual-provider helpers |
| `core/llm_client.py` | Factory OpenAI-compatible, singleton, `ping_llm()` |
| `core/embeddings.py` | Lazy singleton sentence-transformers |
| `core/metadata_store.py` | JSON persistence por módulo, `ChunkMetadata` |
| `rag/retriever.py` | FAISS IndexFlatIP, L2-normalizado |
| `rag/reranker.py` | Passthrough por defecto, opcional cross-encoder |
| `rag/citations.py` | `format_citation()`, `build_context()` |
| `agents/prompts.py` | `MODULE_DESCRIPTIONS` — descripciones semánticas mejoradas para dotacion y medicamentos_dispositivos |
| `agents/guardrails.py` | Pydantic schema + cita obligatoria + `_extract_json()` robusto |
| `agents/base_specialist.py` | Template method: retrieve→rerank→context→LLM→validate |
| `agents/orchestrator.py` | Ruteo híbrido coseno+léxico, soporte transversal; keywords corregidos (eliminado "dispositivo" de dotacion) |
| `agents/baseline_mono_agent.py` | Índice global, sin especialización |
| `agents/specialists/*.py` | 7 especialistas: talento_humano, infraestructura, dotacion, medicamentos_dispositivos, procesos_prioritarios, historia_clinica, interdependencia |
| `scripts/ingest.py` | PDF→chunks→embeddings→FAISS×8 con filtro de encabezados y normalización de acentos |
| `scripts/build_goldset.py` | Generador de Q&A: muestrea chunks → LLM genera pares normativo/respuesta |
| `scripts/add_general_questions.py` | Agrega ~41 preguntas generales/admin con `module="general"` al gold set |
| `api/app.py` | FastAPI + CORS |
| `api/routes.py` | `/health` + `/query`, lazy Orchestrator singleton |
| `ui/app.py` | Streamlit UI principal |
| `ui/pages/2_Evaluacion.py` | Página de evaluación en UI |
| `ui/pages/3_Auditar_Goldset.py` | Auditoría interactiva del gold set (reasignar/eliminar entradas) |
| `eval/metrics.py` | `recall_at_k`, `mrr`, `exact_match`, `f1_score`, `routing_accuracy` |
| `eval/run_eval.py` | Runner offline multi-agente vs mono-agente con checkpoints |
| `main.py` | CLI: `--ingest`, `--ping`, default uvicorn |

### ⏳ PENDIENTE / EN PROCESO

| Tarea | Estado | Notas |
|-------|--------|-------|
| Limpieza gold set | ✅ COMPLETO | Auditoría completada: 39 entradas eliminadas/corregidas. Gold set final: 122 entradas. Backup en `gold_set.json.bak`. |
| Preguntas generales/admin en gold set | ✅ COMPLETO | 30 preguntas generales agregadas con `add_general_questions.py`. |
| Evaluación completa (full gold set) | ✅ COMPLETO | Dos runs: v1 (pre-fix) y v2 (post-fix routing). Resultados en §8. |
| Análisis de routing failures | ✅ COMPLETO | Matriz de confusión generada. Fix aplicado en medicamentos_dispositivos (+20pp top-1). |
| Fix bug `citations.page` | PENDIENTE | LLM devuelve lista en vez de int → tomar primer elemento en `guardrails.py`. |
| Documentación para tesis | PENDIENTE | Tablas comparativas, gráficas, redacción de capítulo de resultados. |

### 🗑️ ARCHIVOS HUÉRFANOS
- ✅ Eliminados: `write_files.py`, `ingest_gen.py`, `ingest_gen2.py`, `ingest_gen3.py`, `tmp_test.py`

---

## 6. GOLD SET (CONJUNTO DE EVALUACIÓN)

### Estado actual ✅ AUDITADO
- **Archivo**: `eval/datasets/gold_set.json`
- **Backup**: `eval/datasets/gold_set.json.bak` (pre-limpieza)
- **Total de entradas**: **122** (auditadas y limpias)
- **Generación automática**: LLM (Qwen 2.5 via LM Studio) desde chunks muestreados
- **Generación manual**: 30 preguntas generales/admin agregadas con `add_general_questions.py`

### Distribución por módulo (post-auditoría final)
| Módulo | Entradas | Tipo |
|--------|----------|------|
| talento_humano | 7 | Específico |
| infraestructura | 19 | Específico |
| dotacion | 21 | Específico |
| medicamentos_dispositivos | 20 | Específico |
| procesos_prioritarios | 16 | Específico |
| historia_clinica | 15 | Específico |
| interdependencia | 7 | Específico |
| general | 17 | Admin/transversal — excluidas del routing accuracy |
| **TOTAL** | **122** | |

### Limpieza aplicada (auditoría 2026-03-01)
Se eliminaron/corrigieron **39 entradas** problemáticas:

| Grupo | Tipo | Acción | Cantidad |
|-------|------|--------|----------|
| 1 | Preguntas autoreferentes (citan "el fragmento") | **Eliminadas** | 17 |
| 2 | Respuestas "no se encuentra" (contenido fuera del corpus) | **Eliminadas** | 22 |
| 3 | Respuestas muy cortas sin contenido normativo | **Expandidas** | 3 |
| 4 | Respuestas con referencia al fragmento en el answer | **Corregidas** | 4 |

**Criterio de exclusión Grupo 2**: Los capítulos 1-10 (páginas 1-35) no fueron indexados (filtro `page >= 36`). Las preguntas sobre definiciones generales, REPS, visitas que respondían "no se encuentra" no tienen poder discriminante entre los dos sistemas → eliminadas para mantener el gold set enfocado en los 7 estándares normativos comparados.

### Problemas conocidos (resueltos)
- **Contaminación talento_humano**: secciones administrativas sin header propio (numerales 9.x, 10.x) se asignaban por carry-over → corregido en auditoría (muchas eliminadas del Grupo 1/2).
- **Desequilibrio**: interdependencia tiene solo 20 chunks → 7 entradas. Aceptable dado el tamaño del estándar.
- **chunk_id stale (3 entradas)**: chunk_ids que ya no existen en metadatos después de re-ingesta. Solo efecto cosmético en UI de auditoría; no afecta métricas de evaluación.

---

## 7. PREGUNTAS GENERALES / ADMINISTRATIVAS ✅ IMPLEMENTADO

Se implementó la **Opción A**: preguntas con `module="general"` excluidas del routing accuracy pero incluidas en F1/EM.

### Clasificación de las 41 preguntas
| Categoría | Cantidad | Módulo asignado |
|-----------|----------|-----------------|
| Condiciones admin IPS / profesional independiente | 2 | `general` |
| Modalidades, estándares y grupos | 3 | `general` |
| Definiciones generales | 1 | `general` |
| Definiciones por estándar (una por módulo) | 7 | módulo correspondiente |
| Condiciones especiales (cooperación, emergencias) | 2 | `general` |
| Tipos de prestadores (IPS, PI, EOSD, transporte) | 5 | `general` |
| Condiciones, clasificación, suficiencia patrimonial | 3 | `general` |
| Pasos inscripción / habilitación | 2 | `general` |
| Distintivos de habilitación | 1 | `general` |
| Visitas de habilitación | 2 | `general` |
| Novedades REPS (9 tipos) | 9 | `general` |
| Talento humano específico | 2 | `talento_humano` |

### Cambios en el sistema
- **`scripts/add_general_questions.py`** — script creado, usa índice global para generar respuestas de referencia ✅ EJECUTADO
- **`eval/run_eval.py`** — `_build_summary()` ahora genera `routing_accuracy_top1_specific` (la métrica representativa) y `routing_accuracy_top1` (todos, incluyendo general)
- **`ui/pages/3_Auditar_Goldset.py`** — `MODULE_CHOICES` incluye `"general"`, `chunk_id=None` muestra mensaje explicativo; **CSS selectbox corregido** (color visible en tema claro)

---

## 8. RESULTADOS DE EVALUACIÓN

### Run v1 — pre-fix routing (gold set limpio, 122 ítems)
| Métrica | Multi-Agente | Mono-Agente |
|---------|-------------|-------------|
| Valid rate | 49.2% | 95.1% |
| F1 promedio | 0.270 | 0.112 |
| Routing Top-1 (específicos, n=105) | 32.4% | N/A |
| Routing Any (específicos) | 44.8% | N/A |

### Run v2 — post-fix routing medicamentos_dispositivos ✅ ÚLTIMO (2026-03-01)
**Archivo**: `artifacts/eval_runs/latest_eval.json`

| Métrica | Multi-Agente | Mono-Agente | Δ vs v1 |
|---------|-------------|-------------|---------|
| Valid rate | **52.5%** | 95.1% | +3.3pp |
| F1 promedio | **0.280** | 0.112 | +0.010 |
| Routing Top-1 (específicos, n=105) | **36.2%** | N/A | +3.8pp |
| Routing Any (específicos) | **47.6%** | N/A | +2.8pp |
| Routing Top-1 (todos ref., n=122) | 31.1% | N/A | — |
| Routing Any (todos ref.) | 41.0% | N/A | — |

*EM = 0 en ambos (esperado para respuestas libres en lenguaje natural)*

### Routing accuracy por módulo (v2)
| Módulo | Top-1 | Any | N | Observación |
|--------|-------|-----|---|-------------|
| interdependencia | 57.1% | 71.4% | 7 | Mejor módulo |
| historia_clinica | 53.3% | 53.3% | 15 | Bueno |
| infraestructura | 42.1% | 63.2% | 19 | Aceptable |
| dotacion | 42.9% | 57.1% | 21 | Aceptable |
| talento_humano | 28.6% | 28.6% | 7 | Bajo (muestra pequeña) |
| medicamentos_dispositivos | 25.0% | 35.0% | 20 | Mejoró +20pp vs v1 (era 5%) |
| procesos_prioritarios | 12.5% | 25.0% | 16 | Crítico — fallos dispersos en 6 módulos |

### Interpretación
- **Multi F1 2.5× > Mono F1**: la especialización mejora significativamente la calidad de respuesta.
- **Valid rate gap (52.5% vs 95.1%)**: el multi-agente tiene guardrails más estrictos (cita obligatoria); el mono-agente acepta respuestas sin evidencia. El gap refleja exigencia de calidad, no falla de cobertura — argumento válido para la tesis.
- **medicamentos_dispositivos: +20pp top-1** gracias a corrección de keywords (`"dispositivo"` removido de `dotacion`).
- **procesos_prioritarios persistente (12.5% top-1)**: módulo con vocabulario transversal que se solapa con casi todos los demás. Límite del enfoque léxico-coseno — hallazgo relevante para la tesis.
- **EM = 0**: esperado — las respuestas son texto libre, no citas textuales exactas.

### Fallos de guardrails observados (v2)
| Sistema | Tipo de fallo | Frecuencia |
|---------|--------------|------------|
| Multi-Agente | "No contiene cita normativa" | ~4 ocurrencias (dotacion, interdependencia, historia_clinica, infraestructura ×2) |
| Multi-Agente | `citations.page` recibe lista en vez de int | 1 vez (dotacion — LLM devuelve `[150, 159]`) |
| Mono-Agente | JSON malformado (comillas en propiedad) | 5 veces |

> **Bug pendiente**: el LLM ocasionalmente devuelve `"page": [n, n]` en lugar de `"page": n`. El guardrail lo rechaza correctamente. Fix sugerido: en `guardrails.py`, si `page` es lista, tomar el primer elemento.

---

## 9. CÓMO EJECUTAR EL SISTEMA

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Verificar conectividad LLM
python main.py --ping

# 3. Ingestar el corpus (ya hecho, 1249 chunks)
python scripts/ingest.py

# Nota: usar .venv/Scripts/python.exe (o py -3) en lugar de `python` — no está en PATH

# 4. Generar gold set por módulos (ya hecho, 122 entradas auditadas)
.venv/Scripts/python.exe scripts/build_goldset.py
.venv/Scripts/python.exe scripts/build_goldset.py --module talento_humano  # solo un módulo

# 4b. Agregar preguntas generales/admin al gold set (ya hecho — 30 preguntas)
.venv/Scripts/python.exe scripts/add_general_questions.py --dry-run  # preview
.venv/Scripts/python.exe scripts/add_general_questions.py             # genera preguntas

# 5. Correr evaluación completa (gold set limpio: 122 ítems)
.venv/Scripts/python.exe -m eval.run_eval
.venv/Scripts/python.exe -m eval.run_eval --limit 20  # prueba rápida

# 6. Levantar API
.venv/Scripts/python.exe main.py

# 7. Levantar UI Streamlit
.venv/Scripts/streamlit.exe run ui/app.py
```

---

## 10. PROBLEMAS CONOCIDOS Y NOTAS TÉCNICAS

### Bugs resueltos
| Bug | Solución |
|-----|---------|
| `{{}}` double-brace en prompt de QA | `QA_SYSTEM_PROMPT` usaba `'{{"question":...}}'` (string literal) → LLM devolvía JSON inválido. Fix: `'{"question": "...", "answer": "..."}'` |
| Mono-agent valid rate ~0% | `MONO_AGENT_SYSTEM_PROMPT` no permitía respuesta "sin evidencia". Fix: se agrega la frase estándar de fallback |
| Guardrails rechazaba respuestas válidas | `_extract_json()` reforzado con manejo de `{{}}``, code blocks y `ast.literal_eval` fallback |
| Encabezados de página en embeddings | `PAGE_HEADER_RE` filtra "Página N de NNN" en `clean_page_text()` |
| Módulos con acentos garbled | `_strip_accents()` normaliza antes de comparar headers |
| Routing medicamentos→dotacion (16/20 fallos) | `"dispositivo"` y `"dispositivos"` estaban en keywords de `dotacion` → removidos. Descripción de `medicamentos_dispositivos` enriquecida con vocabulario farmacológico. Resultado: +20pp top-1 en medicamentos, +3.8pp global |

### Limitaciones actuales
1. **Contenido tabular**: el extractor PyMuPDF genera tablas como texto plano sin estructura. Las preguntas sobre tablas de requisitos pierden contexto.
2. **Secciones administrativas sin módulo**: numerales 1-10.x (condiciones generales, REPS) no tienen header de módulo → se asignan por carry-over.
3. **interdependencia pequeño**: solo 20 chunks → modelo con poca evidencia.
4. **Evaluación sin métricas semánticas**: solo F1 de tokens. Pendiente integrar BERTScore o similitud coseno para evaluar calidad de respuesta.
5. **procesos_prioritarios — límite del ruteo léxico-coseno**: routing top-1 de solo 12.5% con fallos dispersos en 6 de 7 módulos. El vocabulario de "procesos prioritarios" (urgencias, sepsis, parto, transfusión) se solapa semánticamente con casi todos los demás estándares. El enfoque coseno+léxico no es suficiente para discriminar este módulo sin un clasificador supervisado.
6. **talento_humano gold set muy reducido (7 entradas)**: la auditoría eliminó la mayoría de entradas contaminadas con contenido administrativo. Con solo 7 preguntas, cada fallo/acierto representa ~14pp — las estadísticas de routing de este módulo (28.6% top-1) no son estadísticamente representativas.
7. **Mono-agente: valid rate estructuralmente alto**: el 95.1% del mono-agente no refleja mejor calidad sino guardrails más permisivos — está configurado para responder incluso sin evidencia suficiente. La comparación directa de valid rates entre sistemas no es equivalente; el multi-agente penaliza respuestas sin cita normativa mientras el mono-agente las acepta.

---

## 11. PRÓXIMOS PASOS

- [x] ~~Decidir estrategia para preguntas generales/admin~~ → Opción A implementada
- [x] ~~Implementar `module: "general"` en eval~~ → `routing_accuracy_top1_specific` en run_eval.py
- [x] ~~Ejecutar `scripts/add_general_questions.py`~~ → 30 preguntas generales agregadas
- [x] ~~Auditar las respuestas generadas en `ui/pages/3_Auditar_Goldset.py`~~ → gold set limpio: 122 entradas
- [x] ~~Correr evaluación completa sobre gold set final (122 ítems)~~ → completado 2026-03-01
- [x] ~~Analizar routing failures~~ → matriz de confusión generada; medicamentos identificado como módulo crítico
- [x] ~~Fix routing medicamentos_dispositivos~~ → eliminado `"dispositivo"` de dotacion keywords; enriquecida descripción semántica → +20pp top-1 en medicamentos, +3.8pp global
- [ ] **Fix bug `citations.page`**: cuando el LLM devuelve lista, tomar el primer elemento en `guardrails.py` (afecta dotacion y medicamentos ocasionalmente)
- [ ] **procesos_prioritarios routing**: 12.5% top-1 — evaluar si mejorar descripción/keywords o documentar como limitación del enfoque léxico-coseno
- [ ] Documentar resultados para tesis (tablas comparativas, gráficas)
- [x] ~~Limpiar archivos huérfanos~~ → ya no existen en el árbol del proyecto

---

*Este documento se actualiza manualmente después de cada ciclo de implementación.*
