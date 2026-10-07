# ESTADO DEL PROYECTO — RAG Multi-Agente Resolución 3100 de 2019

> **Actualizar este archivo cada vez que se implemente un componente nuevo o se cambie el estado de algo.**
> Última actualización: 2026-10-07 (corrección del baseline mono §8b; estados de respuesta y hoja de auditoría del gold v2 §8c; nueva política de corpus en staging; antes: 2026-03-01 v6)

> ## ⚠️ AVISO (2026-10-06): resultados v1–v4 invalidados para la comparación multi vs. mono
> Una auditoría halló que el índice FAISS global estaba desalineado con sus metadatos y que el prompt del mono tenía llaves dobles literales. Las cifras mono de v1–v4 (F1 ≈ 0.11, valid 95 %) **no miden la capacidad del mono-agente** y la conclusión "multi F1 2.5× > mono" **no debe usarse**. Los archivos `artifacts/eval_runs/latest_eval.json` y `latest_eval_summary.json` (versionados) son esa corrida antigua (v4, pre-corrección; etiqueta git `eval-v4-pre-fix`); **no los cite**. Resultados vigentes (provisionales): §8b.

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
| Max tokens LLM | 4096 (subido de 2048 para evitar truncación JSON) |

Configuración en `.env` (no subir al repo). Cambiar proveedor con `LLM_PROVIDER=lmstudio|ollama`.

---

## 3. DATOS Y CORPUS

### PDF fuente
- `data/raw/resolucion-3100-de-2019.pdf` — documento principal (230 páginas aprox.)
- **Exclusiones en ingesta**: solo el encabezado "Página N de NNN" se filtra (`PAGE_HEADER_RE`). ⚠️ **No existe filtro por página** (versiones anteriores de este documento afirmaban `page >= 36`; era incorrecto). El índice actual incluye 214 chunks de las páginas < 36 y también el articulado/trámites de las págs. 36–58. El capítulo normativo "11. Estándares y criterios de habilitación" empieza en la **pág. 59**. Política de corpus a implementar (ver §11).

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
| `core/config.py` | Pydantic Settings v2, dual-provider helpers; `llm_max_tokens=4096` |
| `core/llm_client.py` | Factory OpenAI-compatible, singleton, `ping_llm()` |
| `core/embeddings.py` | Lazy singleton sentence-transformers |
| `core/metadata_store.py` | JSON persistence por módulo, `ChunkMetadata` |
| `rag/retriever.py` | FAISS IndexFlatIP, L2-normalizado |
| `rag/reranker.py` | Passthrough por defecto, opcional cross-encoder |
| `rag/citations.py` | `format_citation()`, `build_context()` |
| `agents/prompts.py` | `MODULE_DESCRIPTIONS` — descripciones semánticas mejoradas para dotacion y medicamentos_dispositivos |
| `agents/guardrails.py` | Pydantic schema + cita obligatoria + `_extract_json()` robusto; validators `coerce_page` y `coerce_numeral` para tolerar listas del LLM |
| `agents/base_specialist.py` | Template method: retrieve→rerank→context→LLM→validate |
| `agents/orchestrator.py` | Ruteo híbrido coseno+léxico, soporte transversal; mide tiempos por fase (`routing_ms`, `agents_ms`, `total_ms`) en `answer()` |
| `agents/baseline_mono_agent.py` | Índice global, sin especialización |
| `agents/specialists/*.py` | 7 especialistas: talento_humano, infraestructura, dotacion, medicamentos_dispositivos, procesos_prioritarios, historia_clinica, interdependencia |
| `scripts/ingest.py` | PDF→chunks→embeddings→FAISS×8 con filtro de encabezados y normalización de acentos |
| `scripts/build_goldset.py` | Generador de Q&A: muestrea chunks → LLM genera pares normativo/respuesta |
| `scripts/add_general_questions.py` | Agrega ~41 preguntas generales/admin con `module="general"` al gold set |
| `api/app.py` | FastAPI + CORS |
| `api/routes.py` | `/health` + `/query`, lazy Orchestrator singleton |
| `ui/app.py` | Streamlit UI principal; skeleton loader animado + step pills + panel de tiempos de procesamiento |
| `ui/pages/2_Evaluacion.py` | Página de evaluación; métricas corregidas a variantes `_specific` (n=105) |
| `ui/pages/3_Auditar_Goldset.py` | Auditoría interactiva del gold set (reasignar/eliminar entradas) |
| `eval/metrics.py` | `recall_at_k`, `mrr`, `exact_match`, `f1_score`, `routing_accuracy` |
| `eval/run_eval.py` | Runner offline multi-agente vs mono-agente con checkpoints |
| `main.py` | CLI: `--ingest`, `--ping`, default uvicorn |
| `scripts/ingest.py` (reescrito 2026-10-07) | Política de corpus del cap. 11 (desde la pág. 59, detectada), módulo por línea "Estándar de X", servicio por "11.x.y SERVICIO", filtro de ruido de encabezado, `--out-dir` (staging), `--start-page`, `--rebuild-global`. **Staging validado, NO activado** (ver §8c) |
| `scripts/build_gold_v2_audit.py` | Hoja de auditoría del gold v2 (JSON fuente de verdad + CSV, `--import-csv`, huella de evidencia) |
| `tests/` | 81 tests efectivos (alineación de índices, ingesta, prompts, runner, estados, hoja de auditoría); 23 siguen en `skip` (placeholders originales) |

### ⏳ PENDIENTE / EN PROCESO

| Tarea | Estado | Notas |
|-------|--------|-------|
| Limpieza gold set | ✅ COMPLETO | Auditoría completada: 39 entradas eliminadas/corregidas. Gold set final: 122 entradas. Backup en `gold_set.json.bak`. |
| Preguntas generales/admin en gold set | ✅ COMPLETO | 30 preguntas generales agregadas con `add_general_questions.py`. |
| Evaluación completa (full gold set) | ✅ COMPLETO | Dos runs: v1 (pre-fix) y v2 (post-fix routing). Resultados en §8. |
| Análisis de routing failures | ✅ COMPLETO | Matriz de confusión generada. Fix aplicado en medicamentos_dispositivos (+20pp top-1). |
| Fix bug `citations.page` | ✅ COMPLETO | `field_validator("page", mode="before")` en `Citation` — si LLM devuelve lista, toma el primer elemento. |
| Fix bug `checklist.numeral` | ✅ COMPLETO | `field_validator("numeral", mode="before")` en `ChecklistItem` y `Citation` — si LLM devuelve lista, une con `", "`. |
| Fix `llm_max_tokens` insuficiente | ✅ COMPLETO | `config.py`: 2048 → 4096. Respuestas largas (dotación ~5000 chars JSON) se truncaban causando JSON malformado. |
| Fix métricas UI evaluación | ✅ COMPLETO | `2_Evaluacion.py` usaba `routing_accuracy_top1` (n=122) en vez de `routing_accuracy_top1_specific` (n=105). Corregido. |
| UX: skeleton loader + timings | ✅ COMPLETO | `ui/app.py`: shimmer CSS, `_SKELETON_HTML`, `_STEP_PILLS_HTML`. Orchestrator devuelve `timings` por fase. Panel de tiempos en UI. |
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

**Criterio de exclusión Grupo 2**: se supuso que los capítulos 1-10 (páginas 1-35) no estaban indexados (filtro `page >= 36`); ⚠️ **esa suposición era falsa**: el filtro nunca existió y esas páginas sí están en el índice (ver §3). Las preguntas sobre definiciones generales, REPS, visitas que respondían "no se encuentra" no tienen poder discriminante entre los dos sistemas → eliminadas para mantener el gold set enfocado en los 7 estándares normativos comparados.

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

> ⚠️ **Registro histórico (v1–v4).** Las columnas "Mono-Agente" de esta sección provienen del baseline defectuoso y no deben citarse. Resultados vigentes: §8b.

### Run v1 — pre-fix routing (gold set limpio, 122 ítems)
| Métrica | Multi-Agente | Mono-Agente |
|---------|-------------|-------------|
| Valid rate | 49.2% | 95.1% |
| F1 promedio | 0.270 | 0.112 |
| Routing Top-1 (específicos, n=105) | 32.4% | N/A |
| Routing Any (específicos) | 44.8% | N/A |

### Run v2 — post-fix routing medicamentos_dispositivos (2026-03-01)

| Métrica | Multi-Agente | Mono-Agente | Δ vs v1 |
|---------|-------------|-------------|---------|
| Valid rate | **52.5%** | 95.1% | +3.3pp |
| F1 promedio | **0.280** | 0.112 | +0.010 |
| Routing Top-1 (específicos, n=105) | **36.2%** | N/A | +3.8pp |
| Routing Any (específicos) | **47.6%** | N/A | +2.8pp |
| Routing Top-1 (todos ref., n=122) | 31.1% | N/A | — |
| Routing Any (todos ref.) | 41.0% | N/A | — |

### Run v3 — post-fix citations.page; checklist.numeral aún sin corregir (2026-03-01)

| Métrica | Multi-Agente | Mono-Agente | Δ vs v2 |
|---------|-------------|-------------|---------|
| Valid rate | 51.6% | 95.1% | -0.9pp |
| F1 promedio | 0.276 | 0.115 | -0.004 |
| Routing Top-1 (específicos, n=105) | 36.2% | N/A | 0 |
| Routing Any (específicos) | 47.6% | N/A | 0 |

> La leve baja respecto a v2 se debe al fallo de `checklist.numeral` (detectado en este run) y variación aleatoria del LLM. El routing es idéntico — confirma estabilidad del fix de medicamentos.

### Run v4 — post-fix todos los guardrails ✅ ÚLTIMO (2026-03-01)

| Métrica | Multi-Agente | Mono-Agente | Δ vs v2 |
|---------|-------------|-------------|---------|
| Valid rate | 51.6% | 95.1% | -0.9pp |
| F1 promedio | 0.276 | 0.118 | -0.004 |
| Routing Top-1 (específicos, n=105) | **36.2%** | N/A | 0 |
| Routing Any (específicos) | **47.6%** | N/A | 0 |

> La diferencia respecto a v2 (52.5% → 51.6%) está dentro del rango de no-determinismo del LLM (Qwen 2.5 a T=0 varía por redondeo de punto flotante y estado del KV cache). El bug `checklist.numeral` no apareció en este run. Los fixes de guardrails eliminaron los fallos sistemáticos; los fallos restantes son estructurales.

*EM = 0 en ambos runs (esperado para respuestas libres en lenguaje natural)*

### Routing accuracy por módulo (v2/v3 — idéntico)
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
- ~~**Multi F1 2.5× > Mono F1**: la especialización mejora significativamente la calidad de respuesta.~~ **INVALIDADO (2026-10-06)**: el baseline estaba defectuoso. Ver §8b.
- ~~**Valid rate gap (52.5% vs 95.1%)**: refleja exigencia de calidad~~ **REVISADO**: `valid` no mide cobertura (ambos sistemas aceptan la abstención como válida); las "inválidas" del multi son en su mayoría abstenciones. Ver §8b y `docs/decisions.md` §12.
- **medicamentos_dispositivos: +20pp top-1** gracias a corrección de keywords (`"dispositivo"` removido de `dotacion`).
- **procesos_prioritarios persistente (12.5% top-1)**: módulo con vocabulario transversal que se solapa con casi todos los demás. Límite del enfoque léxico-coseno — hallazgo relevante para la tesis.
- **EM = 0**: esperado — las respuestas son texto libre, no citas textuales exactas.

### Fallos de guardrails observados
| Run | Sistema | Tipo de fallo | Estado |
|-----|---------|--------------|--------|
| v2 | Multi-Agente | "No contiene cita normativa" — ~4 ocurrencias | Estructural (guardrail estricto) |
| v2 | Multi-Agente | `citations.page` recibe lista `[150, 159]` — 1 vez | ✅ Corregido en v3 |
| v2/v3 | Mono-Agente | JSON malformado (comillas en propiedad) — 5-6 veces | Estructural (Qwen 2.5 ocasional) |
| v3 | Multi-Agente | `checklist.numeral` recibe lista — 1 vez | ✅ Corregido; no reapareció en v4 |
| v4 | Mono-Agente | "No contiene cita normativa" — 1 vez | Estructural (nuevo tipo vs v2/v3) |

---

## 8b. Corrección del baseline y resultados vigentes (2026-10-06) — PROVISIONALES

**Correcciones aplicadas** (sin cambiar el gold set ni el corpus): (1) `global.faiss` reconstruido en el orden de `load_all()` (`ingest.py --rebuild-global`; test de correspondencia vector–metadato); (2) prompt del mono sin llaves dobles; (3) reintentos del cliente LLM ante `Model unloaded`; (4) runner con `--only`, `--tag`, `--resume` (protegido por huella del dataset y del modo) y `--oracle-routing`; un sistema no ejecutado se registra como `None`, no como 0.

**Ablación (20 primeras preguntas, no aleatorias — solo diagnóstico)**: mono con abstenciones 19→2 y F1 0.101→0.409 (índice) →0.588 (prompt).

**Corrida completa, 122 ítems (`latest_eval_provisional_post_fix.json`)**

| | Mono | Multi (ruteador real) |
|---|---|---|
| Respuestas sustantivas válidas | 89 (73 %) | 64 (52 %) |
| Abstenciones | 31 | ~54 |
| F1 todas | 0.425 | 0.279 |
| F1 en las 52 que ambos responden | 0.513 | 0.438 |
| Routing top-1 (105 específicas) | N/A | 36.2 % |

**Ruteo oráculo de especialista único (105 específicas; `latest_eval_oracle_routing.json`)**: multi con 84 sustantivas (80 %) y F1 0.409. Oráculo−mono = −0.024 [−0.078, +0.030] (sin diferencia demostrable; **no** implica equivalencia); ruteador−mono = −0.150 [−0.216, −0.089]; oráculo−ruteador = +0.125 [+0.082, +0.172].

**Limitaciones de estos resultados** (detalle en `docs/decisions.md` §18): son provisionales (etiquetas del gold set ruidosas, corpus sin depurar, 17 referencias `general` generadas con el índice defectuoso); el oráculo fuerza un solo especialista y es un techo optimista; F1 de tokens penaliza la verbosidad; no hay faithfulness, Recall@k ni MRR; el LLM no es determinista y no se han hecho repeticiones.

**Qué se puede y no se puede afirmar**: con el ruteador actual el multi-agente rinde menos que el baseline; con ruteo perfecto no se demuestra diferencia en F1 y sí mayor cobertura. No se sostiene que la especialización mejore la calidad.

---

## 8c. Estados de respuesta, política de corpus y auditoría del gold v2 (2026-10-07)

### Estados de respuesta (`GuardrailsResult.status`)
| Estado | Significado |
|---|---|
| `answered` | Respuesta sustantiva con al menos una cita |
| `abstained` | El sistema no tiene evidencia: la abstención declarada por el LLM (JSON válido, sin cita) **o** ningún fragmento recuperado |
| `rejected` | El LLM respondió pero los guardrails la rechazan (JSON malformado, esquema inválido, sin cita normativa) |
| `error` | Fallo de ejecución (excepción: LLM, red…), no del contenido |

`valid` se conserva aparte como **validez estructural**, y **no coincide siempre con el estado**: una abstención JSON validada es `abstained` con `valid=True`, pero la falta de fragmentos recuperados es `abstained` con `valid=False`. Por eso las métricas se calculan sobre `status`, no sobre `valid`. El orquestador agrega el estado de varios especialistas con la precedencia answered > error > rejected > abstained. Un fallo de un especialista pasa a `error` sin tumbar la corrida (los índices ausentes, `FileNotFoundError`, siguen propagándose para que la API responda 503). `run_eval` reporta tasas por estado, F1 solo sobre `answered` y F1 sobre las preguntas que ambos sistemas responden sustantivamente. Los checkpoints llevan huella del dataset (preguntas, módulos y respuestas de referencia), del modo y de la **versión del esquema de resultados** (`RESULT_SCHEMA_VERSION`): un checkpoint de otra versión no se reanuda. **No existen resultados completos con estados todavía**: el próximo run debe usar un tag nuevo.

### Política de corpus (implementada en staging, sin activar)
- Alcance: capítulo 11 "Estándares y criterios de habilitación" (desde la pág. 59). El articulado y los capítulos 1–10 (trámites, autoevaluación, novedades) quedan fuera; las 17 preguntas `general` salen del alcance.
- Staging `artifacts/staging_v2/` (no versionado): **848 chunks** (talento_humano 159, infraestructura 201, dotación 135, medicamentos_dispositivos 74, procesos_prioritarios 148, historia_clínica 72, interdependencia 59), 38 servicios detectados (el 11.4.9 sin numeral por el OCR), sin ruido de encabezado. **La descripción de cada servicio (texto previo al primer "Estándar de …") no tiene módulo y queda fuera**: una versión anterior la asignaba a `talento_humano` por defecto (≈67 chunks espurios; error detectado en la preauditoría, caso v1-061). Un numeral por sí solo no demuestra el estándar, tampoco en los dos servicios cuyo primer encabezado se perdió (medicina nuclear, transporte asistencial). Índices vigentes (1.249 chunks) **sin tocar**.
- Impacto en el gold set (corpus de 848 chunks): de 105 preguntas específicas, 96 se emparejan con un chunk del corpus nuevo (39 cambian de módulo, 57 coinciden); 3 salen del corpus y 6 no tienen equivalente (tres de ellas eran descripciones de servicio). 20 preguntas nombran su módulo en el texto (detector estricto: solo cuenta "módulo/estándar/requisito de X"; mencionar el tema, p. ej. "dispositivos médicos", no es nombrar el módulo): las generó el prompt `MODULO: X`, así que reetiquetar sin reescribir crea contradicciones.
- **Activación pendiente**: se activan juntos corpus, índices, metadatos y gold v2 solo después de la auditoría humana.

### Hoja de auditoría del gold v2
`scripts/build_gold_v2_audit.py` → `eval/datasets/gold_v2_audit.json` (fuente de verdad) y `.csv` (vista para Excel). 122 registros: P1 obligatoria 67 (cambios de módulo, excepciones, contradicciones con el nombre de módulo, 17 `general` y toda decisión previa obsoleta sin decisión nueva), P2 recomendada 24, P3 sin banderas 31. **Esquema v2** (una sola decisión por registro): `conservar` | `corregir` | `retirar`. `corregir` admite correcciones **combinadas** (`new_module`, `rewritten_question`, `rewritten_answer`, `evidence_chunk_id_v2`; al menos una) y se valida contra el corpus candidato (el módulo final debe ser compatible con la evidencia: la explícita o, si no hay, el candidato mostrado en la hoja cuando el emparejamiento es confiable; si la pregunta nombra otro módulo exige reescribirla). **`conservar` solo se acepta en registros `modulo_coincide`** (etiqueta v1 igual al módulo del candidato, dentro del corpus); para mantener una etiqueta discutida hay que usar `corregir` con `new_module` y `evidence_chunk_id_v2` de un fragmento válido, y si no hay evidencia, `retirar`. Sin candidato fiable (`sin_correspondencia`, `fuera_de_corpus`, `general`), `corregir` exige `evidence_chunk_id_v2`. `review_level` = `preaudit` (propuesta técnica) | `author` | `expert`: author/expert exigen `reviewer` **y** `reviewed_at` (fecha ISO) y solo ellos cuentan para activar el gold v2; el campo por sí solo no constituye validación experta. `retirar` usa `retire_category` (fuera_de_alcance | sin_evidencia | pregunta_defectuosa | referencia_incorrecta | otro) para distinguir una decisión de alcance de "no existe en el anexo". `history` conserva cada cambio y la migración desde el esquema v1. El revisor edita el CSV y ejecuta `--import-csv`, que valida todo antes de guardar. **Atribución**: promover una propuesta `preaudit` a `author`/`expert`, o cambiarla, exige escribir **explícitamente un `reviewer` distinto del de la propuesta** (nunca se hereda; no autentica a la persona, solo evita una atribución errónea por herencia), además de nivel y fecha. **Reglas de edición del CSV**: una celda vacía conserva el valor guardado; `<borrar>` limpia una celda sin cambiar la decisión; **cambiar la decisión** (p. ej. rechazar una propuesta `corregir` con `retirar`) es una revisión nueva y completa: las celdas vacías se limpian, hay que registrar el propio `reviewer`/`review_level`/`reviewed_at` (si siguen siendo los de la propuesta anterior se rechaza) y la propuesta previa queda en `history` junto con los campos limpiados. Salvaguardas: regenerar nunca borra decisiones; una huella de **toda** la evidencia visible invalida la decisión previa si cambia (pasa a `stale_review` y a P1); un campo vacío del CSV no borra decisiones; IDs duplicados o CSV desactualizado se rechazan; `stale_review` solo se cierra con una decisión nueva y válida. `answer_lexical_overlap_v2` es solo solapamiento de palabras, **no** valida la respuesta. El gold v1 de 122 preguntas se conserva intacto.

**Preauditoría técnica (Codex, 2026-10-07) — estado tras la corrección de la ingesta y la migración a v2:** la preauditoría importó 66 propuestas (31 `retirar`, 14 `reetiquetar`, 1 `conservar` y 20 con observación pero sin decisión). Al corregir la ingesta cambió la evidencia de 43 registros: esas decisiones **no se descartaron**, quedaron en `stale_review` (y en `history`) marcadas para nueva revisión; **23 se conservan vigentes** (20 `retirar`, 3 `corregir`) y todas pasaron a `review_level=preaudit`, es decir, **propuestas técnicas, no revisión del autor ni de un experto**. Los 14 `reetiquetar` compartían un motivo plantilla y deben tratarse como propuestas sin verificación individual (v1-006, con margen de 0,05 entre candidatos, merece revisión caso a caso). Los `retirar` de los 3 `sin_correspondencia` no equivalen a decir que el contenido no exista en el anexo (v1-016 es una descripción de servicio, v1-082 una enumeración de grupo): es una decisión de alcance (siete estándares), a registrar con `retire_category=fuera_de_alcance`. **0 decisiones de nivel author/expert.** No equivale a validación experta ni autoriza activar el gold v2.

### Recall@k y MRR
Las funciones existen en `eval/metrics.py` pero **no se publican cifras** hasta validar qué fragmentos son evidencia relevante en el gold v2 (los `chunk_id` antiguos y los emparejamientos propuestos aún no son verdad de referencia).

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
| `citations.page` recibe lista en vez de int | `field_validator("page", mode="before")` en `Citation` (`guardrails.py`) — toma el primer elemento si el LLM devuelve `[n, m]` en lugar de `n`. |
| `checklist.numeral` recibe lista en vez de str | `field_validator("numeral", mode="before")` en `ChecklistItem` y `Citation` — une la lista con `", "` si el LLM devuelve `["20, 21", "23.1"]`. |
| JSON truncado en respuestas largas | `llm_max_tokens=2048` insuficiente para módulos como dotación (respuesta ~5000 chars, ~14 equipos + citas + checklist). Subido a 4096 en `core/config.py`. |
| Métricas de ruteo incorrectas en UI | `2_Evaluacion.py` mostraba `routing_accuracy_top1` (31.1%, todos n=122) en vez de `routing_accuracy_top1_specific` (36.2%, solo específicos n=105). Corregido. |

### Limitaciones actuales
1. **Contenido tabular**: el extractor PyMuPDF genera tablas como texto plano sin estructura. Las preguntas sobre tablas de requisitos pierden contexto.
2. **Secciones administrativas sin módulo**: numerales 1-10.x (condiciones generales, REPS) no tienen header de módulo → se asignan por carry-over.
3. **interdependencia pequeño**: solo 20 chunks → modelo con poca evidencia.
4. **Evaluación sin métricas semánticas**: solo F1 de tokens. Pendiente integrar BERTScore o similitud coseno para evaluar calidad de respuesta.
5. **procesos_prioritarios — límite del ruteo léxico-coseno**: routing top-1 de solo 12.5% con fallos dispersos en 6 de 7 módulos. El vocabulario de "procesos prioritarios" (urgencias, sepsis, parto, transfusión) se solapa semánticamente con casi todos los demás estándares. El enfoque coseno+léxico no es suficiente para discriminar este módulo sin un clasificador supervisado.
6. **talento_humano gold set muy reducido (7 entradas)**: la auditoría eliminó la mayoría de entradas contaminadas con contenido administrativo. Con solo 7 preguntas, cada fallo/acierto representa ~14pp — las estadísticas de routing de este módulo (28.6% top-1) no son estadísticamente representativas.
7. **`valid` no es cobertura**: ambos sistemas usan el mismo `validate_response()`, que acepta una abstención reconocida como válida. En v1–v4 el 83 % de las respuestas del mono eran abstenciones (índice desalineado). Reportar siempre por separado validez de formato, abstención y respuesta sustantiva.
8. **Brechas respecto al anteproyecto**: metas EM ≥ 0.60 / F1 ≥ 0.70 no alcanzadas (EM = 0 con respuestas libres); gold set "validado por expertos" aún sin validación experta (generado con LLM y auditado por el autor); Recall@10, MRR y faithfulness sin medir; sin OCR/visión en la ingesta; evaluación con usuarios (SUS, tiempos) sin realizar.

---

## 11. PRÓXIMOS PASOS

- [x] ~~Decidir estrategia para preguntas generales/admin~~ → Opción A implementada
- [x] ~~Implementar `module: "general"` en eval~~ → `routing_accuracy_top1_specific` en run_eval.py
- [x] ~~Ejecutar `scripts/add_general_questions.py`~~ → 30 preguntas generales agregadas
- [x] ~~Auditar las respuestas generadas en `ui/pages/3_Auditar_Goldset.py`~~ → gold set limpio: 122 entradas
- [x] ~~Correr evaluación completa sobre gold set final (122 ítems)~~ → completado 2026-03-01
- [x] ~~Analizar routing failures~~ → matriz de confusión generada; medicamentos identificado como módulo crítico
- [x] ~~Fix routing medicamentos_dispositivos~~ → eliminado `"dispositivo"` de dotacion keywords; enriquecida descripción semántica → +20pp top-1 en medicamentos, +3.8pp global
- [x] ~~Fix bug `citations.page`~~ → `field_validator("page", mode="before")` en `Citation.coerce_page()` toma el primer elemento si el LLM devuelve lista
- [x] ~~Fix bug `checklist.numeral`~~ → `field_validator("numeral", mode="before")` en `ChecklistItem` y `Citation` une lista con `", "`
- [x] ~~Correr run v4~~ → multi_valid=51.6%, multi_f1=0.276, routing top-1=36.2% — confirma estabilidad del sistema
- [x] ~~Fix `llm_max_tokens`~~ → 2048→4096 en `config.py`; resuelve JSON truncado en respuestas largas (dotación ~5000 chars)
- [x] ~~Fix métricas UI evaluación~~ → `2_Evaluacion.py` ahora muestra variantes `_specific` (36.2% / 47.6%)
- [x] ~~UX skeleton loader + timings~~ → shimmer CSS + step pills + panel tiempos por fase (routing_ms / agents_ms / total_ms)
- [x] **Política de corpus implementada en staging** (cap. 11 desde la pág. 59; módulo = línea "Estándar de X") — ver §8c
- [ ] **Activar la política de corpus** (índices + metadatos + gold v2 juntos) tras la auditoría humana del gold v2; remapear `chunk_id`
- [ ] **Revisar las 17 referencias `general`** (generadas con el índice defectuoso) y moverlas a un gold set aparte (fuera de alcance)
- [ ] **Cerrar la auditoría del gold v2**: el autor revisa la P1 (43 decisiones obsoletas por la corrección de la ingesta, 23 propuestas vigentes `preaudit`, y los registros aún sin decisión) con `review_level=author`, y se muestrean P2/P3, antes de activar corpus, índices y gold v2 juntos
- [x] Estados explícitos y métricas separadas por estado (§8c) — falta correr un run completo con ellos
- [ ] **Métricas restantes**: Recall@10 y MRR (cifras tras validar evidencia del gold v2), faithfulness, métrica semántica con juez de otra familia (candidato: `google/gemma-4-31b`, calibrado con muestra anotada a mano)
- [ ] **Ruteador supervisado** con partición train/test o validación cruzada
- [ ] **Validación experta** de una muestra del gold set y evaluación con usuarios (comprometidas en el anteproyecto)
- [ ] **procesos_prioritarios routing**: 12.5% top-1 — evaluar si mejorar descripción/keywords o documentar como limitación del enfoque léxico-coseno
- [ ] Documentar resultados para tesis (tablas comparativas, gráficas)
- [x] ~~Limpiar archivos huérfanos~~ → ya no existen en el árbol del proyecto

---

*Este documento se actualiza manualmente después de cada ciclo de implementación.*
