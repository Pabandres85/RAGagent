# Decisiones de Diseño — Sistema RAG Multi-Agente

> **Propósito**: Registrar las decisiones metodológicas y técnicas relevantes para la defensa de tesis.
> Cada entrada responde a la pregunta implícita del jurado: *"¿Por qué lo hiciste así y no de otra manera?"*
>
> Última actualización: 2026-03-01

---

## Índice

1. [Arquitectura — Multi-agente vs. mono-agente](#1-arquitectura--multi-agente-vs-mono-agente)
2. [Ruteo híbrido coseno + léxico](#2-ruteo-híbrido-coseno--léxico)
3. [Chunking: tamaño y solapamiento](#3-chunking-tamaño-y-solapamiento)
4. [Embeddings: elección del modelo](#4-embeddings-elección-del-modelo)
5. [FAISS IndexFlatIP con normalización L2](#5-faiss-indexflatip-con-normalización-l2)
6. [LLM local (LM Studio / Ollama)](#6-llm-local-lm-studio--ollama)
7. [llm_max_tokens = 4096](#7-llm_max_tokens--4096)
8. [Guardrails: por qué se exigen citas obligatorias](#8-guardrails-por-qué-se-exigen-citas-obligatorias)
9. [Gold set: exclusión de preguntas autoreferentes](#9-gold-set-exclusión-de-preguntas-autoreferentes)
10. [Gold set: módulo "general" excluido del routing accuracy](#10-gold-set-módulo-general-excluido-del-routing-accuracy)
11. [Métrica principal: F1 de tokens sobre Exact Match](#11-métrica-principal-f1-de-tokens-sobre-exact-match)
12. [Brecha valid rate: 52 % multi-agente vs. 95 % mono-agente](#12-brecha-valid-rate-52--multi-agente-vs-95--mono-agente)
13. [procesos_prioritarios: limitación documentada, no bug](#13-procesos_prioritarios-limitación-documentada-no-bug)
14. [Corpus restringido a Resolución 3100 de 2019](#14-corpus-restringido-a-resolución-3100-de-2019)
15. [Agente baseline mono-agente como control experimental](#15-agente-baseline-mono-agente-como-control-experimental)
16. [Orden de metadatos = orden de vectores FAISS](#16-orden-de-metadatos--orden-de-vectores-faiss)
17. [Sin streaming LLM en la entrega actual](#17-sin-streaming-llm-en-la-entrega-actual)

---

## 1. Arquitectura — Multi-agente vs. mono-agente

**Decisión**: El sistema usa 7 agentes especialistas + 1 orquestador, no un único agente generalista.

**Justificación**:
La Resolución 3100 de 2019 organiza los estándares de habilitación en estándares temáticos independientes (talento humano, infraestructura, dotación, etc.). Un índice único mezcla chunks de módulos distintos en el espacio vectorial, lo que introduce ruido: la pregunta "¿Cuántos metros cuadrados debe tener un quirófano?" recupera fragmentos de dotación y medicamentos que comparten vocabulario quirúrgico pero no responden la pregunta.

La especialización por módulo reduce el espacio de búsqueda (de 1.249 chunks globales a entre 20 y 719 por módulo; la distribución es muy desigual, ver `ESTADO_PROYECTO.md` §3), con la hipótesis de que mejora la precisión de recuperación.

> **⚠️ Actualización 2026-10-06 — hipótesis NO confirmada.** La evidencia que aquí se citaba (mono_f1=0.112 vs. multi_f1=0.276) provenía de un baseline defectuoso (ver §18). Tras corregirlo, con el ruteador actual el mono-agente obtiene F1 0.425 frente a 0.279 del multi-agente (corrida provisional, 122 ítems). Con ruteo oráculo de especialista único el multi-agente alcanza F1 0.409, sin diferencia demostrable frente al mono. Ver §18.

**Alternativas consideradas**:
- Un único agente con filtrado por metadatos en FAISS → viable pero introduce complejidad en el filtro y pierde la especialización semántica de los prompts.
- Fine-tuning del modelo de embeddings → fuera del alcance de tesis (requiere corpus anotado por expertos).

---

## 2. Ruteo híbrido coseno + léxico

**Decisión**: El orquestador combina similitud coseno entre la consulta y las descripciones de módulo con un bonus léxico por palabras clave de alta señal.

**Justificación**:
El embedding coseno captura semántica general pero puede fallar cuando el vocabulario del módulo es técnico y especializado. Por ejemplo, "farmacovigilancia" y "tecnovigilancia" son términos exclusivos de medicamentos_dispositivos que un modelo multilingüe genérico puede asociar indistintamente con dotación (ambos hablan de "equipos" y "seguimiento").

El bonus léxico actúa como señal de anclaje: si la pregunta contiene "farmacovigilancia", se añade un bonus de 0.12–0.24 al score de `medicamentos_dispositivos`, suficiente para desempatar scores cercanos sin dominar consultas ambiguas.

**Parámetros calibrados empíricamente**:
- Bonus mínimo por 1 hit: 0.12
- Bonus máximo: 0.24 (≥4 hits)
- Umbral de transversalidad: diferencia ≤ 0.08 entre top-1 y top-2

**Resultado**: El fix de routing en v2 (eliminar "dispositivo" de keywords de dotación + enriquecer descripción de medicamentos_dispositivos) produjo +20 pp en routing top-1 de medicamentos y +3.8 pp global.

---

## 3. Chunking: tamaño y solapamiento

**Decisión**: `CHUNK_SIZE=768` caracteres, `CHUNK_OVERLAP=128` caracteres.

**Justificación**:
- **768 caracteres**: alineado con la dimensión del modelo de embeddings (`paraphrase-multilingual-mpnet-base-v2`, 768 dims). Un chunk de ~150-200 tokens cabe completamente en la ventana del encoder (512 tokens) sin truncación. Chunks más grandes degradan la densidad semántica del vector resultante.
- **128 caracteres de overlap**: los numerales de la Resolución 3100 suelen tener encabezados cortos seguidos por el contenido normativo. Sin overlap, un chunk puede empezar en medio de una condición sin el numeral de referencia, perdiendo contexto para la cita.

**Trade-off**: chunks pequeños mejoran precisión de recuperación pero aumentan el índice (más vectores). Con 1.249 chunks globales la búsqueda FAISS es instantánea; el costo es tolerable.

---

## 4. Embeddings: elección del modelo

**Decisión**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dims, multilingüe).

**Justificación**:
- **Multilingüe con énfasis en español**: entrenado en 50+ idiomas con alta cobertura de texto en español. El corpus es 100 % español técnico-jurídico colombiano.
- **Dimensión 768**: equilibrio entre capacidad expresiva y velocidad de cómputo en CPU (sin GPU dedicada en el entorno de desarrollo).
- **Paraphrase**: entrenado para similitud semántica de paráfrasis, no solo NLI. Esto es relevante porque las preguntas del gold set son parafrases de los requisitos normativos.
- **Disponible localmente**: no requiere llamada a API externa; consistente con el requisito de privacidad/despliegue local.

**Alternativas descartadas**:
- `multilingual-e5-large`: mayor capacidad pero ~1.1 GB de modelo, más lento en CPU.
- `text-embedding-ada-002` (OpenAI): dependencia de API externa, costos por token, latencia de red.

---

## 5. FAISS IndexFlatIP con normalización L2

**Decisión**: `IndexFlatIP` (producto interno) con vectores L2-normalizados, equivalente a similitud coseno exacta.

**Justificación**:
- **Coseno sobre distancia euclidiana**: los vectores de embeddings tienen magnitudes variables dependiendo del contenido. La similitud coseno normaliza por magnitud, siendo más robusta para comparar textos de longitud heterogénea.
- **IndexFlatIP vs. IndexIVF**: con ~180 vectores por módulo, una búsqueda exacta (Flat) es instantánea (<1 ms). Los índices aproximados (IVF, HNSW) están justificados a partir de ~100 K vectores; en este caso añadirían complejidad sin beneficio de latencia.
- **Normalización en ingesta**: los vectores se normalizan una vez durante `ingest.py` y se persisten normalizados. Esto evita re-normalizar en cada búsqueda.

---

## 6. LLM local (LM Studio / Ollama)

**Decisión**: El sistema usa un LLM local (Qwen 2.5 7B Instruct vía LM Studio) con interfaz OpenAI-compatible.

**Justificación**:
- **Privacidad y soberanía de datos**: los documentos normativos de habilitación pueden contener contexto institucional sensible. Un LLM local evita enviar datos a servidores externos.
- **Reproducibilidad**: la versión del modelo está fijada en la configuración. Los resultados del gold set son reproducibles sin depender de actualizaciones de API.
- **Costo**: sin costo por token durante el desarrollo y la evaluación (105 preguntas × múltiples runs).
- **Abstracción dual-provider**: la capa `core/llm_client.py` expone una interfaz uniforme para `lmstudio` y `ollama`, permitiendo cambiar de proveedor con una variable de entorno (`LLM_PROVIDER`).

**Trade-off aceptado**: la calidad del modelo local (7B) es inferior a GPT-4 o Claude. Para una tesis de maestría, la comparación se establece entre arquitecturas (multi-agente vs. mono-agente) bajo las mismas condiciones de LLM, no entre modelos.

---

## 7. llm_max_tokens = 4096

**Decisión**: El límite de tokens de salida del LLM se fijó en 4.096 (subido de 2.048 en v6).

**Justificación**:
El módulo `dotacion` requiere listar hasta ~14 equipos biomédicos con sus numerales, más un checklist de verificación y las citas normativas correspondientes. En formato JSON estructurado (requerido por los guardrails), esto genera respuestas de ~5.000–5.500 caracteres, equivalentes a ~1.400–1.600 tokens en Qwen 2.5.

Con `llm_max_tokens=2048` el modelo truncaba la respuesta a mitad del JSON, causando el error `"Expecting ',' delimiter: line 161 column 6 (char 5365)"` al deserializar. El aumento a 4.096 resuelve el problema con margen para los módulos más verbosos.

**Por qué no usar 8.192**: la ventana de contexto del modelo debe acomodar tanto el prompt de sistema + contexto RAG (~2.000–3.000 tokens) como la respuesta. Un límite de 4.096 para la respuesta es conservador dentro de la ventana de 32 K del modelo.

---

## 8. Guardrails: por qué se exigen citas obligatorias

**Decisión**: El schema `AgentResponse` requiere al menos una `Citation` con `text`, `numeral`, `page` y `resolution`. Las respuestas sin citas son rechazadas por los guardrails.

**Justificación**:
En un sistema normativo, una respuesta no trazable a un numeral específico es inutilizable para la habilitación. El auditor de salud necesita verificar cada requisito contra la resolución. Sin citas obligatorias, el LLM puede generar contenido plausible pero incorrecto ("alucinación normativa").

La exigencia de citas tiene el efecto secundario de forzar al LLM a anclar sus afirmaciones en los chunks recuperados, reduciendo la divergencia respecto al corpus.

**Implementación**: `field_validator` con `mode="before"` tolera formatos irregulares del LLM (listas en lugar de escalares), evitando rechazos espurios por variación de formato.

---

## 9. Gold set: exclusión de preguntas autoreferentes

**Decisión**: Se eliminaron 17 preguntas del gold set generado que preguntaban sobre el sistema mismo (e.g., "¿Qué es la Resolución 3100?", "¿Cuál es el objetivo de este módulo?").

**Justificación**:
Estas preguntas no evalúan la capacidad de recuperación normativa; evalúan si el LLM puede describir el documento fuente. Como el LLM las responde trivialmente desde su conocimiento paramétrico (sin necesidad de RAG), incluirlas inflaría artificialmente las métricas de valid rate y F1.

El objetivo del gold set es medir la calidad de la recuperación RAG y la fidelidad de las respuestas a los requisitos específicos de la resolución. Las preguntas autoreferentes no discriminan entre una arquitectura RAG funcional y un LLM genérico.

---

## 10. Gold set: módulo "general" excluido del routing accuracy

**Decisión**: Las 17 preguntas con `module="general"` se incluyen en el gold set (para medir valid rate y F1) pero se excluyen del cálculo de `routing_accuracy_top1`.

**Justificación**:
Las preguntas generales (e.g., "¿Cuál es el proceso para habilitarse?") no tienen un módulo canónico correcto. El orquestador las ruta a cualquier módulo con alta confianza y la respuesta puede ser válida independientemente del módulo seleccionado.

Incluirlas en el routing accuracy crearía ruido: ningún módulo es incorrecto, pero tampoco hay un "correcto" que medir. La métrica representativa para la tesis es `routing_accuracy_top1_specific` (n=105 preguntas específicas), no `routing_accuracy_top1` (n=122 totales).

**Nota de transparencia**: ambas métricas se reportan en la evaluación (36.2 % y 31.1 % respectivamente) para evitar sesgo de selección.

---

## 11. Métrica principal: F1 de tokens sobre Exact Match

**Decisión**: Se usa F1 de solapamiento de tokens (bag-of-words) como métrica principal de calidad de respuesta, no Exact Match (EM).

**Justificación**:
- **EM es demasiado estricto para texto normativo**: una respuesta correcta puede parafrasear el numeral ("se deben tener al menos 2 médicos" vs. "mínimo 2 profesionales médicos") sin coincidir exactamente con la respuesta de referencia.
- **F1 captura solapamiento semántico parcial**: mide qué fracción de tokens relevantes aparece en la respuesta generada, independientemente del orden.
- **Limitación reconocida**: F1 de tokens no distingue entre sinónimos ni detecta paráfrasis semánticamente equivalentes. BERTScore o similitud coseno serían más apropiados semánticamente, pero requieren un modelo adicional de evaluación. Esta limitación se documenta explícitamente en el capítulo de resultados.

---

## 12. Brecha valid rate: 52 % multi-agente vs. 95 % mono-agente

> **⚠️ Decisión REVISADA (2026-10-06).** La interpretación original ("el multi-agente es más riguroso") no se sostiene. Dos hechos: (a) `valid` no mide cobertura: ambos sistemas usan el mismo `validate_response()`, que acepta una abstención como `valid=True`; en el run antiguo el 83 % de las respuestas del mono eran abstenciones causadas por el índice desalineado (§18); (b) en el multi-agente, 54 de las 57 respuestas "inválidas" del run provisional son **abstenciones** (el orquestador devuelve "No se encontró evidencia normativa suficiente" cuando el especialista no recupera evidencia, típicamente por ruteo incorrecto), no rechazos por guardrails. En adelante se reportan por separado: *validez de formato*, *abstención* y *respuesta sustantiva*. El texto original se conserva abajo como registro histórico.

**Decisión (original)**: La brecha se interpreta como evidencia de rigor del multi-agente, no como fallo.

**Justificación**:
El mono-agente tiene un `valid rate ~95 %` estructural porque su prompt acepta cualquier respuesta, incluido el fallback "No se encontró evidencia suficiente". Este fallback cuenta como válido estructuralmente pero genera F1 ≈ 0 (no responde la pregunta).

El multi-agente exige citas obligatorias, schema JSON estricto y coherencia entre la respuesta y el contexto recuperado. Un ~48 % de rechazos indica preguntas donde el contexto recuperado no es suficiente para fundamentar una respuesta trazable — lo cual es una señal de calidad, no de fallo.

**Argumento de tesis**: el multi-agente prioriza precision (respuestas confiables cuando responde) sobre recall (responder siempre). La brecha en valid rate y la ventaja en F1 (0.276 vs. 0.112) son consistentes con esta elección de diseño.

---

## 13. procesos_prioritarios: limitación documentada, no bug

**Decisión**: El módulo `procesos_prioritarios` tiene routing top-1 de 12.5 %. Se documenta como limitación estructural del enfoque léxico-coseno, no como error a corregir.

**Justificación**:
El vocabulario de "procesos prioritarios" (urgencias, sepsis, parto, transfusión, triage) se solapa semánticamente con casi todos los demás estándares. "Urgencias" aparece en dotación, infraestructura, medicamentos y talento humano. "Parto" aparece en infraestructura y dotación. El embedding coseno no puede discriminar el módulo correcto cuando los chunks de múltiples módulos son igualmente relevantes para la consulta.

Las alternativas para mejorar este módulo —clasificador supervisado, re-ranking por módulo— requieren anotaciones adicionales o un corpus de entrenamiento especializado, fuera del alcance de esta tesis.

**Valor como limitación documentada**: identifica una dirección concreta de trabajo futuro y demuestra comprensión de los límites del enfoque, lo cual es valorado en una defensa de tesis.

---

## 14. Corpus restringido a Resolución 3100 de 2019

**Decisión**: El corpus del sistema incluye únicamente la Resolución 3100 de 2019 y sus anexos técnicos. No se incorporan resoluciones relacionadas, circulares del Ministerio de Salud ni jurisprudencia.

**Justificación**:
- **Alcance acotado = evaluación válida**: ampliar el corpus sin ampliar el gold set haría imposible medir la calidad de recuperación sobre el nuevo material.
- **Vigencia jurídica**: la Resolución 3100 de 2019 es el instrumento normativo vigente para habilitación. Las resoluciones anteriores (2003, 2014) fueron derogadas.
- **Foco de tesis**: el objetivo es demostrar la viabilidad del enfoque multi-agente RAG, no construir un sistema exhaustivo de consulta normativa. El alcance acotado es una decisión metodológica, no una limitación técnica.

---

## 15. Agente baseline mono-agente como control experimental

**Decisión**: Se mantiene un `BaselineMonoAgent` con índice global como control experimental para comparación.

**Justificación**:
Sin un baseline, no es posible cuantificar el valor de la especialización. El mono-agente usa exactamente el mismo LLM, el mismo modelo de embeddings y los mismos parámetros de búsqueda, variando únicamente el índice (global vs. por módulo) y el prompt (genérico vs. especializado).

Esta comparación aísla el efecto de la arquitectura multi-agente, que es la variable independiente de la tesis.

**Resultado (original, INVALIDADO)**: multi_f1=0.276 vs. mono_f1=0.112 (+146 % relativo) en el gold set de 122 ítems. Ese baseline estaba roto (§18). Resultado actual, provisional: ver §18.

---

## 16. Orden de metadatos = orden de vectores FAISS

**Decisión**: El `MetadataStore` mantiene los metadatos de chunks en el mismo orden en que fueron añadidos al índice FAISS. La búsqueda retorna índices FAISS que se usan directamente para acceder a `metadata[idx]`.

**Justificación**:
FAISS `IndexFlatIP.search()` retorna índices enteros (posición en el índice). Si los metadatos no están en el mismo orden que los vectores, el sistema asigna citas incorrectas a los fragmentos recuperados.

Esta invariante se garantiza en `ingest.py` para los **índices por módulo**: los chunks se procesan en orden, se añaden al índice FAISS con `index.add()` y se persisten en el mismo orden.

> **⚠️ Para el índice GLOBAL la invariante estaba rota hasta 2026-10-06.** `global.faiss` se construía en orden de página, pero el retriever global cargaba los metadatos agrupados por módulo (`MetadataStore.load_all()`); solo ~8 % de las posiciones coincidían. Corrección: el índice global se construye concatenando los bloques de cada módulo en el orden de `MODULES` (`ingest.py`, y `--rebuild-global` para reconstruirlo sin re-embeber, con validación previa de conteos). La invariante ahora se verifica con `tests/test_index_alignment.py`, que compara vector por vector, no solo totales.

---

## 17. Sin streaming LLM en la entrega actual

**Decisión**: La generación LLM no usa streaming; el sistema espera la respuesta completa antes de mostrarla.

**Justificación**:
Los guardrails requieren deserializar la respuesta completa como JSON antes de validarla. Con streaming, el JSON no está disponible hasta que el LLM termina de generarlo, por lo que el streaming no reduciría la latencia percibida de forma significativa en la arquitectura actual.

La UX mitiga la espera con un skeleton loader y pills de paso que se renderizan antes de la llamada bloqueante, dando feedback visual inmediato.

**Trabajo futuro**: el streaming es viable si se restructura el schema de respuesta para serializar respuesta parcial primero (answer) y metadatos (citations, checklist) al final, permitiendo mostrar el texto mientras se validan las citas.

---

## 18. Corrección del baseline y diagnóstico de ruteo (2026-10-06)

**Qué se encontró**: una auditoría independiente (dos revisiones cruzadas, verificadas contra los artefactos) halló dos defectos en el baseline mono-agente que invalidaban la comparación original:

1. **Índice global desalineado** con sus metadatos (ver §16).
2. **Prompt del mono con llaves dobles literales** (`{{ }}`): se enviaba sin pasar por `.format()`, por lo que el modelo copiaba `{{` dentro de su JSON y producía JSON malformado.

**Ablación sobre las primeras 20 preguntas del gold set** (no aleatorias; es un diagnóstico del arreglo, no un resultado comparativo):

| Estado del mono | Abstenciones | Sustantivas válidas | F1 |
|---|---|---|---|
| Original (índice cruzado + prompt `{{ }}`) | 19/20 | 1/20 | 0.101 |
| Índice alineado | 2/20 | 12/20 | 0.409 |
| + prompt corregido | 2/20 | 18/20 | 0.588 |

**Corrida completa provisional** (122 ítems, `--tag provisional_post_fix`; provisional porque el gold set y el corpus aún no están depurados):

| | Mono | Multi (ruteador real) |
|---|---|---|
| Respuestas sustantivas válidas | 89 (73 %) | 64 (52 %) |
| F1 todas las preguntas | 0.425 | 0.279 |
| F1 solo 105 específicas | 0.433 | 0.284 |
| F1 en las 52 respondidas sustantivamente por ambos | 0.513 | 0.438 |

**Diagnóstico con ruteo oráculo de especialista único** (105 preguntas específicas; el especialista recibe solo la etiqueta de módulo, nunca la respuesta de referencia; `--oracle-routing`): 84 respuestas sustantivas (80 %), F1 0.409 (0.488 sobre las sustantivas). Diferencias de F1 por pregunta, emparejadas, IC 95 % por bootstrap:

| Comparación | Diferencia | IC 95 % |
|---|---|---|
| Oráculo − mono | −0.024 | [−0.078, +0.030] |
| Ruteador − mono | −0.150 | [−0.216, −0.089] |
| Oráculo − ruteador | +0.125 | [+0.082, +0.172] |

**Cómo leer esto (y cómo NO)**:
- El oráculo fuerza **un único especialista**: la mejora combina módulo correcto y eliminación del ruteo transversal. La mejora se concentra en las preguntas donde el ruteador falló (un módulo incorrecto: +0.252; transversal con top-1 incorrecto: +0.134) y es nula donde acertó, lo que apoya, sin demostrarlo causalmente, que el acierto del módulo es el factor decisivo.
- "Sin diferencia demostrable" **no** es "equivalente": el intervalo oráculo−mono no descarta una desventaja moderada. Afirmar equivalencia exigiría fijar un margen a priori y probarlo.
- El oráculo es un **techo optimista**: cada pregunta se generó desde un chunk de su módulo, de modo que el especialista siempre busca en el índice que contiene la respuesta.
- F1 de tokens penaliza la verbosidad (multi: ~57 palabras; mono: ~31; referencia: ~33). No se debe llamar "invención" a las respuestas de casos mal ruteados: F1 y presencia de citas no verifican que la afirmación esté sustentada por el fragmento.
- Resultado válido para la tesis: *con el ruteador actual (36 % top-1) el multi-agente es peor que el baseline; con ruteo perfecto de especialista único no se demuestra diferencia en F1, y sí mayor cobertura (80 % vs 70 % de respuestas sustantivas, sin intervalo calculado)*. **No** se sostiene que la especialización mejore la calidad de las respuestas.

**Pendiente antes de cualquier resultado final**: depurar corpus y etiquetas (el filtro `page >= 36` documentado nunca existió en el código y, además, el cuerpo administrativo llega hasta la pág. 58; el capítulo normativo "11. Estándares y criterios de habilitación" empieza en la pág. 59), revisar las 17 referencias `general` (se generaron con el índice defectuoso), separar abstención / validez / calidad en las métricas y añadir Recall@k, MRR, faithfulness y una métrica semántica.

---

*Este documento es parte del repositorio de tesis. Mantener actualizado ante cambios de diseño.*
