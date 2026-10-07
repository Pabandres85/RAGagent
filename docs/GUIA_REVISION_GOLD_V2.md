# Guía de revisión del gold set v2 (para el autor)

> Documento de trabajo. Complementa `ESTADO_PROYECTO.md` (§8b y §8c) y `docs/decisions.md` (§18).
> Última actualización: 2026-10-07.

---

## 1. Por qué existe esta revisión (contexto en 2 minutos)

1. **El resultado "multi-agente 2,5× mejor que el mono-agente" era un artefacto.** El índice FAISS global estaba desalineado con sus metadatos y el prompt del mono tenía llaves dobles. Corregido eso, con el ruteador actual el mono rinde más (F1 0,425 vs 0,279, corrida provisional); con ruteo perfecto el multi iguala al mono en F1 y responde más preguntas.
2. **Las etiquetas de módulo del gold set eran ruidosas.** El corpus original asignaba módulos por menciones sueltas en la página y arrastraba el módulo anterior. Un corpus corregido (solo el capítulo 11, módulo por encabezado "Estándar de X") cambia el módulo de **39 de 96 preguntas emparejables**.
3. **Las preguntas se generaron con el nombre del módulo viejo escrito en el prompt** (`MODULO: X`): 20 preguntas lo nombran en su texto. Reetiquetar sin reescribir crea contradicciones.
4. **Por eso hace falta una auditoría pregunta por pregunta, y la tienes que hacer tú** (como autor). Una preauditoría técnica de otro agente dejó *propuestas*, pero no son una revisión tuya ni de un experto y no cuentan para activar nada.

**Lo que se activa después de tu revisión, todo junto:** corpus nuevo (848 fragmentos), índices, metadatos y gold v2. Hasta entonces nada de eso está activo: los índices vigentes (1.249 fragmentos) y el gold v1 (122 preguntas) siguen intactos.

---

## 2. Qué hay hecho y qué falta

| Área | Estado |
|---|---|
| Índice global alineado con metadatos + prompt mono corregido | ✅ hecho y con tests |
| Estados de respuesta (`answered`/`abstained`/`rejected`/`error`) y métricas separadas | ✅ implementado; **sin run completo con ellos todavía** |
| Corpus nuevo (cap. 11 desde pág. 59) | ✅ en `artifacts/staging_v2/` (848 fragmentos). **No activado** |
| Hoja de auditoría del gold v2 (`eval/datasets/gold_v2_audit.csv/.json`) | ✅ generada, con validaciones |
| **Tu revisión de la hoja** | ⏳ **pendiente — es lo que sigue** |
| Aplicar decisiones y construir el gold v2 | ⏳ lo hago yo cuando termines |
| Activar corpus + índices + gold v2 | ⏳ con tu visto bueno |
| Run completo con tag nuevo (estados, IC, oráculo) | ⏳ después |
| Recall@k / MRR | ⏳ código listo; cifras tras validar el gold v2 |
| Juez semántico (candidato `google/gemma-4-31b`, calibrar con ~20–30 respuestas tuyas) | ⏳ |
| Validación **experta** de una muestra (prometida en el anteproyecto) | ⏳ **no es esta revisión** |
| Evaluación con usuarios (SUS, tiempos) | ⏳ pendiente (anteproyecto) |
| Ruteador supervisado (train/test) | ⏳ después del run limpio |

**Sin commit aún:** ingesta, estados, hoja de auditoría, tests y documentación (ver `git status`).

---

## 3. Qué es cada cosa en la hoja

Archivos: `eval/datasets/gold_v2_audit.csv` (para Excel) y `gold_v2_audit.json` (**fuente de verdad**; el CSV es solo una vista).

### Cuántas filas y de qué tipo

| Prioridad | Filas | Qué es |
|---|---|---|
| **P1 — obligatoria** | **67** | Debes decidir cada una |
| P2 — recomendada | 24 | Señales débiles (12 emparejamiento ambiguo, 9 numerales ausentes, 9 solapamiento bajo): revísalas |
| P3 — sin banderas | 31 | Basta una muestra (≈10) |

Desglose de la P1:

| Estado | Filas | Situación actual |
|---|---|---|
| `general_fuera_de_alcance` | 17 | Propuesta preaudit vigente: `retirar` |
| `fuera_de_corpus` | 3 | Propuesta preaudit vigente: `retirar` |
| `modulo_cambia` | 39 | 3 con propuesta `corregir` vigente; **36 obsoletas** (la evidencia cambió; ver `stale_review`) |
| `sin_correspondencia` | 6 | Obsoletas (ver `stale_review`) |
| `modulo_coincide` con bandera | 2 | 1 obsoleta y 1 sin decisión |

> Una decisión **obsoleta** (`stale_review`) no vale: se invalidó porque cambió el fragmento o el módulo propuesto. La columna `stale_review` muestra lo que se había propuesto para que lo tengas de referencia.

### Columnas (en el CSV)

**De lectura (no las edites):**
- `audit_id` (v1-001 … v1-122), `priority`, `status`, `flags` (por qué requiere atención).
- `module_v1` = etiqueta del gold v1 · `module_proposed` = módulo del fragmento candidato en el corpus nuevo.
- `question`, `reference_answer`.
- `page_v1/page_v2`, `service_v2` (servicio del fragmento), `numeral_v2`.
- `chunk_text_v2` = texto del candidato (recortado a 500 caracteres; el texto completo está en el JSON o con `find_chunk.py`).
- `match_score`, `match_margin` (diferencia con el 2.º candidato), `second_best_*` (el 2.º candidato completo: módulo, página, servicio, texto).
- `answer_lexical_overlap_v2` = qué fracción de las palabras de la respuesta aparece en el fragmento. **Es solo una señal; no valida la respuesta.**
- `evidence_hash` (no tocar: si lo alteras el importador rechaza la fila).

**Las que completas tú (al final):**
`decision`, `decision_reason`, `new_module`, `rewritten_question`, `rewritten_answer`, `evidence_chunk_id_v2`, `retire_category`, `review_level`, `reviewer`, `reviewed_at`.

### Estados (`status`)
- `modulo_coincide`: el candidato está en el mismo módulo que la etiqueta v1.
- `modulo_cambia`: el candidato está en otro módulo.
- `sin_correspondencia`: no se halló fragmento equivalente en el corpus nuevo.
- `fuera_de_corpus`: la fuente original está en las págs. 1–58 (articulado/trámites).
- `general_fuera_de_alcance`: pregunta `general` (admin/REPS/visitas), no es de los 7 estándares.

---

## 4. Cómo revisar cada fila (el procedimiento)

Para cada fila, en este orden:

1. **Lee `question` y `reference_answer`.** ¿La pregunta es específica y se puede contestar con un requisito concreto de un estándar? ¿Dice "según el fragmento", es genérica, o nombra un módulo ("del módulo de Dotación")?
2. **Lee `chunk_text_v2` y `service_v2`/`page_v2`.** ¿El fragmento candidato responde la pregunta? Si dudas, abre el PDF en la página `page_v2` (`data/raw/resolucion-3100-de-2019.pdf`).
3. **Comprueba el numeral.** Si la pregunta o la respuesta citan un numeral (p. ej. 43.1), ¿está en el fragmento? Si no, ¿está en el `second_best` o en otro fragmento? (ver §6).
4. **Comprueba el módulo.** El módulo correcto es el del encabezado "Estándar de X" bajo el que está el requisito dentro de su servicio, no el tema de la pregunta (p. ej. "registros" bajo "Estándar de historia clínica", aunque la pregunta hable de dotación).
5. **Comprueba la respuesta de referencia.** ¿Está respaldada literalmente por el fragmento? ¿Cita el numeral correcto?
6. **Decide** (§5) y escribe en las columnas de revisión.

> Atención a los casos críticos señalados en las revisiones: **v1-006** (margen de solo 0,05 entre candidatos), **v1-087** (la respuesta es del numeral 43 pero el mejor candidato muestra el 41: mira el `second_best` o busca el 43.1), **v1-045** (decisión previa obsoleta), **v1-061** (la descripción de un servicio había quedado etiquetada como talento humano).

---

## 5. Qué decisión tomar

Solo existen **tres decisiones**: `conservar`, `corregir`, `retirar`.

### `conservar`
Solo para filas `modulo_coincide`. Significa: el fragmento candidato responde la pregunta, el módulo es correcto, la pregunta no nombra otro módulo y la respuesta está respaldada. **No se acepta en ningún otro estado.**

### `corregir`
La pregunta **vale la pena** pero algo debe arreglarse. Rellena **al menos una** de:

| Columna | Cuándo |
|---|---|
| `new_module` | La etiqueta v1 no es la del estándar real del fragmento (normalmente = `module_proposed`) |
| `rewritten_question` | La pregunta nombra el módulo viejo ("módulo de Dotación"), es autorreferente o ambigua |
| `rewritten_answer` | La respuesta está mal, incompleta o cita otro numeral |
| `evidence_chunk_id_v2` | El mejor candidato **no** es el fragmento correcto: pon el `chunk_id` del que sí lo es (p. ej. el `second_best_chunk_id_v2` o uno hallado con `find_chunk.py`) |

Combinaciones permitidas (p. ej. reetiquetar **y** reescribir la pregunta **y** corregir la respuesta). Reglas:
- Si `new_module` ≠ módulo del candidato mostrado, **debes** dar `evidence_chunk_id_v2` de un fragmento de ese módulo.
- Si la pregunta nombra un módulo distinto al final, **debes** reescribirla.
- En `sin_correspondencia` / `fuera_de_corpus` / `general`, `corregir` **exige** `evidence_chunk_id_v2`.
- Para **mantener** una etiqueta discutida: `corregir` con `new_module` = la etiqueta v1 y un `evidence_chunk_id_v2` de un fragmento de ese módulo.

### `retirar`
La pregunta no se puede sostener con el corpus nuevo. Escribe `decision_reason` (obligatorio) y `retire_category`:

| `retire_category` | Cuándo |
|---|---|
| `fuera_de_alcance` | Es de los capítulos 1–10 / administrativa / grupo / descripción de servicio: **no es un criterio de los siete estándares** (es una decisión de alcance, no afirma que "no exista" en la norma) |
| `sin_evidencia` | No hay en el corpus nuevo un fragmento que la respalde |
| `pregunta_defectuosa` | Ambigua, autorreferente, genérica |
| `referencia_incorrecta` | La respuesta está mal y no se puede arreglar sin rehacer la pregunta |
| `otro` | Explícalo en el motivo |

### Qué hacer con los grupos grandes
- **17 `general`:** propuesta vigente `retirar`. Lo razonable es confirmar con `retire_category=fuera_de_alcance` (se conservarán aparte como un conjunto "administrativo", no se pierden).
- **3 `fuera_de_corpus`:** idem, `fuera_de_alcance`.
- **6 `sin_correspondencia`:** decide una a una (hay dos distintas: v1-016 es descripción de servicio y v1-082 una enumeración de grupo → `fuera_de_alcance`; otras pueden ser `sin_evidencia`).
- **36 `modulo_cambia` obsoletas:** son las que más tiempo llevan. Mira `stale_review` (lo que se proponía antes) solo como pista.

---

## 6. Cómo editar el CSV (reglas exactas)

Abre `eval/datasets/gold_v2_audit.csv` en Excel (separador `;`, UTF-8). **Guarda siempre como CSV UTF-8 con el mismo separador.**

Columnas que rellenas, **siempre** para cualquier decisión tuya:

| Columna | Qué poner |
|---|---|
| `decision` | `conservar` / `corregir` / `retirar` |
| `decision_reason` | Breve y específico (obligatorio en `retirar`; recomendado siempre) |
| `review_level` | `author` |
| `reviewer` | Tu nombre (en filas con propuesta preaudit **sobrescribe** el nombre que aparece: no puede quedar el de la propuesta) |
| `reviewed_at` | Fecha ISO, p. ej. `2026-10-08` (no puede estar en el futuro) |

Y según la decisión: `new_module`, `rewritten_question`, `rewritten_answer`, `evidence_chunk_id_v2`, `retire_category`.

**Reglas de edición:**
- **Celda vacía = conserva el valor guardado** (protege contra borrados accidentales).
- **Escribe `<borrar>`** en una celda para vaciarla sin cambiar la decisión.
- **Cambiar la decisión** (p. ej. rechazar una propuesta `corregir` con `retirar`) es una revisión **nueva y completa**: las celdas vacías se limpian, y **debes registrar tus propios** `review_level`/`reviewer`/`reviewed_at`. Sobre una propuesta preaudit, el `reviewer` **debe ser tu nombre escrito explícitamente** (distinto del de la propuesta); si dejas el de la propuesta, se rechaza aunque cambies el nivel o la fecha. Las celdas que dejes con valor de la propuesta anterior (p. ej. `new_module`) se validan: si no aplican a tu nueva decisión, bórralas (`<borrar>` o déjalas vacías al cambiar).
- **Confirmar una propuesta preaudit** (estás de acuerdo): deja `decision` igual y rellena `review_level=author`, `reviewed_at` (y `retire_category` si es `retirar`) y **escribe TU nombre en `reviewer`, sobrescribiendo el de la propuesta** ("Codex (preauditoria tecnica)"). Si lo dejas vacío o igual al de la propuesta, se rechaza: la atribución nunca se hereda.
- **No toques** `evidence_hash`, `audit_id` ni las columnas de lectura.
- Una fila que no tocas queda como estaba. Puedes importar **en tandas** (por ejemplo, 20 filas por sesión).

### Cómo buscar un fragmento (para `evidence_chunk_id_v2`)
```bash
.venv/Scripts/python.exe scripts/find_chunk.py --page 150
.venv/Scripts/python.exe scripts/find_chunk.py --page 150 --module dotacion
.venv/Scripts/python.exe scripts/find_chunk.py --numeral 43.1 --service "Cuidado Intermedio"
.venv/Scripts/python.exe scripts/find_chunk.py --text "tubos endotraqueales"
.venv/Scripts/python.exe scripts/find_chunk.py --id 3f2a9c1b7d44      # fragmento completo
```
Imprime `chunk_id | módulo | página | numeral | servicio` y el texto. Copia el `chunk_id` a `evidence_chunk_id_v2`.

---

## 7. Importar tus decisiones

```bash
.venv/Scripts/python.exe scripts/build_gold_v2_audit.py --import-csv eval/datasets/gold_v2_audit.csv
```
- **Valida todo antes de guardar.** Si **una sola** fila tiene error, **no guarda nada** y te lista **todas** las filas con error y el motivo. Corrige y vuelve a importar.
- Guarda en el JSON (fuente de verdad), con copia de seguridad (`.bak-FECHA`, ignorada por git) y registra el cambio en `history`.
- Mensajes típicos: `falta review_level`, `exige reviewed_at`, `'corregir' requiere al menos uno de …`, `solo se puede 'conservar' un registro con estado 'modulo_coincide'`, `el candidato mostrado pertenece a 'X', incompatible …`, `la pregunta nombra [...] … reescribe la pregunta`, `retirar requiere decision_reason`.

Después de importar, **regenera el CSV** para ver el estado actualizado (no pierde nada):
```bash
.venv/Scripts/python.exe scripts/build_gold_v2_audit.py
```
Si el CSV tiene decisiones sin importar, el generador **se niega a sobrescribirlo** (así no pierdes trabajo).

Para ver tu avance: la salida del generador imprime cuántas decisiones hay por nivel (`author`, `preaudit`, sin decisión).

---

## 8. Orden de trabajo sugerido (≈ 4–6 h en 2–3 sesiones)

1. **Sesión 1 — lo mecánico (≈ 1 h):** confirmar las 20 propuestas de alcance (17 `general` + 3 `fuera_de_corpus`): `decision=retirar` ya está; rellena `review_level=author`, `reviewer`, `reviewed_at`, `retire_category=fuera_de_alcance`. Importa.
2. **Sesión 2 — `sin_correspondencia` (6) y los casos críticos (v1-006, v1-087, v1-045, v1-061):** una a una, con el PDF abierto. Importa.
3. **Sesión 3 — las 39 `modulo_cambia`:** filtra por `status=modulo_cambia`; ordena por `flags` (las 7 con "posible contradicción" y las 10 de margen < 0,1 primero). Importa cada 10–15 filas.
4. **P2 (24 filas):** léelas todas; la mayoría se resolverá con `conservar` o una corrección pequeña.
5. **P3 (31 filas):** muestrea ~10; si encuentras errores en la muestra, amplía.

**Criterio de parada:** todas las P1 con `review_level=author` y las P2 revisadas.

---

## 9. Qué NO hacer

- ❌ No tomes `answer_lexical_overlap_v2` ni `match_score` como aprobación: son señales.
- ❌ No marques `expert`: eso es para un experto externo del sector salud que valide una muestra (compromiso del anteproyecto).
- ❌ No edites el JSON a mano ni borres los `.bak`.
- ❌ No reemplaces `data/metadata/` ni `artifacts/faiss/` (se activan juntos, después, con un solo paso controlado).
- ❌ No conserves una etiqueta "porque así estaba": o hay un fragmento que la respalda, o se corrige/retira.

---

## 10. Qué pasa cuando termines

1. Me avisas. Aplico tus decisiones (`conservar` / `corregir` / `retirar`) y construyo el **gold v2**: preguntas con su módulo, pregunta/respuesta final y `chunk_id` de evidencia validado; las retiradas por alcance pasan a un conjunto aparte; las filas P3 no revisadas individualmente se marcan como "no auditadas".
2. Te propongo las **cifras reales** del gold v2 (cuántas preguntas por módulo; hoy el rango es de ~71 a ~91 específicas según resuelvas los pendientes, frente a las 105 actuales) y cómo presentar el cambio de **122 → N** respecto al anteproyecto (conservando el v1 intacto para trazabilidad).
3. Con tu visto bueno activo **corpus + índices + metadatos + gold v2 juntos** y relanzo la evaluación completa (tag nuevo): estados de respuesta separados, F1 solo sobre respuestas, IC con bootstrap, ruteo oráculo, y después Recall@k/MRR sobre la evidencia validada.

## 11. Decisiones que te tocan a ti (además de la revisión)

1. **Presentar el cambio de tamaño del gold set** (122 del anteproyecto → v2 depurado) a tu director: conviene hablarlo antes de la defensa.
2. **Validación experta:** ¿quién revisará una muestra (30–40 preguntas) y cuándo? Sin eso no se puede decir "validado por expertos".
3. **Metas EM ≥ 0,60 / F1 ≥ 0,70 del anteproyecto:** con respuestas libres el EM no es alcanzable; hay que justificar el cambio a métricas semánticas.
4. **Juez semántico:** confirmar el uso de `google/gemma-4-31b` (otra familia que Qwen) y calibrarlo con unas 20–30 respuestas calificadas por ti.
