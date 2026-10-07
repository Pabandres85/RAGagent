# Resultados de evaluación — leer antes de citar

| Archivo | Qué es | ¿Citable? |
|---|---|---|
| `latest_eval.json`, `latest_eval_summary.json` | Corrida **v4 antigua** (etiqueta git `eval-v4-pre-fix`). El baseline mono estaba defectuoso (índice global desalineado + prompt con `{{ }}`). | **No** |
| `latest_eval_provisional_post_fix.json` (+ `_summary_`) | Corrida completa de 122 ítems con el pipeline corregido. **Provisional**: gold set y corpus aún sin depurar. | Solo como provisional |
| `latest_eval_oracle_routing.json` (+ `_summary_`) | Diagnóstico de ruteo oráculo de especialista único (105 específicas). Techo optimista. | Solo como diagnóstico |

Detalle, intervalos de confianza y limitaciones: `ESTADO_PROYECTO.md` §8b y `docs/decisions.md` §18.

Nota: la UI (`ui/pages/2_Evaluacion.py`) lee `latest_eval_summary.json`, que es la corrida antigua. Se reemplazará con la corrida final tras depurar corpus y gold set.
