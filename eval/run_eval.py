"""
Runner de evaluacion offline con checkpoints y resumen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from agents.baseline_mono_agent import MonoAgent
from agents.guardrails import GuardrailsResult
from agents.orchestrator import Orchestrator
from core.config import settings
from eval.metrics import exact_match, f1_score, routing_accuracy


def load_gold_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No existe gold set en {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _avg(items: list[dict], key: str, as_rate: bool = False) -> float | None:
    """Promedio ignorando valores None (sistema no ejecutado). None si no hay datos."""
    values = [item[key] for item in items if item.get(key) is not None]
    if not values:
        return None
    return mean(1.0 if v else 0.0 for v in values) if as_rate else mean(values)


STATUSES = ("answered", "abstained", "rejected", "error")


def _status_summary(results: list[dict], system: str) -> dict:
    """Tasas por estado y F1 calculado SOLO sobre respuestas sustantivas (answered)."""
    key = f"{system}_status"
    statuses = [item[key] for item in results if item.get(key) is not None]
    summary: dict = {f"{system}_status_count": len(statuses)}
    for status in STATUSES:
        summary[f"{system}_{status}_rate"] = (statuses.count(status) / len(statuses)) if statuses else None
    answered = [item for item in results if item.get(key) == "answered" and item.get(f"{system}_f1") is not None]
    summary[f"{system}_f1_answered_avg"] = mean(item[f"{system}_f1"] for item in answered) if answered else None
    return summary


def _common_answered_summary(results: list[dict]) -> dict:
    """F1 de ambos sistemas sobre las preguntas que los DOS responden sustantivamente."""
    both = [
        item for item in results
        if item.get("multi_status") == "answered" and item.get("mono_status") == "answered"
    ]
    return {
        "common_answered_count": len(both),
        "common_answered_multi_f1": mean(i["multi_f1"] for i in both) if both else None,
        "common_answered_mono_f1": mean(i["mono_f1"] for i in both) if both else None,
    }


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:.3f}"


def _build_summary(results: list[dict]) -> dict:
    if not results:
        return {
            "count": 0,
            "general_count": 0,
            "specific_count": 0,
            "multi_valid_rate": 0.0,
            "mono_valid_rate": 0.0,
            "multi_em_avg": 0.0,
            "mono_em_avg": 0.0,
            "multi_f1_avg": 0.0,
            "mono_f1_avg": 0.0,
            "routing_accuracy_top1": 0.0,
            "routing_hit_rate_any": 0.0,
            "routing_accuracy_top1_specific": 0.0,
            "routing_hit_rate_any_specific": 0.0,
        }

    # Items con módulo específico (excluye "general" del routing accuracy)
    specific = [item for item in results if item.get("module_expected") != "general"]

    expected_all = [item.get("module_expected") for item in results]
    predicted_all = [item.get("module_predicted") for item in results]
    expected_specific = [item.get("module_expected") for item in specific]
    predicted_specific = [item.get("module_predicted") for item in specific]

    any_hits_all = sum(
        1 for item in results
        if item.get("module_expected") and item.get("module_expected") in item.get("module_predicted_all", [])
    )
    any_hits_specific = sum(
        1 for item in specific
        if item.get("module_expected") and item.get("module_expected") in item.get("module_predicted_all", [])
    )

    # Sin orquestador ejecutado (--only mono) el routing no aplica: n/a, no 0%.
    multi_ran = any(item.get("multi_valid") is not None for item in results)

    summary = {
        "count": len(results),
        "general_count": len(results) - len(specific),
        "specific_count": len(specific),
        "multi_valid_rate": _avg(results, "multi_valid", as_rate=True),
        "mono_valid_rate": _avg(results, "mono_valid", as_rate=True),
        "multi_em_avg": _avg(results, "multi_em"),
        "mono_em_avg": _avg(results, "mono_em"),
        "multi_f1_avg": _avg(results, "multi_f1"),
        "mono_f1_avg": _avg(results, "mono_f1"),
        # Routing sobre todos los ítems (general siempre falla → métrica penalizada)
        "routing_accuracy_top1": routing_accuracy(predicted_all, expected_all),
        "routing_hit_rate_any": any_hits_all / len(results),
        # Routing solo sobre módulos especializados (la métrica representativa)
        "routing_accuracy_top1_specific": routing_accuracy(predicted_specific, expected_specific),
        "routing_hit_rate_any_specific": any_hits_specific / len(specific) if specific else 0.0,
    }
    summary.update(_status_summary(results, "multi"))
    summary.update(_status_summary(results, "mono"))
    summary.update(_common_answered_summary(results))

    if not multi_ran:
        for key in (
            "routing_accuracy_top1",
            "routing_hit_rate_any",
            "routing_accuracy_top1_specific",
            "routing_hit_rate_any_specific",
        ):
            summary[key] = None
    return summary


# Version del esquema de resultados por fila. Subirla cuando cambien los campos que las
# metricas necesitan (p. ej. v2 agrego multi_status/mono_status): un checkpoint de otra
# version no se puede reanudar porque sus filas no serian comparables.
RESULT_SCHEMA_VERSION = 2


def run_fingerprint(gold_set: list[dict], only: str | None, oracle_routing: bool) -> dict:
    """Identifica una corrida: mismo dataset (preguntas y modulos, en orden) y mismo modo."""
    payload = json.dumps(
        [(item.get("question"), item.get("module"), item.get("answer")) for item in gold_set],
        ensure_ascii=False,
    )
    return {
        "dataset_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "n_items": len(gold_set),
        "schema_version": RESULT_SCHEMA_VERSION,
        "only": only,
        "oracle_routing": oracle_routing,
    }


def load_checkpoint_for_resume(
    checkpoint_path: Path, meta_path: Path, gold_set: list[dict], fingerprint: dict
) -> list[dict]:
    """
    Carga el checkpoint solo si corresponde EXACTAMENTE a esta corrida. Si el gold set,
    su orden o el modo cambiaron, se niega a reanudar para no mezclar resultados viejos
    con preguntas nuevas.
    """
    if not checkpoint_path.exists():
        return []
    if not meta_path.exists():
        raise RuntimeError(
            f"Checkpoint {checkpoint_path.name} sin metadatos de corrida; no se puede "
            "verificar que corresponda a este dataset. Borralo o corre sin --resume."
        )
    with open(meta_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    if saved != fingerprint:
        raise RuntimeError(
            "El checkpoint no corresponde a esta corrida (dataset, orden o modo "
            f"distintos). checkpoint={saved} | actual={fingerprint}. "
            "Corre sin --resume o usa otro --tag."
        )
    with open(checkpoint_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)
    for position, row in enumerate(results):
        if position >= len(gold_set) or row.get("question") != gold_set[position]["question"]:
            raise RuntimeError(f"El checkpoint difiere del gold set en la posicion {position + 1}.")
    return results


def run_eval(
    limit: int | None = None,
    checkpoint_every: int = 5,
    only: str | None = None,
    tag: str | None = None,
    resume: bool = False,
    oracle_routing: bool = False,
) -> tuple[list[dict], dict]:
    gold_set = load_gold_set(settings.gold_set_path)
    if oracle_routing:
        # Diagnostico: solo preguntas con modulo especifico, solo el multi-agente, y el
        # especialista recibe unicamente la ETIQUETA del modulo (nunca la respuesta).
        gold_set = [item for item in gold_set if item.get("module") not in (None, "general")]
        only = "multi"
    if limit is not None:
        gold_set = gold_set[:limit]

    # only="mono"|"multi" evita ejecutar el otro sistema (pruebas de ablacion).
    orchestrator = Orchestrator() if only != "mono" else None
    baseline = MonoAgent() if only != "multi" else None
    results: list[dict] = []

    suffix = f"_{tag}" if tag else ""
    checkpoint_path = settings.eval_output_dir / f"latest_eval{suffix}.partial.json"
    meta_path = settings.eval_output_dir / f"latest_eval{suffix}.meta.partial.json"
    fingerprint = run_fingerprint(gold_set, only, oracle_routing)

    if resume:
        results = load_checkpoint_for_resume(checkpoint_path, meta_path, gold_set, fingerprint)
        if results:
            print(f"Reanudando desde checkpoint: {len(results)} items ya evaluados.")
    _write_json(meta_path, fingerprint)

    total = len(gold_set)
    for index, item in enumerate(gold_set, start=1):
        if index <= len(results):
            continue
        question = item["question"]
        reference = item.get("answer", "")

        forced = item.get("module") if oracle_routing else None
        try:
            multi_result = orchestrator.answer(question, forced_module=forced) if orchestrator else {}
        except FileNotFoundError:
            raise
        except Exception as exc:  # un fallo aislado no debe tirar toda la corrida
            multi_result = {"valid": False, "status": "error", "errors": [f"{type(exc).__name__}: {exc}"], "response": None}
        try:
            mono_result = baseline.answer(question) if baseline else GuardrailsResult(valid=False)
        except FileNotFoundError:
            raise
        except Exception as exc:
            mono_result = GuardrailsResult(valid=False, status="error", errors=[f"{type(exc).__name__}: {exc}"])
        # Un sistema no ejecutado se registra como None (no como 0 / invalido).
        ran_multi, ran_mono = orchestrator is not None, baseline is not None

        multi_answer = ""
        if multi_result.get("response"):
            multi_answer = multi_result["response"].get("answer", "")

        mono_answer = ""
        if mono_result.response:
            mono_answer = mono_result.response.answer

        row = {
            "index": index,
            "question": question,
            "reference_answer": reference,
            "module_expected": item.get("module"),
            "module_predicted": multi_result.get("routing", {}).get("module"),
            "module_predicted_all": multi_result.get("routing", {}).get("modules", []),
            "routing_is_transversal": multi_result.get("routing", {}).get("is_transversal", False),
            "multi_status": multi_result.get("status", "rejected") if ran_multi else None,
            "mono_status": mono_result.status if ran_mono else None,
            "multi_valid": multi_result.get("valid", False) if ran_multi else None,
            "mono_valid": mono_result.valid if ran_mono else None,
            "multi_em": exact_match(multi_answer, reference) if ran_multi else None,
            "mono_em": exact_match(mono_answer, reference) if ran_mono else None,
            "multi_f1": f1_score(multi_answer, reference) if ran_multi else None,
            "mono_f1": f1_score(mono_answer, reference) if ran_mono else None,
            "multi_answer": multi_answer,
            "mono_answer": mono_answer,
            "multi_errors": multi_result.get("errors", []),
            "multi_warnings": multi_result.get("warnings", []),
            "mono_errors": mono_result.errors,
            "mono_warnings": mono_result.warnings,
        }
        results.append(row)

        if index % checkpoint_every == 0 or index == total:
            _write_json(checkpoint_path, results)

        if index % checkpoint_every == 0 or index == total:
            print(f"[{index}/{total}] checkpoint guardado")

    summary = _build_summary(results)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion offline multi-agente vs baseline mono.")
    parser.add_argument("--limit", type=int, default=None, help="Limitar cantidad de items del gold set.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Guardar checkpoint parcial cada N preguntas.",
    )
    parser.add_argument(
        "--only",
        choices=["mono", "multi"],
        default=None,
        help="Ejecuta solo un sistema (ablaciones). El otro queda en blanco.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Sufijo para los archivos de salida (no pisa latest_eval.json).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continua desde el checkpoint parcial del mismo --tag/--only.",
    )
    parser.add_argument(
        "--oracle-routing",
        action="store_true",
        help="Diagnostico: usa el modulo etiquetado en lugar del ruteador (solo especificos, solo multi).",
    )
    args = parser.parse_args()

    # Una corrida parcial (--only) nunca debe pisar los resultados completos.
    tag = args.tag or ("oracle_routing" if args.oracle_routing else (f"only_{args.only}" if args.only else None))
    results, summary = run_eval(
        limit=args.limit, checkpoint_every=args.checkpoint_every, only=args.only, tag=tag, resume=args.resume, oracle_routing=args.oracle_routing
    )

    settings.eval_output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    output_path = settings.eval_output_dir / f"latest_eval{suffix}.json"
    summary_path = settings.eval_output_dir / f"latest_eval_summary{suffix}.json"

    _write_json(output_path, results)
    _write_json(summary_path, summary)

    print(f"Evaluacion completada: {output_path}")
    print(
        "Resumen | "
        f"items={summary['count']} (general={summary['general_count']}, especificos={summary['specific_count']}) | "
        f"multi_valid={_fmt(summary['multi_valid_rate'], pct=True)} | "
        f"mono_valid={_fmt(summary['mono_valid_rate'], pct=True)} | "
        f"multi_f1={_fmt(summary['multi_f1_avg'])} | "
        f"mono_f1={_fmt(summary['mono_f1_avg'])}"
    )
    if summary.get("multi_status_count") or summary.get("mono_status_count"):
        print(
            "Estados | "
            + " | ".join(
                f"{name}: " + "/".join(f"{st[:3]}={_fmt(summary.get(f'{name}_{st}_rate'), pct=True)}" for st in STATUSES)
                for name in ("multi", "mono")
            )
            + f" | F1 sustantivas: multi={_fmt(summary.get('multi_f1_answered_avg'))} mono={_fmt(summary.get('mono_f1_answered_avg'))}"
            + f" | ambos responden (n={summary.get('common_answered_count')}): "
            f"multi={_fmt(summary.get('common_answered_multi_f1'))} mono={_fmt(summary.get('common_answered_mono_f1'))}"
        )
    print(
        "Routing (especificos) | "
        f"top1={_fmt(summary['routing_accuracy_top1_specific'], pct=True)} | "
        f"any={_fmt(summary['routing_hit_rate_any_specific'], pct=True)} | "
        f"-- Routing (todos, ref.) top1={_fmt(summary['routing_accuracy_top1'], pct=True)} | "
        f"any={_fmt(summary['routing_hit_rate_any'], pct=True)}"
    )


if __name__ == "__main__":
    main()
