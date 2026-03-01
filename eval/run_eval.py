"""
Runner de evaluacion offline con checkpoints y resumen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from agents.baseline_mono_agent import MonoAgent
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

    return {
        "count": len(results),
        "general_count": len(results) - len(specific),
        "specific_count": len(specific),
        "multi_valid_rate": mean(1.0 if item.get("multi_valid") else 0.0 for item in results),
        "mono_valid_rate": mean(1.0 if item.get("mono_valid") else 0.0 for item in results),
        "multi_em_avg": mean(item.get("multi_em", 0.0) for item in results),
        "mono_em_avg": mean(item.get("mono_em", 0.0) for item in results),
        "multi_f1_avg": mean(item.get("multi_f1", 0.0) for item in results),
        "mono_f1_avg": mean(item.get("mono_f1", 0.0) for item in results),
        # Routing sobre todos los ítems (general siempre falla → métrica penalizada)
        "routing_accuracy_top1": routing_accuracy(predicted_all, expected_all),
        "routing_hit_rate_any": any_hits_all / len(results),
        # Routing solo sobre módulos especializados (la métrica representativa)
        "routing_accuracy_top1_specific": routing_accuracy(predicted_specific, expected_specific),
        "routing_hit_rate_any_specific": any_hits_specific / len(specific) if specific else 0.0,
    }


def run_eval(limit: int | None = None, checkpoint_every: int = 5) -> tuple[list[dict], dict]:
    gold_set = load_gold_set(settings.gold_set_path)
    if limit is not None:
        gold_set = gold_set[:limit]

    orchestrator = Orchestrator()
    baseline = MonoAgent()
    results: list[dict] = []

    checkpoint_path = settings.eval_output_dir / "latest_eval.partial.json"

    total = len(gold_set)
    for index, item in enumerate(gold_set, start=1):
        question = item["question"]
        reference = item.get("answer", "")

        multi_result = orchestrator.answer(question)
        mono_result = baseline.answer(question)

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
            "multi_valid": multi_result.get("valid", False),
            "mono_valid": mono_result.valid,
            "multi_em": exact_match(multi_answer, reference),
            "mono_em": exact_match(mono_answer, reference),
            "multi_f1": f1_score(multi_answer, reference),
            "mono_f1": f1_score(mono_answer, reference),
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
    args = parser.parse_args()

    results, summary = run_eval(limit=args.limit, checkpoint_every=args.checkpoint_every)

    settings.eval_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.eval_output_dir / "latest_eval.json"
    summary_path = settings.eval_output_dir / "latest_eval_summary.json"

    _write_json(output_path, results)
    _write_json(summary_path, summary)

    print(f"Evaluacion completada: {output_path}")
    print(
        "Resumen | "
        f"items={summary['count']} (general={summary['general_count']}, especificos={summary['specific_count']}) | "
        f"multi_valid={summary['multi_valid_rate']:.1%} | "
        f"mono_valid={summary['mono_valid_rate']:.1%} | "
        f"multi_f1={summary['multi_f1_avg']:.3f} | "
        f"mono_f1={summary['mono_f1_avg']:.3f}"
    )
    print(
        "Routing (especificos) | "
        f"top1={summary['routing_accuracy_top1_specific']:.1%} | "
        f"any={summary['routing_hit_rate_any_specific']:.1%} | "
        f"-- Routing (todos, ref.) top1={summary['routing_accuracy_top1']:.1%} | "
        f"any={summary['routing_hit_rate_any']:.1%}"
    )


if __name__ == "__main__":
    main()
