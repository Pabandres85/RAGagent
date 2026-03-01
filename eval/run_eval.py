"""
Runner simple de evaluacion offline.
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.baseline_mono_agent import MonoAgent
from agents.orchestrator import Orchestrator
from core.config import settings
from eval.metrics import exact_match, f1_score


def load_gold_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No existe gold set en {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_eval() -> list[dict]:
    gold_set = load_gold_set(settings.gold_set_path)
    orchestrator = Orchestrator()
    baseline = MonoAgent()
    results: list[dict] = []

    for item in gold_set:
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

        results.append(
            {
                "question": question,
                "module_expected": item.get("module"),
                "module_predicted": multi_result.get("routing", {}).get("module"),
                "multi_valid": multi_result.get("valid", False),
                "mono_valid": mono_result.valid,
                "multi_em": exact_match(multi_answer, reference),
                "mono_em": exact_match(mono_answer, reference),
                "multi_f1": f1_score(multi_answer, reference),
                "mono_f1": f1_score(mono_answer, reference),
            }
        )

    return results


def main() -> None:
    results = run_eval()
    settings.eval_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.eval_output_dir / "latest_eval.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Evaluacion completada: {output_path}")


if __name__ == "__main__":
    main()
