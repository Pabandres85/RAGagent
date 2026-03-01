"""
Metricas basicas para evaluacion offline.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def routing_accuracy(predicted: Iterable[str], expected: Iterable[str]) -> float:
    predicted_list = list(predicted)
    expected_list = list(expected)
    if not expected_list or len(predicted_list) != len(expected_list):
        return 0.0
    hits = sum(1 for pred, exp in zip(predicted_list, expected_list) if pred == exp)
    return hits / len(expected_list)
