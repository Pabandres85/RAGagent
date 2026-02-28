"""
Pruebas unitarias para eval/metrics.py.

Cubre:
- Cálculo correcto de Recall@10
- Cálculo correcto de MRR (Mean Reciprocal Rank)
- Cálculo correcto de Exact Match y F1
- Cálculo de Faithfulness
- Precisión de ruteo entre agentes
"""

import pytest


class TestMetrics:
    """Pruebas de las métricas de evaluación del sistema RAG."""

    def test_recall_at_k_perfect_retrieval(self):
        """Recall@k debe ser 1.0 cuando el documento relevante está en top-k."""
        # TODO: from eval.metrics import recall_at_k
        # retrieved = ["doc_1", "doc_2", "doc_3"]
        # relevant = {"doc_1"}
        # assert recall_at_k(retrieved, relevant, k=3) == 1.0
        pytest.skip("Pendiente implementación de metrics")

    def test_recall_at_k_missed_retrieval(self):
        """Recall@k debe ser 0.0 cuando el documento relevante no está en top-k."""
        pytest.skip("Pendiente implementación de metrics")

    def test_mrr_first_position(self):
        """MRR debe ser 1.0 cuando el resultado relevante está en posición 1."""
        pytest.skip("Pendiente implementación de metrics")

    def test_exact_match_identical_strings(self):
        """EM debe ser 1.0 para respuestas idénticas normalizadas."""
        pytest.skip("Pendiente implementación de metrics")

    def test_f1_partial_overlap(self):
        """F1 debe estar entre 0 y 1 para respuestas con solapamiento parcial."""
        pytest.skip("Pendiente implementación de metrics")
