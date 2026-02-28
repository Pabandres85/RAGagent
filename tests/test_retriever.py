"""
Pruebas unitarias para el módulo rag/retriever.py.

Cubre:
- Recuperación semántica con índice FAISS
- Número correcto de fragmentos recuperados (top-k)
- Presencia de metadatos en resultados (numeral, página, módulo)
- Comportamiento ante consulta vacía o sin resultados
"""

import pytest


class TestRetriever:
    """Pruebas del retriever semántico sobre índices FAISS."""

    def test_retrieve_returns_top_k_results(self):
        """Debe retornar exactamente top_k fragmentos."""
        # TODO: instanciar Retriever con un índice de prueba
        # retriever = Retriever(index_path="tests/fixtures/faiss_test")
        # results = retriever.retrieve("requisitos talento humano médico", top_k=5)
        # assert len(results) == 5
        pytest.skip("Pendiente implementación de Retriever")

    def test_retrieve_results_contain_metadata(self):
        """Cada fragmento recuperado debe tener numeral, página y módulo."""
        # TODO: verificar estructura de metadatos en cada resultado
        pytest.skip("Pendiente implementación de Retriever")

    def test_retrieve_scores_are_sorted_descending(self):
        """Los resultados deben ordenarse por similitud de mayor a menor."""
        pytest.skip("Pendiente implementación de Retriever")

    def test_retrieve_empty_query_raises_error(self):
        """Una consulta vacía debe lanzar ValueError."""
        # TODO: assert raises ValueError
        pytest.skip("Pendiente implementación de Retriever")
