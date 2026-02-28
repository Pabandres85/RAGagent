"""
Pruebas unitarias para agents/guardrails.py.

Cubre:
- Rechazo de respuestas sin evidencia normativa (alucinaciones)
- Validación de estructura JSON de salida
- Verificación de vigencia del documento citado
- Bloqueo de consultas fuera del ámbito de habilitación
"""

import pytest


class TestGuardrails:
    """Pruebas de los mecanismos de control y validación de respuestas."""

    def test_response_without_citation_is_rejected(self):
        """Una respuesta generada sin cita normativa debe ser rechazada."""
        # TODO: instanciar Guardrails y pasar respuesta sin metadatos
        # guardrails = Guardrails()
        # response = {"answer": "Se requieren 2 médicos.", "citations": []}
        # assert not guardrails.validate(response)
        pytest.skip("Pendiente implementación de Guardrails")

    def test_valid_response_passes(self):
        """Una respuesta con cita válida (numeral, página, vigencia) debe pasar."""
        pytest.skip("Pendiente implementación de Guardrails")

    def test_outdated_document_is_flagged(self):
        """Una cita con versión derogada de la norma debe ser marcada."""
        pytest.skip("Pendiente implementación de Guardrails")

    def test_out_of_domain_query_blocked(self):
        """Consultas no relacionadas con habilitación deben ser bloqueadas."""
        pytest.skip("Pendiente implementación de Guardrails")

    def test_output_schema_is_valid_json(self):
        """La estructura de salida del agente debe ser JSON válido con campos requeridos."""
        pytest.skip("Pendiente implementación de Guardrails")
