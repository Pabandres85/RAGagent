"""
Pruebas unitarias para agents/orchestrator.py.

Cubre:
- Ruteo correcto de consultas al agente especialista correspondiente
- Precisión de ruteo por módulo (talento_humano, infraestructura, etc.)
- Respuesta coherente ante consultas ambiguas
- Rechazo de consultas fuera del dominio normativo
"""

import pytest


class TestOrchestrator:
    """Pruebas del orquestador central multi-agente."""

    @pytest.mark.parametrize("query,expected_module", [
        ("¿Cuántos médicos se requieren para una UCI?", "talento_humano"),
        ("¿Qué área mínima debe tener una sala de cirugía?", "infraestructura"),
        ("¿Qué equipos son obligatorios en urgencias?", "dotacion"),
        ("¿Cómo debe almacenarse el suero fisiológico?", "medicamentos_dispositivos"),
        ("¿Cuáles son los procesos de triage?", "procesos_prioritarios"),
        ("¿Qué campos obligatorios tiene la historia clínica?", "historia_clinica"),
        ("¿Qué servicios requieren cirugía como soporte?", "interdependencia"),
    ])
    def test_routing_accuracy(self, query, expected_module):
        """El orquestador debe rutear cada consulta al agente correcto."""
        # TODO: instanciar Orchestrator y verificar módulo seleccionado
        # orchestrator = Orchestrator()
        # result = orchestrator.route(query)
        # assert result.agent_module == expected_module
        pytest.skip("Pendiente implementación de Orchestrator")

    def test_out_of_domain_query_is_rejected(self):
        """Consultas fuera del dominio normativo deben ser rechazadas por guardrails."""
        pytest.skip("Pendiente implementación de Orchestrator + Guardrails")

    def test_response_contains_citation(self):
        """La respuesta debe incluir numeral, página y vigencia de la norma."""
        pytest.skip("Pendiente implementación de Orchestrator")
