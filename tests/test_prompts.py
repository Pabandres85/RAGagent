"""Pruebas de los prompts: el que no pasa por .format() no debe tener llaves dobles."""
from agents.prompts import MONO_AGENT_SYSTEM_PROMPT, SPECIALIST_SYSTEM_TEMPLATE


def test_mono_prompt_has_no_literal_double_braces():
    # Se envia tal cual al LLM; '{{' haria que el modelo lo copie en su JSON.
    assert "{{" not in MONO_AGENT_SYSTEM_PROMPT
    assert "}}" not in MONO_AGENT_SYSTEM_PROMPT


def test_specialist_prompt_renders_without_double_braces():
    rendered = SPECIALIST_SYSTEM_TEMPLATE.format(module_name="Dotación", module_key="dotacion")
    assert "{{" not in rendered
    assert '"module": "dotacion"' in rendered
