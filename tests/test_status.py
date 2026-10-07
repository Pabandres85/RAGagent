"""
Estados explicitos de respuesta: answered / abstained / rejected / error.

`valid` sigue siendo validez estructural (formato + cita); `status` describe que paso con la
pregunta. Una abstencion es valida estructuralmente pero NO es una respuesta sustantiva.
"""
import json

import pytest

from agents.guardrails import AgentResponse, Citation, GuardrailsResult, validate_response
from agents.orchestrator import Orchestrator, RoutingDecision
from eval.run_eval import _build_summary

CITATION = {"text": "texto", "numeral": "1.1", "page": 5}


def _raw(**overrides):
    base = {"answer": "respuesta", "citations": [CITATION], "module": "dotacion", "confidence": 0.9}
    base.update(overrides)
    return json.dumps(base)


def test_answer_with_citation_is_answered():
    result = validate_response(_raw(), expected_module="dotacion")
    assert result.valid and result.status == "answered"


def test_declared_abstention_is_valid_but_not_answered():
    raw = _raw(
        answer="La información solicitada no se encuentra en los fragmentos recuperados del estándar de Dotación.",
        citations=[],
    )
    result = validate_response(raw, expected_module="dotacion")
    assert result.valid and result.no_evidence
    assert result.status == "abstained"


@pytest.mark.parametrize(
    "raw",
    [
        _raw(citations=[]),                    # responde sin cita normativa
        "esto no es json",                     # sin JSON
        '{"answer": "x", "citations": [',      # JSON malformado
        '{"answer": "x"}',                     # esquema invalido (faltan campos)
    ],
)
def test_guardrail_failures_are_rejected(raw):
    result = validate_response(raw, expected_module="dotacion")
    assert not result.valid and result.status == "rejected"


def _answered(module="dotacion"):
    response = AgentResponse(answer="ok", citations=[Citation(text="t", numeral="1.1", page=1)], module=module)
    return GuardrailsResult(valid=True, status="answered", response=response)


def _orchestrator(agents):
    orch = Orchestrator.__new__(Orchestrator)  # sin cargar modelos
    orch._agents = agents
    return orch


class _Agent:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    def answer(self, question):
        if self._exc:
            raise self._exc
        return self._result


def _routing(modules):
    return RoutingDecision(module=modules[0], confidence=1.0, reasoning="t", modules=modules,
                           scores={}, is_transversal=len(modules) > 1)


def test_aggregate_prefers_answered_then_error_then_rejected_then_abstained():
    agg = Orchestrator._aggregate_status
    ab, rj, er, an = (GuardrailsResult(valid=False, status=s) for s in ("abstained", "rejected", "error", "answered"))
    assert agg([("a", an), ("b", er)]) == "answered"
    assert agg([("a", ab), ("b", rj), ("c", er)]) == "error"
    assert agg([("a", ab), ("b", rj)]) == "rejected"
    assert agg([("a", ab), ("b", ab)]) == "abstained"


def test_orchestrator_reports_abstention_when_no_specialist_has_evidence():
    abstention = GuardrailsResult(
        valid=True, status="abstained", no_evidence=True,
        response=AgentResponse(answer="La información solicitada no se encuentra en los fragmentos", citations=[], module="dotacion"),
    )
    orch = _orchestrator({"dotacion": _Agent(abstention)})
    orch.route = lambda q: _routing(["dotacion"])
    out = orch.answer("pregunta")
    assert out["status"] == "abstained" and out["valid"] is False


def test_specialist_exception_becomes_error_status_not_crash():
    orch = _orchestrator({"dotacion": _Agent(exc=RuntimeError("LLM caido"))})
    orch.route = lambda q: _routing(["dotacion"])
    out = orch.answer("pregunta")
    assert out["status"] == "error"
    assert any("RuntimeError" in e for e in out["errors"])


def test_missing_index_still_propagates_for_503():
    orch = _orchestrator({"dotacion": _Agent(exc=FileNotFoundError("falta indice"))})
    orch.route = lambda q: _routing(["dotacion"])
    with pytest.raises(FileNotFoundError):
        orch.answer("pregunta")


def test_summary_separates_states_and_scores_f1_only_on_answered():
    def row(ms, mo, mf, of):
        return {"module_expected": "dotacion", "module_predicted": "dotacion", "module_predicted_all": ["dotacion"],
                "multi_status": ms, "mono_status": mo, "multi_valid": ms != "rejected", "mono_valid": mo != "rejected",
                "multi_em": 0.0, "mono_em": 0.0, "multi_f1": mf, "mono_f1": of}

    rows = [
        row("answered", "answered", 0.6, 0.8),
        row("abstained", "answered", 0.0, 0.4),
        row("rejected", "abstained", 0.1, 0.0),
        row("answered", "error", 0.2, 0.0),
    ]
    summary = _build_summary(rows)
    assert summary["multi_answered_rate"] == 0.5 and summary["multi_abstained_rate"] == 0.25
    assert summary["multi_rejected_rate"] == 0.25 and summary["multi_error_rate"] == 0.0
    assert summary["mono_answered_rate"] == 0.5 and summary["mono_error_rate"] == 0.25
    # F1 solo sobre respondidas: multi (0.6+0.2)/2, mono (0.8+0.4)/2
    assert summary["multi_f1_answered_avg"] == pytest.approx(0.4)
    assert summary["mono_f1_answered_avg"] == pytest.approx(0.6)
    # preguntas respondidas por ambos: solo la primera
    assert summary["common_answered_count"] == 1
    assert summary["common_answered_multi_f1"] == pytest.approx(0.6)
    assert summary["common_answered_mono_f1"] == pytest.approx(0.8)
