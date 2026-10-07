"""Pruebas del runner de evaluacion y de la reconstruccion segura del indice global."""
import pytest

from eval.run_eval import _build_summary


def _row(**kw):
    base = {"module_expected": "dotacion", "module_predicted": "dotacion",
            "module_predicted_all": ["dotacion"]}
    base.update(kw)
    return base


def test_summary_ignores_system_not_run():
    rows = [
        _row(multi_valid=None, multi_em=None, multi_f1=None, mono_valid=True, mono_em=0.0, mono_f1=0.5),
        _row(multi_valid=None, multi_em=None, multi_f1=None, mono_valid=False, mono_em=0.0, mono_f1=0.1),
    ]
    summary = _build_summary(rows)
    assert summary["multi_valid_rate"] is None
    assert summary["multi_f1_avg"] is None
    assert summary["mono_valid_rate"] == 0.5
    assert summary["mono_f1_avg"] == pytest.approx(0.3)


def test_rebuild_global_fails_before_writing_when_index_missing(tmp_path, monkeypatch):
    from scripts import ingest

    monkeypatch.setattr(ingest.settings, "faiss_index_dir", tmp_path)
    with pytest.raises(FileNotFoundError):
        ingest.rebuild_global_index()
    assert not (tmp_path / "global.faiss").exists()


def test_summary_routing_is_none_when_multi_not_run():
    rows = [_row(multi_valid=None, multi_em=None, multi_f1=None, mono_valid=True, mono_em=0.0, mono_f1=0.5)]
    summary = _build_summary(rows)
    assert summary["routing_accuracy_top1_specific"] is None
    assert summary["routing_hit_rate_any"] is None


def test_llm_client_retries_transient_unloaded_error(monkeypatch):
    import httpx
    from openai import BadRequestError

    from core import llm_client

    calls = {"n": 0}

    class _Completions:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                req = httpx.Request("POST", "http://x")
                raise BadRequestError("Model unloaded.", response=httpx.Response(400, request=req), body=None)
            msg = type("M", (), {"content": "ok"})
            return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    client = type("C", (), {"chat": type("Ch", (), {"completions": _Completions()})})
    monkeypatch.setattr(llm_client, "get_llm_client", lambda: client)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    assert llm_client.chat_completion([{"role": "user", "content": "x"}]) == "ok"
    assert calls["n"] == 3


def test_orchestrator_forced_module_skips_router():
    from agents.orchestrator import Orchestrator

    calls = []

    class _Agent:
        def answer(self, question):
            from agents.guardrails import GuardrailsResult
            calls.append(question)
            return GuardrailsResult(valid=False, errors=["x"])

    orch = Orchestrator.__new__(Orchestrator)  # sin cargar modelos
    orch._agents = {"dotacion": _Agent(), "infraestructura": _Agent()}
    orch.route = lambda q: (_ for _ in ()).throw(AssertionError("no debe rutear"))
    out = orch.answer("pregunta", forced_module="dotacion")
    assert out["routing"]["modules"] == ["dotacion"]
    assert calls == ["pregunta"]


def _resume_setup(tmp_path, gold):
    from eval.run_eval import _write_json, run_fingerprint

    fp = run_fingerprint(gold, None, False)
    cp, meta = tmp_path / "x.partial.json", tmp_path / "x.meta.partial.json"
    _write_json(cp, [{"question": g["question"]} for g in gold[:2]])
    _write_json(meta, fp)
    return cp, meta, fp


def test_resume_accepts_matching_checkpoint(tmp_path):
    from eval.run_eval import load_checkpoint_for_resume

    gold = [{"question": f"q{i}", "module": "dotacion"} for i in range(4)]
    cp, meta, fp = _resume_setup(tmp_path, gold)
    assert len(load_checkpoint_for_resume(cp, meta, gold, fp)) == 2


def test_resume_refuses_when_dataset_changed(tmp_path):
    from eval.run_eval import load_checkpoint_for_resume, run_fingerprint

    gold = [{"question": f"q{i}", "module": "dotacion"} for i in range(4)]
    cp, meta, _ = _resume_setup(tmp_path, gold)
    changed = gold[:1] + [{"question": "otra", "module": "dotacion"}] + gold[2:]
    with pytest.raises(RuntimeError):
        load_checkpoint_for_resume(cp, meta, changed, run_fingerprint(changed, None, False))


def test_resume_refuses_when_mode_changed(tmp_path):
    from eval.run_eval import load_checkpoint_for_resume, run_fingerprint

    gold = [{"question": f"q{i}", "module": "dotacion"} for i in range(4)]
    cp, meta, _ = _resume_setup(tmp_path, gold)
    with pytest.raises(RuntimeError):
        load_checkpoint_for_resume(cp, meta, gold, run_fingerprint(gold, "mono", False))


def test_resume_refuses_checkpoint_without_metadata(tmp_path):
    from eval.run_eval import load_checkpoint_for_resume, run_fingerprint

    gold = [{"question": "q0", "module": "dotacion"}]
    cp, meta, fp = _resume_setup(tmp_path, gold)
    meta.unlink()
    with pytest.raises(RuntimeError):
        load_checkpoint_for_resume(cp, meta, gold, fp)
