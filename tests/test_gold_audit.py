"""
Hoja de auditoria del gold v2 (esquema v2): no perder decisiones, validar la importacion,
decisiones combinadas (`corregir`) y migracion desde el esquema v1.
"""
import csv
import json

import pytest

from scripts.build_gold_v2_audit import (
    CSV_COLUMNS,
    HUMAN_LEVELS,
    SCHEMA_VERSION,
    evidence_hash,
    import_csv,
    load_records,
    merge_existing_decisions,
    migrate_file,
    migrate_record,
    write_outputs,
)

CORPUS = {
    "c-infra": {"chunk_id": "c-infra", "module": "infraestructura"},
    "c-dota": {"chunk_id": "c-dota", "module": "dotacion"},
}


def _records():
    return [
        {
            "audit_id": f"v1-00{i}", "question": f"pregunta {i}", "flags": ["x"], "priority": 3,
            "evidence_hash": f"h{i}", "stale_review": None, "schema_version": SCHEMA_VERSION, "history": [],
            "status": "modulo_cambia", "module_v1": "dotacion", "module_proposed": "infraestructura",
            "module_named_in_question": [],
            "decision": "", "decision_reason": "", "reviewer": "", "review_level": "", "reviewed_at": "",
            "new_module": "", "rewritten_question": "", "rewritten_answer": "",
            "evidence_chunk_id_v2": "", "retire_category": "",
        }
        for i in (1, 2, 3)
    ]


def _edit_csv(csv_path, updates):
    """Simula al revisor editando el CSV: updates = {audit_id: {campo: valor}}."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    for row in rows:
        row.update(updates.get(row["audit_id"], {}))
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _paths(tmp_path):
    return tmp_path / "gold_v2_audit.json", tmp_path / "gold_v2_audit.csv"


def _saved(json_path):
    return {r["audit_id"]: r for r in json.loads(json_path.read_text(encoding="utf-8"))}


PRE = {"reviewer": "Codex", "review_level": "preaudit"}
AUTHOR = {"reviewer": "PM", "review_level": "author", "reviewed_at": "2026-10-07"}


# ------------------------------------------------------------------ protecciones previas
def test_regeneration_preserves_existing_decisions(tmp_path):
    json_path, _ = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    data = load_records(json_path)
    data[0].update(decision="conservar", reviewer="PM", review_level="author", decision_reason="ok")
    json_path.write_text(json.dumps(data), encoding="utf-8")
    write_outputs(data, tmp_path)

    _, _, kept = write_outputs(_records(), tmp_path)  # el generador produce registros sin decision
    assert kept == 1
    assert _saved(json_path)["v1-001"]["decision"] == "conservar"


def test_regeneration_refuses_to_overwrite_unimported_csv_edits(tmp_path):
    _, csv_path = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    _edit_csv(csv_path, {"v1-001": {"decision": "retirar", "decision_reason": "x", **PRE}})
    with pytest.raises(RuntimeError, match="sin importar"):
        write_outputs(_records(), tmp_path)
    assert "retirar" in csv_path.read_text(encoding="utf-8-sig")


def test_blank_csv_fields_never_erase_saved_decisions(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    _edit_csv(csv_path, {"v1-001": {"decision": "retirar", "decision_reason": "ok", "retire_category": "sin_evidencia", **PRE}})
    import_csv(csv_path, json_path, CORPUS)
    write_outputs(load_records(json_path), tmp_path)
    _edit_csv(csv_path, {"v1-001": {"decision": "", "reviewer": "", "decision_reason": "", "review_level": ""}})
    import_csv(csv_path, json_path, CORPUS)
    saved = _saved(json_path)["v1-001"]
    assert saved["decision"] == "retirar" and saved["reviewer"] == "Codex"


def test_import_rejects_duplicate_ids(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    text = csv_path.read_text(encoding="utf-8-sig").splitlines()
    csv_path.write_text("\n".join(text + [text[1]]) + "\n", encoding="utf-8-sig")  # repite la fila 1
    with pytest.raises(ValueError, match="duplicado"):
        import_csv(csv_path, json_path, CORPUS)


def test_import_rejects_csv_with_outdated_evidence(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    changed = _records()
    changed[0]["evidence_hash"] = "otra-huella"
    json_path.write_text(json.dumps(changed), encoding="utf-8")
    _edit_csv(csv_path, {"v1-001": {"decision": "conservar", "decision_reason": "x", **PRE}})
    with pytest.raises(ValueError, match="desactualizado"):
        import_csv(csv_path, json_path, CORPUS)


def test_changed_evidence_invalidates_previous_decision(tmp_path):
    json_path, _ = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    data = load_records(json_path)
    data[0].update(decision="corregir", new_module="infraestructura", decision_reason="era dotacion", **PRE)
    json_path.write_text(json.dumps(data), encoding="utf-8")
    write_outputs(data, tmp_path)

    regenerated = _records()
    regenerated[0]["evidence_hash"] = "propuesta-nueva"
    _, _, kept = write_outputs(regenerated, tmp_path)
    assert kept == 0
    first = _saved(json_path)["v1-001"]
    assert first["decision"] == "" and first["priority"] == 1
    assert first["stale_review"]["decision"] == "corregir"
    assert any(e["event"] == "decision_obsoleta" for e in first["history"])


def test_partial_edit_does_not_erase_stale_history(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    data = load_records(json_path)
    data[0]["stale_review"] = {"decision": "corregir", "reviewer": "PM"}
    json_path.write_text(json.dumps(data), encoding="utf-8")
    write_outputs(data, tmp_path)

    _edit_csv(csv_path, {"v1-001": {"reviewer": "PM"}})  # solo el nombre
    import_csv(csv_path, json_path, CORPUS)
    first = _saved(json_path)["v1-001"]
    assert first["stale_review"] is not None and first["decision"] == ""

    write_outputs(load_records(json_path), tmp_path)
    _edit_csv(csv_path, {"v1-001": {"decision": "retirar", "decision_reason": "ok", "retire_category": "sin_evidencia", **AUTHOR}})
    import_csv(csv_path, json_path, CORPUS)
    first = _saved(json_path)["v1-001"]
    assert first["decision"] == "retirar" and first["stale_review"] is None


def test_gold_v1_change_under_existing_decision_is_blocked(tmp_path):
    json_path, _ = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    data = load_records(json_path)
    data[0].update(decision="conservar", decision_reason="x", **PRE)
    json_path.write_text(json.dumps(data), encoding="utf-8")
    changed = _records()
    changed[0]["question"] = "pregunta distinta"
    with pytest.raises(RuntimeError, match="cambio"):
        merge_existing_decisions(changed, json_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("page_v2", 999), ("service_v2", "otro servicio"), ("numeral_v2", "99.9"),
        ("match_score", 0.01), ("match_margin", 0.99), ("second_best_score", 0.9),
        ("second_best_module_v2", "talento_humano"), ("page_v1", 1), ("numerals_missing_in_v2", ["1.1"]),
        ("answer_lexical_overlap_v2", 0.01), ("chunk_text_v1", "otro texto"),
    ],
)
def test_evidence_hash_covers_everything_the_reviewer_sees(field, value):
    base = {"question": "q", "reference_answer": "a", "module_v1": "dotacion", "module_proposed": "infraestructura",
            "status": "modulo_cambia", "chunk_id_v1": "a", "chunk_id_v2": "b", "page_v2": 10, "service_v2": "s",
            "numeral_v2": "1.1", "match_score": 0.9, "match_margin": 0.3, "second_best_score": 0.6,
            "second_best_module_v2": "dotacion", "page_v1": 9, "numerals_missing_in_v2": [],
            "answer_lexical_overlap_v2": 0.8, "chunk_text_v1": "t1", "chunk_text_v2": "t2"}
    assert evidence_hash(base) != evidence_hash(dict(base, **{field: value}))


# ------------------------------------------------------------------ esquema v2: decisiones
def test_combined_correction_is_accepted_and_recorded_in_history(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    _edit_csv(csv_path, {"v1-001": {
        "decision": "corregir", "decision_reason": "etiqueta y pregunta eran de otro estandar",
        "new_module": "infraestructura", "rewritten_question": "pregunta nueva",
        "rewritten_answer": "respuesta corregida", "evidence_chunk_id_v2": "c-infra", **AUTHOR,
    }})
    assert import_csv(csv_path, json_path, CORPUS) == 1
    saved = _saved(json_path)["v1-001"]
    assert (saved["new_module"], saved["rewritten_question"], saved["rewritten_answer"]) == (
        "infraestructura", "pregunta nueva", "respuesta corregida")
    assert saved["history"][-1]["event"] == "import" and saved["history"][-1]["previous"]["decision"] == ""


@pytest.mark.parametrize(
    "fields,fragment",
    [
        ({"decision": "quizas", **PRE}, "invalida"),
        ({"decision": "reetiquetar", "new_module": "infraestructura", **PRE}, "ya no existe"),
        ({"decision": "corregir", **PRE}, "al menos uno"),
        ({"decision": "corregir", "new_module": "modulo_inventado", **PRE}, "no es un modulo valido"),
        ({"decision": "corregir", "evidence_chunk_id_v2": "no-existe", "new_module": "infraestructura", **PRE}, "no existe en el corpus"),
        ({"decision": "corregir", "evidence_chunk_id_v2": "c-dota", "new_module": "infraestructura", **PRE}, "incompatible"),
        ({"decision": "conservar", "new_module": "infraestructura", "decision_reason": "x", **PRE}, "solo 'corregir'"),
        ({"decision": "retirar", **PRE}, "requiere decision_reason"),
        ({"decision": "retirar", "decision_reason": "x", "retire_category": "inventada", **PRE}, "invalida"),
        ({"decision": "retirar", "decision_reason": "x", "retire_category": "sin_evidencia"}, "falta review_level"),
        ({"decision": "conservar", "decision_reason": "x", "review_level": "preaudit"}, "falta el revisor"),
        ({"decision": "conservar", "decision_reason": "x", "reviewer": "PM", "review_level": "author"}, "reviewed_at"),
        ({"decision": "conservar", "decision_reason": "x", "reviewer": "PM", "review_level": "author", "reviewed_at": "2999-01-01"}, "futuro"),
        ({"decision": "conservar", "decision_reason": "x", "reviewer": "PM", "review_level": "author", "reviewed_at": "ayer"}, "fecha ISO"),
        ({"decision": "retirar", "decision_reason": "x", **AUTHOR}, "exige retire_category"),
        ({"decision": "conservar", "decision_reason": "mantener", **PRE}, "solo se puede 'conservar'"),  # estado modulo_cambia
    ],
)
def test_invalid_reviews_are_rejected_without_writing(tmp_path, fields, fragment):
    json_path, csv_path = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    before = json_path.read_text(encoding="utf-8")
    _edit_csv(csv_path, {
        "v1-001": {"decision": "retirar", "decision_reason": "valida", **PRE},   # fila valida
        "v1-002": fields,
    })
    with pytest.raises(ValueError, match="rechazada") as exc:
        import_csv(csv_path, json_path, CORPUS)
    assert fragment in str(exc.value)
    assert json_path.read_text(encoding="utf-8") == before  # nada se guardo, ni la fila valida


def test_question_naming_another_module_requires_rewriting(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    records = _records()
    records[1]["module_named_in_question"] = ["dotacion"]      # la pregunta dice "modulo Dotacion"
    write_outputs(records, tmp_path)
    _edit_csv(csv_path, {"v1-002": {"decision": "corregir", "new_module": "infraestructura", "decision_reason": "x", **PRE}})
    with pytest.raises(ValueError, match="reescribe la pregunta"):
        import_csv(csv_path, json_path, CORPUS)
    _edit_csv(csv_path, {"v1-002": {"rewritten_question": "pregunta sin nombre de modulo"}})
    assert import_csv(csv_path, json_path, CORPUS) == 1


# ------------------------------------------------------------------ migracion v1 -> v2
def _legacy(**kw):
    record = {"audit_id": "v1-009", "question": "q", "module_v1": "dotacion", "module_proposed": "infraestructura",
              "decision": "", "decision_reason": "", "reviewer": "", "rewritten_question": ""}
    record.update(kw)
    return record


def test_migration_keeps_old_decisions_as_preaudit_proposals_with_history():
    reetiquetar = _legacy(decision="reetiquetar", reviewer="Codex (preauditoria tecnica)", decision_reason="motivo plantilla")
    reescribir = _legacy(decision="reescribir", rewritten_question="nueva", reviewer="PM")
    retirar = _legacy(decision="retirar", decision_reason="fuera de corpus", reviewer="Codex")
    vacio = _legacy()
    assert all(migrate_record(r) for r in (reetiquetar, reescribir, retirar, vacio))

    assert (reetiquetar["decision"], reetiquetar["new_module"]) == ("corregir", "infraestructura")
    assert (reescribir["decision"], reescribir["rewritten_question"]) == ("corregir", "nueva")
    assert retirar["decision"] == "retirar"
    # NUNCA se convierten en decisiones del autor/experto, ni siquiera si el revisor era humano
    assert {r["review_level"] for r in (reetiquetar, reescribir, retirar)} == {"preaudit"}
    assert reescribir["reviewed_at"] == ""
    # la decision original y su motivo se conservan en el historial
    event = reetiquetar["history"][0]
    assert event["event"] == "migracion_a_v2"
    assert event["legacy"]["decision"] == "reetiquetar" and event["legacy"]["decision_reason"] == "motivo plantilla"
    assert vacio["decision"] == "" and vacio["history"] == [] and vacio["schema_version"] == SCHEMA_VERSION


def test_migration_is_idempotent():
    record = _legacy(decision="reetiquetar", reviewer="Codex")
    assert migrate_record(record) is True
    snapshot = json.dumps(record, sort_keys=True)
    assert migrate_record(record) is False
    assert json.dumps(record, sort_keys=True) == snapshot


def test_migrate_file_backs_up_and_regeneration_accepts_legacy_csv(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    new_fields = ("schema_version", "history", "review_level", "reviewed_at", "new_module",
                  "rewritten_answer", "evidence_chunk_id_v2", "retire_category")
    legacy = [{k: v for k, v in r.items() if k not in new_fields} for r in _records()]
    legacy[0].update(decision="reetiquetar", reviewer="Codex", decision_reason="x")
    json_path.write_text(json.dumps(legacy), encoding="utf-8")
    # CSV del esquema v1 con el valor viejo
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerow({"audit_id": "v1-001", "decision": "reetiquetar", "reviewer": "Codex", "decision_reason": "x"})

    assert migrate_file(json_path) == 3
    assert list(tmp_path.glob("gold_v2_audit.json.bak-*"))
    saved = _saved(json_path)["v1-001"]
    assert saved["decision"] == "corregir" and saved["review_level"] == "preaudit"
    # el CSV viejo ('reetiquetar') no cuenta como edicion pendiente tras migrar
    _, _, kept = write_outputs(_records(), tmp_path)
    assert kept == 1


def test_only_author_or_expert_reviews_count_as_human():
    assert HUMAN_LEVELS == {"author", "expert"}


# ------------------------------------------------------------------ regresiones (revision cruzada)
def test_stale_decision_returns_to_p1_even_when_history_is_carried_over(tmp_path):
    """Regresion: una decision obsoleta no podia quedar en P3 tras regenerar (caso v1-045)."""
    json_path, _ = _paths(tmp_path)
    write_outputs(_records(), tmp_path)
    data = load_records(json_path)
    data[0]["stale_review"] = {"decision": "retirar", "reviewer": "Codex"}   # historial ya obsoleto, sin decision nueva
    data[0]["priority"] = 3                                                    # estado inconsistente heredado
    json_path.write_text(json.dumps(data), encoding="utf-8")
    write_outputs(data, tmp_path)

    regenerated = _records()                                                   # el generador propone P3
    regenerated[0]["priority"] = 3
    write_outputs(regenerated, tmp_path)
    first = _saved(json_path)["v1-001"]
    assert first["stale_review"] is not None
    assert first["priority"] == 1
    assert any("decision previa obsoleta" in f for f in first["flags"])
    # y no se duplica la bandera al regenerar de nuevo
    write_outputs(_records(), tmp_path)
    assert sum("decision previa obsoleta" in f for f in _saved(json_path)["v1-001"]["flags"]) == 1


def test_new_module_incompatible_with_displayed_candidate_requires_evidence(tmp_path):
    """Regresion: corregir hacia otro modulo sin evidencia compatible no debe aceptarse (caso v1-045)."""
    json_path, csv_path = _paths(tmp_path)
    records = _records()
    records[0].update(status="modulo_coincide", module_v1="dotacion", module_proposed="dotacion")  # candidato = dotacion
    write_outputs(records, tmp_path)

    _edit_csv(csv_path, {"v1-001": {"decision": "corregir", "new_module": "infraestructura",
                                    "decision_reason": "x", **AUTHOR}})
    with pytest.raises(ValueError, match="candidato mostrado pertenece a 'dotacion'"):
        import_csv(csv_path, json_path, CORPUS)

    # con evidencia explicita de infraestructura SI se acepta
    _edit_csv(csv_path, {"v1-001": {"evidence_chunk_id_v2": "c-infra"}})
    assert import_csv(csv_path, json_path, CORPUS) == 1
    # y una evidencia de otro modulo sigue rechazandose
    write_outputs(load_records(json_path), tmp_path)
    _edit_csv(csv_path, {"v1-001": {"evidence_chunk_id_v2": "c-dota"}})
    with pytest.raises(ValueError, match="incompatible"):
        import_csv(csv_path, json_path, CORPUS)


def test_unreliable_match_requires_explicit_evidence_to_correct(tmp_path):
    """En 'sin_correspondencia' no hay candidato fiable: corregir exige evidence_chunk_id_v2."""
    json_path, csv_path = _paths(tmp_path)
    records = _records()
    records[0].update(status="sin_correspondencia", module_v1="dotacion", module_proposed="dotacion")
    write_outputs(records, tmp_path)
    _edit_csv(csv_path, {"v1-001": {"decision": "corregir", "new_module": "infraestructura",
                                    "decision_reason": "x", **PRE}})
    with pytest.raises(ValueError, match="sin candidato fiable"):
        import_csv(csv_path, json_path, CORPUS)
    _edit_csv(csv_path, {"v1-001": {"evidence_chunk_id_v2": "c-infra"}})
    assert import_csv(csv_path, json_path, CORPUS) == 1



# ------------------------------------------------------------------ conservar: solo con evidencia compatible y en alcance
@pytest.mark.parametrize(
    "status",
    ["modulo_cambia", "sin_correspondencia", "fuera_de_corpus", "general_fuera_de_alcance", "sin_chunk_v1"],
)
def test_conservar_is_rejected_unless_module_matches_the_corpus(tmp_path, status):
    json_path, csv_path = _paths(tmp_path)
    records = _records()
    records[0].update(status=status)
    write_outputs(records, tmp_path)
    _edit_csv(csv_path, {"v1-001": {"decision": "conservar", "decision_reason": "mantener la etiqueta", **AUTHOR}})
    with pytest.raises(ValueError, match="solo se puede 'conservar'"):
        import_csv(csv_path, json_path, CORPUS)


def test_conservar_is_accepted_when_module_matches(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    records = _records()
    records[0].update(status="modulo_coincide", module_v1="dotacion", module_proposed="dotacion")
    write_outputs(records, tmp_path)
    _edit_csv(csv_path, {"v1-001": {"decision": "conservar", **AUTHOR}})
    assert import_csv(csv_path, json_path, CORPUS) == 1


def test_keeping_a_disputed_label_goes_through_corregir_with_valid_evidence(tmp_path):
    """Para mantener una etiqueta discutida: corregir con new_module=<v1> y un fragmento de ese modulo."""
    json_path, csv_path = _paths(tmp_path)
    records = _records()                                     # v1=dotacion, candidato=infraestructura
    write_outputs(records, tmp_path)
    _edit_csv(csv_path, {"v1-001": {"decision": "corregir", "new_module": "dotacion", "decision_reason": "x", **AUTHOR}})
    with pytest.raises(ValueError, match="candidato mostrado"):
        import_csv(csv_path, json_path, CORPUS)              # sin evidencia: rechazado
    _edit_csv(csv_path, {"v1-001": {"evidence_chunk_id_v2": "c-dota"}})
    assert import_csv(csv_path, json_path, CORPUS) == 1       # con fragmento de dotacion: aceptado


# --- los tres casos reales reportados en la revision cruzada (v1-004, v1-007, v1-106)
REAL_JSON = __import__("pathlib").Path(__file__).resolve().parent.parent / "eval" / "datasets" / "gold_v2_audit.json"
REAL_STAGING = __import__("pathlib").Path(__file__).resolve().parent.parent / "artifacts" / "staging_v2"


@pytest.mark.skipif(not (REAL_JSON.exists() and REAL_STAGING.exists()), reason="requiere la hoja real y staging_v2")
@pytest.mark.parametrize("audit_id,expected_status", [
    ("v1-004", "modulo_cambia"),            # etiqueta v1 distinta del candidato mostrado
    ("v1-007", "fuera_de_corpus"),          # fuente fuera del capitulo 11
    ("v1-106", "general_fuera_de_alcance"), # general: fuera de alcance
])
def test_real_cases_cannot_be_conservar(audit_id, expected_status):
    from scripts.build_gold_v2_audit import REVIEW_FIELDS, corpus_by_id, validate_review

    record = {r["audit_id"]: r for r in load_records(REAL_JSON)}[audit_id]
    assert record["status"] == expected_status
    state = {f: "" for f in REVIEW_FIELDS}
    state.update(decision="conservar", decision_reason="mantener", **AUTHOR)
    errors = validate_review(record, state, corpus_by_id(REAL_STAGING))
    assert any("solo se puede 'conservar'" in e for e in errors), errors


# ------------------------------------------------------------------ cambiar de decision / limpiar campos
def _store_decision(tmp_path, records, index, **fields):
    """Deja una decision ya guardada (p. ej. una propuesta preaudit) y refresca el CSV."""
    json_path, _ = _paths(tmp_path)
    write_outputs(records, tmp_path)
    data = load_records(json_path)
    data[index].update(fields)
    json_path.write_text(json.dumps(data), encoding="utf-8")
    write_outputs(data, tmp_path)


def test_rejecting_a_corregir_proposal_with_retirar_clears_new_module(tmp_path):
    """Caso real v1-068: propuesta corregir+new_module rechazada con retirar y new_module vacio."""
    json_path, csv_path = _paths(tmp_path)
    _store_decision(tmp_path, _records(), 0, decision="corregir", new_module="infraestructura",
                    decision_reason="propuesta tecnica", **PRE)
    _edit_csv(csv_path, {"v1-001": {"decision": "retirar", "new_module": "", "decision_reason": "rechazo la propuesta",
                                    "retire_category": "sin_evidencia", "review_level": "author", "reviewer": "PM",
                                    "reviewed_at": "2026-10-07"}})
    assert import_csv(csv_path, json_path, CORPUS) == 1
    saved = _saved(json_path)["v1-001"]
    assert saved["decision"] == "retirar" and saved["new_module"] == ""
    assert (saved["review_level"], saved["reviewer"]) == ("author", "PM")
    event = saved["history"][-1]                       # la propuesta anterior queda en el historial
    assert event["previous"]["decision"] == "corregir" and event["previous"]["new_module"] == "infraestructura"
    assert event["previous"]["review_level"] == "preaudit"
    assert event["decision_changed"] == ["corregir", "retirar"]
    assert "new_module" in event["cleared_fields"]


def test_changing_from_retirar_to_conservar_clears_retire_fields(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    records = _records()
    records[0].update(status="modulo_coincide", module_v1="dotacion", module_proposed="dotacion")
    _store_decision(tmp_path, records, 0, decision="retirar", decision_reason="sin evidencia",
                    retire_category="sin_evidencia", **PRE)
    _edit_csv(csv_path, {"v1-001": {"decision": "conservar", "decision_reason": "", "retire_category": "", **AUTHOR}})
    assert import_csv(csv_path, json_path, CORPUS) == 1
    saved = _saved(json_path)["v1-001"]
    assert saved["decision"] == "conservar" and saved["retire_category"] == "" and saved["decision_reason"] == ""
    assert saved["history"][-1]["previous"]["retire_category"] == "sin_evidencia"


def test_cells_left_unchanged_with_a_new_decision_are_validated_not_silently_kept(tmp_path):
    """Si al cambiar de decision se deja el campo de la anterior, la validacion lo senala."""
    json_path, csv_path = _paths(tmp_path)
    _store_decision(tmp_path, _records(), 0, decision="corregir", new_module="infraestructura",
                    decision_reason="propuesta", **PRE)
    _edit_csv(csv_path, {"v1-001": {"decision": "retirar", "decision_reason": "x", "retire_category": "sin_evidencia", **AUTHOR}})
    # new_module sigue escrito en la celda (no se borro) -> retirar no lo admite
    with pytest.raises(ValueError, match="solo 'corregir' admite"):
        import_csv(csv_path, json_path, CORPUS)


def test_changing_decision_requires_the_reviewers_own_identity(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    _store_decision(tmp_path, _records(), 0, decision="corregir", new_module="infraestructura",
                    decision_reason="propuesta", **PRE)
    _edit_csv(csv_path, {"v1-001": {"decision": "retirar", "new_module": "", "decision_reason": "x",
                                    "retire_category": "sin_evidencia"}})   # reviewer/level siguen siendo los de Codex
    with pytest.raises(ValueError, match="escribe TU nombre"):
        import_csv(csv_path, json_path, CORPUS)


def test_explicit_clear_token_empties_one_field_without_changing_the_decision(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    _store_decision(tmp_path, _records(), 0, decision="corregir", new_module="infraestructura",
                    rewritten_question="vieja", rewritten_answer="resp", decision_reason="r", **PRE)
    # celda vacia = conserva; <borrar> = limpia
    _edit_csv(csv_path, {"v1-001": {"rewritten_question": "<borrar>", "rewritten_answer": ""}})
    assert import_csv(csv_path, json_path, CORPUS) == 1
    saved = _saved(json_path)["v1-001"]
    assert saved["rewritten_question"] == "" and saved["rewritten_answer"] == "resp"
    assert saved["decision"] == "corregir" and saved["new_module"] == "infraestructura"
    assert saved["history"][-1]["cleared_fields"] == ["rewritten_question"]


def test_clearing_does_not_bypass_validation(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    _store_decision(tmp_path, _records(), 0, decision="corregir", new_module="infraestructura", decision_reason="r", **PRE)
    _edit_csv(csv_path, {"v1-001": {"new_module": "<borrar>"}})   # era la unica correccion
    with pytest.raises(ValueError, match="al menos uno"):
        import_csv(csv_path, json_path, CORPUS)
    _edit_csv(csv_path, {"v1-001": {"new_module": "infraestructura", "reviewer": "<borrar>"}})
    with pytest.raises(ValueError, match="falta el revisor"):
        import_csv(csv_path, json_path, CORPUS)


def test_blank_cells_still_protect_when_the_decision_does_not_change(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    _store_decision(tmp_path, _records(), 0, decision="corregir", new_module="infraestructura", decision_reason="r", **PRE)
    _edit_csv(csv_path, {"v1-001": {"new_module": "", "decision_reason": "", "reviewer": ""}})  # CSV desactualizado/vacio
    import_csv(csv_path, json_path, CORPUS)
    saved = _saved(json_path)["v1-001"]
    assert saved["new_module"] == "infraestructura" and saved["reviewer"] == "Codex"



# ------------------------------------------------------------------ atribucion: nunca se hereda de una propuesta preaudit
def _preaudit_corregir(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    _store_decision(tmp_path, _records(), 0, decision="corregir", new_module="infraestructura",
                    decision_reason="propuesta tecnica", reviewer="Codex (preauditoria tecnica)", review_level="preaudit")
    return json_path, csv_path


@pytest.mark.parametrize("level", ["author", "expert"])
@pytest.mark.parametrize(
    "reviewer",
    ["", "Codex (preauditoria tecnica)", "codex (preauditoria tecnica)", "<borrar>"],
    ids=["vacio", "mismo_de_la_propuesta", "mismo_distinta_capitalizacion", "borrado"],
)
def test_promoting_a_preaudit_proposal_requires_an_explicit_different_reviewer(tmp_path, level, reviewer):
    json_path, csv_path = _preaudit_corregir(tmp_path)
    before = json_path.read_text(encoding="utf-8")
    _edit_csv(csv_path, {"v1-001": {"review_level": level, "reviewer": reviewer, "reviewed_at": "2026-10-07"}})
    with pytest.raises(ValueError, match="escribe TU nombre"):
        import_csv(csv_path, json_path, CORPUS)
    assert json_path.read_text(encoding="utf-8") == before   # no quedo ninguna atribucion heredada


def test_confirming_a_preaudit_proposal_with_my_own_name_is_accepted(tmp_path):
    json_path, csv_path = _preaudit_corregir(tmp_path)
    _edit_csv(csv_path, {"v1-001": {"review_level": "author", "reviewer": "PM", "reviewed_at": "2026-10-07"}})
    assert import_csv(csv_path, json_path, CORPUS) == 1
    saved = _saved(json_path)["v1-001"]
    assert (saved["decision"], saved["review_level"], saved["reviewer"]) == ("corregir", "author", "PM")
    assert saved["history"][-1]["previous"]["reviewer"] == "Codex (preauditoria tecnica)"   # queda trazado


def test_changing_a_preaudit_decision_keeping_the_proposals_reviewer_is_rejected(tmp_path):
    """Aunque cambien nivel y fecha, el revisor de la propuesta no puede quedar como autor del cambio."""
    json_path, csv_path = _preaudit_corregir(tmp_path)
    _edit_csv(csv_path, {"v1-001": {"decision": "retirar", "new_module": "", "decision_reason": "x",
                                    "retire_category": "sin_evidencia", "review_level": "author",
                                    "reviewed_at": "2026-10-07"}})   # reviewer sigue siendo el de la propuesta
    with pytest.raises(ValueError, match="escribe TU nombre"):
        import_csv(csv_path, json_path, CORPUS)
    _edit_csv(csv_path, {"v1-001": {"reviewer": "PM"}})
    assert import_csv(csv_path, json_path, CORPUS) == 1
    assert _saved(json_path)["v1-001"]["reviewer"] == "PM"


def test_a_human_decision_can_be_changed_by_the_same_reviewer_with_new_date(tmp_path):
    json_path, csv_path = _paths(tmp_path)
    first_pass = dict(AUTHOR, reviewed_at="2026-10-01")
    _store_decision(tmp_path, _records(), 0, decision="retirar", decision_reason="x", retire_category="sin_evidencia", **first_pass)
    _edit_csv(csv_path, {"v1-001": {"decision": "corregir", "new_module": "infraestructura", "decision_reason": "revise otra vez",
                                    "retire_category": "", "reviewed_at": "2026-10-07"}})
    assert import_csv(csv_path, json_path, CORPUS) == 1       # el autor puede corregirse a si mismo
    assert _saved(json_path)["v1-001"]["decision"] == "corregir"
