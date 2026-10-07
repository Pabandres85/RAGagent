"""Busqueda de fragmentos para la auditoria."""
from scripts.find_chunk import search_chunks

CHUNKS = [
    {"chunk_id": "a1", "module": "dotacion", "page": 182, "numeral": "43.1", "service": "11.4.9 Servicio De Cuidado Intensivo Adultos", "text": "Tubos endotraqueales de varios calibres"},
    {"chunk_id": "b2", "module": "infraestructura", "page": 182, "numeral": None, "service": "11.4.9 Servicio De Cuidado Intensivo Adultos", "text": "Área para el depósito de equipos"},
    {"chunk_id": "c3", "module": "dotacion", "page": 90, "numeral": "5.2", "service": "11.2.3 Servicio De Vacunacion", "text": "Cadena de frío"},
]


def test_filters_are_combined_with_and():
    assert [c["chunk_id"] for c in search_chunks(CHUNKS, page=182)] == ["a1", "b2"]
    assert [c["chunk_id"] for c in search_chunks(CHUNKS, page=182, module="dotacion")] == ["a1"]


def test_text_and_service_ignore_accents_and_case():
    assert [c["chunk_id"] for c in search_chunks(CHUNKS, text="AREA PARA EL DEPOSITO")] == ["b2"]
    assert [c["chunk_id"] for c in search_chunks(CHUNKS, service="vacunacion")] == ["c3"]


def test_numeral_matches_metadata_or_text_and_id_is_exact():
    assert [c["chunk_id"] for c in search_chunks(CHUNKS, numeral="43.1")] == ["a1"]
    assert [c["chunk_id"] for c in search_chunks(CHUNKS, chunk_id="c3")] == ["c3"]
    assert search_chunks(CHUNKS, page=1) == []
