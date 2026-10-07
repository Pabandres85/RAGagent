"""
Pruebas de la politica de corpus de scripts/ingest.py.

El PDF es un manual OCR: el nucleo normativo es el capitulo 11 ("Estandares y criterios
de habilitacion", desde la pag. 59) y, dentro de cada servicio, cada modulo empieza en
una linea propia "Estandar de <modulo>". Las menciones sueltas dentro del texto no deben
cambiar el modulo.
"""
import pytest

from core.config import settings
from core.metadata_store import MODULES
from scripts.ingest import (
    build_chunks,
    extract_pages,
    find_corpus_start_page,
    is_header_noise,
    match_group_heading,
    match_service_heading,
    match_standard_heading,
)

PDF = settings.data_raw_dir / "resolucion-3100-de-2019.pdf"


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Estándar de talento humano", "talento_humano"),
        ("11.1.1. Estándar de talento humano", "talento_humano"),
        ("Estándar de infraestructura", "infraestructura"),
        ("11.1.2. Estandar de dotación", "dotacion"),
        ("Estándar de medicamentos, dispositivos médicos e insumos", "medicamentos_dispositivos"),
        ("Estándar de procesos prioritarios", "procesos_prioritarios"),
        ("Estandar de historia clinica y ,registros", "historia_clinica"),
        ("Estándar de interdependencia", "interdependencia"),
        ("Estándar de fnfraeslructura", "infraestructura"),  # ruido de OCR
    ],
)
def test_standard_headings_are_detected(line, expected):
    assert match_standard_heading(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "estándar de procesos prioritarios.",  # referencia cruzada (minuscula, con punto)
        "documentado en el estándar de procesos prioritarios.",
        "Conforme al estándar de dotación del servicio",  # no empieza la linea
        "El estándar de talento humano exige lo siguiente",  # sentencia, no encabezado
        "Estándar de calidad",  # modulo inexistente
    ],
)
def test_cross_references_do_not_change_module(line):
    assert match_standard_heading(line) is None


def test_service_headings_tolerate_ocr_noise():
    assert match_service_heading("11.2.3. SERVICIO DE VACUNACiON") == "11.2.3 Servicio De Vacunacion"
    assert match_service_heading("11.3.12 SERVICIO DE LABORATORIO CLiNICO") is not None
    assert match_service_heading("11.1. Lavamanos.") is None  # criterio, no servicio
    assert match_service_heading("11.2. Meson de trabajo.") is None


def test_group_headings_cut_context():
    assert match_group_heading("11.4 GRUPO INTERNACIÓN") == "GROUP"
    assert match_group_heading("11.1. ESTANDARES y CRITERIOS APLICABLES A TODOS LOS SERVICIOS") == "ALL"
    assert match_group_heading("11.4.1 SERVICIO DE HOSPITALIZACION") is None


def test_header_noise_lines_are_recognised():
    assert is_header_noise("RESOLUCiÓN NÚMERO C003100")
    assert is_header_noise("RESOLUCI?N N?MERO 0003100")
    assert is_header_noise("C003100")
    assert is_header_noise("Continuación ele la resolución: \"Por la cual se definen")
    assert is_header_noise("Continuación de [a resolución: \"Por la cual")
    assert is_header_noise('de Prestadores y Habilitación de Servicios de Salud"')
    assert is_header_noise("DE 2019")
    assert is_header_noise('Continuación de la resolución: "Por la cual se definen los procedimientos')
    assert not is_header_noise("11.1. Lavamanos.")
    assert not is_header_noise("El prestador cuenta con camillas para el traslado de pacientes.")


@pytest.fixture(scope="module")
def pages():
    if not PDF.exists():
        pytest.skip(f"Falta el PDF {PDF}")
    return extract_pages(PDF)


def test_corpus_starts_at_chapter_11(pages):
    assert find_corpus_start_page(pages) == 59


def test_chunks_respect_corpus_policy(pages):
    chunks = build_chunks(pages, PDF.name)
    assert chunks
    # Nada del articulado ni de los capitulos 1-10.
    assert min(c.page for c in chunks) >= 59
    # Los siete modulos tienen contenido y ninguno concentra la mayoria del corpus
    # (antes, talento_humano acumulaba el 58 % por arrastre de modulo).
    counts = {m: sum(1 for c in chunks if c.module == m) for m in MODULES}
    assert all(n > 0 for n in counts.values()), counts
    assert max(counts.values()) / len(chunks) < 0.35, counts
    # El encabezado repetido de pagina no debe llegar a los chunks (patrones amplios:
    # el OCR deforma fecha, codigo, "Pagina N de 230" y "Continuacion de la resolucion").
    import re

    noise = re.compile(
        r"RESOLUC.{0,4}N\s+N.{0,4}MERO"
        r"|Continua.{1,4}n\b.{0,14}resol"
        r"|P.gina\s*\d+\s*de\s*230"
        r"|adopt.\s+el\s+Manual\s+de\s+Inscri"
        r"|(?:^|\n)\W{0,4}\d?\s?\d?\W{0,3}N[O0]\W?[VIJ\\!]\W{0,3}\s?20\w{1,3}\W*(?:\n|$)",
        re.IGNORECASE,
    )
    offenders = [(c.page, m.group(0)[:40]) for c in chunks if (m := noise.search(c.text))]
    assert not offenders, offenders[:5]


def test_manual_start_page_is_a_working_fallback(pages):
    auto = build_chunks(pages, PDF.name)
    manual = build_chunks(pages, PDF.name, start_page=59)
    assert len(manual) == len(auto) > 0


def test_services_missed_by_ocr_are_recovered():
    # 11.4.4 llega con coma ("11,4.4") y 11.4.9 sin numeral.
    assert match_service_heading("11,4.4 SERVICIO DE CUIDADO INTERMEDIO NEONATAL") == (
        "11.4.4 Servicio De Cuidado Intermedio Neonatal"
    )
    assert match_service_heading("SERVICIO DE CUIDADO INTENSIVO ADULTOS") == "Servicio De Cuidado Intensivo Adultos"
    # Texto corrido de un criterio no es un titulo de servicio.
    assert match_service_heading("Servicio de cuidado intensivo adulto cuenta con camas") is None
    assert match_service_heading("El servicio de hospitalizacion garantiza lo siguiente") is None


def test_services_are_assigned_to_the_right_pages(pages):
    chunks = build_chunks(pages, PDF.name)
    by_page = {}
    for c in chunks:
        by_page.setdefault(c.page, set()).add(c.service or "")
    assert any("11.4.4" in s for s in by_page.get(166, set())), by_page.get(166)
    assert any("Intensivo Adultos" in s for s in by_page.get(184, set())), by_page.get(184)


def test_start_page_works_when_chapter_heading_is_absent():
    """Respaldo real: un documento sin el titulo del cap. 11 solo se ingesta con --start-page."""
    synthetic = [
        (1, "Texto introductorio sin encabezados de capitulo."),
        (
            2,
            "Estándar de talento humano\n1. El prestador cuenta con el talento humano necesario "
            "para la prestación del servicio de salud habilitado.",
        ),
        (
            3,
            "Estándar de infraestructura\n2. El prestador cuenta con áreas y ambientes adecuados "
            "para la prestación del servicio de salud habilitado.",
        ),
    ]
    with pytest.raises(RuntimeError):
        build_chunks(synthetic, "sintetico.pdf")

    chunks = build_chunks(synthetic, "sintetico.pdf", start_page=2)
    assert {c.module for c in chunks} == {"talento_humano", "infraestructura"}
    assert min(c.page for c in chunks) == 2


def test_service_description_before_first_standard_heading_has_no_module():
    """La descripcion de un servicio no es criterio de ningun estandar (antes se asignaba a talento_humano)."""
    synthetic = [
        (
            1,
            "11. ESTÁNDARES Y CRITERIOS DE HABILITACIÓN\n"
            "11.2.1. SERVICIO DE CONSULTA EXTERNA GENERAL\n"
            "Descripción: servicio ambulatorio que presta atención general a los usuarios del prestador.\n"
            "No se podrá habilitar como servicio único en el prestador de servicios de salud.\n"
            "Estándar de talento humano\n"
            "1. El prestador cuenta con el talento humano necesario para el servicio habilitado.\n"
            "Estándar de infraestructura\n"
            "2. El prestador cuenta con las áreas y ambientes necesarios para el servicio habilitado.\n",
        ),
    ]
    chunks = build_chunks(synthetic, "sintetico.pdf")
    assert {c.module for c in chunks} == {"talento_humano", "infraestructura"}
    assert not any("servicio único" in c.text or "Descripción" in c.text for c in chunks)
    assert all(c.service == "11.2.1 Servicio De Consulta Externa General" for c in chunks)


def test_service_whose_first_heading_is_not_talento_keeps_unlabelled_text_out():
    synthetic = [
        (
            1,
            "11. ESTÁNDARES Y CRITERIOS DE HABILITACIÓN\n"
            "11.6.2 SERVICIO DE TRANSPORTE ASISTENCIAL\n"
            "1. Criterio numerado cuyo encabezado de estándar se perdió en el OCR de la página.\n"
            "Estándar de infraestructura\n"
            "2. El prestador cuenta con las áreas y ambientes necesarios para el servicio habilitado.\n",
        ),
    ]
    chunks = build_chunks(synthetic, "sintetico.pdf")
    assert {c.module for c in chunks} == {"infraestructura"}  # un numeral solo no prueba el estandar
