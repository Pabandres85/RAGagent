"""
Interfaz Streamlit del asistente normativo - RAG Multi-Agente.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
QUERY_TIMEOUT = 120
HEALTH_TIMEOUT = 5

MODULE_LABELS = {
    "talento_humano": "Talento Humano",
    "infraestructura": "Infraestructura",
    "dotacion": "Dotación",
    "medicamentos_dispositivos": "Medicamentos y Dispositivos",
    "procesos_prioritarios": "Procesos Prioritarios",
    "historia_clinica": "Historia Clínica",
    "interdependencia": "Interdependencia",
    "transversal": "Consulta Transversal",
}

EXAMPLE_QUESTIONS = [
    "Que titulos o certificaciones debe tener el personal de talento humano en salud?",
    "Que requisitos de infraestructura aplican para un servicio quirurgico?",
    "Que equipos son obligatorios para un servicio de urgencias?",
]

STATUS_ICONS = {
    "cumple": "OK",
    "no_cumple": "X",
    "no_aplica": "NA",
    "pendiente": "...",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* =========================================
           1. FONDOS GENERALES Y PROTECCION DE TEMA
           ========================================= */
        [data-testid="stAppViewContainer"] {
            background-color: #f8fafc !important;
        }

        /* =========================================
           2. ÁREA PRINCIPAL (LETRAS OSCURAS)
           ========================================= */
        /* Forzar texto oscuro solo en el area principal para no dañar el sidebar */
        /* Agregamos h4, h5 y h6 para que los títulos de las tarjetas no se pierdan */
        [data-testid="stMain"] p,
        [data-testid="stMain"] span,
        [data-testid="stMain"] label,
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] h5,
        [data-testid="stMain"] h6,
        [data-testid="stMain"] li {
            color: #0f172a !important;
        }

        /* =========================================
           3. SIDEBAR (MENU LATERAL) - TEMA OSCURO
           ========================================= */
        [data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid #1e293b !important;
        }

        /* Forzar texto claro solo en el sidebar */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] li {
            color: #f1f5f9 !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: #334155 !important;
        }

        /* =========================================
           4. HERO SECTION (ENCABEZADO)
           ========================================= */
        .hero {
            background: linear-gradient(135deg, #0d6b5f, #1e2a2f);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        .hero * {
            color: #ffffff !important;
        }

        .hero h1 {
            margin-bottom: 0.5rem;
            font-size: 2.2rem;
            font-weight: 800;
        }

        .hero p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* =========================================
           5. AREA DE CONSULTA Y BOTONES
           ========================================= */
        div[data-baseweb="base-input"] {
            background-color: #ffffff !important;
            border: 2px solid #cbd5e1 !important;
            border-radius: 12px !important;
        }

        div[data-baseweb="textarea"] textarea {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            background-color: transparent !important;
            font-size: 1.05rem !important;
            padding: 0.5rem !important;
        }

        div[data-baseweb="textarea"] textarea::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
        }

        div[data-baseweb="base-input"]:focus-within {
            border-color: #0d6b5f !important;
            box-shadow: 0 0 0 2px rgba(13, 107, 95, 0.2) !important;
        }

        div[data-testid="stButton"] button {
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            background-color: #ffffff !important;
            color: #0f172a !important;
            font-weight: 600 !important;
            transition: all 0.2s ease-in-out;
        }

        div[data-testid="stButton"] button:hover {
            border-color: #0d6b5f !important;
            color: #0d6b5f !important;
            background-color: #f0fdfa !important;
        }

        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #0d6b5f !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 4px 6px -1px rgba(13, 107, 95, 0.4) !important;
        }

        div[data-testid="stButton"] button[kind="primary"] * {
            color: #ffffff !important;
        }

        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #0a5249 !important;
            box-shadow: 0 6px 8px -1px rgba(13, 107, 95, 0.5) !important;
        }

        /* =========================================
           6. TARJETAS Y CONTENEDORES
           ========================================= */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        }

        [data-testid="stMetricLabel"] * {
            color: #64748b !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] div {
            color: #0d6b5f !important;
            font-weight: 800 !important;
        }

        [data-testid="stExpander"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
        }

        [data-testid="stExpander"] p {
            color: #0f172a !important;
        }

        [data-testid="stHeadingWithActionElements"] a {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Asistente Normativo - Habilitacion en Salud</h1>
            <p>Sistema multi-agente RAG para verificacion rapida y fundamentada de la Resolucion 3100 de 2019.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_health() -> dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=HEALTH_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"status": "down", "llm_provider": "-", "llm_ok": False}


def render_sidebar(health: dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown("### Monitoreo del Sistema")

        api_ok = health.get("status") == "ok"
        llm_ok = health.get("llm_ok", False)

        st.markdown(f"{'[OK]' if api_ok else '[X]'} **API Backend:** {'Operativa' if api_ok else 'Caida'}")
        st.markdown(f"{'[OK]' if llm_ok else '[X]'} **LLM Local:** {'Conectado' if llm_ok else 'Desconectado'}")
        st.markdown(f"**Motor LLM:** `{health.get('llm_provider', 'Desconocido')}`")

        st.divider()
        st.markdown("### Agentes Especialistas")
        for key, label in MODULE_LABELS.items():
            if key != "transversal":
                st.markdown(f"- {label}")


def render_top_strip(health: dict[str, Any], payload: dict[str, Any] | None) -> None:
    routing = payload.get("routing", {}) if payload else {}
    module_key = routing.get("module")
    module_label = MODULE_LABELS.get(module_key, "Ninguno") if module_key else "Esperando consulta..."
    confidence = float(routing.get("confidence", 0.0)) if payload else 0.0
    confidence_display = confidence
    search_status = "Completada" if payload else "En espera"
    if health.get("status") != "ok":
        search_status = "API no disponible"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Agente Especializado", module_label)
    with col2:
        st.metric("Confianza de Ruteo", f"{confidence_display:.1%}" if payload else "-")
    with col3:
        st.metric("Estado de Busqueda", search_status)

    st.markdown("<br>", unsafe_allow_html=True)


def call_query(question: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/query",
        json={"question": question},
        timeout=QUERY_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def render_response(payload: dict[str, Any]) -> None:
    routing = payload.get("routing", {})
    response = payload.get("response")
    valid = payload.get("valid", False)
    warnings = payload.get("warnings", [])
    errors = payload.get("errors", [])
    routing_confidence = float(routing.get("confidence", 0.0))
    progress_value = max(0.0, min(1.0, routing_confidence))

    col_main, col_side = st.columns([2.2, 1], gap="large")

    with col_side:
        with st.container(border=True):
            st.markdown("#### Decision del Orquestador")
            st.markdown(
                f"**Modulo Asignado:**<br>`{MODULE_LABELS.get(routing.get('module'), routing.get('module', '-'))}`",
                unsafe_allow_html=True,
            )
            st.progress(
                progress_value,
                text=f"Confianza de ruteo: {routing_confidence:.1%}",
            )
            st.info(f"**Criterio:** {routing.get('reasoning', 'Sin justificacion')}")

            routed_modules = routing.get("modules", [])
            if isinstance(routed_modules, list) and len(routed_modules) > 1:
                labels = [MODULE_LABELS.get(module, module) for module in routed_modules]
                st.caption("Complementa con: " + ", ".join(labels))

        with st.container(border=True):
            st.markdown("#### Validacion (Guardrails)")
            if valid:
                st.success("Respuesta validada y fundamentada en la norma.")
            else:
                st.error("Alerta: respuesta fuera del formato o la norma.")

            filtered_warnings = []
            for warning in warnings:
                if warning.startswith("Confianza baja:") and response:
                    continue
                filtered_warnings.append(warning)

            for warning in filtered_warnings:
                st.warning(warning)
            for error in errors:
                st.error(error)

    with col_main:
        if response:
            st.markdown("### Analisis Normativo")
            st.info(response.get("answer", "Sin respuesta"))

            checklist = response.get("checklist", [])
            if checklist:
                st.markdown("### Elementos de Verificacion")
                with st.container(border=True):
                    for item in checklist:
                        status_icon = STATUS_ICONS.get(item.get("status", "pendiente"), "-")
                        numeral_badge = f"`Numeral {item['numeral']}`" if item.get("numeral") else ""
                        st.markdown(f"**{status_icon}** {item.get('item', '')} {numeral_badge}")

            citations = response.get("citations", [])
            if citations:
                st.markdown("### Evidencia Extraida (FAISS)")
                for index, citation in enumerate(citations, 1):
                    header = (
                        f"Resolucion 3100 - Numeral {citation.get('numeral', 'N/A')} "
                        f"(Pagina {citation.get('page', 'N/A')})"
                    )
                    with st.expander(f"Fragmento {index}: {header}"):
                        st.write(citation.get("text", "Sin texto extraido."))
                        st.caption(f"Estado normativo: {citation.get('vigencia', 'Vigente')}")
        else:
            st.warning("El motor de IA no devolvio una respuesta estructurada.")


def main() -> None:
    st.set_page_config(
        page_title="Asistente 3100 | RAG",
        page_icon="RAG",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    health = get_health()
    render_sidebar(health)
    render_hero()

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    current_result = st.session_state.last_result
    top_strip_slot = st.empty()

    with st.container(border=True):
        st.markdown("### Nueva Consulta")
        question = st.text_area(
            "Consulta Normativa",
            value=st.session_state.question_input,
            placeholder="Escribe tu consulta aqui... (ej. Cuantas camas necesito para cuidado intermedio adulto?)",
            height=120,
            label_visibility="collapsed",
        )

        col_btn, col_help = st.columns([1, 4])
        with col_btn:
            run_query = st.button("Consultar Sistema", type="primary", use_container_width=True)
        with col_help:
            st.caption("Tus preguntas se procesan localmente. Ningun dato sale de tu entorno.")

    st.markdown("<br>**No sabes que preguntar? Prueba un ejemplo:**", unsafe_allow_html=True)
    example_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if example_cols[index].button(example, use_container_width=True):
            st.session_state.question_input = example
            st.rerun()

    if run_query:
        st.session_state.question_input = question
        if not question.strip():
            st.warning("Por favor, escribe una pregunta valida.")
        else:
            with st.status("Despertando al Orquestador...", expanded=True) as status:
                try:
                    time.sleep(0.5)
                    st.write("Analizando intencion y definiendo ruteo...")
                    time.sleep(0.5)
                    st.write("Buscando fragmentos normativos en FAISS...")
                    st.write("Construyendo respuesta fundamentada...")

                    st.session_state.last_result = call_query(question.strip())
                    current_result = st.session_state.last_result

                    status.update(label="Consulta procesada con exito.", state="complete", expanded=False)
                except Exception as exc:
                    st.session_state.last_result = None
                    current_result = None
                    status.update(label="Error durante el procesamiento", state="error", expanded=True)
                    st.error(f"Error de comunicacion con la API local: {exc}")

    with top_strip_slot.container():
        render_top_strip(health, current_result)

    if current_result:
        st.divider()
        render_response(current_result)


if __name__ == "__main__":
    main()