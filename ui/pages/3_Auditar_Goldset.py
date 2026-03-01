"""
Pagina Streamlit para auditar y limpiar el gold set de forma interactiva.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
GOLD_SET_PATH = BASE_DIR / "eval" / "datasets" / "gold_set.json"
METADATA_DIR = BASE_DIR / "data" / "metadata"

MODULE_CHOICES = [
    "general",
    "talento_humano",
    "infraestructura",
    "dotacion",
    "medicamentos_dispositivos",
    "procesos_prioritarios",
    "historia_clinica",
    "interdependencia",
]


def load_data() -> list[dict]:
    if not GOLD_SET_PATH.exists():
        return []
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_data(data: list[dict]) -> None:
    with open(GOLD_SET_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def get_chunk_text(module: str, chunk_id: str | None) -> str:
    if not chunk_id:
        return (
            "Pregunta elaborada manualmente (sin chunk_id).\n"
            "Corresponde a una pregunta general/administrativa de la Resolución 3100."
        )

    meta_path = METADATA_DIR / f"{module}.json"
    if not meta_path.exists():
        return f"Archivo de metadatos no encontrado para módulo '{module}'."

    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            chunks = json.load(handle)
    except Exception as exc:  # pragma: no cover
        return f"Error leyendo metadatos: {exc}"

    for chunk in chunks:
        if chunk.get("chunk_id") == chunk_id:
            return chunk.get("text", "Sin texto.")

    return "Chunk no encontrado."


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

        /* Forzar texto oscuro SOLO en elementos tipograficos (sin dañar botones/inputs) */
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] h5,
        [data-testid="stAppViewContainer"] h6,
        [data-testid="stAppViewContainer"] li {
            color: #0f172a !important;
        }

        /* =========================================
           2. SIDEBAR (MENU LATERAL) - TEMA OSCURO
           ========================================= */
        [data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid #1e293b !important;
        }

        [data-testid="stSidebar"] p,
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
           3. HERO SECTION Y TARJETAS
           ========================================= */
        .hero {
            background: linear-gradient(135deg, #0d6b5f, #1e2a2f);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        .hero * {
            color: #ffffff !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
        }

        /* =========================================
           4. SELECTBOX (MENU DESPLEGABLE)
           ========================================= */
        /* Fondo de la caja selectora */
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border-color: #cbd5e1 !important;
            color: #0f172a !important;
        }
        /* Texto de la opcion seleccionada (span, div, p dentro del select) */
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] p {
            color: #0f172a !important;
        }
        /* Fondo de la lista que se despliega */
        ul[data-baseweb="menu"] {
            background-color: #ffffff !important;
        }
        /* Texto de las opciones en la lista */
        ul[data-baseweb="menu"] li,
        ul[data-baseweb="menu"] li span,
        ul[data-baseweb="menu"] li div {
            color: #0f172a !important;
        }

        /* =========================================
           5. BOTONES
           ========================================= */
        /* Botones secundarios (Eliminar, Saltar) */
        div[data-testid="stButton"] button {
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            background-color: #ffffff !important;
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stButton"] button p,
        div[data-testid="stButton"] button span {
            color: #0f172a !important; /* Letras oscuras */
            font-weight: 600 !important;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #0d6b5f !important;
            background-color: #f0fdfa !important;
        }
        div[data-testid="stButton"] button:hover p,
        div[data-testid="stButton"] button:hover span {
            color: #0d6b5f !important;
        }

        /* Boton primario (Guardar y Avanzar) */
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #0d6b5f !important;
            border: none !important;
        }
        div[data-testid="stButton"] button[kind="primary"] p,
        div[data-testid="stButton"] button[kind="primary"] span {
            color: #ffffff !important; /* Letras blancas para contraste */
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #0a5249 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Auditor Gold Set", page_icon="🧹", layout="wide")
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Auditoria Semantica del Gold Set</h1>
            <p>Verifica que las preguntas pertenezcan realmente al modulo asignado y depura entradas ambiguas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = load_data()
    if not data:
        st.warning("No se encontro el archivo gold_set.json")
        return

    if "audit_goldset_index" not in st.session_state:
        st.session_state.audit_goldset_index = 0

    total = len(data)
    idx = st.session_state.audit_goldset_index

    if idx >= total:
        st.success("Has revisado todo el Gold Set.")
        if st.button("Volver a empezar", use_container_width=True):
            st.session_state.audit_goldset_index = 0
            st.rerun()
        return

    item = data[idx]
    current_module = item.get("module")
    chunk_id = item.get("chunk_id")

    progress_value = 0.0 if total == 0 else idx / total
    st.progress(progress_value, text=f"Progreso: pregunta {idx + 1} de {total}")

    col_ctx, col_qa = st.columns([1.2, 1], gap="large")

    with col_ctx:
        with st.container(border=True):
            st.markdown("### Fragmento Fuente")
            st.caption(
                f"Modulo actual: `{current_module}` | Pagina: {item.get('page')} | Numeral: {item.get('numeral')}"
            )
            st.info(get_chunk_text(str(current_module), chunk_id))

    with col_qa:
        with st.container(border=True):
            st.markdown("### Pregunta Generada")
            st.warning(item.get("question", ""))
            st.markdown("### Respuesta Esperada")
            st.success(item.get("answer", ""))

    st.divider()
    st.markdown("### Acciones")

    col1, col2, col3 = st.columns(3)

    with col1:
        new_module = st.selectbox(
            "Corregir modulo:",
            MODULE_CHOICES,
            index=MODULE_CHOICES.index(current_module) if current_module in MODULE_CHOICES else 0,
        )
        if st.button("Guardar y avanzar", type="primary", use_container_width=True):
            data[idx]["module"] = new_module
            save_data(data)
            st.session_state.audit_goldset_index += 1
            st.rerun()

    with col2:
        # Espaciador vertical para alinear los botones con el selectbox
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Eliminar pregunta", use_container_width=True):
            data.pop(idx)
            save_data(data)
            st.rerun()

    with col3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Saltar", use_container_width=True):
            st.session_state.audit_goldset_index += 1
            st.rerun()


if __name__ == "__main__":
    main()