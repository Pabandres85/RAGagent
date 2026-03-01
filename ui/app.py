"""
Interfaz Streamlit del asistente normativo.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Asistente Normativo", page_icon="RAG", layout="wide")

st.title("Asistente Normativo de Habilitacion")
st.caption("Resolucion 3100 de 2019")

with st.sidebar:
    st.subheader("Estado")
    try:
        health = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5).json()
        st.success(f"API: {health.get('status', 'desconocido')}")
        st.write(f"Proveedor LLM: {health.get('llm_provider', '-')}")
        st.write(f"LLM OK: {'Si' if health.get('llm_ok') else 'No'}")
    except Exception:
        st.error("No se pudo consultar la API")

question = st.text_area(
    "Escribe tu consulta normativa:",
    placeholder="Ejemplo: Que requisitos exige talento humano para urgencias?",
    height=120,
)

if st.button("Consultar", type="primary", disabled=not question.strip()):
    with st.spinner("Consultando..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/query",
                json={"question": question},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            st.error(f"Error consultando API: {exc}")
            st.stop()

    routing = data.get("routing", {})
    st.subheader("Ruteo")
    st.write(routing)

    if data.get("valid") and data.get("response"):
        payload = data["response"]
        st.subheader("Respuesta")
        st.write(payload.get("answer", ""))

        citations = payload.get("citations", [])
        if citations:
            st.subheader("Citas")
            for citation in citations:
                label = " | ".join(
                    part
                    for part in [
                        citation.get("resolution"),
                        f"Numeral {citation['numeral']}" if citation.get("numeral") else "",
                        f"Pagina {citation['page']}" if citation.get("page") else "",
                    ]
                    if part
                )
                st.info(label)
                if citation.get("text"):
                    with st.expander("Ver fragmento"):
                        st.write(citation["text"])

        checklist = payload.get("checklist", [])
        if checklist:
            st.subheader("Checklist")
            for item in checklist:
                st.write(f"- {item.get('item', '')} [{item.get('status', 'pendiente')}]")
    else:
        st.warning("No se obtuvo una respuesta valida.")
        for error in data.get("errors", []):
            st.error(error)
