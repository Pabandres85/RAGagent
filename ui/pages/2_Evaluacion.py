"""
Página Streamlit para visualizar resultados de evaluación offline.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_PATH = BASE_DIR / "artifacts" / "eval_runs" / "latest_eval.json"
SUMMARY_PATH = BASE_DIR / "artifacts" / "eval_runs" / "latest_eval_summary.json"


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* =========================================
           1. FONDOS GENERALES
           ========================================= */
        [data-testid="stAppViewContainer"] {
            background-color: #f8fafc !important;
        }

        /* =========================================
           2. ÁREA PRINCIPAL (LETRAS OSCURAS)
           ========================================= */
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
           4. HERO SECTION
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

        /* =========================================
           5. TARJETAS DE MÉTRICAS Y ACORDEONES
           ========================================= */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
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
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            margin-bottom: 0.5rem;
        }
        
        [data-testid="stExpander"] p, [data-testid="stExpander"] span {
            color: #0f172a !important;
        }
        
        /* Estilizar alertas de errores/warnings */
        [data-testid="stAlert"] * {
             color: inherit !important; /* Permitir que los colores nativos de la alerta operen */
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Evaluación | RAG", page_icon="📊", layout="wide")
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>📊 Resultados de Evaluación Offline</h1>
            <p>Comparativa de desempeño: Sistema Multi-Agente vs Baseline Mono-Agente.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = _load_json(RESULTS_PATH)
    summary = _load_json(SUMMARY_PATH)

    if not results:
        st.warning("No hay resultados de evaluación todavía. Ejecuta `python eval/run_eval.py` primero.")
        return

    if not summary:
        st.info("No existe resumen guardado. Se mostrarán solo los resultados detallados.")
        summary = {}

    st.markdown("### 📈 Métricas de Rendimiento")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Items Evaluados", summary.get("count", len(results)))
    with col2: st.metric("✅ Validez Estructural (Multi)", f"{summary.get('multi_valid_rate', 0.0):.1%}")
    with col3: st.metric("✅ Validez Estructural (Mono)", f"{summary.get('mono_valid_rate', 0.0):.1%}")
    with col4: st.metric("🎯 Ruteo Any-Hit Específicos", f"{summary.get('routing_hit_rate_any_specific', 0.0):.1%}")

    col5, col6, col7, col8 = st.columns(4)
    with col5: st.metric("🥇 F1-Score (Multi)", f"{summary.get('multi_f1_avg', 0.0):.3f}")
    with col6: st.metric("🥈 F1-Score (Mono)", f"{summary.get('mono_f1_avg', 0.0):.3f}")
    with col7: st.metric("🎯 Ruteo Top-1 Específicos", f"{summary.get('routing_accuracy_top1_specific', 0.0):.1%}")
    with col8: st.metric("🥇 Exact Match (Multi)", f"{summary.get('multi_em_avg', 0.0):.3f}")

    st.divider()

    st.markdown("### 🔍 Inspección Detallada por Pregunta")
    only_failures = st.toggle("Mostrar solo casos donde el Multi-Agente falló o fue superado", value=False)

    filtered = results
    if only_failures:
        filtered = [
            item
            for item in results
            if (not item.get("multi_valid"))
            or item.get("multi_f1", 0.0) < item.get("mono_f1", 0.0)
        ]

    st.caption(f"Mostrando {len(filtered)} de {len(results)} resultados.")

    for item in filtered[:50]:
        # Decorador visual para el título del expander
        icon = "🟢" if item.get('multi_valid') and item.get('multi_f1', 0) >= item.get('mono_f1', 0) else "🔴"
        title = (
            f"{icon} [{item.get('index')}] "
            f"Esperado: {item.get('module_expected')} ➔ Predicho: {item.get('module_predicted')}"
        )
        
        with st.expander(title):
            st.markdown(f"**Pregunta del Gold Set:**")
            st.info(item.get('question', ''))
            
            # Contexto de Ruteo
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"**Esperado:** `{item.get('module_expected')}`")
            with col_r2:
                predicted_all = item.get("module_predicted_all", [])
                st.markdown(f"**Predicho (Todos):** `{', '.join(predicted_all) if predicted_all else item.get('module_predicted')}`")

            st.markdown("---")

            # Comparativa Lado a Lado
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🧠 Multi-Agente")
                st.caption(f"Válido: `{item.get('multi_valid')}` | EM: `{item.get('multi_em', 0.0):.1f}` | F1: `{item.get('multi_f1', 0.0):.3f}`")
                
                if item.get("multi_answer"):
                    st.write(item.get("multi_answer"))
                else:
                    st.write("*(Respuesta vacía o inválida)*")
                
                if item.get("multi_errors"):
                    for e in item["multi_errors"]: st.error(e)
                if item.get("multi_warnings"):
                    for w in item["multi_warnings"]: st.warning(w)

            with c2:
                st.markdown("#### 🤖 Baseline Mono-Agente")
                st.caption(f"Válido: `{item.get('mono_valid')}` | EM: `{item.get('mono_em', 0.0):.1f}` | F1: `{item.get('mono_f1', 0.0):.3f}`")
                
                if item.get("mono_answer"):
                    st.write(item.get("mono_answer"))
                else:
                    st.write("*(Respuesta vacía o inválida)*")
                
                if item.get("mono_errors"):
                    for e in item["mono_errors"]: st.error(e)
                if item.get("mono_warnings"):
                    for w in item["mono_warnings"]: st.warning(w)

            st.markdown("---")
            st.markdown("**Respuesta de Referencia (Gold Truth):**")
            st.success(item.get('reference_answer', ''))


if __name__ == "__main__":
    main()