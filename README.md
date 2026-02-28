# Sistema Multi-Agente RAG Local — Habilitación en Salud (Resolución 3100 de 2019)

Asistente normativo inteligente que responde preguntas y genera checklists citadas por
numeral para el proceso de Habilitación de Servicios de Salud en Colombia, usando una
arquitectura **Agentic RAG local multi-agente** desplegada completamente on-premise.

**Autor:** Pablo Andrés Muñoz Martínez
**Director:** Dr. Juan Diego Pulgarín
**Programa:** Maestría en Inteligencia Artificial — Universidad Autónoma de Occidente
**Año:** 2025

---

## Descripción del Proyecto

Las IPS deben demostrar el cumplimiento de los requisitos del Sistema Único de Habilitación,
pero la normatividad se encuentra dispersa en resoluciones, anexos y guías con formatos
heterogéneos. Este sistema implementa un pipeline RAG local con **siete agentes especialistas**
(uno por módulo de la Resolución 3100) coordinados por un **orquestador central**, sin uso
de APIs externas ni datos sensibles.

### Módulos cubiertos (Resolución 3100 de 2019)

| Agente especialista | Módulo normativo |
|---|---|
| `talento_humano` | Talento Humano |
| `infraestructura` | Infraestructura |
| `dotacion` | Dotación |
| `medicamentos_dispositivos` | Medicamentos y Dispositivos Médicos |
| `procesos_prioritarios` | Procesos Prioritarios |
| `historia_clinica` | Historia Clínica y Registros |
| `interdependencia` | Interdependencia de Servicios |

---

## Arquitectura

```
Consulta del usuario
        │
        ▼
┌─────────────────┐
│   Orquestador   │  ← ruteo semántico + guardrails
└────────┬────────┘
         │
    ┌────┴────┐
    │ Agente  │  (uno de los 7 especialistas)
    │especial.│
    └────┬────┘
         │
┌────────▼────────┐
│  FAISS Index    │  ← índice vectorial por módulo
│  (embeddings)   │
└────────┬────────┘
         │
┌────────▼────────┐
│   LLM Local     │  ← Llama 3 8B / Mistral 7B via Ollama
│   (Ollama)      │
└────────┬────────┘
         │
  Respuesta citada
  (numeral + página + vigencia)
```

---

## Requisitos del Sistema

### Hardware mínimo
- **CPU:** Intel Core 10a gen+ / AMD Ryzen
- **GPU:** NVIDIA con CUDA, mínimo 8 GB VRAM (para modelos cuantizados 4-bit/8-bit)
- **RAM:** 32 GB mínimo
- **Almacenamiento:** 50 GB SSD libre

### Software
- Python 3.10+
- Docker y Docker Compose
- [Ollama](https://ollama.ai) instalado localmente

---

## Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd proyecto_grado_rag
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con los valores correspondientes al entorno local
```

### 3. Instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 4. Descargar modelo en Ollama

```bash
ollama pull llama3:8b
# o alternativamente:
ollama pull mistral:7b
```

### 5. Ejecutar ingesta del corpus normativo

Coloque los documentos PDF de la Resolución 3100 y sus anexos en `data/raw/`.

```bash
python scripts/ingest.py
```

### 6. Levantar con Docker Compose

```bash
docker compose up --build
```

La API estará disponible en `http://localhost:8000`
La interfaz de usuario en `http://localhost:8501`

---

## Estructura del Proyecto

```
proyecto_grado_rag/
├── agents/                     # Orquestador y 7 agentes especialistas
│   ├── orchestrator.py
│   ├── base_specialist.py
│   ├── baseline_mono_agent.py  # Agente mono para comparación
│   ├── guardrails.py
│   ├── prompts.py
│   └── specialists/            # Un agente por módulo Res. 3100
├── api/                        # API REST (FastAPI)
├── artifacts/
│   ├── faiss/                  # Índices vectoriales generados
│   └── eval_runs/              # Resultados de evaluaciones
├── core/                       # Infraestructura transversal
│   ├── config.py
│   ├── embeddings.py
│   ├── llm_client.py
│   └── metadata_store.py
├── data/
│   ├── raw/                    # Documentos normativos originales (no versionados)
│   ├── processed/              # Chunks normalizados
│   └── metadata/               # Metadatos: servicio, numeral, vigencia, página
├── docs/                       # Documentación técnica y decisiones de arquitectura
├── eval/
│   ├── datasets/               # Gold set: 210 preguntas Q&A
│   ├── metrics.py              # Recall@10, MRR, EM, F1, Faithfulness
│   ├── run_eval.py             # Runner offline multi-agente vs mono-agente
│   └── user_eval/              # Evaluación con usuarios (SUS, tiempos, entrevistas)
├── notebooks/                  # Exploración y análisis
├── rag/                        # Pipeline RAG
│   ├── retriever.py
│   ├── reranker.py
│   └── citations.py
├── scripts/
│   ├── ingest.py               # Ingesta y construcción de índices FAISS
│   └── build_goldset.py        # Construcción del gold set de evaluación
├── tests/                      # Pruebas unitarias
├── ui/                         # Interfaz de usuario (Streamlit)
├── main.py
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Evaluación

El sistema se evalúa contra un **gold set de 210 preguntas** (30 por módulo) con las
siguientes métricas:

| Categoría | Métricas |
|---|---|
| Recuperación | Recall@10, MRR |
| Generación | Exact Match, F1, Faithfulness |
| Orquestación | Precisión de ruteo entre agentes |

Para ejecutar la evaluación offline:

```bash
python eval/run_eval.py --config configs/eval_config.json
```

---

## Corpus Normativo

Los documentos utilizados son de **dominio público** y se obtienen del portal oficial del
Ministerio de Salud y Protección Social de Colombia:

- Resolución 3100 de 2019
- Anexos técnicos por servicio
- Guías de verificación oficiales

> Los archivos no están incluidos en el repositorio. Ver instrucciones en `docs/architecture.md`.

---

## Licencia

Este proyecto es de carácter académico. El corpus normativo utilizado es de dominio público.
El código fuente se distribuye bajo licencia MIT.

---

## Referencia

> Muñoz Martínez, P.A. (2025). *Sistema Multi-Agente Basado en RAG Local para la Consulta
> Normativa de Habilitación en Salud (Resolución 3100 de 2019)*. Anteproyecto de Maestría
> en Inteligencia Artificial. Universidad Autónoma de Occidente. Santiago de Cali, Colombia.
