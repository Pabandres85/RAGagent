# ── Etapa 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar paquetes C
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Etapa 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copiar paquetes instalados desde el builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar el código fuente (excluye data/raw por .dockerignore)
COPY . .

# Variables de entorno por defecto (sobreescribibles en docker-compose)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
