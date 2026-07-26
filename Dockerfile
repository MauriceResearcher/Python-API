FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# curl nur für den HEALTHCHECK unten
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Erst nur requirements.txt kopieren -> Docker-Layer-Cache bleibt beim
# Code-Ändern erhalten, pip install muss nicht bei jedem Build neu laufen.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Nicht als root laufen lassen
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /code/qdrant_db \
    && chown -R appuser:appuser /code
USER appuser

# Standardpfad für die lokale Qdrant-Datenbank im Container.
# In docker-compose.yml wird darüber ein Volume gemountet, damit der
# Index einen Container-Neustart übersteht.
ENV QDRANT_PATH=/code/qdrant_db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
