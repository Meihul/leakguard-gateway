FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the code and documents the gateway needs to run are copied in.
# Secrets (.env) are never part of the image: they're injected at container
# runtime by docker-compose's env_file, straight from the host's .env.
COPY gateway/ gateway/
COPY data/ data/

EXPOSE 8000

# Rebuild the RAG index (safe to run every start; it's a full rebuild from
# data/docs) before serving, so a fresh container always has a usable index.
CMD ["sh", "-c", "python -m gateway.rag.ingest && uvicorn gateway.main:app --host 0.0.0.0 --port 8000"]
