# Personal AI Research & Knowledge Assistant – Backend
# Requires Postgres (pgvector) for vector store, checkpoint, and store.
FROM python:3.12-slim

WORKDIR /app

# Install system deps if needed (e.g. for unstructured)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
# Create uploads dir
RUN mkdir -p uploads

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
