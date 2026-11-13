FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY ie/ ie/
COPY serve/ serve/
COPY artifacts/schema.json artifacts/schema.json

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
