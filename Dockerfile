# Dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .env .env
COPY celery_app.py celery_app.py
COPY tasks.py tasks.py
COPY fetcher/ ./fetcher/
COPY bot/ ./bot/
COPY ml/ ./ml/
COPY shared/ ./shared/
COPY data/ ./data/
COPY models/ ./models

CMD ["celery", "-A", "celery_app", "worker", "-l", "INFO"]